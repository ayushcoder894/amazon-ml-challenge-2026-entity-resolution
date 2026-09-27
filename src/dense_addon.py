"""Dense add-on (A100 E1 bi-encoder, e5-base): the pairs the dense retriever finds that are NOT in the v12 candidate
set get their own matcher; decisions are taken on the union with the v12 final probabilities (argmax owner per record,
per-entity expected-F0.5 top-k - the v12 rule).

  feat <split>      pair features + dense + v12-context features of the new pairs -> DN/<split>_nfeat_<country>.parquet
  fit               2-fold honest (v12 folds, harder split): v12 alone vs v12 + dense pairs, per country
  test <name>       both fold models averaged on the test new pairs -> output/matching_results_<name>.tsv
                    (countries absent from train keep the base submission's rows)

New pairs = dense forward rank <= FWD or reverse rank <= REV (env), not already v12 candidates.
Train harder split: kept entities only, S1 name frequencies recomputed on the kept entities (as make_drop_folds).
"""
import gc
import glob
import json
import os
import sys
import time

import numpy as np
import polars as pl
import xgboost as xgb

import decide
import evaluate as ev
import features as F

D = "C:/ML/Challenges/amazon/work/data/"
OUT = "C:/ML/Challenges/amazon/work/output/"
DN = os.environ.get("DN", "D:/amazon_er/dense/")
BLK = "D:/amazon_er/data/"
FWD, REV = int(os.environ.get("FWD", 10)), int(os.environ.get("REV", 1))
COUNTRIES = ("India", "US")
UNSEEN = ("France",)  # test-only countries: stage-1 base + dense_france model, rule ef(0, 4)
FREQ = ["n_css_s1_1", "n_css_s1_2", "n_soft_s1_1", "n_soft_s1_2"]
NON_FEAT = {"rid", "oid", "y", "fold", "country"}
XP = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda",
          grow_policy="lossguide", max_leaves=127, max_depth=0, learning_rate=0.06, subsample=0.8,
          colsample_bytree=0.8, min_child_weight=20, reg_lambda=1.0, max_bin=128)
ROUNDS = int(os.environ.get("ROUNDS", 600))
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def base_k(split):
    """v12 final candidate set (rid, oid, country, p1, pf) + dense cosine of its pairs."""
    f = "train_oof_final_v12_twin" if split == "train" else "test_final_v12_ce_cectx_twin"
    K = pl.read_parquet(f"{D}{f}.parquet", columns=["rid", "oid", "country", "p1", "p2", "pf"])
    cos = pl.read_parquet(f"{DN}dense_{split}_cos.parquet")
    return K.join(cos, on=["rid", "oid"], how="left")


def new_pairs(split):
    f = f"{DN}dense_train_new_small.parquet" if split == "train" else f"{DN}dense_test_new.parquet"
    n = pl.read_parquet(f).filter((pl.col("fwd_rank") <= FWD) | (pl.col("rev_rank") <= REV))
    K = pl.read_parquet(f"{D}{'train_oof_final_v12_twin' if split == 'train' else 'test_final_v12_ce_cectx_twin'}.parquet",
                        columns=["rid", "oid"])
    return n.join(K, on=["rid", "oid"], how="anti")


def context(N, K):
    """N: new pairs (rid, oid, cos, ...); K: v12 pairs (rid, oid, pf, p1, dense_cos)."""
    c = pl.col("u")
    u = pl.concat([K.select("rid", "oid", u=pl.col("dense_cos").cast(pl.Float32), inK=pl.lit(1, pl.Int8)),
                   N.select("rid", "oid", u=pl.col("cos").cast(pl.Float32), inK=pl.lit(0, pl.Int8))])
    u = u.with_columns(
        u_rank1=c.rank("ordinal", descending=True).over("rid").cast(pl.UInt16), u_gap1=c.max().over("rid") - c,
        u_n1=pl.len().over("rid").cast(pl.UInt16),
        u_rank2=c.rank("ordinal", descending=True).over("oid").cast(pl.UInt16), u_gap2=c.max().over("oid") - c,
        u_n2=pl.len().over("oid").cast(pl.UInt16),
    ).filter(pl.col("inK") == 0).drop("inK", "u")
    pf = pl.col("pf")
    kr = K.group_by("rid").agg(k_pfmax1=pf.max(), k_pfsum1=pf.sum(), k_nconf1=(pf > 0.5).sum().cast(pl.UInt16),
                               k_n1=pl.len().cast(pl.UInt16), k_cosmax1=pl.col("dense_cos").max(),
                               k_cosconf1=pl.col("dense_cos").filter(pf > 0.5).min(),
                               k_p1max1=pl.col("p1").max())
    ko = K.group_by("oid").agg(k_pfmax2=pf.max(), k_pfsum2=pf.sum(), k_n2=pl.len().cast(pl.UInt16),
                               k_cosmax2=pl.col("dense_cos").max(),
                               k_cos_at_best2=pl.col("dense_cos").sort_by("pf").last())
    x = N.join(u, on=["rid", "oid"], how="left").join(kr, on="rid", how="left").join(ko, on="oid", how="left")
    return x.with_columns(d_cos_kmax2=pl.col("cos") - pl.col("k_cosmax2"), d_cos_best2=pl.col("cos") - pl.col("k_cos_at_best2"),
                          d_cos_conf1=pl.col("cos") - pl.col("k_cosconf1"), d_cos_kmax1=pl.col("cos") - pl.col("k_cosmax1"))


def feat(split, chunk=300_000):
    """Memory-lean: context + blocking lookups first (then freed), records loaded once per country, small chunks."""
    os.makedirs(DN, exist_ok=True)
    N = new_pairs(split).join(pl.read_parquet(f"{D}{split}_s1_norm.parquet", columns=["rid", "country"]), on="rid")
    K = base_k(split)
    cl = COUNTRIES + (UNSEEN if split == "test" and os.environ.get("FRANCE", "1") == "1" else ())
    for c in cl:
        Nc = N.filter(pl.col("country") == c).drop("country").sort("rid", "oid")
        bl = (pl.scan_parquet(f"{BLK}{split}_cand.parquet").select("rid", "oid", "score", "sn", "sa", "sx", "sz", "nk", "r1", "r2")
              .join(Nc.lazy().select("rid", "oid"), on=["rid", "oid"], how="semi").collect())
        Kc = K.filter(pl.col("country") == c)
        if c in UNSEEN:  # unseen country: decided from stage 1, so its context comes from p1 (as in dense_france)
            Kc = Kc.with_columns(pf=pl.col("p1"))
        x = (context(Nc, Kc).join(bl, on=["rid", "oid"], how="left")
             .with_columns(in_blk=pl.col("score").is_not_null().cast(pl.UInt8),
                           **{k: pl.col(k).fill_null(0.0) for k in ("score", "sn", "sa", "sx", "sz")},
                           nk=pl.col("nk").fill_null(0), r1=pl.col("r1").fill_null(999), r2=pl.col("r2").fill_null(999)))
        x.with_columns(pl.col(pl.Float64).cast(pl.Float32)).write_parquet(f"{DN}_ctx_{split}_{c}.parquet")
        log(f"[{split} {c}] {Nc.height:,} new pairs; context + blocking columns ready")
        del Nc, bl, x
        gc.collect()
    del N, K
    gc.collect()
    keep = pl.read_parquet(f"{D}keep_v11.parquet") if split == "train" else None
    for c in cl:
        s1, o = F.load_records(split, c)
        if keep is not None:  # harder split: S1 name frequencies over the kept entities only (as make_drop_folds)
            ks = s1.join(keep, on="rid", how="semi")
            for key in ("css", "soft"):
                cnt = ks.group_by(key).agg(pl.len().cast(pl.UInt32).alias("_n"))
                s1 = s1.join(cnt, on=key, how="left").with_columns(pl.col("_n").fill_null(0).alias(f"n_{key}_s1")).drop("_n")
                o = o.join(cnt, on=key, how="left").with_columns(pl.col("_n").fill_null(0).alias(f"n_{key}_s1")).drop("_n")
            del ks
        tabs = F.token_tables(s1, o)
        pairs = pl.read_parquet(f"{DN}_ctx_{split}_{c}.parquet", columns=["rid", "oid"])
        tmp = f"{DN}_parts_{split}_{c}"
        os.makedirs(tmp, exist_ok=True)
        for j, i in enumerate(range(0, pairs.height, chunk)):
            F.pair_features(pairs.slice(i, chunk), s1, o, tabs).with_columns(
                pl.col(pl.Float64).cast(pl.Float32)).write_parquet(f"{tmp}/p{j:04d}.parquet")
            if j % 5 == 0:
                log(f"   [{c}] chunk {j + 1}/{(pairs.height + chunk - 1) // chunk} done")
        del s1, o, tabs, pairs
        gc.collect()
        (pl.scan_parquet(f"{DN}_ctx_{split}_{c}.parquet")
         .join(pl.scan_parquet(f"{tmp}/p*.parquet"), on=["rid", "oid"], how="left")
         .with_columns(country=pl.lit(c)).sink_parquet(f"{DN}{split}_nfeat_{c}.parquet"))
        n = pl.scan_parquet(f"{DN}{split}_nfeat_{c}.parquet").select(pl.len()).collect().item()
        log(f"[{split} {c}] wrote {n:,} rows")
        for f in glob.glob(f"{tmp}/p*.parquet"):
            os.remove(f)
        os.rmdir(tmp)
        os.remove(f"{DN}_ctx_{split}_{c}.parquet")
        gc.collect()


def load_nfeat(split):
    x = pl.concat([pl.read_parquet(f"{DN}{split}_nfeat_{c}.parquet") for c in COUNTRIES], how="diagonal_relaxed")
    return x.with_columns(is_india=(pl.col("country") == "India").cast(pl.UInt8))


def feat_cols(x):
    return [c for c in x.columns if c not in NON_FEAT and x[c].dtype != pl.String]


def fit():
    x = load_nfeat("train")
    x = x.with_columns(fold=(pl.col("rid").hash(seed=11) % 2).cast(pl.Int8))  # the v12 folds (train3.fold_expr)
    cols = feat_cols(x)
    x = x.select("rid", "oid", "country", "fold", "y", *[pl.col(c).cast(pl.Float32) for c in cols])  # float32 numpy, no float64 copy
    gc.collect()
    log(f"train new pairs {x.height:,}, positives {x['y'].sum():,}, {len(cols)} features")
    pn = np.zeros(x.height, dtype=np.float32)
    os.makedirs(f"{DN}models", exist_ok=True)
    for f in (0, 1):
        tr = (x["fold"] == f).to_numpy()
        dm = xgb.QuantileDMatrix(x.filter(pl.Series(tr)).select(cols).to_numpy(), label=x.filter(pl.Series(tr))["y"].to_numpy(), max_bin=XP["max_bin"])
        bst = xgb.train(XP, dm, ROUNDS)
        bst.save_model(f"{DN}models/addon_fold{f}.json")
        te = ~tr
        pn[te] = bst.predict(xgb.DMatrix(x.filter(pl.Series(te)).select(cols).to_numpy()))
        log(f"fold {f} model trained; scored {te.sum():,} held-out pairs")
        del dm, bst
        gc.collect()
    json.dump(cols, open(f"{DN}models/addon_cols.json", "w"))
    x = x.with_columns(pn=pl.Series(pn))
    x.select("rid", "oid", "country", "fold", "y", "cos", "fwd_rank", "rev_rank", "pn").write_parquet(f"{DN}train_pn.parquet")
    evaluate(x.select("rid", "oid", "y", "pn", "fwd_rank", "rev_rank"))


def evaluate(N):
    from sklearn.metrics import roc_auc_score
    log(f"AUC of the add-on model on new pairs: {roc_auc_score(N['y'].to_numpy(), N['pn'].to_numpy()):.4f}")
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    s1 = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).join(keep, on="rid", how="semi")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").join(keep, on="rid", how="semi").select("rid", "oid")
    K = pl.read_parquet(f"{D}train_oof_final_v12_twin.parquet", columns=["rid", "oid", "pf"])

    def score(U):
        pred = decide.ef_predict(U.select("rid", "oid", pl.col("p").alias("p2")), "p2", 0.0, 1.0, 0.05)
        out = {}
        for c in COUNTRIES:
            r = s1.filter(pl.col("country") == c)["rid"]
            out[c] = ev.macro_f05(pred.join(pl.DataFrame({"rid": r}), on="rid", how="semi"),
                                  gt.join(pl.DataFrame({"rid": r}), on="rid", how="semi"), r)
        return out, pred.height
    b, nb = score(K.select("rid", "oid", p="pf"))
    log(f"v12 alone             India {b['India']:.5f}  US {b['US']:.5f}  ({nb:,} predicted pairs)")
    for name, flt in [("+ dense fwd5|rev1", (pl.col("fwd_rank") <= 5) | (pl.col("rev_rank") <= 1)),
                      (f"+ dense fwd{FWD}|rev{REV}", pl.lit(True))]:
        U = pl.concat([K.select("rid", "oid", p="pf"), N.filter(flt).select("rid", "oid", p="pn")])
        s, ns = score(U)
        log(f"v12 {name:<18s} India {s['India']:.5f} ({s['India'] - b['India']:+.5f})  US {s['US']:.5f} "
            f"({s['US'] - b['US']:+.5f})  ({ns:,} predicted pairs)")


def union_frame(K, N):
    """All pairs (v12 + new) with their own probability p and competition features over the union."""
    v2 = JV == "2"
    ek = [pl.col("p2")] if v2 else []
    en = [pl.col("fwd_rank").cast(pl.Float32), pl.col("rev_rank").cast(pl.Float32)] if v2 else []
    U = pl.concat([K.select("rid", "oid", p=pl.col("pf"), p1=pl.col("p1"), cosu=pl.col("dense_cos"), is_new=pl.lit(0, pl.UInt8),
                            *ek, *([pl.lit(None, pl.Float32).alias("fwd_rank"), pl.lit(None, pl.Float32).alias("rev_rank")] if v2 else [])),
                   N.select("rid", "oid", p=pl.col("pn"), p1=pl.lit(None, pl.Float32), cosu=pl.col("cos"), is_new=pl.lit(1, pl.UInt8),
                            *([pl.lit(None, pl.Float32).alias("p2")] if v2 else []), *en,
                            *[pl.col(c) for c in ("ce_new", "cectx_new") if c in N.columns])],
                  how="diagonal_relaxed")
    if v2:
        nw = pl.col("is_new") == 1
        U = U.with_columns(n_new1=nw.sum().over("rid").cast(pl.UInt16), n_new2=nw.sum().over("oid").cast(pl.UInt16),
                           new_max1=pl.col("p").filter(nw).max().over("rid"), old_max1=pl.col("p").filter(~nw).max().over("rid"),
                           p2_max2=pl.col("p2").max().over("oid"))
    p, c = pl.col("p"), pl.col("cosu")
    U = U.with_columns(
        p_rank2=p.rank("ordinal", descending=True).over("oid").cast(pl.UInt16), p_max2=p.max().over("oid"),
        p_sum2=p.sum().over("oid"), n2=pl.len().over("oid").cast(pl.UInt16),
        p_second2=p.sort(descending=True).slice(1, 1).first().over("oid"),
        p_rank1=p.rank("ordinal", descending=True).over("rid").cast(pl.UInt16), p_max1=p.max().over("rid"),
        p_sum1=p.sum().over("rid"), n1=pl.len().over("rid").cast(pl.UInt16), n_conf1=(p > 0.5).sum().over("rid").cast(pl.UInt16),
        c_rank2=c.rank("ordinal", descending=True).over("oid").cast(pl.UInt16), c_gap2=c.max().over("oid") - c,
        c_rank1=c.rank("ordinal", descending=True).over("rid").cast(pl.UInt16), c_gap1=c.max().over("rid") - c,
        new_max2=p.filter(pl.col("is_new") == 1).max().over("oid"), old_max2=p.filter(pl.col("is_new") == 0).max().over("oid"))
    return U.with_columns(p_margin2=pl.when(pl.col("p_rank2") == 1).then(p - pl.col("p_second2").fill_null(0)).otherwise(p - pl.col("p_max2")),
                          p_other1=pl.col("p_sum1") - p)


JCOLS = ["p", "p1", "cosu", "is_new", "p_rank2", "p_max2", "p_sum2", "n2", "p_second2", "p_rank1", "p_max1", "p_sum1", "n1",
         "n_conf1", "c_rank2", "c_gap2", "c_rank1", "c_gap1", "new_max2", "old_max2", "p_margin2", "p_other1", "is_india"]
JP = dict(XP, max_leaves=63, learning_rate=0.05)
JV = os.environ.get("JOINT_V", "1")  # "2": richer union features, models joint2_fold*.json
if JV == "2":
    JCOLS = JCOLS + ["p2", "fwd_rank", "rev_rank", "n_new1", "n_new2", "new_max1", "old_max1", "p2_max2"]
RAW = ["n_ratio", "n_tset", "n_partial", "c_jw", "c_tset", "a_tset", "a_ratio", "w_tset", "num1_lev", "first_num_eq",
       "num_prefix", "n_inter", "core_eq", "css_eq", "legal_conflict", "soft_eq", "addr_empty2", "core_cov1", "core_cov2",
       "words_cov1", "words_cov2", "nums_cov1", "n_css_s1_1", "indic", "src"]
if JV in ("3", "4", "5", "6", "7"):  # + cross-encoder score(s) of the uncertain new pairs (ce_dense.py; "4" adds the context CE)
    JCOLS = JCOLS + ["ce_new"] + (["cectx_new"] if JV == "4" else []) + (["ceu"] if JV in ("5", "6", "7") else [])
SURF = ["raw_eq", "raw_sub", "raw_tok", "raw_ratio", "raw_tset", "raw_tok_n2", "raw_tok_uniq", "raw_eq_n2", "raw_miss1", "raw_extra2"]
if JV in ("6", "7"):  # + raw pair evidence for every pair of the union
    JCOLS = JCOLS + RAW
if JV == "7":  # + surface form of the ORIGINAL names (legal-form spelling, punctuation) and its uniqueness per record
    JCOLS = JCOLS + SURF
if JV == "8":  # v5 + e5-large cross-encoder (A100, out-of-fold on train / mean of both folds on test)
    JCOLS = JCOLS + ["ce_new", "ceu", "e5l"]
if JV == "9":  # v8 + bge-m3 cross-encoder (A100)
    JCOLS = JCOLS + ["ce_new", "ceu", "e5l", "bgm3"]
JNAME = {"1": "joint", "2": "joint2", "3": "joint3", "4": "joint4", "5": "joint5", "6": "joint6", "7": "joint7", "8": "joint8",
         "9": "joint9"}[JV]


def surf_feats(split, U):
    """Raw (un-normalised, lower-cased) name comparisons for the union pairs, cached per split."""
    path = f"{DN}surf_{split}.parquet"
    if os.path.exists(path):
        return pl.read_parquet(path)
    from rapidfuzz import fuzz
    from rapidfuzz.process import cpdist
    s1 = pl.read_parquet(f"{D}{split}_source1.parquet", columns=["business_name"]).with_row_index("rid").select(
        pl.col("rid").cast(pl.UInt32), sn=pl.col("business_name").fill_null("").str.to_lowercase().str.strip_chars())
    o = pl.concat([pl.read_parquet(f"{D}{split}_source{k}.parquet", columns=["business_name"]) for k in (2, 3)]).with_row_index(
        "oid").select(pl.col("oid").cast(pl.UInt32), rn=pl.col("business_name").fill_null("").str.to_lowercase().str.strip_chars())
    keys = U.select("rid", "oid")
    parts = []
    st, rt = pl.col("sn").str.split(" "), pl.col("rn").str.split(" ")
    for i in range(0, keys.height, 2_000_000):  # chunked: strings of 20M pairs do not fit at once
        x = keys.slice(i, 2_000_000).join(s1, on="rid", how="left").join(o, on="oid", how="left")
        x = x.with_columns(raw_eq=(pl.col("sn") == pl.col("rn")).cast(pl.UInt8),
                           raw_sub=pl.col("rn").str.contains(pl.col("sn"), literal=True).cast(pl.UInt8),
                           raw_tok=(st.list.set_difference(rt).list.len() == 0).cast(pl.UInt8),
                           raw_miss1=st.list.set_difference(rt).list.len().cast(pl.UInt8),
                           raw_extra2=rt.list.set_difference(st).list.len().cast(pl.UInt8))
        sn, rn = x["sn"].to_list(), x["rn"].to_list()
        x = x.with_columns(raw_ratio=pl.Series(cpdist(sn, rn, scorer=fuzz.ratio, workers=-1, dtype=np.float32)),
                           raw_tset=pl.Series(cpdist(sn, rn, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)))
        parts.append(x.drop("sn", "rn"))
        del x, sn, rn
        gc.collect()
    del s1, o
    x = pl.concat(parts)
    del parts
    x = x.with_columns(raw_tok_n2=pl.col("raw_tok").cast(pl.UInt16).sum().over("oid"),
                       raw_eq_n2=pl.col("raw_eq").cast(pl.UInt16).sum().over("oid")).with_columns(
        raw_tok_uniq=((pl.col("raw_tok") == 1) & (pl.col("raw_tok_n2") == 1)).cast(pl.UInt8))
    out = x.select("rid", "oid", *[pl.col(c).cast(pl.Float32) for c in SURF])
    out.write_parquet(path)
    return out


def raw_feats(split):
    """Raw pair features (no competition features) for v12 pairs (stage feature files) and new pairs (nfeat), cached."""
    path = f"{DN}raw_{split}.parquet"
    if os.path.exists(path):
        return pl.read_parquet(path)
    if split == "train":
        kf = [f"D:/amazon_er/archive/data/train_feat_v11_fold{f}.parquet" for f in (0, 1)]
        keys = pl.read_parquet(f"{D}train_oof_final_v12_twin.parquet", columns=["rid", "oid"])
    else:
        kf = [f"D:/amazon_er/data/test_feat_v10_{c}.parquet" for c in COUNTRIES]
        keys = pl.read_parquet(f"{D}test_final_v12_ce_cectx_twin.parquet", columns=["rid", "oid"])
    Kr = (pl.concat([pl.scan_parquet(f).select("rid", "oid", *RAW) for f in kf])
          .join(keys.lazy(), on=["rid", "oid"], how="semi").collect())
    Nr = pl.concat([pl.read_parquet(f"{DN}{split}_nfeat_{c}.parquet", columns=["rid", "oid", *RAW]) for c in COUNTRIES])
    out = pl.concat([Kr.with_columns([pl.col(c).cast(pl.Float32) for c in RAW]),
                     Nr.with_columns([pl.col(c).cast(pl.Float32) for c in RAW])]).unique(["rid", "oid"])
    out.write_parquet(path)
    return out


def with_ceu(U, split):
    """v5: union-band cross-encoder score (ce_union.py) for v12 and new pairs; v6 adds the raw pair features."""
    if JV not in ("5", "6", "7", "8", "9"):
        return U
    U = U.join(pl.read_parquet(f"{DN}ceu_{split}.parquet"), on=["rid", "oid"], how="left")
    if JV in ("8", "9"):
        e = pl.read_parquet(f"D:/amazon_er/gpu3_results/ce3_e5l_inc_{split}.parquet").rename({"s": "e5l"})
        U = U.join(e, on=["rid", "oid"], how="left")
    if JV == "9":
        g = pl.read_parquet(f"D:/amazon_er/gpu3_results/ce3_bgem3_inc_{split}.parquet").rename({"s": "bgm3"})
        U = U.join(g, on=["rid", "oid"], how="left")
    if JV in ("6", "7"):
        U = U.join(raw_feats(split), on=["rid", "oid"], how="left")
    if JV == "7":
        U = U.join(surf_feats(split, U), on=["rid", "oid"], how="left")
    return U


def with_ce(N, split):
    if JV not in ("3", "4", "5", "6", "7", "8", "9"):
        return N
    N = N.join(pl.read_parquet(f"{DN}ce_{split}_new.parquet"), on=["rid", "oid"], how="left")
    if JV == "4":
        N = N.join(pl.read_parquet(f"{DN}cectx_{split}_new.parquet"), on=["rid", "oid"], how="left")
    return N


def joint():
    """Second pass over the union: every pair re-scored knowing the new competitors (2-fold honest)."""
    K = base_k("train")
    N = with_ce(pl.read_parquet(f"{DN}train_pn.parquet"), "train")
    U = with_ceu(union_frame(K, N), "train").join(pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]), on="rid")
    U = U.with_columns(is_india=(pl.col("country") == "India").cast(pl.UInt8), fold=(pl.col("rid").hash(seed=11) % 2).cast(pl.Int8))
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").select("rid", "oid").with_columns(y=pl.lit(1, pl.Int8))
    U = U.join(gt, on=["rid", "oid"], how="left").with_columns(pl.col("y").fill_null(0))
    # memory-lean: one float32 feature matrix, then the table keeps only what the evaluation needs
    X = U.select([pl.col(c).cast(pl.Float32) for c in JCOLS]).to_numpy()
    U = U.select("rid", "oid", "country", "is_new", "p", "y", "fold")
    gc.collect()
    yv, fv = U["y"].to_numpy(), U["fold"].to_numpy()
    pj = np.zeros(U.height, dtype=np.float32)
    jp = dict(JP, max_leaves=int(os.environ.get("JLEAVES", JP["max_leaves"])))
    groups = [("", np.ones(U.height, bool))] if os.environ.get("JPER") != "1" else [
        (f"_{c}", (U["country"] == c).to_numpy()) for c in COUNTRIES]
    for f in (0, 1):
        for gname, gm in groups:
            tr = np.flatnonzero((fv == f) & gm)
            te = np.flatnonzero((fv != f) & gm)
            dm = xgb.QuantileDMatrix(X[tr], label=yv[tr], max_bin=jp["max_bin"])
            bst = xgb.train(jp, dm, int(os.environ.get("JROUNDS", 400)))
            del dm
            gc.collect()
            bst.save_model(f"{DN}models/{JNAME}{os.environ.get('JSUFFIX', '')}{gname}_fold{f}.json")
            for i in range(0, len(te), 2_000_000):
                pj[te[i:i + 2_000_000]] = bst.predict(xgb.DMatrix(X[te[i:i + 2_000_000]]))
            del bst
            gc.collect()
        log(f"joint fold {f} done")
    del X
    gc.collect()
    U = U.with_columns(pj=pl.Series(pj))
    U.select("rid", "oid", "country", "is_new", "p", "pj", "y").write_parquet(f"{DN}train_{JNAME}{os.environ.get('JSUFFIX', '')}.parquet")
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    s1 = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).join(keep, on="rid", how="semi")
    gtk = gt.join(keep, on="rid", how="semi").select("rid", "oid")
    for col in ("p", "pj"):
        pred = decide.ef_predict(U.select("rid", "oid", pl.col(col).alias("p2")), "p2", 0.0, 1.0, 0.05)
        sc = {c: ev.macro_f05(pred.join(s1.filter(pl.col("country") == c), on="rid", how="semi"),
                              gtk.join(s1.filter(pl.col("country") == c), on="rid", how="semi"),
                              s1.filter(pl.col("country") == c)["rid"]) for c in COUNTRIES}
        log(f"union decided on {col:<3s}: India {sc['India']:.5f}  US {sc['US']:.5f}  ({pred.height:,} pairs)")


def test(name, base="v12_FRp1"):
    x = load_nfeat("test")
    cols = json.load(open(f"{DN}models/addon_cols.json"))
    X = x.select(cols).to_numpy().astype(np.float32)
    pn = np.mean([xgb.Booster(model_file=f"{DN}models/addon_fold{f}.json").predict(xgb.DMatrix(X)) for f in (0, 1)], axis=0)
    x = x.with_columns(pn=pl.Series(pn))
    x.select("rid", "oid", "country", "cos", "fwd_rank", "rev_rank", "pn").write_parquet(f"{DN}test_pn.parquet")
    K = base_k("test").filter(pl.col("country").is_in(COUNTRIES))
    if os.environ.get("JOINT") == "1":  # second pass over the union (models from `joint`)
        U = (with_ceu(union_frame(K, with_ce(x.select("rid", "oid", "pn", "cos", "fwd_rank", "rev_rank"), "test")), "test")
             .join(pl.read_parquet(f"{D}test_s1_norm.parquet", columns=["rid", "country"]), on="rid")
             .with_columns(is_india=(pl.col("country") == "India").cast(pl.UInt8)))
        XU = U.select([pl.col(c).cast(pl.Float32) for c in JCOLS]).to_numpy()
        pj = np.mean([xgb.Booster(model_file=f"{DN}models/{JNAME}_fold{f}.json").predict(xgb.DMatrix(XU)) for f in (0, 1)], axis=0)
        U = U.select("rid", "oid").with_columns(p=pl.Series(pj))
        del XU
    else:
        U = pl.concat([K.select("rid", "oid", p="pf"), x.select("rid", "oid", p="pn")])
    U.select("rid", "oid", "p").write_parquet(f"{DN}test_union_p.parquet")  # final India/US probabilities (for variants)
    best = decide.ef_predict(U.select("rid", "oid", pl.col("p").alias("p2")), "p2", 0.0, 1.0, 0.05)
    log(f"test: {x.height:,} new pairs; predicted pairs India+US {best.height:,}; from the add-on "
        f"{best.join(x, on=['rid', 'oid'], how='semi').height:,}")
    done_c = list(COUNTRIES)
    if os.environ.get("FRANCE", "1") == "1":
        xf = pl.concat([pl.read_parquet(f"{DN}test_nfeat_{c}.parquet") for c in UNSEEN], how="diagonal_relaxed")
        fcols = json.load(open(f"{DN}models/france_cols.json"))
        pf_ = xgb.Booster(model_file=f"{DN}models/france_addon.json").predict(
            xgb.DMatrix(xf.select([pl.col(c).cast(pl.Float32) for c in fcols]).to_numpy()))
        xf = xf.select("rid", "oid").with_columns(pn=pl.Series(pf_))
        Kf = base_k("test").filter(pl.col("country").is_in(UNSEEN))
        Uf = pl.concat([Kf.select("rid", "oid", p="p1"), xf.select("rid", "oid", p="pn")])
        bf = decide.ef_predict(Uf.select("rid", "oid", pl.col("p").alias("p2")), "p2", 0.0, 4.0, 0.05)
        base_f = decide.ef_predict(Kf.select("rid", "oid", pl.col("p1").alias("p2")), "p2", 0.0, 4.0, 0.05)
        log(f"France: {xf.height:,} new pairs; predicted {bf.height:,} (stage-1 alone {base_f.height:,}); from the add-on "
            f"{bf.join(xf, on=['rid', 'oid'], how='semi').height:,}")
        best = pl.concat([best, bf])
        x = pl.concat([x.select("rid", "oid", "pn"), xf], how="vertical_relaxed")
        done_c += list(UNSEEN)
    s1 = pl.read_parquet(D + "test_s1_norm.parquet", columns=["rid", "entity_id", "country"])
    o = pl.concat([pl.read_parquet(D + f"test_s{k}_norm.parquet", columns=["entity_id"]) for k in (2, 3)]).with_row_index("oid")
    ids = s1.filter(pl.col("country").is_in(done_c))["entity_id"]

    def to_ids(pairs, col):
        return (pairs.join(s1.select("rid", s="entity_id"), on="rid").join(o.select("oid", m="entity_id"), on="oid")
                .group_by("s").agg(pl.col("m").sort().str.join(",")).rename({"s": "source1_entity_id", "m": col}))
    rd = dict(separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
    bdf = pl.read_csv(f"{OUT}matching_results_{base}.tsv", **rd)
    new = (bdf.join(to_ids(best, "new"), on="source1_entity_id", how="left")
           .with_columns(matched_entity_ids=pl.when(pl.col("source1_entity_id").is_in(ids.implode()))
                         .then(pl.col("new").fill_null("")).otherwise(pl.col("matched_entity_ids"))).drop("new"))
    new.write_csv(f"{OUT}matching_results_{name}.tsv", separator="\t", quote_style="never")
    # Candidates are the full pool actually scored by the final matcher. A cutoff
    # applied only at export time is not a pre-inference blocking/filtering stage.
    cdf = pl.read_csv(f"{OUT}candidate_pairs_{base}.tsv", **rd)
    add = to_ids(x.select("rid", "oid").unique(), "add")
    cdf = (cdf.join(add, on="source1_entity_id", how="left")
           .with_columns(candidate_entity_ids=pl.when(pl.col("add").is_null()).then(pl.col("candidate_entity_ids"))
                         .when(pl.col("candidate_entity_ids") == "").then(pl.col("add"))
                         .otherwise(pl.col("candidate_entity_ids") + "," + pl.col("add"))).drop("add"))
    cdf = cdf.with_columns(candidate_entity_ids=pl.col("candidate_entity_ids").str.split(",").list.unique()
                          .list.eval(pl.element().filter(pl.element() != "")).list.sort().list.join(","))
    cdf.write_csv(f"{OUT}candidate_pairs_{name}.tsv", separator="\t", quote_style="never")
    changed = new.join(bdf, on="source1_entity_id").filter(pl.col("matched_entity_ids") != pl.col("matched_entity_ids_right")).height
    log(f"wrote matching_results_{name}.tsv ({new.height:,} rows, {changed:,} entities changed) and candidate_pairs_{name}.tsv")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "feat":
        feat(sys.argv[2])
    elif cmd == "fit":
        fit()
    elif cmd == "joint":
        joint()
    elif cmd == "surf":  # precompute the raw-surface features of the union pairs (separate process: RAM)
        sp = sys.argv[2]
        K = base_k(sp)
        if sp == "test":
            K = K.filter(pl.col("country").is_in(COUNTRIES))
        N = pl.read_parquet(f"{DN}{sp}_pn.parquet", columns=["rid", "oid"])
        surf_feats(sp, pl.concat([K.select("rid", "oid"), N]).unique())
    elif cmd == "test":
        test(sys.argv[2])
