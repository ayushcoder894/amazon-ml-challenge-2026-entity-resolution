"""Self-training for an unseen country (France), validated on the proxy (India / US treated as unseen, true labels).

Base  = France recipe on the target: 50/50 blend of (stage-1 | dense add-on) probability and the joint pass, ef(0, 4).
Labels= pseudo: accepted by the base with pb >= POS -> 1, rejected with pb <= NEG -> 0, the rest unlabelled.
Model = XGBoost on RAW evidence only (84 pair features + dense cosine and its ranks) - never the base probability,
        otherwise it would just copy the labelling function. 2-fold cross-fit by entity inside the target.
Output= pt; decisions on pt and on blends of pt with pb; F0.5 against the TRUE labels of the target.

  python selftrain.py proxy India|US      (validation)
"""
import gc
import os
import sys

import numpy as np
import polars as pl
import xgboost as xgb

import decide
import evaluate as ev

D = "C:/ML/Challenges/amazon/work/data/"
DN = "D:/amazon_er/dense/"
NIGHT = "D:/amazon_er/france_night/"
ARCH = "D:/amazon_er/archive/data/"
FEATS = ['a_partial', 'a_ratio', 'a_tset', 'a_tsort', 'addr_empty2', 'c_cont', 'c_inter', 'c_jac', 'c_jw', 'c_len1', 'c_len2',
         'c_lev', 'c_partial', 'c_ratio', 'c_tset', 'cl1', 'cl2', 'core_cov1', 'core_cov2', 'core_eq', 'core_imax', 'core_isum',
         'core_miss1', 'core_miss2', 'cs_ratio', 'css_eq', 'first_core_eq', 'first_num_eq', 'indic', 'legal_both',
         'legal_conflict', 'legal_eq', 'n_cont1', 'n_css_o_1', 'n_css_o_2', 'n_css_s1_1', 'n_css_s1_2', 'n_inter', 'n_jac',
         'n_len1', 'n_len2', 'n_partial', 'n_ratio', 'n_soft_o_1', 'n_soft_o_2', 'n_soft_s1_1', 'n_soft_s1_2', 'n_tset',
         'n_tsort', 'nk', 'num1_anagram', 'num1_in2', 'num1_lev', 'num2_in1', 'num_maxlen', 'num_prefix', 'nums_cov1',
         'nums_cov2', 'nums_imax', 'nums_isum', 'nums_miss1', 'nums_miss2', 'r1', 'r2', 'sa', 'score', 'sn', 'soft_eq', 'src',
         'sx', 'sz', 'w_cont', 'w_inter', 'w_jac', 'w_len1', 'w_len2', 'w_ratio', 'w_tset', 'words_cov1', 'words_cov2',
         'words_imax', 'words_isum', 'words_miss1', 'words_miss2']
XP = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda", grow_policy="lossguide",
          max_leaves=127, max_depth=0, learning_rate=0.06, subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
          reg_lambda=1.0, max_bin=128)
POS, NEG = float(os.environ.get("ST_POS", 0.7)), float(os.environ.get("ST_NEG", 0.1))


def log(*a):
    import time
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def proxy_frame(c):
    """Union of the target c under the France recipe with pb, dense cosine and the 84 raw features; y = truth.
    Memory-lean: target rows only, float32 at once, cosine from the feature files (no 30M-row dense join)."""
    frac = int(float(os.environ.get("ST_FRAC", 1.0)) * 100)  # random share of the target's entities (RAM)
    U = (pl.scan_parquet(f"{NIGHT}data/loco_joint_U.parquet").filter((pl.col("country") == c) & (pl.col("rid").hash(seed=77) % 100 < frac))
         .select("rid", "oid", "y", "is_new", pb=(0.5 * pl.col("p") + 0.5 * pl.col("pj")).cast(pl.Float32)).collect())
    keysK = U.filter(pl.col("is_new") == 0).select("rid", "oid")
    cosK = pl.scan_parquet(f"{DN}dense_train_cos.parquet").join(keysK.lazy(), on=["rid", "oid"], how="semi").collect().rename({"dense_cos": "cos"})
    FK = (pl.concat([pl.scan_parquet(f"{ARCH}train_feat_v11_fold{f}.parquet").select("rid", "oid", *[pl.col(x).cast(pl.Float32) for x in FEATS])
                     for f in (0, 1)])
          .join(keysK.lazy(), on=["rid", "oid"], how="semi").collect(engine="streaming"))
    FK = FK.unique(["rid", "oid"]).join(cosK, on=["rid", "oid"], how="left")
    del cosK
    FN = (pl.scan_parquet(f"{DN}loco_feat_{c}.parquet").select("rid", "oid", "cos", *[pl.col(x).cast(pl.Float32) for x in FEATS])
          .join(U.lazy().filter(pl.col("is_new") == 1).select("rid", "oid"), on=["rid", "oid"], how="semi").collect())
    FK = FK.with_columns(pl.col("cos").cast(pl.Float32))
    F = pl.concat([FK, FN.with_columns(pl.col("cos").cast(pl.Float32)).select(FK.columns)])
    del FK, FN
    gc.collect()
    U = U.join(F, on=["rid", "oid"], how="left")
    del F
    gc.collect()
    return add_cos_ctx(U)


def add_cos_ctx(U):
    c = pl.col("cos")
    return U.with_columns(cos_rank1=c.rank("ordinal", descending=True).over("rid").cast(pl.Float32),
                          cos_gap1=(c.max().over("rid") - c).cast(pl.Float32),
                          cos_rank2=c.rank("ordinal", descending=True).over("oid").cast(pl.Float32),
                          cos_gap2=(c.max().over("oid") - c).cast(pl.Float32),
                          n1u=pl.len().over("rid").cast(pl.Float32), n2u=pl.len().over("oid").cast(pl.Float32))


XCOLS = FEATS + ["cos", "cos_rank1", "cos_gap1", "cos_rank2", "cos_gap2", "n1u", "n2u", "is_new"]


def self_train(U, rounds=1, c_rule=4.0):
    """Returns U with pt (cross-fitted self-trained probability)."""
    X = U.select([pl.col(x).cast(pl.Float32) for x in XCOLS]).to_numpy()
    U = U.select([c for c in ("rid", "oid", "pb", "y", "is_new") if c in U.columns])  # free the feature columns
    gc.collect()
    fold = (U["rid"].hash(seed=11) % 2).to_numpy()
    pb = U["pb"].to_numpy()
    rng = np.random.default_rng(0)
    cur = pb.copy()
    for rd in range(rounds):
        acc = decide.ef_predict(U.select("rid", "oid").with_columns(p2=pl.Series(cur)), "p2", 0.0, c_rule, 0.05).with_columns(a=pl.lit(1))
        a = U.select("rid", "oid").join(acc, on=["rid", "oid"], how="left")["a"].fill_null(0).to_numpy()
        lab = np.full(len(cur), -1, np.int8)
        lab[(a == 1) & (cur >= POS)] = 1
        lab[(a == 0) & (cur <= NEG)] = 0
        log(f"round {rd + 1}: pseudo-labels pos {int((lab == 1).sum()):,} neg {int((lab == 0).sum()):,} unlabelled {int((lab < 0).sum()):,}")
        pt = np.zeros(len(cur), np.float32)
        for f in (0, 1):
            pos = np.flatnonzero((fold == f) & (lab == 1))
            neg = np.flatnonzero((fold == f) & (lab == 0))
            frac = min(1.0, 2.0 * len(pos) / max(1, len(neg)))  # negatives subsampled (RAM), weighted back
            neg = neg[rng.random(len(neg)) < frac]
            tr = np.concatenate([pos, neg])
            w = np.concatenate([np.ones(len(pos), np.float32), np.full(len(neg), 1.0 / frac, np.float32)])
            te = np.flatnonzero(fold != f)
            bst = xgb.train(XP, xgb.QuantileDMatrix(X[tr], label=lab[tr], weight=w, max_bin=XP["max_bin"]), 400)
            del tr, w
            for i in range(0, len(te), 2_000_000):
                pt[te[i:i + 2_000_000]] = bst.predict(xgb.DMatrix(X[te[i:i + 2_000_000]]))
            del bst
            gc.collect()
        cur = 0.5 * pb + 0.5 * pt
        U = U.with_columns(pl.Series(f"pt{rd + 1}", pt))
    del X
    gc.collect()
    return U


def proxy(c):
    U = proxy_frame(c)
    log(f"{c}: {U.height:,} union pairs, {U['y'].sum():,} true")
    U = self_train(U, rounds=int(os.environ.get("ST_ROUNDS", 2)))
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    s1 = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == c).join(keep, on="rid", how="semi")
    s1 = s1.filter(pl.col("rid").hash(seed=77) % 100 < int(float(os.environ.get("ST_FRAC", 1.0)) * 100))
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").join(s1, on="rid", how="semi").select("rid", "oid")
    cols = {"base pb": pl.col("pb")}
    for rd in (1, 2):
        if f"pt{rd}" in U.columns:
            cols[f"pt{rd}"] = pl.col(f"pt{rd}")
            cols[f"0.5 pb + 0.5 pt{rd}"] = 0.5 * pl.col("pb") + 0.5 * pl.col(f"pt{rd}")
    U = U.with_columns(**{k.replace(" ", "_").replace(".", "").replace("+", "p"): v for k, v in cols.items()})
    for k in cols:
        kk = k.replace(" ", "_").replace(".", "").replace("+", "p")
        for cr in (2.0, 4.0):
            pred = decide.ef_predict(U.select("rid", "oid", pl.col(kk).alias("p2")), "p2", 0.0, cr, 0.05)
            log(f"{c} {k:<18s} c={cr}: F0.5 {ev.macro_f05(pred.join(s1, on='rid', how='semi'), gt, s1['rid']):.5f}")
    U.select("rid", "oid", "pb", *[x for x in U.columns if x.startswith("pt")], "y").write_parquet(f"{NIGHT}data/selftrain_{c}.parquet")


def france_frame():
    """France union (france_v14.py joint: p, pj, pb) + the 84 raw features + dense cosine."""
    U = pl.read_parquet(f"{NIGHT}data/france_union.parquet").select("rid", "oid", "is_new", pb=pl.col("pb").cast(pl.Float32))
    keysK = U.filter(pl.col("is_new") == 0).select("rid", "oid")
    cosK = pl.scan_parquet(f"{DN}dense_test_cos.parquet").join(keysK.lazy(), on=["rid", "oid"], how="semi").collect().rename({"dense_cos": "cos"})
    FK = (pl.scan_parquet(f"{ARCH}test_feat_v10_France.parquet").select("rid", "oid", *[pl.col(x).cast(pl.Float32) for x in FEATS])
          .join(keysK.lazy(), on=["rid", "oid"], how="semi").collect(engine="streaming").unique(["rid", "oid"])
          .join(cosK, on=["rid", "oid"], how="left").with_columns(pl.col("cos").cast(pl.Float32)))
    FN = (pl.scan_parquet(f"{DN}test_nfeat_France.parquet").select("rid", "oid", "cos", *[pl.col(x).cast(pl.Float32) for x in FEATS])
          .join(U.lazy().filter(pl.col("is_new") == 1).select("rid", "oid"), on=["rid", "oid"], how="semi").collect()
          .with_columns(pl.col("cos").cast(pl.Float32)))
    F = pl.concat([FK, FN.select(FK.columns)])
    del FK, FN, cosK
    gc.collect()
    U = U.join(F, on=["rid", "oid"], how="left")
    del F
    gc.collect()
    miss = U["a_tset"].null_count()
    log(f"France frame {U.height:,} pairs; pairs without features {miss:,}")
    return add_cos_ctx(U)


def france():
    U = self_train(france_frame(), rounds=int(os.environ.get("ST_ROUNDS", 2)))
    rd = int(os.environ.get("ST_ROUNDS", 2))
    U = U.with_columns(pst=0.5 * pl.col("pb") + 0.5 * pl.col(f"pt{rd}"))
    U.select("rid", "oid", "is_new", "pb", *[x for x in U.columns if x.startswith("pt")], "pst").write_parquet(f"{NIGHT}data/france_selftrain.parquet")
    for col in ("pb", "pst"):
        for cr in (2.0, 4.0):
            n = decide.ef_predict(U.select("rid", "oid", pl.col(col).alias("p2")), "p2", 0.0, cr, 0.05).height
            log(f"France {col} c={cr}: {n:,} predicted pairs ({n / 259452:.3f} per S1)")


if __name__ == "__main__":
    if sys.argv[1] == "proxy":
        proxy(sys.argv[2])
    elif sys.argv[1] == "france":
        france()
