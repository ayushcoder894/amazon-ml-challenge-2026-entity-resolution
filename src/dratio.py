"""Density-ratio decoy estimator (India/US). Generator logic: train and test entities get TRUE matches from the same
process, so per entity the density of accepted-and-true pairs at feature value x is the same on both. A classifier
separating  train accepted-TRUE pairs (weighted n_test/n_train)  from  test ACCEPTED pairs  gives
r(x) = q/(1-q) = test accepted density / true density, hence the false fraction of test accepted pairs at x:
    f(x) = 1 - 1/r(x).
This generalises the hand-made shift cells (A', A2, A3-A5) to a smooth per-pair estimate. Features are computed the same
way on both sides (canonical text, number sets and gaps, agreement with the entity's other accepted records); the model
probability is optional (USE_P=1) because true pairs score lower on test (deflation) which would look like excess.
  python dratio.py self  US     calibration on labelled train: half A true vs half B accepted -> f vs actual false rate
  python dratio.py test  US     train true vs test accepted (v21) -> D:/amazon_er/france_night/data/dratio_US[_p].parquet
"""
import os
import sys
import time

import numpy as np
import polars as pl
import xgboost as xgb
from rapidfuzz import fuzz, process

import decide
import evaluate as ev
from nearby_number import canon2, digits

D = "C:/ML/Challenges/amazon/work/data/"
DN = "D:/amazon_er/dense/"
OUT = "C:/ML/Challenges/amazon/work/output/"
N = "D:/amazon_er/france_night/data/"
rd = dict(separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
USE_P = os.environ.get("USE_P", "0") == "1"
XP = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda", max_depth=int(os.environ.get("DEPTH", 5)), learning_rate=0.05,
          subsample=0.8, colsample_bytree=0.8, min_child_weight=float(os.environ.get("MCW", 2000)), reg_lambda=5.0, max_bin=128)
ROUNDS = int(os.environ.get("ROUNDS", 200))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def feats(P, split, c):
    """P: accepted pairs (rid, oid, ...) of one country; siblings = the other accepted records of the same rid."""
    s1, o = canon2(split, c, "s1"), canon2(split, c, "o")
    na = s1.filter(pl.col("cak") != "").group_by("cak").agg(na=pl.len())
    s1 = s1.join(na, on="cak", how="left")
    x = (P.join(s1.select(pl.col("idx").cast(pl.UInt32).alias("rid"), n1="cnk", a1="cak", w1="w", u1="nums", na="na"), on="rid")
         .join(o.select(pl.col("idx").cast(pl.UInt32).alias("oid"), n2="cnk", a2="cak", w2="w", u2="nums", ae="aempty",
                        src3=pl.col("entity_id").str.starts_with("S3").cast(pl.Int8)), on="oid"))
    del s1, o
    inter = pl.col("u1").list.set_intersection(pl.col("u2")).list.len()
    x = x.with_columns(
        name_eq=(pl.col("n1") == pl.col("n2")).cast(pl.Int8), addr_eq=(pl.col("a1") == pl.col("a2")).cast(pl.Int8),
        words_eq=(pl.col("w1") == pl.col("w2")).cast(pl.Int8), ae=pl.col("ae").cast(pl.Int8), a1e=(pl.col("a1") == "").cast(pl.Int8),
        na=pl.col("na").fill_null(0), nl1=pl.col("n1").str.len_chars(), nl2=pl.col("n2").str.len_chars(),
        nt1=pl.col("n1").str.count_matches(" ") + 1, nt2=pl.col("n2").str.count_matches(" ") + 1,
        nn1=pl.col("u1").list.len(), nn2=pl.col("u2").list.len(), ninter=inter)
    n1, n2, a1, a2, w1, w2 = (x[k].to_list() for k in ("n1", "n2", "a1", "a2", "w1", "w2"))
    x = x.with_columns(name_r=pl.Series(process.cpdist(n1, n2, scorer=fuzz.ratio, workers=-1), dtype=pl.Float32),
                       name_ts=pl.Series(process.cpdist(n1, n2, scorer=fuzz.token_set_ratio, workers=-1), dtype=pl.Float32),
                       addr_r=pl.Series(process.cpdist(a1, a2, scorer=fuzz.ratio, workers=-1), dtype=pl.Float32),
                       words_ts=pl.Series(process.cpdist(w1, w2, scorer=fuzz.token_set_ratio, workers=-1), dtype=pl.Float32))
    del n1, n2, a1, a2, w1, w2
    gap, trunc = [], []
    for a, b in zip(x["u1"].to_list(), x["u2"].to_list()):
        u, v = set(a) - set(b), set(b) - set(a)
        if u and v:
            t = any(p in q or q in p for p in u for q in v)
            gap.append(float(np.log10(1 + min(abs(digits(p) - digits(q)) for p in u for q in v))))
            trunc.append(int(t))
        else:
            gap.append(-1.0)
            trunc.append(-1)
    x = x.with_columns(gap=pl.Series(gap, dtype=pl.Float32), trunc=pl.Series(trunc, dtype=pl.Int8))
    # sibling agreement (other accepted records of the same S1)
    S = x.select("rid", soid="oid", sn="n2", sa="a2", su="u2", sae="ae")
    y = (x.select("rid", "oid", "n2", "a2", "u1", "u2").join(S, on="rid").filter(pl.col("soid") != pl.col("oid"))
         .with_columns(hasn=(pl.col("su").list.len() > 0) & (pl.col("sae") == 0),
                       s_name=(pl.col("sn") == pl.col("n2")), s_addr=(pl.col("sa") == pl.col("a2")) & (pl.col("sae") == 0),
                       s_nums=pl.col("su").list.join(" ") == pl.col("u2").list.join(" "),
                       s_share=pl.col("su").list.set_intersection(pl.col("u2")).list.len() > 0,
                       s_s1=pl.col("su").list.set_intersection(pl.col("u1")).list.len() == pl.col("u1").list.len()))
    g = y.group_by("rid", "oid").agg(
        sib_name=pl.col("s_name").mean(), sib_addr=pl.col("s_addr").mean(),
        sib_nums=pl.col("s_nums").filter(pl.col("hasn")).mean(), sib_share=pl.col("s_share").filter(pl.col("hasn")).mean(),
        sib_s1=pl.col("s_s1").filter(pl.col("hasn")).mean(), sib_nnum=pl.col("hasn").sum().cast(pl.Float32))
    del y, S
    x = x.join(g, on=["rid", "oid"], how="left").with_columns(has_sib=pl.col("sib_name").is_not_null().cast(pl.Int8))
    return x.drop("n1", "n2", "a1", "a2", "w1", "w2", "u1", "u2")


FEATS = ["name_eq", "addr_eq", "words_eq", "ae", "a1e", "na", "nl1", "nl2", "nt1", "nt2", "nn1", "nn2", "ninter", "name_r", "name_ts",
         "addr_r", "words_ts", "gap", "trunc", "src3", "sib_name", "sib_addr", "sib_nums", "sib_share", "sib_s1", "sib_nnum", "has_sib"]


def crossfit(A, B, wA):
    """A: 'true' side rows, B: 'accepted' side rows (both with column 'cf' in {0,1}); returns f for every B row (out of fold)."""
    cols = FEATS + (["p"] if USE_P else [])
    f = np.zeros(B.height, np.float32)
    for k in (0, 1):
        ta, tb = A.filter(pl.col("cf") == k), B.filter(pl.col("cf") == k)
        X = np.concatenate([ta.select(pl.col(cols).cast(pl.Float32)).to_numpy(), tb.select(pl.col(cols).cast(pl.Float32)).to_numpy()])
        lab = np.r_[np.zeros(ta.height), np.ones(tb.height)]
        w = np.r_[np.full(ta.height, wA), np.ones(tb.height)]
        bst = xgb.train(XP, xgb.QuantileDMatrix(X, label=lab, weight=w, max_bin=128), ROUNDS)
        m = (B["cf"] != k).to_numpy()
        q = bst.predict(xgb.DMatrix(B.filter(pl.Series(m)).select(pl.col(cols).cast(pl.Float32)).to_numpy()))
        f[m] = np.clip(1 - (1 - q) / np.maximum(q, 1e-6), 0, 1)
        del bst, X
    return f


def train_side(c):
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    s1c = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == c).select("rid").join(keep, on="rid", how="semi")
    U = pl.read_parquet(f"{DN}train_joint9.parquet", columns=["rid", "oid", "pj", "y", "country"]).filter(pl.col("country") == c).join(s1c, on="rid", how="semi")
    pred = decide.ef_predict(U.select("rid", "oid", p2="pj"), "p2", 0.0, 1.0, 0.05)
    A = U.select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32), "y", p="pj").join(pred.select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32)), on=["rid", "oid"], how="semi")
    return A, s1c


def pairs_of(v, c):
    s1t = pl.read_parquet(f"{D}p1_canon_test_{c}_s1.parquet", columns=["idx", "entity_id"])
    ot = pl.read_parquet(f"{D}p1_canon_test_{c}_o.parquet", columns=["idx", "entity_id"])
    fin = pl.read_csv(f"{OUT}matching_results_{v}.tsv", **rd).filter(pl.col("matched_entity_ids") != "").with_columns(m=pl.col("matched_entity_ids").str.split(",")).explode("m")
    return (fin.join(s1t.rename({"entity_id": "source1_entity_id", "idx": "rid"}), on="source1_entity_id").join(ot.rename({"entity_id": "m", "idx": "oid"}), on="m")
            .select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32))), s1t.height


def cf_col(seed):
    return (pl.col("rid").hash(seed) % 2).cast(pl.Int8).alias("cf")


def self_check(c):
    A, s1c = train_side(c)
    A = feats(A, "train", c).with_columns(h=(pl.col("rid").hash(7) % 2).cast(pl.Int8), cf=cf_col(3))
    nh = s1c.with_columns(h=(pl.col("rid").hash(7) % 2).cast(pl.Int8)).group_by("h").len().sort("h")["len"].to_list()
    Ta, Bb = A.filter((pl.col("h") == 0) & (pl.col("y") == 1)), A.filter(pl.col("h") == 1)
    log(f"{c} self-check: half A true accepted {Ta.height:,} ({nh[0]:,} S1) vs half B accepted {Bb.height:,} ({nh[1]:,} S1), "
        f"B false rate {1 - Bb['y'].mean():.4f}")
    f = crossfit(Ta, Bb, nh[1] / nh[0])
    Bb = Bb.with_columns(f=pl.Series(f))
    log(f"  mean f {f.mean():.4f} vs actual false rate {1 - Bb['y'].mean():.4f}")
    t = (Bb.with_columns(fb=pl.col("f").cut([0.05, 0.1, 0.2, 0.33, 0.5, 0.7])).group_by("fb")
         .agg(n=pl.len(), f_mean=pl.col("f").mean(), false_rate=1 - pl.col("y").mean()).sort("fb"))
    with pl.Config(tbl_rows=20, tbl_hide_column_data_types=True, float_precision=4):
        print(t)
    for th in (0.33, 0.5, 0.7):
        r = Bb.filter(pl.col("f") > th)
        log(f"  remove f>{th}: {r.height:,} pairs, of which false {1 - r['y'].mean() if r.height else 0:.3f}")


def test_run(c):
    A, s1c = train_side(c)
    ntr = s1c.height
    A = feats(A, "train", c)   # siblings from all accepted pairs, as on test
    A = A.filter(pl.col("y") == 1).with_columns(cf=cf_col(3))
    B, nte = pairs_of("v21", c)
    B = B.join(pl.read_parquet(f"{DN}test_union_p.parquet").select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32), "p"), on=["rid", "oid"], how="left").with_columns(pl.col("p").fill_null(1.0))
    B = feats(B, "test", c).with_columns(cf=cf_col(3))
    log(f"{c} test: train true accepted {A.height:,} ({ntr:,} S1, {A.height / ntr:.4f}/S1) vs test v21 accepted {B.height:,} ({nte:,} S1, {B.height / nte:.4f}/S1); "
        f"implied false fraction overall {1 - (A.height / ntr) / (B.height / nte):.4f}")
    f = crossfit(A, B, nte / ntr)
    B = B.with_columns(f=pl.Series(f))
    log(f"  mean f {f.mean():.4f}")
    rem = {}
    for a, b, name in (("v19", "probeA19", "A' (LB +0.0011)"), ("probeA", "probeA2", "A2 (LB +0.00085)"), ("probeA2", "probeA3", "A3"),
                       ("probeA3", "probeA4", "A4"), ("probeA4", "probeA5", "A5")):
        pa, _ = pairs_of(a, c)
        pb, _ = pairs_of(b, c)
        rem[name] = pa.join(pb, on=["rid", "oid"], how="anti")
        r = B.join(rem[name], on=["rid", "oid"], how="semi")
        log(f"  removed by {name:<18s}: {rem[name].height:>6,} pairs, in v21 {r.height:>6,}, mean f {r['f'].mean() if r.height else float('nan'):.3f}, "
            f"f>0.5 {(r['f'] > 0.5).mean() if r.height else 0:.3f}")
    allrem = pl.concat(list(rem.values())).unique()
    left = B.join(allrem, on=["rid", "oid"], how="anti")
    t = (left.with_columns(fb=pl.col("f").cut([0.1, 0.2, 0.33, 0.5, 0.7, 0.85]), lowp=pl.col("p") < 0.97).group_by("fb", "lowp")
         .agg(n=pl.len(), f_mean=pl.col("f").mean(), p_mean=pl.col("p").mean()).sort("fb", "lowp"))
    log("  pairs still in A5, by estimated false fraction:")
    with pl.Config(tbl_rows=30, tbl_hide_column_data_types=True, float_precision=4):
        print(t)
    B.select("rid", "oid", "p", "f", *FEATS).write_parquet(f"{N}dratio_{c}{'_p' if USE_P else ''}.parquet")
    A.select("rid", "oid", "p", *FEATS).write_parquet(f"{N}dratio_A_{c}.parquet")
    pl.DataFrame({"ntr": [ntr], "nte": [nte]}).write_parquet(f"{N}dratio_n_{c}.parquet")
    log("saved")


if __name__ == "__main__":
    (self_check if sys.argv[1] == "self" else test_run)(sys.argv[2])
