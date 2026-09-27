"""True share of accepted nearby-number pairs per probability band, estimated from the number-edit profile alone:
P(profile | true) and P(profile | decoy) from TRAIN candidates (same buckets), then EM for the mixture proportion in
each test subset. Checked on train subsets (known truth). Buckets: siblings odd/mixed/none, |diff| <= 50; and, for
reference, siblings-share-record, |diff| <= 50."""
import numpy as np
import polars as pl
from rapidfuzz.distance import Levenshtein

import decide
from nearby_number import D, DN, pattern, sib_agree
from nn_followup import load_pred

keep = pl.read_parquet(f"{D}keep_v11.parquet")
gt = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid"])


def cells(x):
    a = [u[0] if len(u) == 1 else "" for u in x["da"].to_list()]
    b = [u[0] if len(u) == 1 else "" for u in x["db"].to_list()]
    lev = [min(Levenshtein.distance(u, v), 3) if u and v else 4 for u, v in zip(a, b)]
    sl = [int(len(u) == len(v)) for u, v in zip(a, b)]
    return x.with_columns(cell=pl.Series([f"{min(d, 12) if d <= 12 else (20 if d <= 20 else 50)}|{l}|{s}" for d, l, s in zip(x["diff"].to_list(), lev, sl)]))


def em(cellcol, lr):
    r = np.array([lr.get(c, 1.0) for c in cellcol])
    x = 0.5
    for _ in range(200):
        x = float(np.mean(x * r / (x * r + 1 - x)))
    return x


def bands(T, lr, lab, truth):
    print(f"   {lab}")
    for lo, hi in ((0.99, 1.01), (0.95, 0.99), (0.9, 0.95), (0.7, 0.9), (0.0, 0.7), (0.0, 1.01)):
        y = T.filter((pl.col("acc") == 1) & (pl.col("p") >= lo) & (pl.col("p") < hi))
        if y.height == 0:
            continue
        est = em(y["cell"].to_list(), lr)
        tr = f"  actual {y['y'].mean():.3f}" if truth else ""
        print(f"      p in [{lo:.2f},{min(hi, 1):.2f}) {'(all)' if lo == 0 and hi > 1 else '     '} accepted {y.height:>7,}  est. true share {est:.3f}{tr}")


for c in ("US", "India"):
    print(f"\n==== {c}")
    s1c = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == c).select("rid").join(keep, on="rid", how="semi")
    U = pl.read_parquet(f"{DN}train_joint9.parquet", columns=["rid", "oid", "pj", "country"]).filter(pl.col("country") == c).drop("country")
    pred = decide.ef_predict(U.select("rid", "oid", pl.col("pj").alias("p2")), "p2", 0.0, 1.0, 0.05)
    P, o = pattern("train", c)
    P = P.join(s1c, on="rid", how="semi").join(gt.with_columns(y=pl.lit(1, pl.Int8)), on=["rid", "oid"], how="left").join(
        pred.with_columns(acc=pl.lit(1, pl.Int8)), on=["rid", "oid"], how="left").join(U.rename({"pj": "p"}), on=["rid", "oid"], how="left").with_columns(
        pl.col("y", "acc").fill_null(0), pl.col("p").fill_null(0.0))
    del U
    P = cells(sib_agree(P, pred, o, "sib").filter(pl.col("diff") <= 50))
    s1 = pl.read_parquet(f"{D}p1_canon_test_{c}_s1.parquet", columns=["idx", "entity_id"])
    ot = pl.read_parquet(f"{D}p1_canon_test_{c}_o.parquet", columns=["idx", "entity_id"])
    v19 = load_pred("v19", s1, ot)
    Pt, oc = pattern("test", c)
    tp = pl.read_parquet(f"{DN}test_union_p.parquet").select("rid", "oid", "p")
    Pt = Pt.join(v19.with_columns(acc=pl.lit(1, pl.Int8)), on=["rid", "oid"], how="left").join(tp, on=["rid", "oid"], how="left").with_columns(
        pl.col("acc").fill_null(0), pl.col("p").fill_null(0.0))
    Pt = cells(sib_agree(Pt, v19, oc, "sib").filter(pl.col("diff") <= 50))
    for grp, sibs in (("odd/mixed/none", ["odd one out", "mixed", "none"]), ("siblings share rec", ["siblings share rec"])):
        T = P.filter(pl.col("sib").is_in(sibs))
        g = T.group_by("cell").agg(nt=(pl.col("y") == 1).sum(), nf=(pl.col("y") == 0).sum())
        st, sf = g["nt"].sum(), g["nf"].sum()
        lr = {r["cell"]: ((r["nt"] + 0.5) / (st + 0.5 * g.height)) / ((r["nf"] + 0.5) / (sf + 0.5 * g.height)) for r in g.iter_rows(named=True)}
        print(f"-- {grp}, |diff| <= 50 (train candidates {T.height:,}, true {T['y'].mean():.3f}; EM on all train candidates -> {em(T['cell'].to_list(), lr):.3f})")
        bands(T, lr, "TRAIN harder split (check: estimate vs actual)", True)
        bands(Pt.filter(pl.col("sib").is_in(sibs)), lr, "TEST v19", False)
