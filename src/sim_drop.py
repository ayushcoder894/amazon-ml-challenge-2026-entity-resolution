"""Simulate the test condition on training data: remove a random share of Source-1 entities (their S2/S3 records stay,
now as distractors). Test has 5.75 records per S1 vs 4.68 in train -> about 18.6% of entities missing from S1.
Evaluates the honest out-of-fold probabilities under that condition, and a prior-shift recalibration
p' = r*p / (r*p + 1 - p) whose r is tuned on one fold and reported on the other.
"""
import sys

import numpy as np
import polars as pl

import decide
import evaluate as ev

D = "C:/ML/Challenges/amazon/work/data/"


def run(src="train_oof_ce_v10", pcol="p3", drop=0.186, seed=0):
    K = pl.read_parquet(f"{D}{src}.parquet", columns=["rid", "oid", "y", "fold", "country", pcol])
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid", "country"])
    s1 = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).with_columns(
        fold=(pl.col("rid").hash(seed=11) % 2).cast(pl.UInt8),
        keep=(pl.col("rid").hash(seed=100 + seed) % 1000) >= int(drop * 1000))
    kept = s1.filter(pl.col("keep"))
    Kd = K.join(kept.select("rid"), on="rid", how="semi")
    gtd = gt.join(kept.select("rid"), on="rid", how="semi")
    print(f"kept S1 {kept.height:,} of {s1.height:,}; candidate pairs {Kd.height:,} of {K.height:,}")

    def score(KK, g, rids, p, mu=0.0, c=1.5):
        pred = decide.ef_predict(KK.select("rid", "oid", pl.col(p).alias("p2")), "p2", mu, c, 0.05).select("rid", "oid")
        return ev.macro_f05(pred.filter(pl.col("rid").is_in(rids.implode())), g.filter(pl.col("rid").is_in(rids.implode())), rids)
    print(f"full condition   : F = {score(K, gt.select('rid', 'oid'), s1['rid'], pcol):.5f}")
    print(f"test-like (drop) : F = {score(Kd, gtd.select('rid', 'oid'), kept['rid'], pcol):.5f}")
    for c in ("India", "US"):
        r = kept.filter(pl.col("country") == c)["rid"]
        print(f"   {c}: {score(Kd, gtd.select('rid', 'oid'), r, pcol):.5f}")
    # prior-shift recalibration, honest over folds
    res = {}
    for r in (1.0, 0.7, 0.5, 0.35, 0.25):
        Kr = Kd.with_columns(pr=(r * pl.col(pcol)) / (r * pl.col(pcol) + 1 - pl.col(pcol)))
        res[r] = [score(Kr, gtd.select("rid", "oid"), kept.filter(pl.col("fold") == f)["rid"], "pr") for f in (0, 1)]
        print(f"   r={r}: fold0 {res[r][0]:.5f} fold1 {res[r][1]:.5f}", flush=True)
    honest = np.mean([res[max(res, key=lambda k: res[k][f])][1 - f] for f in (0, 1)])
    best = max(res, key=lambda k: sum(res[k]))
    print(f"test-like with recalibration (honest): {honest:.5f}; best r = {best}")


if __name__ == "__main__":
    run(*sys.argv[1:2])
