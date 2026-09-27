"""Harder split v2 at the decision level: test has no name-only orphans (name-only records per S1 are the same in test
as in train and ~97.5% of them belong to an S1), while the harder split turns the dropped entities' name-only records
into orphans. Here:
  1. drop the pairs of name-only orphan records (records owned by a dropped S1, empty address) from the out-of-fold
     final probabilities -> score of the current model on the test-like composition;
  2. shift the logit of name-only pairs by delta (per country), chosen on one half of the S1s and scored on the other
     (2-fold by rid hash) -> does a prior correction for name-only records gain?
"""
import sys

import numpy as np
import polars as pl

import decide
import evaluate as ev

D = "C:/ML/Challenges/amazon/work/data/"


def load(tag="_v12_twin"):
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    s1 = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).join(keep, on="rid", how="semi")
    gt_all = pl.read_parquet(f"{D}train_gt_rid.parquet").select("rid", "oid")
    gt = gt_all.join(keep, on="rid", how="semi")
    raw = pl.concat([pl.read_parquet(f"{D}train_source{k}.parquet", columns=["business_address"]) for k in (2, 3)]
                    ).with_row_index("oid").select(pl.col("oid").cast(pl.UInt32),
                                                   ae=pl.col("business_address").fill_null("").str.strip_chars() == "")
    own = gt_all.rename({"rid": "owner"})
    rec = raw.join(own, on="oid", how="left").with_columns(
        orphan=pl.col("owner").is_not_null() & ~pl.col("owner").is_in(keep["rid"].implode()))
    K = pl.read_parquet(f"{D}train_oof_final{tag}.parquet", columns=["rid", "oid", "pf"])
    K = K.join(rec.select("oid", "ae", "orphan"), on="oid", how="left")
    return s1, gt, K, rec


def score(pred, gt, s1):
    out = {}
    for c in ("India", "US"):
        r = s1.filter(pl.col("country") == c)["rid"]
        out[c] = ev.macro_f05(pred.filter(pl.col("rid").is_in(r.implode())).select("rid", "oid"),
                              gt.join(pl.DataFrame({"rid": r}), on="rid", how="semi"), r)
    return out


def decide_(K, col="pf"):
    return decide.ef_predict(K.select("rid", "oid", pl.col(col).alias("p2")), "p2", 0.0, 1.0, 0.05).select("rid", "oid")


def shifted(K, d):
    lg = (pl.col("pf").clip(1e-6, 1 - 1e-6) / (1 - pl.col("pf").clip(1e-6, 1 - 1e-6))).log()
    return K.with_columns(pf=pl.when(pl.col("ae")).then(1 / (1 + (-(lg + d)).exp())).otherwise(pl.col("pf")))


def main(tag="_v12_twin"):
    s1, gt, K, rec = load(tag)
    n_or = rec.filter(pl.col("orphan") & pl.col("ae")).height
    print(f"name-only orphan records: {n_or:,}; their candidate pairs: {K.filter(pl.col('orphan') & pl.col('ae')).height:,}")
    base = score(decide_(K), gt, s1)
    K2 = K.filter(~(pl.col("orphan") & pl.col("ae")))
    v2 = score(decide_(K2), gt, s1)
    print(f"harder split (v1)                 India {base['India']:.5f}  US {base['US']:.5f}")
    print(f"harder split v2 (no n-o orphans)  India {v2['India']:.5f}  US {v2['US']:.5f}")
    fold = s1.with_columns(f=pl.col("rid").hash(seed=11) % 2)
    grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    res = {}
    for d in grid:
        pr = decide_(shifted(K2, d))
        for f in (0, 1):
            ss = fold.filter(pl.col("f") == f)
            res[(d, f)] = score(pr.join(ss, on="rid", how="semi"), gt.join(ss, on="rid", how="semi"), ss)
        print(f"  delta {d:+.2f}: India {np.mean([res[(d, f)]['India'] for f in (0, 1)]):.5f}  "
              f"US {np.mean([res[(d, f)]['US'] for f in (0, 1)]):.5f}", flush=True)
    for c in ("India", "US"):
        hon = []
        for f in (0, 1):
            best = max(grid, key=lambda d: res[(d, 1 - f)][c])
            hon.append(res[(best, f)][c])
        print(f"{c}: honest (delta chosen on the other half) {np.mean(hon):.5f} vs {v2[c]:.5f} ({np.mean(hon) - v2[c]:+.5f})")


if __name__ == "__main__":
    main(*sys.argv[1:2])
