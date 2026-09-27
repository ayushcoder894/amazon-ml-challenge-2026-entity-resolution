"""Model-implied expected macro F0.5 per country (assumes calibrated probabilities; ignores blocking misses)."""
import sys

import polars as pl

import decide

D = "C:/ML/Challenges/amazon/work/data/"


def self_ef(K, pcol, split, rule=("ef", 0.0, 1.5)):
    pred = decide.ef_predict(K.select("rid", "oid", pl.col(pcol)), pcol, rule[1], rule[2], 0.05).with_columns(sel=pl.lit(1))
    x = K.select("rid", "oid", "country", pcol).join(pred, on=["rid", "oid"], how="left").with_columns(pl.col("sel").fill_null(0))
    # a record can only belong to one S1: use each record's probability for this S1 as P(match)
    g = x.group_by("rid", "country").agg(tp=(pl.col(pcol) * pl.col("sel")).sum(), npred=pl.col("sel").sum(),
                                         ng=pl.col(pcol).sum())
    s1 = pl.read_parquet(f"{D}{split}_s1_norm.parquet", columns=["rid", "country"])
    g = s1.join(g, on=["rid", "country"], how="left").fill_null(0)
    g = g.with_columns(f=pl.when((pl.col("npred") == 0) & (pl.col("ng") < 0.5)).then(1.0 - pl.col("ng"))
                       .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("ng") + pl.col("npred")).clip(1e-9)))
    return g.group_by("country").agg(pl.col("f").mean().round(5), pl.len()).sort("country")


if __name__ == "__main__":
    tr = pl.read_parquet(f"{D}train_oof_ce_v10.parquet", columns=["rid", "oid", "country", "p3"])
    print("train OOF (v10 p3):\n", self_ef(tr, "p3", "train"))
    te = pl.read_parquet(f"{D}test_pred_ce_v10.parquet", columns=["rid", "oid", "country", "p3"])
    print("test (v10 p3):\n", self_ef(te, "p3", "test"))
