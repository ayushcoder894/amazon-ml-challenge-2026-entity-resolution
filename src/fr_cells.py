"""France recipe cell audit. Proxy (India/US scored with the France recipe = v18 + rule S + v21 name-only adds, true labels):
precision of ACCEPTED and of REJECTED (p >= 0.3) pairs per (pattern key x blend-probability band). France test: counts per cell.
  python fr_cells.py India|US|France"""
import importlib
import os
import sys
import polars as pl
from shift_all import keyed

D = "C:/ML/Challenges/amazon/work/data/"
N = "D:/amazon_er/france_night/data/"
PB = (pl.when(pl.col("p") >= 0.97).then(pl.lit("5 >=.97")).when(pl.col("p") >= 0.9).then(pl.lit("4 .90-.97")).when(pl.col("p") >= 0.7).then(pl.lit("3 .70-.90"))
      .when(pl.col("p") >= 0.5).then(pl.lit("2 .50-.70")).when(pl.col("p") >= 0.3).then(pl.lit("1 .30-.50")).otherwise(pl.lit("0 <.30")))
c = sys.argv[1]
if c == "France":
    U = pl.read_parquet(f"{N}france_union.parquet").select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32), p="pb")
    A = pl.read_parquet(f"{N}france_v21_pairs.parquet").select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32))
    U = pl.concat([U, A.join(U, on=["rid", "oid"], how="anti").with_columns(p=pl.lit(1.0, pl.Float32))], how="vertical_relaxed")
    X = keyed(U.join(A.with_columns(acc=pl.lit(1, pl.Int8)), on=["rid", "oid"], how="left").with_columns(pl.col("acc").fill_null(0), y=pl.lit(None, pl.Int8)), "test", "France")
    n = 259452
else:
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    s1 = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == c).join(keep, on="rid", how="semi").select(pl.col("rid").cast(pl.UInt32))
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid"]).join(s1, on="rid", how="semi").with_columns(y=pl.lit(1, pl.Int8))
    import france_v18 as v18
    from france_v19 import rule_s
    _, _, C, _ = v18.v18(c)
    P = pl.concat([C, rule_s(c, "train", C)]).unique().select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32)).join(s1, on="rid", how="semi")
    os.environ["W_S"] = "0"
    import france_v15 as f0
    importlib.reload(f0)
    U, _ = f0.frames(c)
    U = U.select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32), p="pb").join(s1, on="rid", how="semi")
    ae = pl.read_parquet(f"{D}p1_canon_train_{c}_o.parquet", columns=["idx", "aempty"]).filter(pl.col("aempty")).select(pl.col("idx").cast(pl.UInt32).alias("oid"))
    t2 = (U.join(ae, on="oid", how="semi").join(P.select("oid"), on="oid", how="anti").sort("p", descending=True)
          .group_by("oid", maintain_order=True).agg(rid=pl.col("rid").first(), p1=pl.col("p").first(), p2=pl.col("p").slice(1, 1).first()).with_columns(pl.col("p2").fill_null(0.0)))
    P = pl.concat([P, t2.filter((pl.col("p1") >= 0.5) & (pl.col("p2") < 0.1)).select("rid", "oid")]).unique()
    X = keyed(U.join(P.with_columns(acc=pl.lit(1, pl.Int8)), on=["rid", "oid"], how="left").join(gt, on=["rid", "oid"], how="left")
              .with_columns(pl.col("acc").fill_null(0), pl.col("y").fill_null(0)), "train", c)
    n = s1.height
X = X.with_columns(pb=PB)
t = X.group_by("key", "pb").agg(n_acc=(pl.col("acc") == 1).sum(), n_rej=((pl.col("acc") == 0) & (pl.col("p") >= 0.3)).sum(),
                                prec_acc=pl.col("y").filter(pl.col("acc") == 1).mean(), prec_rej=pl.col("y").filter((pl.col("acc") == 0) & (pl.col("p") >= 0.3)).mean())
t = t.with_columns(acc_per_s1=pl.col("n_acc") / n, rej_per_s1=pl.col("n_rej") / n)
t.write_parquet(f"{N}frcells_{c}.parquet")
if c == "France":
    X.filter((pl.col("acc") == 1) | (pl.col("p") >= 0.3)).select("rid", "oid", "key", "pb", "acc", "p").write_parquet(f"{N}frcells_pairs_France.parquet")
print(c, "done", t.height, "cells")
