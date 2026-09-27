"""How much signal does each true pair carry? Estimates the noise ceiling for macro F0.5."""
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

import features as F
import norm

D = F.D
gt = pl.read_parquet(f"{D}train_gt_rid.parquet")
cand = pl.read_parquet(f"{D}train_cand.parquet", columns=["rid", "oid", "r1", "r2"]).filter(
    (pl.col("r1") <= 10) | (pl.col("r2") <= 3)).select("rid", "oid", inc=pl.lit(1, pl.UInt8))
outs = []
for c in ["India", "US"]:
    s1, o = F.load_records("train", c)
    g = gt.filter(pl.col("country") == c).join(cand, on=["rid", "oid"], how="left").with_columns(pl.col("inc").fill_null(0))
    x = (g.join(s1.select("rid", "cs", "nums", "ac"), on="rid")
         .join(o.select(pl.col("rid").alias("oid"), pl.col("cs").alias("cs2"), pl.col("nums").alias("nums2"),
                        pl.col("ac").alias("ac2"), "indic"), on="oid"))
    x = x.with_columns(
        name_ts=pl.Series(cpdist(x["cs"].to_list(), x["cs2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)),
        addr_ts=pl.Series(cpdist(x["ac"].to_list(), x["ac2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)),
        num_share=pl.col("nums").list.set_intersection("nums2").list.eval(pl.element().str.len_chars()).list.max().fill_null(0),
        addr_empty=(pl.col("ac2").str.len_chars() == 0),
    ).with_columns(
        name_ok=pl.col("name_ts") >= 60,
        addr_ok=(pl.col("num_share") >= 2) | (pl.col("addr_ts") >= 60),
    )
    outs.append(x.select("rid", "oid", "country", "src", "inc", "indic", "name_ts", "addr_ts", "num_share",
                         "addr_empty", "name_ok", "addr_ok"))
x = pl.concat(outs)
x.write_parquet(f"{D}gt_signal.parquet")
x = x.with_columns(cls=pl.when(pl.col("name_ok") & pl.col("addr_ok")).then(pl.lit("name+addr"))
                   .when(pl.col("addr_ok")).then(pl.lit("addr only"))
                   .when(pl.col("name_ok") & pl.col("addr_empty")).then(pl.lit("name only, addr EMPTY"))
                   .when(pl.col("name_ok")).then(pl.lit("name only, addr weak"))
                   .when(pl.col("addr_empty")).then(pl.lit("NO signal, addr EMPTY"))
                   .otherwise(pl.lit("NO signal, addr weak")))
pl.Config.set_tbl_width_chars(200)
print(x.group_by("cls").agg(pairs=pl.len(), share=pl.len() / x.height, in_cands=pl.col("inc").mean(),
                            indic=pl.col("indic").mean()).sort("pairs", descending=True))
miss = x.filter(pl.col("inc") == 0)
print(f"\nMISSED by current candidates: {miss.height:,} pairs")
print(miss.group_by("cls").agg(pairs=pl.len(), share_of_missed=pl.len() / miss.height).sort("pairs", descending=True))
