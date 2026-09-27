"""Classify the true pairs missed by the v5 candidate set by how much signal they carry."""
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

import features as F

D = F.D
gt = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid", "country", "src"])
found = pl.read_parquet(f"{D}train_oof_v9.parquet", columns=["rid", "oid", "y"]).filter(pl.col("y") == 1).select("rid", "oid")
miss = gt.join(found, on=["rid", "oid"], how="anti")
outs = []
for c in ["India", "US"]:
    s1, o = F.load_records("train", c)
    x = (miss.filter(pl.col("country") == c).join(s1.select("rid", "cs", "nums", "ac"), on="rid")
         .join(o.select(pl.col("rid").alias("oid"), cs2="cs", nums2="nums", ac2="ac", indic="indic"), on="oid"))
    coll = s1.select("rid", n_same=pl.len().over(pl.col("css")))
    x = x.join(coll, on="rid").with_columns(
        name_ts=pl.Series(cpdist(x["cs"].to_list(), x["cs2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)),
        addr_ts=pl.Series(cpdist(x["ac"].to_list(), x["ac2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)),
        num_share=pl.col("nums").list.set_intersection("nums2").list.eval(pl.element().str.len_chars()).list.max().fill_null(0),
        addr_empty=pl.col("ac2").str.len_chars() == 0)
    outs.append(x)
x = pl.concat(outs, how="diagonal")
x = x.with_columns(name_ok=pl.col("name_ts") >= 60, addr_ok=(pl.col("num_share") >= 2) | (pl.col("addr_ts") >= 60))
x = x.with_columns(cls=pl.when(pl.col("name_ok") & pl.col("addr_ok")).then(pl.lit("A name+addr (recoverable)"))
                   .when(pl.col("addr_ok")).then(pl.lit("B addr only (recoverable via address)"))
                   .when(pl.col("name_ok") & (pl.col("n_same") == 1)).then(pl.lit("C name only, unique name (recoverable)"))
                   .when(pl.col("name_ok")).then(pl.lit("D name only, ambiguous name (mostly unrecoverable)"))
                   .otherwise(pl.lit("E no signal (unrecoverable)")))
print(f"missed pairs: {x.height:,}")
print(x.group_by("cls", "country").agg(pairs=pl.len(), indic=pl.col("indic").mean(), empty_addr=pl.col("addr_empty").mean())
      .sort("cls", "country"))
x.select("rid", "oid", "country", "cls").write_parquet(f"{D}miss_v9.parquet")
