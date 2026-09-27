"""Sibling features for stage 2: compare each candidate record with the other records already confidently
assigned (by out-of-fold stage-1 probability) to the same S1 entity.

Records of one entity tend to resemble each other (same S2 formatting, same garbled name variant, same partial
address) - often more than they resemble the S1 record. A candidate close to its confident siblings is likely a
match; one that looks like the S1 but unlike every sibling is a likely false merge.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

import features as F

SIB_COLS = ["sib_n", "sib_name_max", "sib_addr_max", "sib_both_max", "sib_name_mean", "sib_p_best"]


def _cp(a, b, scorer):
    return cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def sibling_features(K: pl.DataFrame, split: str, country: str, conf=0.5, top_sib=3, chunk_rids=150_000):
    """K: all candidate pairs of one country with columns rid, oid, p1. Returns (rid, oid, SIB_COLS...)."""
    _, o = F.load_records(split, country)
    txt = o.select(pl.col("rid").alias("oid"), "cs", "ac")
    sib = (K.filter(pl.col("p1") >= conf)
           .filter(pl.col("p1").rank("ordinal", descending=True).over("rid") <= top_sib + 1)
           .select("rid", sib="oid", sp="p1"))
    rids = K["rid"].unique().sort()
    outs = []
    for i in range(0, len(rids), chunk_rids):
        r = rids.slice(i, chunk_rids)
        k = K.filter(pl.col("rid").is_in(r.implode())).select("rid", "oid")
        x = k.join(sib.filter(pl.col("rid").is_in(r.implode())), on="rid").filter(pl.col("sib") != pl.col("oid"))
        if x.height == 0:
            continue
        x = (x.join(txt, on="oid", how="left")
             .join(txt.rename({"oid": "sib", "cs": "cs_s", "ac": "ac_s"}), on="sib", how="left"))
        x = x.with_columns(
            sn=pl.Series(_cp(x["cs"].to_list(), x["cs_s"].to_list(), fuzz.token_set_ratio)),
            sa=pl.Series(_cp(x["ac"].to_list(), x["ac_s"].to_list(), fuzz.token_set_ratio)))
        x = x.with_columns(sb=(pl.col("sn") + pl.col("sa")) / 2)
        outs.append(x.group_by("rid", "oid").agg(
            sib_n=pl.len().cast(pl.UInt8),
            sib_name_max=pl.col("sn").max(), sib_addr_max=pl.col("sa").max(), sib_both_max=pl.col("sb").max(),
            sib_name_mean=pl.col("sn").mean(),
            sib_p_best=pl.col("sp").sort_by("sb", descending=True).first()))
    out = pl.concat(outs) if outs else pl.DataFrame(schema={"rid": pl.UInt32, "oid": pl.UInt32})
    return K.select("rid", "oid").join(out, on=["rid", "oid"], how="left").with_columns(
        pl.col("sib_n").fill_null(0))


def run_file(split, in_path, out_path):
    """Subprocess entry point: K (rid, oid, p1, country) parquet -> sibling feature parquet (keeps RAM isolated)."""
    K = pl.read_parquet(in_path)
    outs = [sibling_features(K.filter(pl.col("country") == c).select("rid", "oid", "p1"), split, c)
            for c in sorted(K["country"].unique().to_list())]
    pl.concat(outs).write_parquet(out_path)


if __name__ == "__main__":
    import sys
    run_file(*sys.argv[1:4])
