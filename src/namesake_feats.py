"""Namesake-resolution features for every candidate pair (competition-aware, per record).

When a record's canonical name equals the canonical name of several candidate S1 entities, the raw surface still
carries owner information: the record's legal-form set equals only one namesake's legal form -> that S1 is the owner
~76% of the time; an exact surface-spelling match to only one namesake -> 65-81% (train, namesakes.py).
Per pair (rid, oid):
  ns_self   this S1's canonical name equals the record's
  ns_n      number of candidate S1 for the record with that canonical name
  lg_rec    the record carries a legal form;  lg_eq  legal-form sets equal;  lg_cnt  namesakes with equal legal form;
  lg_uniq   lg_eq and this S1 is the only namesake with it
  sf_eq / sf_cnt / sf_uniq  same for the exact surface spelling (lowercase, punctuation collapsed)
Writes data/ns_{split}{tag}.parquet.
"""
import sys

import polars as pl

import normalize as N

D = "C:/ML/Challenges/amazon/work/data/"
NS_COLS = ["ns_self", "ns_n", "lg_rec", "lg_eq", "lg_cnt", "lg_uniq", "sf_eq", "sf_cnt", "sf_uniq"]


def legal_key(col):
    t = (pl.col(col).str.to_lowercase().str.replace_all(r"\b([a-z])\.", "$1").str.replace_all(r"[^a-z0-9]+", " ")
         .str.strip_chars().str.split(" "))
    return t.list.eval(pl.element().filter(pl.element().is_in(list(N.LEGAL)))).list.unique().list.sort().list.join(" ")


def surface(col):
    return pl.col(col).str.to_lowercase().str.replace_all(r"[^a-z0-9&]+", " ").str.strip_chars()


def main(split, src, tag):
    K = pl.read_parquet(f"{D}{src}.parquet", columns=["rid", "oid", "country"])
    outs = []
    for c in sorted(K["country"].unique().to_list()):
        s1 = pl.read_parquet(f"{D}p1_canon_{split}_{c}_s1.parquet", columns=["idx", "cnk"])
        o = pl.read_parquet(f"{D}p1_canon_{split}_{c}_o.parquet", columns=["idx", "cnk"])
        r1 = (pl.read_parquet(f"{D}{split}_source1.parquet", columns=["business_name"]).with_row_index("idx")
              .join(s1, on="idx").select(rid="idx", k1="cnk", l1=legal_key("business_name"), f1=surface("business_name")))
        r2 = (pl.concat([pl.read_parquet(f"{D}{split}_source{k}.parquet", columns=["business_name"]) for k in (2, 3)])
              .with_row_index("idx").join(o, on="idx")
              .select(oid="idx", k2="cnk", l2=legal_key("business_name"), f2=surface("business_name")))
        x = K.filter(pl.col("country") == c).select("rid", "oid").join(r1, on="rid").join(r2, on="oid")
        x = x.with_columns(ns_self=((pl.col("k1") == pl.col("k2")) & (pl.col("k1") != "")).cast(pl.Int8))
        x = x.with_columns(lg_eq=(pl.col("ns_self").cast(pl.Boolean) & (pl.col("l1") == pl.col("l2"))).cast(pl.Int8),
                           sf_eq=(pl.col("ns_self").cast(pl.Boolean) & (pl.col("f1") == pl.col("f2"))).cast(pl.Int8),
                           lg_rec=(pl.col("l2") != "").cast(pl.Int8))
        x = x.with_columns(ns_n=pl.col("ns_self").cast(pl.Int16).sum().over("oid"),
                           lg_cnt=pl.col("lg_eq").cast(pl.Int16).sum().over("oid"),
                           sf_cnt=pl.col("sf_eq").cast(pl.Int16).sum().over("oid"))
        x = x.with_columns(lg_uniq=((pl.col("lg_eq") == 1) & (pl.col("lg_cnt") == 1)).cast(pl.Int8),
                           sf_uniq=((pl.col("sf_eq") == 1) & (pl.col("sf_cnt") == 1)).cast(pl.Int8))
        outs.append(x.select("rid", "oid", *NS_COLS))
        print(f"{split} {c}: {x.height:,} pairs, namesake pairs {x['ns_self'].sum():,}, lg_uniq {x['lg_uniq'].sum():,}, "
              f"sf_uniq {x['sf_uniq'].sum():,}", flush=True)
    pl.concat(outs).write_parquet(f"{D}ns_{split}{tag}.parquet")


if __name__ == "__main__":
    main(*sys.argv[1:4])
