"""Namesake name-only records: does 'the namesake S1 with fewer other confident matches owns it' pick the owner,
and does the model already exploit it? Harder split, final OOF probabilities."""
import sys

import polars as pl

D = "C:/ML/Challenges/amazon/work/data/"


def main(c="US"):
    K = pl.read_parquet(f"{D}train_oof_final_v12_twin.parquet", columns=["rid", "oid", "y", "pf", "country"]).filter(pl.col("country") == c)
    s1 = pl.read_parquet(f"{D}p1_canon_train_{c}_s1.parquet", columns=["idx", "cnk"])
    o = pl.read_parquet(f"{D}p1_canon_train_{c}_o.parquet", columns=["idx", "cnk", "aempty"])
    conf = K.filter(pl.col("pf") >= 0.9).group_by("rid").agg(nconf=pl.len(), sconf=pl.col("pf").sum())
    x = (K.join(s1.select(rid="idx", k1="cnk"), on="rid").join(o.select(oid="idx", k2="cnk", ae="aempty"), on="oid")
         .filter(pl.col("ae") & (pl.col("k1") == pl.col("k2")) & (pl.col("k1") != "")))
    x = x.with_columns(nc=pl.len().over("oid")).filter(pl.col("nc") == 2)             # exactly two namesake candidates
    x = x.join(conf, on="rid", how="left").with_columns(pl.col("nconf").fill_null(0))
    x = x.with_columns(own=pl.col("y").sum().over("oid")).filter(pl.col("own") == 1)   # owner among the two
    x = x.with_columns(fewer=(pl.col("nconf") < pl.col("nconf").max().over("oid")) & (pl.col("nconf").n_unique().over("oid") == 2),
                       tie=pl.col("nconf").n_unique().over("oid") == 1,
                       top=pl.col("pf") == pl.col("pf").max().over("oid"))
    rec = x.filter(pl.col("y") == 1)
    print(f"\n=== {c}: {rec.height:,} name-only records with exactly 2 namesake candidates, owner among them")
    print(f"  tie on confident-match count: {rec['tie'].mean():.3f}")
    nt = rec.filter(~pl.col("tie"))
    print(f"  no tie: owner is the one with FEWER other confident matches: {nt['fewer'].mean():.3f} (n={nt.height:,})")
    print(f"  model: owner has the higher probability: {rec['top'].mean():.3f}; mean owner p {rec['pf'].mean():.3f}; "
          f"owner predicted (p>=0.5) {(rec['pf'] >= 0.5).mean():.3f}")
    print(f"  model when no tie: owner higher p {nt['top'].mean():.3f}")
    print(x.filter(pl.col("y") == 1).group_by("nconf").agg(n=pl.len()).sort("nconf").head(8).transpose(include_header=True))
    print(x.filter(pl.col("y") == 0).group_by("nconf").agg(n=pl.len()).sort("nconf").head(8).transpose(include_header=True))


if __name__ == "__main__":
    main(*sys.argv[1:2])
