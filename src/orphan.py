"""Orphan-cluster features for stage 2 (test-like condition: some Source-1 entities are absent, their records remain).

A name-only record "First Business Dynamics" is easily merged into the one Source-1 entity of that name - unless
other records with the same name exist that no Source-1 entity explains ("First Business Dynamics LLC | 2780
Mountain View Dr, Salem, OR" three times, no candidate). Such an unexplained same-name cluster that agrees on a house
number is the footprint of an absent entity that owns the ambiguous record.

Per candidate pair (rid, oid), with peers = the other records of the country carrying the same sorted core name:
  orph_n5 / orph_n1  peers with an address whose best stage-1 probability to any S1 is < 0.5 / < 0.1
  orph_clu           largest group of such unexplained peers (p < 0.5) sharing one house number
  orph_clu_s1eq      that group's house number equals the candidate S1's house number
  orph_nclu2         number of such groups with at least 2 records (each one = a likely absent entity)
  peer_n, orph_share all peers, share unexplained
Stage-1 probabilities are out-of-fold, so the features are honest.
"""
import os
import sys

import polars as pl

import features as F

D = "C:/ML/Challenges/amazon/work/data/"
ORPH_COLS = ["orph_n5", "orph_n1", "orph_clu", "orph_clu_s1eq", "peer_n", "orph_share", "orph_nclu2"]


def keys(split, country):
    """(oid, css, n1, aempty) for records and (rid, n1_s1) for S1, cached on disk."""
    p_o, p_s = f"{D}reckeys_{split}_{country}.parquet", f"{D}s1keys_{split}_{country}.parquet"
    if not (os.path.exists(p_o) and os.path.exists(p_s)):
        s1, o = F.load_records(split, country)
        o.select(oid="rid", css=pl.when(pl.col("css") != "").then(pl.col("css").hash()),
                 n1=pl.when(pl.col("n1") != "").then(pl.col("n1").hash()),
                 aempty=(pl.col("ac") == "")).write_parquet(p_o)
        s1.select("rid", n1_s1=pl.when(pl.col("n1") != "").then(pl.col("n1").hash())).write_parquet(p_s)
    o, s1 = pl.read_parquet(p_o), pl.read_parquet(p_s)
    if os.environ.get("WORLD") and split == "train":  # a sub-world (make_world.py): only its records are peers
        o = o.join(pl.read_parquet(f"{D}world{os.environ['WORLD']}.parquet"), on="oid", how="semi")
    return o, s1


def orphan_features(K, split, country):
    """K: candidate pairs of one country (rid, oid, p1). Returns (rid, oid, ORPH_COLS...)."""
    o, s1 = keys(split, country)
    pmax = K.group_by("oid").agg(pmax=pl.col("p1").max())
    r = (o.join(pmax, on="oid", how="left").with_columns(pl.col("pmax").fill_null(0.0))
         .with_columns(u5=(pl.col("pmax") < 0.5) & ~pl.col("aempty") & pl.col("css").is_not_null(),
                       u1=(pl.col("pmax") < 0.1) & ~pl.col("aempty") & pl.col("css").is_not_null()))
    g = r.filter(pl.col("css").is_not_null()).group_by("css").agg(
        peer=pl.len(), U5=pl.col("u5").sum(), U1=pl.col("u1").sum())
    cl = r.filter(pl.col("u5") & pl.col("n1").is_not_null()).group_by("css", "n1").agg(c=pl.len())
    clu = (cl.sort(["css", "c"], descending=[False, True])
           .group_by("css", maintain_order=True).agg(top1=pl.col("c").first(), top1_n1=pl.col("n1").first(),
                                                     top2=pl.col("c").get(1, null_on_oob=True),
                                                     N2=(pl.col("c") >= 2).sum()))
    x = (K.select("rid", "oid").join(r.select("oid", "css", "n1", "u5", "u1"), on="oid", how="left")
         .join(g, on="css", how="left").join(clu, on="css", how="left").join(s1, on="rid", how="left")
         .join(cl.rename({"c": "c_self"}), on=["css", "n1"], how="left")
         .with_columns(pl.col("peer", "U5", "U1", "top1", "top2", "N2", "c_self").fill_null(0)))
    self5, self1 = pl.col("u5").cast(pl.Int32), pl.col("u1").cast(pl.Int32)
    in_top = pl.col("u5") & (pl.col("n1") == pl.col("top1_n1")).fill_null(False)
    x = x.with_columns(
        orph_n5=(pl.col("U5").cast(pl.Int32) - self5),
        orph_n1=(pl.col("U1").cast(pl.Int32) - self1),
        orph_clu=pl.when(in_top).then(pl.max_horizontal(pl.col("top1").cast(pl.Int32) - 1, pl.col("top2").cast(pl.Int32)))
        .otherwise(pl.col("top1").cast(pl.Int32)),
        orph_clu_s1eq=((pl.col("top1_n1") == pl.col("n1_s1")) & (pl.col("top1") > 0)).fill_null(False).cast(pl.Int8),
        peer_n=(pl.col("peer").cast(pl.Int32) - 1).clip(0),
        orph_nclu2=(pl.col("N2").cast(pl.Int32) - (pl.col("u5").fill_null(False) & (pl.col("c_self") == 2)).cast(pl.Int32)),
    ).with_columns(orph_share=pl.col("orph_n5") / pl.col("peer_n").clip(1))
    # records without a usable name get -1 everywhere
    x = x.with_columns([pl.when(pl.col("css").is_null()).then(-1).otherwise(pl.col(c)).cast(pl.Float32).alias(c)
                        for c in ORPH_COLS])
    return x.select("rid", "oid", *ORPH_COLS)


def run_file(split, in_path, out_path):
    """Subprocess entry point: K (rid, oid, p1, country) parquet -> orphan feature parquet."""
    K = pl.read_parquet(in_path)
    outs = [orphan_features(K.filter(pl.col("country") == c).select("rid", "oid", "p1"), split, c)
            for c in sorted(K["country"].unique().to_list())]
    pl.concat(outs).write_parquet(out_path)


if __name__ == "__main__":
    run_file(*sys.argv[1:4])
