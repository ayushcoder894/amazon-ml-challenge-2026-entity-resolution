"""Test-like training data: remove 18.6% of Source-1 entities (test has 5.75 records per S1 vs 4.68 in train, i.e. about
that share of entities is absent from S1 while their records remain) and recompute every feature that depends on
which S1 entities exist:
  - per-record competition features (rel2, gap2, ncand2, cis_gap2 from the kept pairs; the ordinal ranks r2, cis_rank2,
    wis_rank2 by shifting the original rank past the removed entities, which keeps the original tie order),
  - S1 name frequencies (n_css_s1_*, n_soft_s1_*: how many S1 entities carry the same name).
Writes data/train_feat{out}_fold{f}.parquet and data/keep{out}.parquet.
"""
import gc
import sys
import time

import polars as pl

D = "C:/ML/Challenges/amazon/work/data/"
RANKS = ["r2", "cis_rank2", "wis_rank2"]
PER_OID = RANKS + ["rel2", "gap2", "ncand2", "cis_gap2"]
FREQ = ["n_css_s1_1", "n_css_s1_2", "n_soft_s1_1", "n_soft_s1_2"]


def name_freq(keep, countries=("India", "US")):
    """(rid -> kept-S1 counts of its css / soft name) and (oid -> the same for the record's name)."""
    from features import load_records
    r_tabs, o_tabs = [], []
    for c in countries:
        s1, o = load_records("train", c)
        s1 = s1.select("rid", css=pl.col("css").hash(), soft=pl.col("soft").hash())
        o = o.select(oid=pl.col("rid"), css=pl.col("css").hash(), soft=pl.col("soft").hash())
        ks = s1.join(keep, on="rid", how="semi")
        cnt = {k: ks.group_by(k).agg(pl.len().cast(pl.UInt32).alias(f"n_{k}")) for k in ("css", "soft")}
        r_tabs.append(s1.join(cnt["css"], on="css", how="left").join(cnt["soft"], on="soft", how="left")
                      .select("rid", n_css_s1_1="n_css", n_soft_s1_1="n_soft"))
        o_tabs.append(o.join(cnt["css"], on="css", how="left").join(cnt["soft"], on="soft", how="left")
                      .select("oid", n_css_s1_2="n_css", n_soft_s1_2="n_soft"))
        del s1, o, ks
        gc.collect()
    fill = lambda df: df.with_columns(pl.exclude("rid", "oid").fill_null(0))
    return fill(pl.concat(r_tabs)), fill(pl.concat(o_tabs))


def main(src="_v10", out="_v11", drop=0.186, seed=0):
    t = time.time()
    s1 = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"])
    keep = s1.filter((pl.col("rid").hash(seed=100 + seed) % 1000) >= int(drop * 1000)).select("rid")
    keep.write_parquet(f"{D}keep{out}.parquet")
    rfreq, ofreq = name_freq(keep)
    print(f"kept S1 {keep.height:,}; name frequencies recomputed ({time.time() - t:.0f}s)", flush=True)
    files = [f"{D}train_feat{src}_fold{f}.parquet" for f in (0, 1)]
    slim = (pl.concat([pl.scan_parquet(p).select("rid", "oid", "score", "core_isum", "words_isum", *RANKS) for p in files])
            .collect().with_columns(k=pl.col("rid").is_in(keep["rid"].implode())))
    # new rank = old rank minus the removed pairs ranked ahead of it on the same record
    for r in RANKS:
        slim = slim.sort("oid", r).with_columns(
            (pl.col(r).cast(pl.Int32) - ((~pl.col("k")).cast(pl.Int32).cum_sum().over("oid")
                                         - (~pl.col("k")).cast(pl.Int32))).cast(pl.UInt16).alias(r))
    slim = slim.filter(pl.col("k")).drop("k")
    s = pl.col("score")
    per = slim.with_columns(
        rel2=s / s.max().over("oid"), gap2=s.max().over("oid") - s,
        ncand2=pl.len().over("oid").cast(pl.UInt16),
        cis_gap2=pl.col("core_isum").max().over("oid") - pl.col("core_isum"),
    ).select("rid", "oid", *PER_OID).with_columns(pl.col(pl.Float64).cast(pl.Float32))
    del slim
    gc.collect()
    print(f"per-record competition features recomputed for {per.height:,} pairs ({time.time() - t:.0f}s)", flush=True)
    for f, p in enumerate(files):
        sch = pl.scan_parquet(p).collect_schema()
        cols = sch.names()
        (pl.scan_parquet(p).join(keep.lazy(), on="rid", how="semi").drop(PER_OID + FREQ)
         .join(per.lazy(), on=["rid", "oid"], how="left")
         .join(rfreq.lazy(), on="rid", how="left").join(ofreq.lazy(), on="oid", how="left")
         .with_columns([pl.col(c).cast(sch[c]) for c in PER_OID + FREQ]).select(cols)
         .sink_parquet(f"{D}train_feat{out}_fold{f}.parquet", row_group_size=500_000))
        n = pl.scan_parquet(f"{D}train_feat{out}_fold{f}.parquet").select(pl.len()).collect().item()
        print(f"fold {f}: {n:,} rows ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
