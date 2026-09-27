"""Where do the true pairs missed by blocking rank under the blocking score itself?
For the S1 entities (kept, harder split) that lost a true record at blocking, score every record that shares a key
(no top-K cut) and report the rank of the missed record on the S1 side (r1) and on the record side (r2), plus the
recall that larger top-K / reverse top-K would give. Also: how many missed pairs share no usable key at all.
"""
import gc
import sys
import time

import polars as pl

import block

D = "C:/ML/Challenges/amazon/work/data/"


def main(c="India", cap=5000):
    t = time.time()
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").filter(pl.col("country") == c).join(keep, on="rid", how="semi").select("rid", "oid")
    C = pl.read_parquet("D:/amazon_er/data/train_cand.parquet", columns=["rid", "oid"])
    miss = gt.join(C, on=["rid", "oid"], how="anti")
    del C
    gc.collect()
    s1, o = block.load("train")
    s1, o = s1.filter(pl.col("country") == c), o.filter(pl.col("country") == c)
    # load() re-indexes the other sources over all countries: keep the global oid
    k1, k2 = block.keys(s1), block.keys(o)
    df1 = k1.group_by("h").agg(d1=pl.len())
    df2 = k2.group_by("h").agg(d2=pl.len())
    n = s1.height + o.height
    import math
    dd = df1.join(df2, on="h").filter(pl.col("d1") * pl.col("d2") <= cap).with_columns(
        w=(math.log(n) - (pl.col("d1") + pl.col("d2")).cast(pl.Float64).log()).cast(pl.Float32))
    k1 = k1.join(dd.select("h", "w"), on="h")
    k2 = k2.join(dd.select("h"), on="h").select(pl.col("rid").alias("oid"), "h")
    print(f"[{c}] missed {miss.height:,} pairs of {gt.height:,}; keys ready ({time.time() - t:.0f}s)", flush=True)
    # S1 side: all records of the missed S1s
    mr = miss.select("rid").unique()
    outs = []
    rl = mr["rid"].sort()
    for i in range(0, rl.len(), 20_000):
        sub = k1.join(pl.DataFrame({"rid": rl[i:i + 20_000]}), on="rid", how="semi")
        p = sub.join(k2, on="h").group_by("rid", "oid").agg(score=pl.col("w").sum())
        p = p.with_columns(r1=pl.col("score").rank("ordinal", descending=True).over("rid"))
        outs.append(p.join(miss, on=["rid", "oid"], how="semi"))
        del p, sub
        gc.collect()
    s_side = pl.concat(outs)
    # record side: all S1s of the missed records
    ol = miss["oid"].unique().sort()
    outs = []
    for i in range(0, ol.len(), 40_000):
        sub = k2.join(pl.DataFrame({"oid": ol[i:i + 40_000]}), on="oid", how="semi")
        p = sub.join(k1, on="h").group_by("rid", "oid").agg(score=pl.col("w").sum())
        p = p.with_columns(r2=pl.col("score").rank("ordinal", descending=True).over("oid"))
        outs.append(p.join(miss, on=["rid", "oid"], how="semi").select("rid", "oid", "r2"))
        del p, sub
        gc.collect()
    r_side = pl.concat(outs)
    x = miss.join(s_side, on=["rid", "oid"], how="left").join(r_side, on=["rid", "oid"], how="left")
    print(f"[{c}] missed pairs sharing no usable key: {x['score'].is_null().sum():,} ({time.time() - t:.0f}s)")
    tot = gt.height
    base = tot - miss.height
    for K1, K2 in [(60, 3), (80, 3), (100, 3), (150, 3), (40, 10), (40, 20), (40, 50), (100, 10), (150, 20), (300, 50)]:
        got = x.filter((pl.col("r1") <= K1) | (pl.col("r2") <= K2)).height
        print(f"  top {K1:>3d} per S1 + top {K2:>2d} per record: recovers {got:>6,} -> recall {(base + got) / tot:.4f}")
    print(x.select(pl.col("r1").quantile(q).alias(f"r1_q{q}") for q in (0.25, 0.5, 0.75, 0.9)))
    print(x.select(pl.col("r2").quantile(q).alias(f"r2_q{q}") for q in (0.25, 0.5, 0.75, 0.9)))
    x.write_parquet(f"{D}miss_rank_{c}.parquet")


if __name__ == "__main__":
    main(*sys.argv[1:2])
