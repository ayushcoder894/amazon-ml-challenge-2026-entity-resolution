"""Smaller synthetic world for one country (test has half as many US entities as train).

world = a hash-selected share of the country's Source-1 entities with their own records, plus the same share of the
unowned records; then the test-like drop (keep_v11: 18.6% of the entities removed, their records kept). Every feature
that depends on which entities / records exist is recomputed:
  per record  : r2 / cis_rank2 / wis_rank2 (rank shift past removed pairs), rel2, gap2, ncand2, cis_gap2
  per entity  : r1 / r1src (rank shift past removed records), rel1, gap1, ncand1
  frequencies : n_css_* / n_soft_* over the world's entities and records
Writes data/train_feat{out}_fold{f}.parquet, data/keep{out}.parquet (entities) and data/world{out}.parquet (records).
"""
import sys
import time

import polars as pl

D = "C:/ML/Challenges/amazon/work/data/"
R2 = ["r2", "cis_rank2", "wis_rank2"]
R1 = ["r1"]
PER_OID = R2 + ["rel2", "gap2", "ncand2", "cis_gap2"]
PER_RID = R1 + ["r1src", "rel1", "gap1", "ncand1"]
FREQ = ["n_css_s1_1", "n_css_s1_2", "n_soft_s1_1", "n_soft_s1_2", "n_css_o_1", "n_css_o_2", "n_soft_o_1", "n_soft_o_2"]


def shift(df, col, group, removed):
    """new rank = old rank minus the removed rows ranked ahead of it in the same group (original tie order kept)."""
    g = [group] if isinstance(group, str) else list(group)
    return df.sort(g + [col]).with_columns(
        (pl.col(col).cast(pl.Int32) - (pl.col(removed).cast(pl.Int32).cum_sum().over(group)
                                       - pl.col(removed).cast(pl.Int32))).cast(pl.UInt16).alias(col))


def name_keys(country):
    from features import load_records
    s1, o = load_records("train", country)
    return (s1.select("rid", css=pl.col("css").hash(), soft=pl.col("soft").hash()),
            o.select(oid=pl.col("rid"), css=pl.col("css").hash(), soft=pl.col("soft").hash()))


def main(country="US", share=0.5, out="_w1", src="_v10"):
    t = time.time()
    share = float(share)
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").filter(pl.col("country") == country).select(owner="rid", oid="oid")
    s1 = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == country)
    half = s1.filter((pl.col("rid").hash(seed=200) % 1000) < int(share * 1000)).select("rid")
    keep = half.join(pl.read_parquet(f"{D}keep_v11.parquet"), on="rid", how="semi")
    s1k, ok = name_keys(country)
    rec = ok.select("oid").join(gt, on="oid", how="left").with_columns(
        w=pl.col("owner").is_in(half["rid"].implode())
        | (pl.col("owner").is_null() & ((pl.col("oid").hash(seed=201) % 1000) < int(share * 1000))))
    world = rec.filter(pl.col("w")).select("oid")
    keep.write_parquet(f"{D}keep{out}.parquet")
    world.write_parquet(f"{D}world{out}.parquet")
    print(f"{country}: world of {half.height:,} entities ({keep.height:,} kept in S1) and {world.height:,} records "
          f"({world.height / keep.height:.2f} per kept S1)", flush=True)
    # name frequencies inside the world
    ks, ko = s1k.join(keep, on="rid", how="semi"), ok.join(world, on="oid", how="semi")
    cnt = {(side, k): (ks if side == "s1" else ko).group_by(k).agg(pl.len().cast(pl.UInt32).alias("n"))
           for side in ("s1", "o") for k in ("css", "soft")}

    def freq(tab, idcol, sfx):
        x = tab
        for side in ("s1", "o"):
            for k in ("css", "soft"):
                x = x.join(cnt[(side, k)].rename({"n": f"n_{k}_{side}_{sfx}"}), on=k, how="left")
        return x.drop("css", "soft").with_columns(pl.exclude(idcol).fill_null(0))
    rf, of = freq(s1k, "rid", "1"), freq(ok, "oid", "2")
    files = [f"{D}train_feat{src}_fold{f}.parquet" for f in (0, 1)]
    cols_needed = ["rid", "oid", "src", "score", "core_isum", "words_isum"] + R2 + R1 + ["r1src"]
    slim = (pl.concat([pl.scan_parquet(p).filter(pl.col("country") == country).select(cols_needed) for p in files])
            .collect())
    slim = slim.with_columns(ks_=pl.col("rid").is_in(keep["rid"].implode()),
                             kw_=pl.col("oid").is_in(world["oid"].implode()))
    slim = slim.filter(pl.col("kw_") | ~pl.col("ks_"))  # removed records only matter for per-rid ranks of kept S1
    slim = slim.with_columns(rm_s=~pl.col("ks_"), rm_o=~pl.col("kw_"))
    for c in R2:
        slim = shift(slim, c, "oid", "rm_s")
    for c in R1:
        slim = shift(slim, c, "rid", "rm_o")
    slim = shift(slim, "r1src", ["rid", "src"], "rm_o")
    slim = slim.filter(pl.col("ks_") & pl.col("kw_"))
    s = pl.col("score")
    per = slim.with_columns(
        rel2=s / s.max().over("oid"), gap2=s.max().over("oid") - s, ncand2=pl.len().over("oid").cast(pl.UInt16),
        cis_gap2=pl.col("core_isum").max().over("oid") - pl.col("core_isum"),
        rel1=s / s.max().over("rid"), gap1=s.max().over("rid") - s, ncand1=pl.len().over("rid").cast(pl.UInt16),
    ).select("rid", "oid", *PER_OID, *PER_RID).with_columns(pl.col(pl.Float64).cast(pl.Float32))
    del slim
    print(f"recomputed per-record / per-entity features for {per.height:,} pairs ({time.time() - t:.0f}s)", flush=True)
    for f, p in enumerate(files):
        sch = pl.scan_parquet(p).collect_schema()
        cols = sch.names()
        (pl.scan_parquet(p).filter(pl.col("country") == country).join(per.lazy().select("rid", "oid"), on=["rid", "oid"], how="semi")
         .drop(PER_OID + PER_RID + FREQ).join(per.lazy(), on=["rid", "oid"], how="left")
         .join(rf.lazy(), on="rid", how="left").join(of.lazy(), on="oid", how="left")
         .with_columns([pl.col(c).cast(sch[c]) for c in PER_OID + PER_RID + FREQ]).select(cols)
         .sink_parquet(f"{D}train_feat{out}_fold{f}.parquet", row_group_size=500_000))
        n = pl.scan_parquet(f"{D}train_feat{out}_fold{f}.parquet").select(pl.len()).collect().item()
        print(f"fold {f}: {n:,} rows ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:5])
