"""Blocking recall + macro F0.5 scorer (identical to the official definition)."""
import polars as pl

D = "C:/ML/Challenges/amazon/work/data/"


def gt_pairs_rid(split="train"):
    """Ground-truth pairs as (rid, oid) using the same indexing as block.load()."""
    s1 = pl.read_parquet(f"{D}{split}_s1_norm.parquet", columns=["rid", "entity_id", "country"])
    o = pl.concat([pl.read_parquet(f"{D}{split}_s2_norm.parquet", columns=["entity_id"]),
                   pl.read_parquet(f"{D}{split}_s3_norm.parquet", columns=["entity_id"])]).with_row_index("oid")
    p = pl.read_parquet(f"{D}train_pairs.parquet")
    p = (p.join(s1.select(pl.col("entity_id").alias("s1"), "rid", "country"), on="s1")
         .join(o.select(pl.col("entity_id").alias("m"), "oid"), on="m"))
    return s1, o, p.select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32), "src", "country")


def macro_f05(pred: pl.DataFrame, gt: pl.DataFrame, s1_ids: pl.Series) -> float:
    """pred/gt: pair tables (rid, oid). s1_ids: all S1 rids in the eval set (singletons included)."""
    base = pl.DataFrame({"rid": s1_ids.cast(pl.UInt32)})
    tp = pred.join(gt, on=["rid", "oid"]).group_by("rid").agg(tp=pl.len())
    npred = pred.group_by("rid").agg(np_=pl.len())
    ngt = gt.group_by("rid").agg(ng=pl.len())
    m = (base.join(tp, on="rid", how="left").join(npred, on="rid", how="left").join(ngt, on="rid", how="left")
         .fill_null(0))
    p = pl.when(pl.col("np_") > 0).then(pl.col("tp") / pl.col("np_")).otherwise(0.0)
    r = pl.when(pl.col("ng") > 0).then(pl.col("tp") / pl.col("ng")).otherwise(0.0)
    m = m.with_columns(p=p, r=r)
    f = (pl.when((pl.col("ng") == 0) & (pl.col("np_") == 0)).then(1.0)
         .when(pl.col("tp") == 0).then(0.0)
         .otherwise(1.25 * pl.col("p") * pl.col("r") / (0.25 * pl.col("p") + pl.col("r"))))
    return m.select(f.mean()).item()


def blocking_report(cand: pl.DataFrame, gt: pl.DataFrame, n_s1: int):
    c = cand.select("rid", "oid", "r1", "r2").join(
        gt.select("rid", "oid", y=pl.lit(1, pl.UInt8)), on=["rid", "oid"], how="left").with_columns(pl.col("y").fill_null(0))
    tot = gt.height
    print(f"GT pairs {tot:,}; cand pairs {c.height:,}; recall ceiling {c['y'].sum() / tot:.4f}")
    exprs = []
    for k in [1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 60]:
        for rk in [0, 1, 3]:
            m = (pl.col("r1") <= k) | (pl.col("r2") <= rk)
            exprs += [(m.cast(pl.UInt32).sum() / n_s1).alias(f"pp_{k}_{rk}"),
                      ((m & (pl.col("y") == 1)).cast(pl.UInt32).sum() / tot).alias(f"rec_{k}_{rk}")]
    r = c.select(exprs).row(0, named=True)
    rows = [dict(k=k, recall_r0=r[f"rec_{k}_0"], recall_r1=r[f"rec_{k}_1"], recall_r3=r[f"rec_{k}_3"],
                 ppS1_r0=r[f"pp_{k}_0"], ppS1_r1=r[f"pp_{k}_1"], ppS1_r3=r[f"pp_{k}_3"])
            for k in [1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 60]]
    print(pl.DataFrame(rows))
    hit = c.filter(pl.col("y") == 1).select("rid", "oid")
    print("recall by country/src:")
    print(gt.join(hit.with_columns(h=pl.lit(1)), on=["rid", "oid"], how="left").group_by("country", "src")
          .agg(n=pl.len(), rec=pl.col("h").fill_null(0).mean()).sort("country", "src"))
