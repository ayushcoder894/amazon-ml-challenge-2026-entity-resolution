"""Remaining loss of the current best (joint3: v12 + dense add-on + CE, honest harder split), per country:
error type x record has address x (for misses) whether the true owner is the record's top candidate (threshold issue)
or a rival outranks it (ranking issue), with the number of candidate entities of the record.
"""
import polars as pl

import decide

D = "C:/ML/Challenges/amazon/work/data/"
DN = "D:/amazon_er/dense/"


def main(tag="joint3"):
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    s1 = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).join(keep, on="rid", how="semi")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").join(keep, on="rid", how="semi").select("rid", "oid")
    U = pl.read_parquet(f"{DN}train_{tag}.parquet", columns=["rid", "oid", "pj", "is_new"])
    U = U.with_columns(r2=pl.col("pj").rank("ordinal", descending=True).over("oid"), n2=pl.len().over("oid"))
    pred = decide.ef_predict(U.select("rid", "oid", pl.col("pj").alias("p2")), "p2", 0.0, 1.0, 0.05).select("rid", "oid")
    raw = pl.concat([pl.read_parquet(f"{D}train_source{k}.parquet", columns=["business_address"]) for k in (2, 3)]).with_row_index(
        "oid").select(pl.col("oid").cast(pl.UInt32), ae=pl.col("business_address").fill_null("").str.strip_chars() == "")
    tp = pred.join(gt, on=["rid", "oid"]).group_by("rid").agg(tp=pl.len())
    m = (s1.join(gt.group_by("rid").agg(nt=pl.len()), on="rid", how="left").join(pred.group_by("rid").agg(np_=pl.len()), on="rid", how="left")
         .join(tp, on="rid", how="left").fill_null(0))
    p = pl.when(pl.col("np_") > 0).then(pl.col("tp") / pl.col("np_")).otherwise(0.0)
    r = pl.when(pl.col("nt") > 0).then(pl.col("tp") / pl.col("nt")).otherwise(0.0)
    m = m.with_columns(f=pl.when((pl.col("nt") == 0) & (pl.col("np_") == 0)).then(1.0).when(pl.col("tp") == 0).then(0.0)
                       .otherwise(1.25 * p * r / (0.25 * p + r)))
    m = m.with_columns(loss=(1 - pl.col("f")) / pl.len().over("country"))
    fp = pred.join(gt, on=["rid", "oid"], how="anti").with_columns(t=pl.lit("FP"))
    fn = gt.join(pred, on=["rid", "oid"], how="anti").join(U.select("rid", "oid", "r2"), on=["rid", "oid"], how="left").with_columns(
        t=pl.when(pl.col("r2").is_null()).then(pl.lit("FN not a candidate"))
        .when(pl.col("r2") == 1).then(pl.lit("FN owner is top (threshold)")).otherwise(pl.lit("FN rival on top (ranking)"))).drop("r2")
    E = pl.concat([fp, fn]).join(raw, on="oid").join(m.select("rid", "country", "loss"), on="rid").join(
        U.select("oid", "n2").unique("oid"), on="oid", how="left")
    E = E.with_columns(share=pl.col("loss") / pl.len().over("rid"),
                       ncand=pl.when(pl.col("n2").is_null()).then(pl.lit("-")).when(pl.col("n2") == 1).then(pl.lit("1"))
                       .when(pl.col("n2") <= 3).then(pl.lit("2-3")).otherwise(pl.lit("4+")))
    print(m.group_by("country").agg(F=pl.col("f").mean()).sort("country"))
    with pl.Config(tbl_rows=40, tbl_width_chars=160, tbl_hide_column_data_types=True):
        for c in ("India", "US"):
            x = E.filter(pl.col("country") == c)
            print(f"\n=== {c}: total loss {x['share'].sum():.5f}")
            print(x.group_by("t", "ae").agg(pairs=pl.len(), loss=pl.col("share").sum().round(5)).sort("loss", descending=True))
            print(x.group_by("t", "ae", "ncand").agg(pairs=pl.len(), loss=pl.col("share").sum().round(5)).sort("loss", descending=True).head(10))
    E.write_parquet(f"{DN}loss_{tag}.parquet")


if __name__ == "__main__":
    main()
