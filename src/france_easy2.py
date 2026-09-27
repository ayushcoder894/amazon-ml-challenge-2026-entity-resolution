"""Drill into the France bucket "name exact, house number differs or missing" (0.95 pairs per S1, France accepts 20%,
train truth 80% India / 53% US): split by record-has-no-number vs number-differs, S1-name unique among S1s or not,
and street similarity; compare France acceptance with the train truth rates of the same sub-buckets."""
import polars as pl

import decide

D = "C:/ML/Challenges/amazon/work/data/"
ARCH = "D:/amazon_er/archive/data/"
COLS = ["rid", "oid", "css_eq", "first_num_eq", "num_prefix", "addr_empty2", "legal_conflict", "n_len1", "n_len2",
        "n_css_s1_1", "a_tset", "w_tset"]


def sub(F):
    F = F.filter((pl.col("css_eq") == 1) & (pl.col("addr_empty2") == 0) & (pl.col("first_num_eq") == 0)
                 & (pl.col("num_prefix") == 0) & (pl.col("legal_conflict") == 0))
    return F.with_columns(
        num=pl.when(pl.col("n_len2") == 0).then(pl.lit("rec no number")).when(pl.col("n_len1") == 0).then(pl.lit("S1 no number"))
        .otherwise(pl.lit("numbers differ")),
        uniq=pl.when(pl.col("n_css_s1_1") <= 1).then(pl.lit("name unique")).otherwise(pl.lit("name shared")),
        street=pl.when(pl.col("w_tset") >= 90).then(pl.lit("street~")).otherwise(pl.lit("street differs")))


def main():
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").select("rid", "oid").with_columns(y=pl.lit(1, pl.Int8))
    Kin = pl.read_parquet(f"{D}train_oof_final_v12_twin.parquet", columns=["rid", "oid", "country"]).join(keep, on="rid", how="semi")
    Ftr = (pl.concat([pl.scan_parquet(f"{ARCH}train_feat_v11_fold{f}.parquet").select(COLS) for f in (0, 1)])
           .join(Kin.lazy(), on=["rid", "oid"], how="semi").collect())
    tr = sub(Ftr).join(Kin, on=["rid", "oid"]).join(gt, on=["rid", "oid"], how="left").with_columns(pl.col("y").fill_null(0))
    trs = tr.group_by("country", "num", "uniq", "street").agg(true=pl.col("y").mean()).pivot(
        on="country", index=["num", "uniq", "street"], values="true")
    Kf = pl.read_parquet(f"{D}test_final_v12_ce_cectx_twin.parquet", columns=["rid", "oid", "country", "p1"]).filter(pl.col("country") == "France")
    pf = decide.ef_predict(Kf.select("rid", "oid", pl.col("p1").alias("p2")), "p2", 0.0, 4.0, 0.05).with_columns(pred=pl.lit(1, pl.Int8))
    Ff = pl.scan_parquet(f"{ARCH}test_feat_v10_France.parquet").select(COLS).join(Kf.lazy().select("rid", "oid"), on=["rid", "oid"], how="semi").collect()
    fr = sub(Ff).join(Kf, on=["rid", "oid"]).join(pf, on=["rid", "oid"], how="left").with_columns(pl.col("pred").fill_null(0))
    n_fr = pl.read_parquet(f"{D}test_s1_norm.parquet", columns=["country"]).filter(pl.col("country") == "France").height
    t = fr.group_by("num", "uniq", "street").agg(fr_S1=pl.len() / n_fr, fr_p1=pl.col("p1").mean(), fr_pred=pl.col("pred").mean()).join(
        trs, on=["num", "uniq", "street"], how="left").rename({"India": "IN_true", "US": "US_true"})
    with pl.Config(tbl_rows=20, tbl_width_chars=200, float_precision=3, tbl_hide_column_data_types=True):
        print(t.sort("fr_S1", descending=True))
    fr.write_parquet("D:/amazon_er/dense/france_bucket_pairs.parquet")


if __name__ == "__main__":
    main()
