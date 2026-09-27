"""Does France (unseen country, stage-1 + ef(0,4)) under-accept easy pairs like the LOCO model does?
Buckets of simple pair properties: train truth rate (India / US, harder split) and in-country predicted share vs the
France predicted share (v12_FRp1 France decisions) and France mean p1. A bucket that is ~99% true in both train
countries but rarely accepted in France is a label-free fix candidate.
"""
import polars as pl

import decide

D = "C:/ML/Challenges/amazon/work/data/"
ARCH = "D:/amazon_er/archive/data/"
COLS = ["rid", "oid", "css_eq", "addr_empty2", "first_num_eq", "n_tset", "a_tset", "legal_conflict", "num_prefix"]


def bucket(F):
    return F.with_columns(
        name=pl.when(pl.col("css_eq") == 1).then(pl.lit("name exact")).when(pl.col("n_tset") >= 90).then(pl.lit("name ~")).otherwise(pl.lit("name differs")),
        addr=pl.when(pl.col("addr_empty2") == 1).then(pl.lit("no address")).when(pl.col("first_num_eq") == 1).then(
            pl.when(pl.col("a_tset") >= 90).then(pl.lit("num= street~")).otherwise(pl.lit("num= street differs")))
        .when(pl.col("num_prefix") == 1).then(pl.lit("num prefix")).otherwise(pl.lit("num differs/missing")),
        legal=pl.when(pl.col("legal_conflict") == 1).then(pl.lit("legal conflict")).otherwise(pl.lit("")))


def main():
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").select("rid", "oid").with_columns(y=pl.lit(1, pl.Int8))
    Kin = pl.read_parquet(f"{D}train_oof_final_v12_twin.parquet", columns=["rid", "oid", "country", "pf"]).join(keep, on="rid", how="semi")
    pin = decide.ef_predict(Kin.select("rid", "oid", pl.col("pf").alias("p2")), "p2", 0.0, 1.0, 0.05).with_columns(pred=pl.lit(1, pl.Int8))
    Ftr = (pl.concat([pl.scan_parquet(f"{ARCH}train_feat_v11_fold{f}.parquet").select(COLS) for f in (0, 1)])
           .join(Kin.lazy().select("rid", "oid"), on=["rid", "oid"], how="semi").collect())
    tr = (bucket(Ftr).join(Kin, on=["rid", "oid"]).join(gt, on=["rid", "oid"], how="left").join(pin, on=["rid", "oid"], how="left")
          .with_columns(pl.col("y").fill_null(0), pl.col("pred").fill_null(0)))
    trs = tr.group_by("country", "name", "addr", "legal").agg(n=pl.len(), true=pl.col("y").mean(), pred=pl.col("pred").mean())
    trs = trs.pivot(on="country", index=["name", "addr", "legal"], values=["n", "true", "pred"])
    Kf = pl.read_parquet(f"{D}test_final_v12_ce_cectx_twin.parquet", columns=["rid", "oid", "country", "p1"]).filter(pl.col("country") == "France")
    pf = decide.ef_predict(Kf.select("rid", "oid", pl.col("p1").alias("p2")), "p2", 0.0, 4.0, 0.05).with_columns(pred=pl.lit(1, pl.Int8))
    Ff = pl.scan_parquet(f"{ARCH}test_feat_v10_France.parquet").select(COLS).join(Kf.lazy().select("rid", "oid"), on=["rid", "oid"], how="semi").collect()
    fr = bucket(Ff).join(Kf, on=["rid", "oid"]).join(pf, on=["rid", "oid"], how="left").with_columns(pl.col("pred").fill_null(0))
    n_fr = pl.read_parquet(f"{D}test_s1_norm.parquet", columns=["country"]).filter(pl.col("country") == "France").height
    frs = fr.group_by("name", "addr", "legal").agg(n_France=pl.len(), p1_France=pl.col("p1").mean(), pred_France=pl.col("pred").mean())
    t = frs.join(trs, on=["name", "addr", "legal"], how="left").with_columns(
        per_S1_France=pl.col("n_France") / n_fr, missed_if_true=(pl.col("n_France") * (pl.col("true_US") - pl.col("pred_France"))) / n_fr)
    t.write_parquet("D:/amazon_er/dense/france_easy.parquet")
    with pl.Config(tbl_rows=40, tbl_width_chars=250, float_precision=3, tbl_hide_column_data_types=True):
        print(t.select("name", "addr", "legal", fr_S1="per_S1_France", fr_p1="p1_France", fr_pred="pred_France",
                       IN_true="true_India", US_true="true_US").sort("fr_S1", descending=True).head(16))


if __name__ == "__main__":
    main()
