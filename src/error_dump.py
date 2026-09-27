"""Dump every case that costs score (train, harder test-like split, honest out-of-fold final probabilities).

analysis/errors_pairs.csv   one row per wrong pair:
    FP              predicted match that is wrong (with the record's true owner, if any)
    FN_candidate    true match that was a candidate but not predicted
    FN_missing      true match that never became a candidate
analysis/errors_by_s1.csv   one row per Source-1 entity with F0.5 < 1 (true set vs predicted set, loss)
analysis/errors_summary.csv score loss by country x error type x category
Score loss of an entity = (1 - F0.5) / #entities of its country (so the losses of a country sum to 1 - macro F0.5).
"""
import os

import polars as pl

import decide
from p1_catalog import addr_rel, name_rel
from p1_decoy2 import refine
from p2_france_diag import addr_cls, canon

D = "C:/ML/Challenges/amazon/work/data/"
OUT = "C:/ML/Challenges/amazon/work/analysis/"


def write_csv(df, path):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:  # BOM: Excel shows native scripts correctly
        f.write(df.write_csv())


def main():
    os.makedirs(OUT, exist_ok=True)
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    s1c = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).join(keep, on="rid", how="semi")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").join(keep, on="rid", how="semi").select("rid", "oid")
    own = pl.read_parquet(f"{D}train_gt_rid.parquet").select(owner="rid", oid="oid")
    K = pl.read_parquet(f"{D}train_oof_final_v12_twin.parquet", columns=["rid", "oid", "pf", "country"])
    pred = decide.ef_predict(K.select("rid", "oid", pl.col("pf").alias("p2")), "p2", 0.0, 1.0, 0.05).select("rid", "oid")
    # per-entity F0.5 and loss
    tp = pred.join(gt, on=["rid", "oid"]).group_by("rid").agg(tp=pl.len())
    m = (s1c.join(gt.group_by("rid").agg(n_true=pl.len()), on="rid", how="left")
         .join(pred.group_by("rid").agg(n_pred=pl.len()), on="rid", how="left").join(tp, on="rid", how="left").fill_null(0))
    p = pl.when(pl.col("n_pred") > 0).then(pl.col("tp") / pl.col("n_pred")).otherwise(0.0)
    r = pl.when(pl.col("n_true") > 0).then(pl.col("tp") / pl.col("n_true")).otherwise(0.0)
    m = m.with_columns(f05=pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") == 0)).then(1.0)
                       .when(pl.col("tp") == 0).then(0.0).otherwise(1.25 * p * r / (0.25 * p + r)))
    m = m.with_columns(loss=(1 - pl.col("f05")) / pl.len().over("country"))
    for c in ("India", "US"):
        x = m.filter(pl.col("country") == c)
        print(f"{c}: macro F0.5 {x['f05'].mean():.5f}; entities with F0.5 < 1: {(x['f05'] < 1).sum():,} of {x.height:,}")
    # error pairs
    fp = pred.join(gt, on=["rid", "oid"], how="anti").with_columns(type=pl.lit("FP"))
    fn = gt.join(pred, on=["rid", "oid"], how="anti")
    fn = fn.join(K.select("rid", "oid"), on=["rid", "oid"], how="left", coalesce=True).with_columns(
        type=pl.when(pl.struct("rid", "oid").is_in(K.select(pl.struct("rid", "oid")).to_series().implode()))
        .then(pl.lit("FN_candidate")).otherwise(pl.lit("FN_missing")))
    E = pl.concat([fp, fn.select("rid", "oid", "type")]).join(K.select("rid", "oid", "pf"), on=["rid", "oid"], how="left")
    E = E.join(s1c, on="rid").join(own, on="oid", how="left").join(m.select("rid", "f05", "loss", "n_true", "n_pred"), on="rid")
    raw1 = pl.read_parquet(f"{D}train_source1.parquet").with_row_index("rid").select(
        "rid", s1_id="entity_id", s1_name="business_name", s1_address="business_address")
    raw2 = pl.concat([pl.read_parquet(f"{D}train_source{k}.parquet") for k in (2, 3)]).with_row_index("oid").select(
        "oid", rec_id="entity_id", rec_name="business_name", rec_address="business_address")
    E = (E.join(raw1, on="rid").join(raw2, on="oid")
         .join(raw1.rename({"rid": "owner", "s1_id": "true_owner_id", "s1_name": "true_owner_name", "s1_address": "true_owner_address"}),
               on="owner", how="left")
         .with_columns(true_owner_status=pl.when(pl.col("type") != "FP").then(pl.lit("this entity"))
                       .when(pl.col("owner").is_null()).then(pl.lit("nobody (decoy)"))
                       .when(pl.col("owner").is_in(keep["rid"].implode())).then(pl.lit("another entity"))
                       .otherwise(pl.lit("an entity absent from S1"))))
    # generator category of each error pair
    cats = []
    for c in ("India", "US"):
        s1, o, vocab = canon("train", c)
        x = (E.filter(pl.col("country") == c).select("rid", "oid")
             .join(s1.select(rid="idx", cn1="cn", cns1="cns", cak1="cak"), on="rid")
             .join(o.select(oid="idx", cn2="cn", cns2="cns", cak2="cak", ae2="aempty"), on="oid"))
        rows = []
        for q in x.iter_rows(named=True):
            A, B = list(q["cn1"]), list(q["cn2"])
            nr = refine(name_rel(A, B, q["cns1"], q["cns2"]), A, B, vocab)
            ar = addr_cls(addr_rel(q["cak1"].split(" ") if q["cak1"] else [], q["cak2"].split(" ") if q["cak2"] else [], q["ae2"]))
            rows.append((q["rid"], q["oid"], nr, ar))
        cats.append(pl.DataFrame(rows, schema=["rid", "oid", "name_relation", "address_relation"], orient="row")
                    .with_columns(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32)))
    E = E.join(pl.concat(cats), on=["rid", "oid"], how="left")
    cols = ["type", "country", "s1_id", "s1_name", "s1_address", "rec_id", "rec_name", "rec_address", "pf",
            "name_relation", "address_relation", "true_owner_status", "true_owner_id", "true_owner_name",
            "true_owner_address", "f05", "loss", "n_true", "n_pred"]
    E = E.with_columns(pl.col("pf").round(4), pl.col("f05").round(4), pl.col("loss") * 1e6).rename({"loss": "loss_ppm"})
    cols = [c if c != "loss" else "loss_ppm" for c in cols]
    E = E.sort("loss_ppm", descending=True).select(cols)
    write_csv(E, f"{OUT}errors_pairs.csv")
    # per entity: true vs predicted sets
    tset = gt.join(raw2, on="oid").group_by("rid").agg(true_records=(pl.col("rec_name") + " | " + pl.col("rec_address")).str.join(" ;; "))
    pset = pred.join(raw2, on="oid").group_by("rid").agg(predicted_records=(pl.col("rec_name") + " | " + pl.col("rec_address")).str.join(" ;; "))
    S = (m.filter(pl.col("f05") < 1).join(raw1, on="rid").join(tset, on="rid", how="left").join(pset, on="rid", how="left")
         .with_columns(pl.col("f05").round(4), loss_ppm=pl.col("loss") * 1e6).sort("loss_ppm", descending=True)
         .select("country", "s1_id", "s1_name", "s1_address", "n_true", "n_pred", "tp", "f05", "loss_ppm", "true_records", "predicted_records"))
    write_csv(S, f"{OUT}errors_by_s1.csv")
    # summary: score loss attributed to error categories (an entity's loss split evenly over its error pairs)
    E2 = E.with_columns(share=pl.col("loss_ppm") / pl.len().over("s1_id"))
    summ = (E2.group_by("country", "type", "name_relation", "address_relation", "true_owner_status")
            .agg(pairs=pl.len(), score_loss=(pl.col("share").sum() / 1e6).round(6))
            .sort("score_loss", descending=True))
    write_csv(summ, f"{OUT}errors_summary.csv")
    print(f"wrote {OUT}errors_pairs.csv ({E.height:,} rows), errors_by_s1.csv ({S.height:,} rows), errors_summary.csv ({summ.height:,} rows)")
    print(E.group_by("country", "type").agg(pairs=pl.len()).sort("country", "type"))
    print(summ.head(15))


if __name__ == "__main__":
    main()
