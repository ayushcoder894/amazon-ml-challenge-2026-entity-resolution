"""Combine stage-2 probabilities with cross-encoder scores on the uncertain band (honest, 2-fold), then score.

Outside the band p3 = p2. Inside, a small XGBoost combiner sees p2, the CE score and CE competition features
(rank / margin among the S1s competing for the same record). Combiner f trains on fold-f band pairs (whose CE
scores are themselves out-of-fold) and predicts fold 1-f. A plain average (p2 + ce) / 2 is reported as a baseline.
"""
import json
import os
import sys

import numpy as np
import polars as pl
import xgboost as xgb

from train2 import report

D = "C:/ML/Challenges/amazon/work/data/"
M = "C:/ML/Challenges/amazon/work/models_ce/"
CP = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda", max_depth=6,
          learning_rate=0.05, subsample=0.8, colsample_bytree=0.9, min_child_weight=20)
FEATS = ["p1", "p2", "ce", "lg2", "lgce", "ce_rank2", "ce_max2", "ce_margin2", "ce_rank1", "ce_max1", "nband1", "nband2",
         "p2_rank2"]


def logit(c):
    x = pl.col(c).clip(1e-5, 1 - 1e-5)
    return (x / (1 - x)).log()


def ce_features(K):
    b = K.filter(pl.col("ce").is_not_null())
    b = b.with_columns(
        lg2=logit("p2"), lgce=logit("ce"),
        ce_rank2=pl.col("ce").rank("ordinal", descending=True).over("oid").cast(pl.UInt16),
        ce_max2=pl.col("ce").max().over("oid"),
        ce_rank1=pl.col("ce").rank("ordinal", descending=True).over("rid").cast(pl.UInt16),
        ce_max1=pl.col("ce").max().over("rid"),
        nband1=pl.len().over("rid").cast(pl.UInt16), nband2=pl.len().over("oid").cast(pl.UInt16),
        p2_rank2=pl.col("p2").rank("ordinal", descending=True).over("oid").cast(pl.UInt16),
        ce_second2=pl.col("ce").sort(descending=True).slice(1, 1).first().over("oid").fill_null(0.0),
    ).with_columns(ce_margin2=pl.when(pl.col("ce_rank2") == 1).then(pl.col("ce") - pl.col("ce_second2"))
                   .otherwise(pl.col("ce") - pl.col("ce_max2")))
    return b


def main(tag="_v9"):
    K = pl.read_parquet(f"{D}train_oof{tag}.parquet", columns=["rid", "oid", "y", "fold", "country", "p1", "p2"])
    ce = pl.read_parquet(f"{D}ce_train{tag}.parquet")
    K = K.join(ce, on=["rid", "oid"], how="left")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").filter(pl.col("country").is_in(["India", "US"])).select("rid", "oid")
    s1_all = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(
        pl.col("country").is_in(["India", "US"]))
    if os.environ.get("KEEP"):  # test-like condition (make_drop_folds.py): only these S1 entities exist
        kr = pl.read_parquet(f"{D}keep{os.environ['KEEP']}.parquet")
        gt, s1_all = gt.join(kr, on="rid", how="semi"), s1_all.join(kr, on="rid", how="semi")
    b = ce_features(K)
    preds = []
    for f in (0, 1):
        tr, te = b.filter(pl.col("fold") == f), b.filter(pl.col("fold") != f)
        m = xgb.train(CP, xgb.DMatrix(tr.select(FEATS).to_numpy().astype(np.float32), tr["y"].to_numpy()), 400)
        m.save_model(f"{M}comb{tag}_fold{f}.json")
        preds.append(te.select("rid", "oid").with_columns(
            pc=pl.Series(m.predict(xgb.DMatrix(te.select(FEATS).to_numpy().astype(np.float32))))))
    K = K.join(pl.concat(preds), on=["rid", "oid"], how="left").with_columns(
        p3=pl.coalesce("pc", "p2"),
        pavg=pl.when(pl.col("ce").is_not_null()).then((pl.col("p2") + pl.col("ce")) / 2).otherwise(pl.col("p2")))
    s_avg, _ = report(K, gt, s1_all, "pavg")
    s3, rule = report(K, gt, s1_all, "p3")
    from train2 import per_country
    per_country(K, gt, s1_all, "p3", rule)
    json.dump(dict(stage3=s3, avg=s_avg, rule=list(rule), feats=FEATS), open(f"{M}comb_meta{tag}.json", "w"), indent=1)
    K.select("rid", "oid", "y", "fold", "country", "p1", "p2", "ce", "p3").write_parquet(f"{D}train_oof_ce{tag}.parquet")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "_v9")
