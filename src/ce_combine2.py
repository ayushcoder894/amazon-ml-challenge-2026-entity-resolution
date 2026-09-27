"""Second-round combiner: on the hard band (p3 in (LO, HI)) blend p3, the first CE and the second CE (honest 2-fold)."""
import json
import sys

import numpy as np
import polars as pl
import xgboost as xgb

from ce_combine import CP, logit
from train2 import per_country, report

D = "C:/ML/Challenges/amazon/work/data/"
M = "C:/ML/Challenges/amazon/work/models_ce/"
LO, HI = 0.02, 0.98
FEATS2 = ["p2", "p3", "ce", "ce2", "lg3", "lgce2", "ce2_rank2", "ce2_max2", "ce2_margin2", "ce2_rank1", "ce2_max1",
          "n2band1", "n2band2"]


def feats2(K):
    b = K.filter(pl.col("ce2").is_not_null())
    return b.with_columns(
        lg3=logit("p3"), lgce2=logit("ce2"),
        ce2_rank2=pl.col("ce2").rank("ordinal", descending=True).over("oid").cast(pl.UInt16),
        ce2_max2=pl.col("ce2").max().over("oid"),
        ce2_rank1=pl.col("ce2").rank("ordinal", descending=True).over("rid").cast(pl.UInt16),
        ce2_max1=pl.col("ce2").max().over("rid"),
        n2band1=pl.len().over("rid").cast(pl.UInt16), n2band2=pl.len().over("oid").cast(pl.UInt16),
        ce2_second2=pl.col("ce2").sort(descending=True).slice(1, 1).first().over("oid").fill_null(0.0),
    ).with_columns(ce2_margin2=pl.when(pl.col("ce2_rank2") == 1).then(pl.col("ce2") - pl.col("ce2_second2"))
                   .otherwise(pl.col("ce2") - pl.col("ce2_max2")))


def main(tag):
    K = pl.read_parquet(f"{D}train_oof_ce{tag}.parquet").join(
        pl.read_parquet(f"{D}ce2_train{tag}.parquet"), on=["rid", "oid"], how="left")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").filter(pl.col("country").is_in(["India", "US"])).select("rid", "oid")
    s1_all = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(
        pl.col("country").is_in(["India", "US"]))
    b = feats2(K)
    preds = []
    for f in (0, 1):
        tr, te = b.filter(pl.col("fold") == f), b.filter(pl.col("fold") != f)
        m = xgb.train(CP, xgb.DMatrix(tr.select(FEATS2).to_numpy().astype(np.float32), tr["y"].to_numpy()), 400)
        m.save_model(f"{M}comb2{tag}_fold{f}.json")
        preds.append(te.select("rid", "oid").with_columns(
            pc2=pl.Series(m.predict(xgb.DMatrix(te.select(FEATS2).to_numpy().astype(np.float32))))))
    K = K.join(pl.concat(preds), on=["rid", "oid"], how="left").with_columns(p4=pl.coalesce("pc2", "p3"))
    s4, rule = report(K, gt, s1_all, "p4")
    per_country(K, gt, s1_all, "p4", rule)
    json.dump(dict(stage4=s4, rule=list(rule), feats=FEATS2), open(f"{M}comb2_meta{tag}.json", "w"), indent=1)
    K.select("rid", "oid", "y", "fold", "country", "p2", "p3", "ce", "ce2", "p4").write_parquet(
        f"{D}train_oof_ce2{tag}.parquet")


if __name__ == "__main__":
    main(sys.argv[1])
