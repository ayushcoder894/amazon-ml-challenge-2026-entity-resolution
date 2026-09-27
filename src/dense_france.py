"""Dense add-on for countries absent from train (France). Validated by dense_loco.py (leave-one-country-out, France rule
ef(0, c=4): India +0.0037, US +0.0009 over stage-1 alone).

  prep   per country: new pairs not in the LOCO candidate set, context features from the LOCO stage-1 probabilities
         (out-of-country quality, like France's p1) -> DN/loco_feat_<country>.parquet   (one country at a time: low RAM)
  fit    one model on India + US -> DN/models/france_addon.json (+ france_cols.json)
Test use: dense_addon.py test (France rows: union of stage-1 p1 and this model, rule ef(0, 4)).
"""
import gc
import json
import sys

import numpy as np
import polars as pl
import xgboost as xgb

from dense_addon import COUNTRIES, D, DN, XP, ROUNDS, context, feat_cols, log

CTX_PREFIX = ("u_", "k_", "d_cos")


def prep():
    Kl = (pl.read_parquet(f"{D}train_oof_locoh.parquet", columns=["rid", "oid", "country", "p1"])
          .join(pl.read_parquet(f"{DN}dense_train_cos.parquet"), on=["rid", "oid"], how="left").with_columns(pf=pl.col("p1")))
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").select("rid", "oid").with_columns(y=pl.lit(1, pl.Int8))
    for c in COUNTRIES:
        N = pl.read_parquet(f"{DN}train_nfeat_{c}.parquet")
        N = N.drop([x for x in N.columns if x.startswith(CTX_PREFIX)] + ["y"])
        N = N.join(Kl.select("rid", "oid"), on=["rid", "oid"], how="anti").join(gt, on=["rid", "oid"], how="left")
        N = context(N.with_columns(pl.col("y").fill_null(0)), Kl.filter(pl.col("country") == c))
        cols = [x for x in feat_cols(N) if x != "is_india"]
        N.select("rid", "oid", "y", *[pl.col(x).cast(pl.Float32) for x in cols]).write_parquet(f"{DN}loco_feat_{c}.parquet")
        log(f"{c}: {N.height:,} new pairs, {N['y'].sum():,} positives")
        del N
        gc.collect()


def fit():
    parts = [pl.read_parquet(f"{DN}loco_feat_{c}.parquet") for c in COUNTRIES]
    cols = [x for x in parts[0].columns if x not in ("rid", "oid", "y")]
    X = np.concatenate([p.select(cols).to_numpy() for p in parts])
    y = np.concatenate([p["y"].to_numpy() for p in parts])
    del parts
    gc.collect()
    bst = xgb.train(XP, xgb.QuantileDMatrix(X, label=y, max_bin=XP["max_bin"]), ROUNDS)
    bst.save_model(f"{DN}models/france_addon.json")
    json.dump(cols, open(f"{DN}models/france_cols.json", "w"))
    log(f"France add-on trained on {len(y):,} India+US new pairs ({y.sum():,} positives), {len(cols)} features")


if __name__ == "__main__":
    {"prep": prep, "fit": fit}[sys.argv[1]]()
