"""France proxy: does the joint re-scoring pass (mostly relative / competition features) transfer to an unseen country?
Union = LOCO stage-1 p1 (candidate set of the LOCO cascade) + dense new pairs scored by the LOCO add-on (dense_loco.py).
Joint model trained on one country's union, applied to the other; decided with the France rule ef(0, c=4) and ef(0,1).
Baseline = the same union decided on its own probabilities (what v13 does for France).
"""
import gc

import numpy as np
import polars as pl
import xgboost as xgb

import decide
import evaluate as ev
from dense_addon import COUNTRIES, D, DN, JCOLS, JP, log, union_frame


def main():
    Kl = (pl.read_parquet(f"{D}train_oof_locoh.parquet", columns=["rid", "oid", "country", "p1"])
          .join(pl.read_parquet(f"{DN}dense_train_cos.parquet"), on=["rid", "oid"], how="left").with_columns(pf=pl.col("p1")))
    N = (pl.read_parquet(f"{DN}train_pn_loco.parquet", columns=["rid", "oid", "pn"])
         .join(pl.read_parquet(f"{DN}dense_train_new_small.parquet", columns=["rid", "oid", "cos"]), on=["rid", "oid"], how="left"))
    U = union_frame(Kl, N).join(pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]), on="rid")
    U = U.with_columns(is_india=(pl.col("country") == "India").cast(pl.UInt8))
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").select("rid", "oid").with_columns(y=pl.lit(1, pl.Int8))
    U = U.join(gt, on=["rid", "oid"], how="left").with_columns(pl.col("y").fill_null(0))
    cols = [c for c in JCOLS if c != "is_india"]
    pj = np.zeros(U.height, dtype=np.float32)
    for c in COUNTRIES:
        tr = (U["country"] != c).to_numpy()
        X = U.filter(pl.Series(tr)).select([pl.col(x).cast(pl.Float32) for x in cols]).to_numpy()
        bst = xgb.train(JP, xgb.QuantileDMatrix(X, label=U.filter(pl.Series(tr))["y"].to_numpy(), max_bin=JP["max_bin"]), 400)
        pj[~tr] = bst.predict(xgb.DMatrix(U.filter(pl.Series(~tr)).select([pl.col(x).cast(pl.Float32) for x in cols]).to_numpy()))
        log(f"joint trained on {'US' if c == 'India' else 'India'}, scored {c}")
        del X, bst
        gc.collect()
    U = U.with_columns(pj=pl.Series(pj))
    U.select("rid", "oid", "country", "p", "pj", "y", "is_new").write_parquet("D:/amazon_er/france_night/data/loco_joint_U.parquet")
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    s1 = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).join(keep, on="rid", how="semi")
    gtk = gt.join(keep, on="rid", how="semi").select("rid", "oid")
    for rule in ((0.0, 4.0), (0.0, 1.0)):
        for col in ("p", "pj"):
            pred = decide.ef_predict(U.select("rid", "oid", pl.col(col).alias("p2")), "p2", rule[0], rule[1], 0.05)
            sc = {c: ev.macro_f05(pred.join(s1.filter(pl.col("country") == c), on="rid", how="semi"),
                                  gtk.join(s1.filter(pl.col("country") == c), on="rid", how="semi"),
                                  s1.filter(pl.col("country") == c)["rid"]) for c in COUNTRIES}
            log(f"rule ef{rule} on {col:<2s}: India {sc['India']:.5f}  US {sc['US']:.5f}")


if __name__ == "__main__":
    main()
