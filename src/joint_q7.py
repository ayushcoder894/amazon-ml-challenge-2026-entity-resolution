"""Joint variant 10 = joint-9 features + Qwen-7B score (A100 v6, cross-country LoRA models) as a 28th input, same XGBoost
settings and 2 entity folds. Reports harder-split F0.5 vs joint 9, the test/train predicted-pairs ratio (safety), and writes
D:/amazon_er/dense/test_union_p_q7.parquet (mean of both fold models, as the current joint).
  python joint_q7.py"""
import glob
import os
import time

os.environ.setdefault("JOINT_V", "9")
import numpy as np
import polars as pl
import xgboost as xgb

import dense_addon as da
import decide
import evaluate as ev

D = "C:/ML/Challenges/amazon/work/data/"
DN = "D:/amazon_er/dense/"
G = "D:/amazon_er/gpu5_results/"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def q7(meta, split):
    fs = glob.glob(f"{G}ce6_q7_iu_{split}_*.parquet")
    s = pl.concat([pl.read_parquet(f).select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32), pl.col("s").cast(pl.Float32)) for f in fs])
    v = meta.select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32)).join(s, on=["rid", "oid"], how="left")["s"]
    log(f"q7 {split}: files {[os.path.basename(f) for f in fs]}; scored {v.is_not_null().sum():,} of {meta.height:,}")
    return v.to_numpy().astype(np.float32)[:, None]


def main():
    X = np.load(f"{DN}jalt_X9.npy")
    meta = pl.read_parquet(f"{DN}jalt_meta9.parquet")
    XT = np.load(f"{DN}jalt_X9_test.npy")
    tm = pl.read_parquet(f"{DN}jalt_meta9_test.parquet")
    X, XT = np.concatenate([X, q7(meta, "train")], 1), np.concatenate([XT, q7(tm, "test")], 1)
    y, fv = meta["y"].to_numpy(), meta["fold"].to_numpy()
    oof, tst = np.zeros(len(y), np.float32), np.zeros(XT.shape[0], np.float32)
    for f in (0, 1):
        tr, te = np.flatnonzero(fv == f), np.flatnonzero(fv != f)
        dm = xgb.QuantileDMatrix(X[tr], label=y[tr], max_bin=da.JP["max_bin"])
        bst = xgb.train(da.JP, dm, int(os.environ.get("JROUNDS", 400)))
        del dm
        oof[te] = bst.predict(xgb.DMatrix(X[te]))
        tst += bst.predict(xgb.DMatrix(XT)) / 2
        bst.save_model(f"{DN}models/joint10q7_fold{f}.json")
        log(f"fold {f} done")
    xg = meta.select("rid", "oid").join(pl.read_parquet(f"{DN}train_joint9.parquet", columns=["rid", "oid", "pj"]), on=["rid", "oid"], how="left")["pj"].to_numpy()
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    s1 = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).join(keep, on="rid", how="semi")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid"]).join(keep, on="rid", how="semi")
    npairs = {}
    for name, p in (("joint 9 (current)", xg), ("joint 10 (+Qwen)", oof)):
        pr = decide.ef_predict(meta.select("rid", "oid").with_columns(p2=pl.Series(p)), "p2", 0.0, 1.0, 0.05)
        sc = {}
        for c in ("India", "US"):
            sc[c] = ev.macro_f05(pr.join(s1.filter(pl.col("country") == c), on="rid", how="semi"),
                                 gt.join(s1.filter(pl.col("country") == c), on="rid", how="semi"), s1.filter(pl.col("country") == c)["rid"])
            npairs[(name, c, "train")] = pr.join(s1.filter(pl.col("country") == c), on="rid", how="semi").height
        log(f"  {name:<18s} India {sc['India']:.5f}  US {sc['US']:.5f}")
    ref = pl.read_parquet(f"{DN}test_union_p.parquet")
    new = tm.select("rid", "oid", "country").with_columns(p=pl.Series(tst))
    for c in ("India", "US"):
        a = decide.ef_predict(ref.join(new.filter(pl.col("country") == c).select("rid", "oid"), on=["rid", "oid"], how="semi").select("rid", "oid", p2="p"), "p2", 0.0, 1.0, 0.05).height
        b = decide.ef_predict(new.filter(pl.col("country") == c).select("rid", "oid", p2="p"), "p2", 0.0, 1.0, 0.05).height
        tr_ratio = npairs[("joint 10 (+Qwen)", c, "train")] / npairs[("joint 9 (current)", c, "train")]
        log(f"  {c}: predicted pairs joint10/joint9  train {tr_ratio:.4f}  test {b / a:.4f}")
    new.select("rid", "oid", "p").write_parquet(f"{DN}test_union_p_q7.parquet")
    log("wrote test_union_p_q7.parquet")


if __name__ == "__main__":
    main()
