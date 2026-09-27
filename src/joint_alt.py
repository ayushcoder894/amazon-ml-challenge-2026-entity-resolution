"""Alternative learners for the India/US final rescoring model (joint variant 9 features, same 2 entity folds).
Builds (and caches) the exact joint-9 training matrix, trains LightGBM (CPU) and CatBoost (GPU) 2-fold honestly, and
compares harder-split macro F0.5 per country for XGBoost (current), each alternative, and probability averages.
  JOINT_V=9 python joint_alt.py [lgb] [cat]
"""
import gc
import os
import sys
import time

os.environ.setdefault("JOINT_V", "9")
import numpy as np
import polars as pl

import dense_addon as da
import decide
import evaluate as ev

D = "C:/ML/Challenges/amazon/work/data/"
DN = "D:/amazon_er/dense/"
XF, MF = f"{DN}jalt_X9.npy", f"{DN}jalt_meta9.parquet"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def build():
    if os.path.exists(XF) and os.path.exists(MF):
        return np.load(XF, mmap_mode="r"), pl.read_parquet(MF)
    K = da.base_k("train")
    N = da.with_ce(pl.read_parquet(f"{DN}train_pn.parquet"), "train")
    U = da.with_ceu(da.union_frame(K, N), "train").join(pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]), on="rid")
    U = U.with_columns(is_india=(pl.col("country") == "India").cast(pl.UInt8), fold=(pl.col("rid").hash(seed=11) % 2).cast(pl.Int8))
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").select("rid", "oid").with_columns(y=pl.lit(1, pl.Int8))
    U = U.join(gt, on=["rid", "oid"], how="left").with_columns(pl.col("y").fill_null(0))
    X = U.select([pl.col(c).cast(pl.Float32) for c in da.JCOLS]).to_numpy()
    meta = U.select("rid", "oid", "country", "y", "fold")
    del U, K, N
    gc.collect()
    np.save(XF, X)
    meta.write_parquet(MF)
    log(f"built X {X.shape} ({X.nbytes / 1e9:.1f} GB), cols {da.JCOLS}")
    return X, meta


def fit_lgb(X, y, fv):
    import lightgbm as lgb
    P = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=20, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=127, num_threads=0, verbose=-1)
    out = np.zeros(len(y), np.float32)
    for f in (0, 1):
        tr, te = np.flatnonzero(fv == f), np.flatnonzero(fv != f)
        ds = lgb.Dataset(X[tr], label=y[tr], free_raw_data=True)
        t0 = time.time()
        bst = lgb.train(P, ds, num_boost_round=int(os.environ.get("JROUNDS", 400)))
        del ds
        out[te] = bst.predict(X[te]).astype(np.float32)
        log(f"lgb fold {f} trained in {(time.time() - t0) / 60:.1f} min")
        del bst
        gc.collect()
    return out


def fit_cat(X, y, fv):
    from catboost import CatBoostClassifier, Pool
    out = np.zeros(len(y), np.float32)
    for f in (0, 1):
        tr, te = np.flatnonzero(fv == f), np.flatnonzero(fv != f)
        m = CatBoostClassifier(iterations=int(os.environ.get("CAT_IT", 1500)), learning_rate=0.08, depth=8, loss_function="Logloss",
                               task_type="GPU", border_count=64, verbose=300, thread_count=8, used_ram_limit="6gb", gpu_ram_part=0.85)
        t0 = time.time()
        m.fit(Pool(X[tr], label=y[tr]))
        out[te] = m.predict_proba(X[te])[:, 1].astype(np.float32)
        log(f"cat fold {f} trained in {(time.time() - t0) / 60:.1f} min")
        del m
        gc.collect()
    return out


def evaluate(meta, cols):
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    s1 = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).join(keep, on="rid", how="semi")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid"]).join(keep, on="rid", how="semi")
    res = {}
    for name, p in cols.items():
        pred = decide.ef_predict(meta.select("rid", "oid").with_columns(p2=pl.Series(p)), "p2", 0.0, 1.0, 0.05)
        sc = {c: ev.macro_f05(pred.join(s1.filter(pl.col("country") == c), on="rid", how="semi"),
                              gt.join(s1.filter(pl.col("country") == c), on="rid", how="semi"),
                              s1.filter(pl.col("country") == c)["rid"]) for c in ("India", "US")}
        res[name] = sc
        log(f"  {name:<22s} India {sc['India']:.5f}  US {sc['US']:.5f}")
    return res


def main(which):
    X, meta = build()
    y, fv = meta["y"].to_numpy(), meta["fold"].to_numpy()
    xgb = meta.select("rid", "oid").join(pl.read_parquet(f"{DN}train_joint9.parquet", columns=["rid", "oid", "pj"]), on=["rid", "oid"], how="left")["pj"].to_numpy()
    log(f"rows {len(y):,}, xgb pj missing {np.isnan(xgb).sum():,}")
    cols = {"xgb (current)": xgb}
    for w in which:
        f = f"{DN}jalt_{w}9.parquet"
        if os.path.exists(f):
            cols[w] = pl.read_parquet(f)["p"].to_numpy()
        else:
            p = (fit_lgb if w == "lgb" else fit_cat)(X, y, fv)
            pl.DataFrame({"p": p}).write_parquet(f)
            cols[w] = p
    names = [w for w in which]
    for w in names:
        cols[f"avg(xgb,{w})"] = (xgb + cols[w]) / 2
    if len(names) == 2:
        cols["avg(xgb,lgb,cat)"] = (xgb + cols["lgb"] + cols["cat"]) / 3
    evaluate(meta, cols)


if __name__ == "__main__":
    main(sys.argv[1:] or ["lgb", "cat"])
