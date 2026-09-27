"""Country-wise variants of the India/US final model (joint-9 features, same 2 entity folds, same XGBoost settings):
 shared (current) | per-country models | shared + within-country standardised competition features |
 shared without the train->test shifting features. Harder-split F0.5 + predicted-pairs ratio vs current (train, test)."""
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
COLS = da.JCOLS
COMP = ["p_rank2", "p_max2", "p_sum2", "n2", "p_second2", "p_rank1", "p_max1", "p_sum1", "n1", "n_conf1", "c_rank2", "c_gap2", "c_rank1", "c_gap1", "p_margin2", "p_other1"]
SHIFT = ["n1", "n2", "p_rank1", "c_rank1", "is_new"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


X = np.load(f"{DN}jalt_X9.npy"); XT = np.load(f"{DN}jalt_X9_test.npy")
meta = pl.read_parquet(f"{DN}jalt_meta9.parquet"); tm = pl.read_parquet(f"{DN}jalt_meta9_test.parquet")
y, fv = meta["y"].to_numpy(), meta["fold"].to_numpy()
ci, cti = (meta["country"] == "India").to_numpy(), (tm["country"] == "India").to_numpy()
keep = pl.read_parquet(f"{D}keep_v11.parquet")
s1 = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).join(keep, on="rid", how="semi")
gt = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid"]).join(keep, on="rid", how="semi")
xg = meta.select("rid", "oid").join(pl.read_parquet(f"{DN}train_joint9.parquet", columns=["rid", "oid", "pj"]), on=["rid", "oid"], how="left")["pj"].to_numpy()
xgt = tm.select("rid", "oid").join(pl.read_parquet(f"{DN}test_union_p.parquet"), on=["rid", "oid"], how="left")["p"].to_numpy()


def fit(Xa, XTa, groups):
    oof, tst = np.zeros(len(y), np.float32), np.zeros(XTa.shape[0], np.float32)
    for gm, gtm in groups:
        for f in (0, 1):
            tr, te = np.flatnonzero(gm & (fv == f)), np.flatnonzero(gm & (fv != f))
            bst = xgb.train(da.JP, xgb.QuantileDMatrix(Xa[tr], label=y[tr], max_bin=da.JP["max_bin"]), 400)
            oof[te] = bst.predict(xgb.DMatrix(Xa[te]))
            tst[gtm] += bst.predict(xgb.DMatrix(XTa[gtm])) / 2
            del bst
    return oof, tst


def country_std(A, AT):
    idx = [COLS.index(c) for c in COMP]
    Z = np.zeros((A.shape[0], len(idx)), np.float32); ZT = np.zeros((AT.shape[0], len(idx)), np.float32)
    for m, mt in ((ci, cti), (~ci, ~cti)):
        s = A[m][:, idx]
        mu, sd = np.nanmean(s, 0), np.nanstd(s, 0) + 1e-6
        Z[m], ZT[mt] = (s - mu) / sd, (AT[mt][:, idx] - mu) / sd
    return np.concatenate([A, Z], 1), np.concatenate([AT, ZT], 1)


base = {}


def report(name, oof, tst):
    out = []
    for c, m, mt in (("India", ci, cti), ("US", ~ci, ~cti)):
        pr = decide.ef_predict(meta.filter(pl.Series(m)).select("rid", "oid").with_columns(p2=pl.Series(oof[m])), "p2", 0.0, 1.0, 0.05)
        f = ev.macro_f05(pr, gt.join(s1.filter(pl.col("country") == c), on="rid", how="semi"), s1.filter(pl.col("country") == c)["rid"])
        te = decide.ef_predict(tm.filter(pl.Series(mt)).select("rid", "oid").with_columns(p2=pl.Series(tst[mt])), "p2", 0.0, 1.0, 0.05).height
        if c not in base:
            base[c] = (pr.height, te)
        out.append(f"{c} {f:.5f} (pairs train {pr.height / base[c][0]:.4f}, test {te / base[c][1]:.4f})")
    log(f"  {name:<36s} " + " | ".join(out))


import sys
report("shared (current, saved)", xg, xgt)
allg = [(np.ones(len(y), bool), np.ones(XT.shape[0], bool))]
v = sys.argv[1]
if v == "country":
    o, t = fit(X, XT, [(ci, cti), (~ci, ~cti)]); report("per-country models", o, t)
elif v == "std":
    X, XT = country_std(X, XT); o, t = fit(X, XT, allg); report("shared + country-standardised comp.", o, t)
else:
    kc = [i for i, c in enumerate(COLS) if c not in SHIFT]
    X, XT = np.ascontiguousarray(X[:, kc]), np.ascontiguousarray(XT[:, kc])
    o, t = fit(X, XT, allg); report("shared without shifting features", o, t)
    tm.select("rid", "oid").with_columns(p=pl.Series(t)).write_parquet(f"{DN}test_p_noshift.parquet")
