"""Pseudo-labelling for the India/US joint-9 model.
  val : labelled fold 0 + pseudo-labelled fold 1 (OOF pj >= HI -> 1, <= LO -> 0) -> predict fold 1; F0.5 vs baseline on fold 1.
  test: train (harder split, true labels) + test pseudo labels from the current submission (A14):
        accepted & p >= HI -> 1; p <= LO -> 0; pairs removed by the test-shift corrections -> 0 (weight W_DEC).
        Cross-fitted over test entities (2 halves): the model predicting half k never saw half k's pseudo labels.
        Writes D:/amazon_er/dense/test_union_p_pseudo.parquet
  python pseudo.py val | test"""
import os
import sys
import time
os.environ.setdefault("JOINT_V", "9")
import numpy as np
import polars as pl
import xgboost as xgb
import dense_addon as da
import decide
import evaluate as ev

D = "C:/ML/Challenges/amazon/work/data/"; DN = "D:/amazon_er/dense/"; OUT = "C:/ML/Challenges/amazon/work/output/"
rd = dict(separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
HI, LO = float(os.environ.get("HI", 0.98)), float(os.environ.get("LO", 0.02))
W_PS, W_DEC = float(os.environ.get("W_PS", 1.0)), float(os.environ.get("W_DEC", 0.5))
NEG_FRAC = float(os.environ.get("NEG_FRAC", 0.3))       # subsample of confident test negatives (they are the bulk)
ROUNDS = int(os.environ.get("JROUNDS", 400))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fit(Xa, ya, wa):
    return xgb.train(da.JP, xgb.QuantileDMatrix(Xa, label=ya, weight=wa, max_bin=da.JP["max_bin"]), ROUNDS)


def val():
    X = np.load(f"{DN}jalt_X9.npy")
    meta = pl.read_parquet(f"{DN}jalt_meta9.parquet")
    pj = meta.select("rid", "oid").join(pl.read_parquet(f"{DN}train_joint9.parquet", columns=["rid", "oid", "pj"]), on=["rid", "oid"], how="left")["pj"].to_numpy()
    y, fv = meta["y"].to_numpy(), meta["fold"].to_numpy()
    lab = fv == 0
    ps = (fv == 1) & ((pj >= HI) | (pj <= LO))
    idx = np.flatnonzero(lab | ps)
    yy = np.where(lab, y, (pj >= HI).astype(y.dtype))[idx]
    ww = np.where(lab, 1.0, W_PS)[idx].astype(np.float32)
    log(f"val: labelled {lab.sum():,} + pseudo {ps.sum():,} (pseudo-1 {((fv == 1) & (pj >= HI)).sum():,}; pseudo label error {(y[ps] != (pj[ps] >= HI)).mean():.4f})")
    bst = fit(X[idx], yy, ww)
    te = np.flatnonzero(fv == 1)
    pn = bst.predict(xgb.DMatrix(X[te]))
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    s1 = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).join(keep, on="rid", how="semi")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid"]).join(keep, on="rid", how="semi")
    M = meta[te].select("rid", "oid")
    for c in ("India", "US"):
        sc = s1.filter(pl.col("country") == c).join(M.select("rid").unique(), on="rid", how="semi")
        g = gt.join(sc, on="rid", how="semi")
        res = []
        for name, p in (("baseline (fold-0 model OOF)", pj[te]), ("self-trained", pn)):
            pr = decide.ef_predict(M.with_columns(p2=pl.Series(p)).join(sc, on="rid", how="semi"), "p2", 0.0, 1.0, 0.05)
            res.append(f"{name} {ev.macro_f05(pr, g, sc['rid']):.5f}")
        log(f"  {c} fold-1: " + " | ".join(res))


def test():
    import gc
    X = np.load(f"{DN}jalt_X9.npy", mmap_mode="r")
    meta = pl.read_parquet(f"{DN}jalt_meta9.parquet")
    y = meta["y"].to_numpy().astype(np.float32)
    XT = np.load(f"{DN}jalt_X9_test.npy", mmap_mode="r")
    tm = pl.read_parquet(f"{DN}jalt_meta9_test.parquet").select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32), "country")
    p = tm.join(pl.read_parquet(f"{DN}test_union_p.parquet").select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32), "p"), on=["rid", "oid"], how="left")["p"].to_numpy()
    acc, rem = [], []
    for c in ("US", "India"):
        s1t = pl.read_parquet(f"{D}p1_canon_test_{c}_s1.parquet", columns=["idx", "entity_id"]).select(pl.col("idx").cast(pl.UInt32).alias("rid"), s="entity_id")
        ot = pl.read_parquet(f"{D}p1_canon_test_{c}_o.parquet", columns=["idx", "entity_id"]).select(pl.col("idx").cast(pl.UInt32).alias("oid"), m="entity_id")
        for v, lst in (("probeA14", acc), ("v21", rem)):
            f = pl.read_csv(f"{OUT}matching_results_{v}.tsv", **rd).filter(pl.col("matched_entity_ids") != "").with_columns(m=pl.col("matched_entity_ids").str.split(",")).explode("m")
            lst.append(f.join(s1t, left_on="source1_entity_id", right_on="s").join(ot, on="m").select("rid", "oid"))
    A14 = pl.concat(acc).unique()
    removed = pl.concat(rem).unique().join(A14, on=["rid", "oid"], how="anti")          # pairs our corrections removed
    t = tm.with_row_index("_i").with_columns(p=pl.Series(p))
    t = (t.join(A14.with_columns(a=pl.lit(1)), on=["rid", "oid"], how="left").join(removed.with_columns(r=pl.lit(1)), on=["rid", "oid"], how="left")
         .with_columns(pl.col("a").fill_null(0), pl.col("r").fill_null(0), half=(pl.col("rid").hash(5) % 2).cast(pl.Int8),
                       u=(pl.col("rid").hash(9) % 1000).cast(pl.Float32) / 1000))
    pos = (t["a"] == 1) & (t["p"] >= HI)
    neg = (t["a"] == 0) & (t["r"] == 0) & (t["p"] <= LO) & (t["u"] < NEG_FRAC)
    dec = t["r"] == 1
    t = t.with_columns(ps_y=pl.when(pos).then(1.0).when(neg | dec).then(0.0).otherwise(None),
                       ps_w=pl.when(dec).then(W_DEC).when(neg).then(W_PS / NEG_FRAC).otherwise(W_PS))
    log(f"test pseudo labels: pos {pos.sum():,}, neg {neg.sum():,} (x{1 / NEG_FRAC:.1f} weight), decoy-removed neg {dec.sum():,} (w {W_DEC})")
    out = np.zeros(len(t), np.float32)
    S = [t.filter(pl.col("ps_y").is_not_null() & (pl.col("half") == k)).select("_i", "ps_y", "ps_w") for k in (0, 1)]
    T = [t.filter(pl.col("half") != k)["_i"].to_numpy() for k in (0, 1)]
    del t, meta
    gc.collect()
    for k in (0, 1):
        sel = S[k]
        ii = np.sort(sel["_i"].to_numpy())
        sel = sel.sort("_i")
        Xa = np.concatenate([X, XT[ii]])
        ya = np.concatenate([y, sel["ps_y"].to_numpy().astype(np.float32)])
        wa = np.concatenate([np.ones(len(y), np.float32), sel["ps_w"].to_numpy().astype(np.float32)])
        bst = fit(Xa, ya, wa)
        del Xa, ya, wa
        gc.collect()
        tgt = T[k]
        out[tgt] = bst.predict(xgb.DMatrix(XT[tgt]))
        log(f"half {k}: trained on train + {len(ii):,} pseudo rows; predicted {len(tgt):,}")
    tm.select("rid", "oid").with_columns(p=pl.Series(out)).write_parquet(f"{DN}test_union_p_pseudo.parquet")
    for c in ("US", "India"):
        m = (tm["country"] == c).to_numpy()
        a = decide.ef_predict(tm.filter(pl.Series(m)).select("rid", "oid").with_columns(p2=pl.Series(p[m])), "p2", 0.0, 1.0, 0.05)
        b = decide.ef_predict(tm.filter(pl.Series(m)).select("rid", "oid").with_columns(p2=pl.Series(out[m])), "p2", 0.0, 1.0, 0.05)
        log(f"{c}: ef pairs old {a.height:,} new {b.height:,}; only-old {a.join(b, on=['rid', 'oid'], how='anti').height:,}, only-new {b.join(a, on=['rid', 'oid'], how='anti').height:,}")


if __name__ == "__main__":
    (val if sys.argv[1] == "val" else test)()
