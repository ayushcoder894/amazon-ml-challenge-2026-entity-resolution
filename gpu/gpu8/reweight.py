"""Test-reweighted retraining of the India/US final model (A100 box: 256 cores, 1 TB RAM).
Generator logic: train and test come from the same generator, so the number of TRUE matches per entity of each pattern is
the same; if test has more CANDIDATES of a pattern per entity, the extra ones are decoys. Per country and pattern
(name relation x address relation x single/cluster x house-number gap):
    decoys/entity on train = (cand_tr - true_tr) / n_tr ;  on test (estimated) = cand_te / n_te - true_tr / n_tr
    weight of train NEGATIVES in the pattern = clip(test / train decoy rate, 0.5, 4)   (positives keep weight 1)
Then the joint-9 XGBoost (same features, params, 2 entity folds) is retrained with these weights; an unweighted twin is
trained the same way as the baseline. Outputs in --out:
    test_union_p_rw.parquet (rid, oid, p)   weights_<country>.csv   report.txt (printed too)
  python reweight.py --data data --out out
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import polars as pl
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{HERE}/code")
import france_cats as fc          # noqa: E402  (canon + category, pattern labelling)
import decide                     # noqa: E402
import evaluate as ev             # noqa: E402

JP = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda", grow_policy="lossguide",
          max_leaves=63, max_depth=0, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
          reg_lambda=1.0, max_bin=128)
G = {}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _num_gap(a, b):
    na = {t for t in (a or "").split(" ") if t.isdigit()}
    nb = {t for t in (b or "").split(" ") if t.isdigit()}
    u, v = na - nb, nb - na
    if not u or not v:
        return ""
    d = min(abs(int(p[:12]) - int(q[:12])) for p in u for q in v)
    return "<=10" if d <= 10 else "11-50" if d <= 50 else "51-500" if d <= 500 else ">500"


def _label(chunk):
    vocab = G["vocab"]
    out = []
    for r in chunk:
        d = dict(zip(G["keys"], r))
        nr, ar, cl = fc.category(d, vocab)
        gap = _num_gap(d["cak1"], d["cak2"]) if "changed" in ar else ""
        out.append(f"{nr}|{ar}|{cl}|{gap}")
    return out


def patterns(P, split, c, procs):
    """P: (rid, oid) pairs of one country -> pattern key per row (same order)."""
    s1, o, vocab = fc.canon(split, c)
    x = (P.with_row_index("_i").join(s1.select(pl.col("idx").cast(pl.UInt32).alias("rid"), cn1="cn", cns1="cns", cak1="cak", na="na"), on="rid", how="left")
         .join(o.select(pl.col("idx").cast(pl.UInt32).alias("oid"), cn2="cn", cns2="cns", cak2="cak", ae2="aempty", csize="csize"), on="oid", how="left")
         .sort("_i"))
    keys = ["cn1", "cn2", "cns1", "cns2", "cak1", "cak2", "ae2", "na", "csize"]
    x = x.with_columns(pl.col("cn1", "cn2").fill_null([]), pl.col("cns1", "cns2", "cak1", "cak2").fill_null(""), pl.col("ae2").fill_null(False))
    G["vocab"], G["keys"] = vocab, keys
    rows = list(zip(*[x[k].to_list() for k in keys]))
    step = max(20000, len(rows) // (procs * 4))
    with Pool(procs) as pool:
        res = pool.map(_label, [rows[i:i + step] for i in range(0, len(rows), step)])
    return [k for part in res for k in part]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{HERE}/data")
    ap.add_argument("--out", default=f"{HERE}/out")
    ap.add_argument("--procs", type=int, default=240)
    ap.add_argument("--wmin", type=float, default=0.5)
    ap.add_argument("--wmax", type=float, default=4.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fc.D = a.data.rstrip("/") + "/"
    D = fc.D
    X = np.load(f"{D}jalt_X9.npy"); XT = np.load(f"{D}jalt_X9_test.npy")
    meta = pl.read_parquet(f"{D}jalt_meta9.parquet"); tm = pl.read_parquet(f"{D}jalt_meta9_test.parquet")
    s1k = pl.read_parquet(f"{D}s1_kept.parquet")               # rid, country (kept train entities)
    n_te = pl.read_parquet(f"{D}s1_test_counts.parquet")        # country, n
    y, fv = meta["y"].to_numpy(), meta["fold"].to_numpy()
    w = np.ones(len(y), np.float32)
    rep = []
    for c in ("India", "US"):
        m, mt = (meta["country"] == c).to_numpy(), (tm["country"] == c).to_numpy()
        t0 = time.time()
        ptr = patterns(meta.filter(pl.Series(m)).select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32)), "train", c, a.procs)
        pte = patterns(tm.filter(pl.Series(mt)).select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32)), "test", c, a.procs)
        log(f"{c}: labelled {len(ptr):,} train + {len(pte):,} test pairs in {(time.time() - t0) / 60:.1f} min")
        ntr, nte = s1k.filter(pl.col("country") == c).height, n_te.filter(pl.col("country") == c)["n"][0]
        tr = pl.DataFrame({"g": ptr, "y": y[m]}).group_by("g").agg(cand_tr=pl.len(), true_tr=pl.col("y").sum())
        te = pl.DataFrame({"g": pte}).group_by("g").agg(cand_te=pl.len())
        t = tr.join(te, on="g", how="full", coalesce=True).fill_null(0).with_columns(
            dec_tr=(pl.col("cand_tr") - pl.col("true_tr")) / ntr, dec_te=pl.col("cand_te") / nte - pl.col("true_tr") / ntr)
        t = t.with_columns(w=pl.when((pl.col("cand_tr") >= 300) & (pl.col("dec_tr") > 0))
                           .then((pl.col("dec_te") / pl.col("dec_tr")).clip(a.wmin, a.wmax)).otherwise(1.0))
        t.sort("cand_tr", descending=True).write_csv(f"{a.out}/weights_{c}.csv")
        wm = pl.DataFrame({"g": ptr}).join(t.select("g", "w"), on="g", how="left")["w"].fill_null(1.0).to_numpy()
        idx = np.flatnonzero(m)
        w[idx] = np.where(y[idx] == 0, wm, 1.0)
        big = t.filter(pl.col("cand_tr") >= 300).with_columns(ex=(pl.col("dec_te") - pl.col("dec_tr")) * nte).sort("ex", descending=True)
        line = f"{c}: patterns {t.height}; negatives weighted mean {w[idx][y[idx] == 0].mean():.3f}; top excess patterns:\n" + \
               "\n".join(f"   {r['g']:<70s} w={r['w']:.2f} test-extra decoys ~{r['ex']:,.0f}" for r in big.head(12).iter_rows(named=True))
        log(line); rep.append(line)

    def fit(weights):
        oof, tst = np.zeros(len(y), np.float32), np.zeros(XT.shape[0], np.float32)
        for f in (0, 1):
            tr, te = np.flatnonzero(fv == f), np.flatnonzero(fv != f)
            bst = xgb.train(JP, xgb.QuantileDMatrix(X[tr], label=y[tr], weight=None if weights is None else weights[tr], max_bin=128), 400)
            oof[te] = bst.predict(xgb.DMatrix(X[te]))
            tst += bst.predict(xgb.DMatrix(XT)) / 2
        return oof, tst

    o0, t0_ = fit(None)
    o1, t1 = fit(w)
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid"]).join(s1k.select("rid"), on="rid", how="semi")
    for c in ("India", "US"):
        m, mt = (meta["country"] == c).to_numpy(), (tm["country"] == c).to_numpy()
        s = s1k.filter(pl.col("country") == c)
        g = gt.join(s, on="rid", how="semi")
        res = []
        for name, o, tt in (("unweighted", o0, t0_), ("test-reweighted", o1, t1)):
            pr = decide.ef_predict(meta.filter(pl.Series(m)).select("rid", "oid").with_columns(p2=pl.Series(o[m])), "p2", 0.0, 1.0, 0.05)
            pt = decide.ef_predict(tm.filter(pl.Series(mt)).select("rid", "oid").with_columns(p2=pl.Series(tt[mt])), "p2", 0.0, 1.0, 0.05)
            res.append((name, ev.macro_f05(pr, g, s["rid"]), pr.height, pt.height))
        line = f"{c}: " + " | ".join(f"{n} train F0.5 {f:.5f}, pairs train {a_:,} test {b_:,}" for n, f, a_, b_ in res) + \
               f"\n   test pairs change {res[1][3] / res[0][3] - 1:+.4f} (train {res[1][2] / res[0][2] - 1:+.4f})"
        log(line); rep.append(line)
    tm.select("rid", "oid").with_columns(p=pl.Series(t1)).write_parquet(f"{a.out}/test_union_p_rw.parquet")
    tm.select("rid", "oid").with_columns(p=pl.Series(t0_)).write_parquet(f"{a.out}/test_union_p_unw.parquet")
    open(f"{a.out}/report.txt", "w").write("\n".join(rep))
    log("DONE - send back out/test_union_p_rw.parquet, out/test_union_p_unw.parquet, out/report.txt, out/weights_*.csv")


if __name__ == "__main__":
    main()
