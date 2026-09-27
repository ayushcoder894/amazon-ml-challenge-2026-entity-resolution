"""A100 v9: bagged density-ratio decoy estimator (XGBoost bag + MLP bag), US first then India.
Class 0 = train accepted TRUE pairs (weight n_test/n_train), class 1 = test accepted pairs (v21). Out-of-fold (2 entity
folds) q -> f = 1 - (1-q)/q = estimated false fraction of each test accepted pair.
NULL check with the identical pipeline: train-true half 0 vs train-true half 1 (no real shift) -> f should be ~0;
its 99.9% quantile is the noise floor reported per model family.
  python dr_ens.py --data data --out out [--bags 8 --mlp_seeds 3]
Outputs out/f_<country>.parquet (rid, oid, p, f_xgb, f_mlp, f_ens, sd_xgb) and out/report.txt"""
import argparse
import os
import time

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import xgboost as xgb

FEATS = ["name_eq", "addr_eq", "words_eq", "ae", "a1e", "na", "nl1", "nl2", "nt1", "nt2", "nn1", "nn2", "ninter", "name_r", "name_ts",
         "addr_r", "words_ts", "gap", "trunc", "src3", "sib_name", "sib_addr", "sib_nums", "sib_share", "sib_s1", "sib_nnum", "has_sib"]
REP = []


def log(*a):
    s = time.strftime("%H:%M:%S") + " " + " ".join(str(x) for x in a)
    print(s, flush=True)
    REP.append(s)


def fold_of(rid, seed):
    h = (rid.astype(np.uint64) * np.uint64(2654435761) + np.uint64(seed)) % np.uint64(4294967296)
    return ((h >> np.uint64(16)) & np.uint64(1)).astype(np.int8)


def xgb_bag(XA, XB, fa, fb, wA, bags):
    """returns (mean q, sd q) out of fold for B rows"""
    qs = np.zeros((bags, XB.shape[0]), np.float32)
    for b in range(bags):
        P = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda",
                 max_depth=[4, 5, 6][b % 3], learning_rate=0.05, subsample=0.7, colsample_bytree=0.7,
                 min_child_weight=[2000, 3000, 1500][b % 3], reg_lambda=5.0, max_bin=128, seed=b)
        for k in (0, 1):
            ta, tb = XA[fa == k], XB[fb == k]
            X = np.concatenate([ta, tb])
            y = np.r_[np.zeros(len(ta)), np.ones(len(tb))]
            w = np.r_[np.full(len(ta), wA), np.ones(len(tb))]
            bst = xgb.train(P, xgb.QuantileDMatrix(X, label=y, weight=w, max_bin=128), 200)
            m = fb != k
            qs[b, m] = bst.predict(xgb.DMatrix(XB[m]))
    return qs.mean(0), qs.std(0)


def prep(X, stats=None):
    miss = np.isnan(X)
    X = np.nan_to_num(X, nan=0.0)
    if stats is None:
        stats = (X.mean(0), X.std(0) + 1e-6)
    return np.concatenate([(X - stats[0]) / stats[1], miss], 1).astype(np.float32), stats


def mlp_bag(XA, XB, fa, fb, wA, seeds, epochs=3):
    q = np.zeros(XB.shape[0], np.float32)
    for k in (0, 1):
        ta, tb = XA[fa == k], XB[fb == k]
        X = np.concatenate([ta, tb])
        Z, st = prep(X)
        Zt, _ = prep(XB[fb != k], st)
        y = np.r_[np.zeros(len(ta)), np.ones(len(tb))].astype(np.float32)
        w = np.r_[np.full(len(ta), wA), np.ones(len(tb))].astype(np.float32)
        Zg, yg, wg = torch.from_numpy(Z).cuda(), torch.from_numpy(y).cuda(), torch.from_numpy(w).cuda()
        Ztg = torch.from_numpy(Zt).cuda()
        for s in range(seeds):
            torch.manual_seed(1000 * k + s)
            net = nn.Sequential(nn.Linear(Z.shape[1], 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 256), nn.GELU(),
                                nn.Dropout(0.1), nn.Linear(256, 1)).cuda()
            opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
            bs = 16384
            steps = epochs * (len(y) // bs + 1)
            sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
            lf = nn.BCEWithLogitsLoss(reduction="none")
            for ep in range(epochs):
                perm = torch.randperm(len(y), device="cuda")
                net.train()
                for i in range(0, len(y), bs):
                    ix = perm[i:i + bs]
                    loss = (lf(net(Zg[ix]).squeeze(1), yg[ix]) * wg[ix]).sum() / wg[ix].sum()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    sch.step()
            net.eval()
            with torch.inference_mode():
                out = torch.cat([torch.sigmoid(net(Ztg[i:i + 262144]).squeeze(1)) for i in range(0, len(Ztg), 262144)]).cpu().numpy()
            q[fb != k] += out / seeds
            del net, opt
        del Zg, yg, wg, Ztg
        torch.cuda.empty_cache()
    return q


def f_of(q):
    return np.clip(1 - (1 - q) / np.maximum(q, 1e-6), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="out")
    ap.add_argument("--bags", type=int, default=8)
    ap.add_argument("--mlp_seeds", type=int, default=3)
    ap.add_argument("--countries", default="US,India")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for c in a.countries.split(","):
        t0 = time.time()
        A = pl.read_parquet(f"{a.data}/dratio_A_{c}.parquet")
        B = pl.read_parquet(f"{a.data}/dratio_{c}.parquet")
        n = pl.read_parquet(f"{a.data}/dratio_n_{c}.parquet")
        ntr, nte = int(n["ntr"][0]), int(n["nte"][0])
        XA = A.select(pl.col(FEATS).cast(pl.Float32)).to_numpy()
        XB = B.select(pl.col(FEATS).cast(pl.Float32)).to_numpy()
        ra, rb = A["rid"].to_numpy(), B["rid"].to_numpy()
        fa, fb = fold_of(ra, 3), fold_of(rb, 3)
        wA = nte / ntr
        log(f"==== {c}: train-true {len(XA):,} ({ntr:,} S1) vs test accepted {len(XB):,} ({nte:,} S1)")
        # NULL: true half 0 vs true half 1 (split by entity), same pipeline
        h = (((ra.astype(np.uint64) * np.uint64(40503) + np.uint64(12345)) % np.uint64(1000003)) % np.uint64(2)).astype(np.int8)
        N0, N1 = XA[h == 0], XA[h == 1]
        nf0, nf1 = fold_of(ra[h == 0], 3), fold_of(ra[h == 1], 3)
        wn = len(N1) / len(N0)
        qn, _ = xgb_bag(N0, N1, nf0, nf1, wn, max(2, a.bags // 2))
        fn = f_of(qn)
        log(f"  NULL xgb: mean f {fn.mean():.4f}, q99 {np.quantile(fn, .99):.3f}, q99.9 {np.quantile(fn, .999):.3f}, max {fn.max():.3f}")
        qm = mlp_bag(N0, N1, nf0, nf1, wn, max(1, a.mlp_seeds - 1))
        fm = f_of(qm)
        log(f"  NULL mlp: mean f {fm.mean():.4f}, q99 {np.quantile(fm, .99):.3f}, q99.9 {np.quantile(fm, .999):.3f}, max {fm.max():.3f}")
        # REAL
        qx, sx = xgb_bag(XA, XB, fa, fb, wA, a.bags)
        log(f"  xgb bag done ({(time.time() - t0) / 60:.1f} min)")
        qm = mlp_bag(XA, XB, fa, fb, wA, a.mlp_seeds)
        log(f"  mlp bag done ({(time.time() - t0) / 60:.1f} min)")
        fx, fmm = f_of(qx), f_of(qm)
        fe = f_of((qx + qm) / 2)
        o = B.select("rid", "oid", "p").with_columns(f_xgb=pl.Series(fx), f_mlp=pl.Series(fmm), f_ens=pl.Series(fe), sd_xgb=pl.Series(sx))
        o.write_parquet(f"{a.out}/f_{c}.parquet")
        for name, f in (("xgb", fx), ("mlp", fmm), ("ens", fe)):
            log(f"  {name}: mean f {f.mean():.4f}; pairs f>0.3 {(f > 0.3).sum():,}, >0.4 {(f > 0.4).sum():,}, >0.5 {(f > 0.5).sum():,}, >0.7 {(f > 0.7).sum():,}")
        log(f"  corr(xgb, mlp) {np.corrcoef(fx, fmm)[0, 1]:.3f}; wrote {a.out}/f_{c}.parquet")
        open(f"{a.out}/report.txt", "w").write("\n".join(REP))
    log("DONE - send back out/f_US.parquet, out/f_India.parquet, out/report.txt")
    open(f"{a.out}/report.txt", "w").write("\n".join(REP))


if __name__ == "__main__":
    main()
