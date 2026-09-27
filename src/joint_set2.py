"""Set transformer final model for India/US, production version: S seeds x 2 entity folds; out-of-fold predictions on train
(harder-split evaluation vs XGBoost) and test predictions = mean over all fold/seed models (same scheme as the XGBoost joint).
Test features = the exact joint-9 union table (checked against the saved XGBoost test probabilities).
Optional extra feature columns (e.g. Qwen) via EXTRA=q7 (joined from D:/amazon_er/gpu5_results/ce6_q7_iu_{split}_*.parquet).
  python joint_set2.py            -> D:/amazon_er/dense/test_union_p_set.parquet (+ OOF D:/amazon_er/dense/jalt_set2_oof.parquet)
"""
import gc
import glob
import os
import time

os.environ.setdefault("JOINT_V", "9")
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import xgboost as xgb

import dense_addon as da
import decide
import evaluate as ev
from joint_set import SetModel

D = "C:/ML/Challenges/amazon/work/data/"
DN = "D:/amazon_er/dense/"
L = int(os.environ.get("SET_L", 48))
EPOCHS = int(os.environ.get("SET_EP", 3))
SEEDS = int(os.environ.get("SET_SEEDS", 3))
EXTRA = [e for e in os.environ.get("EXTRA", "").split(",") if e]
TAG = os.environ.get("SET_TAG", "set2" + ("_" + "_".join(EXTRA) if EXTRA else ""))
DEV = "cuda"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def build_test():
    xf, mf = f"{DN}jalt_X9_test.npy", f"{DN}jalt_meta9_test.parquet"
    if os.path.exists(xf) and os.path.exists(mf):
        return np.load(xf), pl.read_parquet(mf)
    x = pl.read_parquet(f"{DN}test_pn.parquet")
    K = da.base_k("test").filter(pl.col("country").is_in(da.COUNTRIES))
    U = (da.with_ceu(da.union_frame(K, da.with_ce(x.select("rid", "oid", "pn", "cos", "fwd_rank", "rev_rank"), "test")), "test")
         .join(pl.read_parquet(f"{D}test_s1_norm.parquet", columns=["rid", "country"]), on="rid")
         .with_columns(is_india=(pl.col("country") == "India").cast(pl.UInt8)))
    XT = U.select([pl.col(c).cast(pl.Float32) for c in da.JCOLS]).to_numpy()
    tm = U.select("rid", "oid", "country")
    del U, K, x
    gc.collect()
    # sanity: the saved XGBoost joint-9 models on this table must reproduce test_union_p
    pj = np.mean([xgb.Booster(model_file=f"{DN}models/{da.JNAME}_fold{f}.json").predict(xgb.DMatrix(XT[:200000])) for f in (0, 1)], axis=0)
    ref = tm.head(200000).join(pl.read_parquet(f"{DN}test_union_p.parquet"), on=["rid", "oid"], how="left")["p"].to_numpy()
    log(f"test table {XT.shape}; XGBoost reproduces saved test p: max abs diff {np.nanmax(np.abs(pj - ref)):.2e}")
    np.save(xf, XT)
    tm.write_parquet(mf)
    return XT, tm


def extra_cols(meta, split):
    cols = []
    for e in EXTRA:
        fs = glob.glob(f"D:/amazon_er/gpu5_results/ce6_{e}_iu_{split}_*.parquet")
        s = pl.concat([pl.read_parquet(f) for f in fs]) if fs else pl.DataFrame({"rid": [], "oid": [], "s": []}, schema={"rid": pl.UInt32, "oid": pl.UInt32, "s": pl.Float32})
        v = meta.select("rid", "oid").join(s.select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32), "s"), on=["rid", "oid"], how="left")["s"]
        log(f"extra {e} ({split}): {v.is_not_null().sum():,} of {meta.height:,} rows scored, files {len(fs)}")
        cols.append(v.to_numpy().astype(np.float32))
    return np.stack(cols, 1) if cols else None


def transform(X, stats=None):
    miss = np.isnan(X)
    X = np.nan_to_num(X, nan=0.0)
    X = np.sign(X) * np.log1p(np.abs(X))
    if stats is None:
        stats = (X.mean(0), X.std(0) + 1e-6)
    X = ((X - stats[0]) / stats[1]).astype(np.float32)
    return np.concatenate([X, miss.astype(np.float32)], axis=1), stats


def groups(meta, p0, with_fold):
    g = meta.with_row_index("i").select("i", "rid", *(["fold"] if with_fold else []), p=pl.Series(p0)).sort(["rid", "p"], descending=[False, True])
    g = g.with_columns(r=pl.int_range(pl.len()).over("rid")).filter(pl.col("r") < L)
    sets = g.group_by("rid", maintain_order=True).agg(idx=pl.col("i"), **({"fold": pl.col("fold").first()} if with_fold else {}))
    lens = sets["idx"].list.len().to_numpy()
    IDX = np.full((sets.height, L), -1, dtype=np.int64)
    IDX[np.repeat(np.arange(sets.height), lens), np.concatenate([np.arange(m) for m in lens])] = np.concatenate(sets["idx"].to_list())
    return IDX, (sets["fold"].to_numpy() if with_fold else None)


def main():
    X = np.load(f"{DN}jalt_X9.npy")
    meta = pl.read_parquet(f"{DN}jalt_meta9.parquet")
    XT, tmeta = build_test()
    ex, ext = extra_cols(meta, "train"), extra_cols(tmeta, "test")
    if ex is not None:
        X, XT = np.concatenate([X, ex], 1), np.concatenate([XT, ext], 1)
    F, stats = transform(X)
    FT, _ = transform(XT, stats)
    p0, p0t = X[:, 0].copy(), XT[:, 0].copy()
    del X, XT
    gc.collect()
    y = meta["y"].to_numpy().astype(np.float32)
    IDX, fold = groups(meta, p0, True)
    IDXT, _ = groups(tmeta, p0t, False)
    log(f"train {F.shape} {IDX.shape[0]:,} entities; test {FT.shape} {IDXT.shape[0]:,} entities; seeds {SEEDS}, epochs {EPOCHS}, tag {TAG}")
    Ft, FTt, yt = torch.from_numpy(F), torch.from_numpy(FT), torch.from_numpy(y)
    oof = np.zeros(len(y), np.float32)
    tst = np.zeros(FT.shape[0], np.float32)

    def gather(M, T, ids):
        ix = torch.from_numpy(M[ids])
        pad = ix < 0
        return T[ix.clamp(min=0)].to(DEV, non_blocking=True), pad.to(DEV), ix

    def predict(model, M, T, rows, out, w):
        model.eval()
        with torch.inference_mode():
            for b in range(0, len(rows), 2048):
                xb, pad, ix = gather(M, T, rows[b:b + 2048])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pr = torch.sigmoid(model(xb, pad).float()).cpu().numpy()
                m = ix.numpy() >= 0
                out[ix.numpy()[m]] += w * pr[m]

    lossf = nn.BCEWithLogitsLoss(reduction="none")
    for s in range(SEEDS):
        for f in (0, 1):
            tr, te = np.flatnonzero(fold == f), np.flatnonzero(fold != f)
            torch.manual_seed(100 * s + f)
            model = SetModel(F.shape[1]).to(DEV)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            bs = 512
            steps = EPOCHS * (len(tr) // bs + 1)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
            t0 = time.time()
            for ep in range(EPOCHS):
                model.train()
                perm = np.random.default_rng(1000 * s + 10 * ep + f).permutation(tr)
                for b in range(0, len(perm), bs):
                    ids = perm[b:b + bs]
                    xb, pad, ix = gather(IDX, Ft, ids)
                    yb = yt[ix.clamp(min=0)].to(DEV, non_blocking=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logit = model(xb, pad)
                    l = lossf(logit.float(), yb)
                    loss = (l * (~pad)).sum() / (~pad).sum()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    sched.step()
            predict(model, IDX, Ft, te, oof, 1.0 / SEEDS)
            predict(model, IDXT, FTt, np.arange(IDXT.shape[0]), tst, 1.0 / (2 * SEEDS))
            log(f"seed {s} fold {f} done in {(time.time() - t0) / 60:.1f} min")
            del model
            torch.cuda.empty_cache()
    meta.select("rid", "oid").with_columns(p=pl.Series(oof)).write_parquet(f"{DN}jalt_{TAG}_oof.parquet")
    tmeta.select("rid", "oid").with_columns(p=pl.Series(tst)).write_parquet(f"{DN}test_union_p_{TAG}.parquet")
    xg = meta.select("rid", "oid").join(pl.read_parquet(f"{DN}train_joint9.parquet", columns=["rid", "oid", "pj"]), on=["rid", "oid"], how="left")["pj"].to_numpy()
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    s1 = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).join(keep, on="rid", how="semi")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid"]).join(keep, on="rid", how="semi")
    for name, p in (("xgb (current)", xg), (f"set x{SEEDS} seeds", oof), ("avg(xgb,set)", (xg + oof) / 2)):
        pr = decide.ef_predict(meta.select("rid", "oid").with_columns(p2=pl.Series(p)), "p2", 0.0, 1.0, 0.05)
        sc = {c: ev.macro_f05(pr.join(s1.filter(pl.col("country") == c), on="rid", how="semi"),
                              gt.join(s1.filter(pl.col("country") == c), on="rid", how="semi"),
                              s1.filter(pl.col("country") == c)["rid"]) for c in ("India", "US")}
        log(f"  {name:<18s} India {sc['India']:.5f}  US {sc['US']:.5f}")
    tp = pl.read_parquet(f"{DN}test_union_p.parquet").join(tmeta.select("rid", "oid").with_columns(ps=pl.Series(tst)), on=["rid", "oid"], how="left")
    log(f"test: rows {tp.height:,}, set-model coverage {tp['ps'].is_not_null().mean():.4f}, corr with XGBoost {np.corrcoef(tp.drop_nulls()['p'], tp.drop_nulls()['ps'])[0, 1]:.4f}")


if __name__ == "__main__":
    main()
