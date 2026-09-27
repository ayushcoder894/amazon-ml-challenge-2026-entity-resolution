"""Set transformer as the India/US final rescoring model: each S1 entity's candidate list is one sequence; candidates attend
to each other (competition inside the entity), one probability per candidate. Same joint-9 features and the same 2 entity
folds as the XGBoost model (joint_alt.py cache). Harder-split macro F0.5 vs XGBoost and the average of both.
  python joint_set.py
"""
import gc
import os
import time

import numpy as np
import polars as pl
import torch
import torch.nn as nn

import decide
import evaluate as ev

D = "C:/ML/Challenges/amazon/work/data/"
DN = "D:/amazon_er/dense/"
L = int(os.environ.get("SET_L", 48))       # max candidates per entity (top by p)
EPOCHS = int(os.environ.get("SET_EP", 3))
DEV = "cuda"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class SetModel(nn.Module):
    def __init__(self, d_in, d=128, layers=3, heads=4):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x, pad):
        h = self.enc(self.inp(x), src_key_padding_mask=pad)
        return self.out(h).squeeze(-1)


def main():
    X = np.load(f"{DN}jalt_X9.npy")
    meta = pl.read_parquet(f"{DN}jalt_meta9.parquet").with_row_index("i")
    n, k = X.shape
    miss = np.isnan(X)
    X = np.nan_to_num(X, nan=0.0)
    # scale: counts / ranks -> log1p; everything standardised on the whole table (label-free)
    X = np.sign(X) * np.log1p(np.abs(X))
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X = ((X - mu) / sd).astype(np.float32)
    F = np.concatenate([X, miss.astype(np.float32)], axis=1)
    del X, miss
    gc.collect()
    y = meta["y"].to_numpy().astype(np.float32)
    # groups: rows of each entity, top-L by the stage-1/union score p (column 0), padded index matrix
    g = meta.select("i", "rid", "fold", p=pl.Series(F[:, 0])).sort(["rid", "p"], descending=[False, True])
    g = g.with_columns(r=pl.int_range(pl.len()).over("rid")).filter(pl.col("r") < L)
    sets = g.group_by("rid", maintain_order=True).agg(idx=pl.col("i"), fold=pl.col("fold").first())
    lens = sets["idx"].list.len().to_numpy()
    log(f"rows {n:,}, entities {sets.height:,}, candidates/entity mean {lens.mean():.1f} max {lens.max()} (cap {L}); rows kept {lens.sum():,}")
    IDX = np.full((sets.height, L), -1, dtype=np.int64)
    flat = np.concatenate(sets["idx"].to_list())
    rowpos = np.repeat(np.arange(sets.height), lens)
    colpos = np.concatenate([np.arange(m) for m in lens])
    IDX[rowpos, colpos] = flat
    fold = sets["fold"].to_numpy()
    del g, sets, flat, rowpos, colpos
    gc.collect()
    Ft = torch.from_numpy(F)          # CPU, gathered per batch
    yt = torch.from_numpy(y)
    pred = np.zeros(n, dtype=np.float32)

    def batch(ids):
        ix = torch.from_numpy(IDX[ids])
        pad = ix < 0
        ixc = ix.clamp(min=0)
        xb = Ft[ixc].to(DEV, non_blocking=True)
        yb = yt[ixc].to(DEV, non_blocking=True)
        return xb, yb, pad.to(DEV), ix

    for f in (0, 1):
        tr, te = np.flatnonzero(fold == f), np.flatnonzero(fold != f)
        torch.manual_seed(f)
        model = SetModel(F.shape[1]).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        bs = 512
        steps = EPOCHS * (len(tr) // bs + 1)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
        lossf = nn.BCEWithLogitsLoss(reduction="none")
        t0, s = time.time(), 0
        for ep in range(EPOCHS):
            model.train()
            perm = np.random.default_rng(ep + 10 * f).permutation(tr)
            run = 0.0
            for b in range(0, len(perm), bs):
                xb, yb, pad, _ = batch(perm[b:b + bs])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logit = model(xb, pad)
                l = lossf(logit.float(), yb)
                loss = (l * (~pad)).sum() / (~pad).sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                run += loss.item()
                s += 1
                if s % 500 == 0:
                    log(f"fold {f} ep {ep} step {s}/{steps} loss {run / 500:.4f} ({(time.time() - t0) / 60:.1f} min)")
                    run = 0.0
        model.eval()
        with torch.inference_mode():
            for b in range(0, len(te), 2048):
                xb, _, pad, ix = batch(te[b:b + 2048])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pr = torch.sigmoid(model(xb, pad).float()).cpu().numpy()
                m = ix.numpy() >= 0
                pred[ix.numpy()[m]] = pr[m]
        log(f"fold {f} done in {(time.time() - t0) / 60:.1f} min")
        del model
        torch.cuda.empty_cache()
    pl.DataFrame({"p": pred}).write_parquet(f"{DN}jalt_set9.parquet")
    # evaluation (rows beyond the cap keep probability 0)
    xgb = meta.select("rid", "oid").join(pl.read_parquet(f"{DN}train_joint9.parquet", columns=["rid", "oid", "pj"]), on=["rid", "oid"], how="left")["pj"].to_numpy()
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    s1 = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).join(keep, on="rid", how="semi")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid"]).join(keep, on="rid", how="semi")
    for name, p in (("xgb (current)", xgb), ("set transformer", pred), ("avg(xgb,set)", (xgb + pred) / 2)):
        pr = decide.ef_predict(meta.select("rid", "oid").with_columns(p2=pl.Series(p)), "p2", 0.0, 1.0, 0.05)
        sc = {c: ev.macro_f05(pr.join(s1.filter(pl.col("country") == c), on="rid", how="semi"),
                              gt.join(s1.filter(pl.col("country") == c), on="rid", how="semi"),
                              s1.filter(pl.col("country") == c)["rid"]) for c in ("India", "US")}
        log(f"  {name:<18s} India {sc['India']:.5f}  US {sc['US']:.5f}")


if __name__ == "__main__":
    main()
