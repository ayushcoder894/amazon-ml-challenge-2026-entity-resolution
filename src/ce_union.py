"""Cross-encoder fine-tuned on the UNION uncertain band (v12 candidates with pf in (0.02, 0.98) + dense new pairs with pn
in (0.02, 0.98)). Warm start from v12's pair CE of the same fold (ce_v10_fold{f}, trained on fold-f entities only),
one more epoch on the fold-f union band, then scores the other fold (honest, v12 folds). Test: mean of both models.
  train -> DN/ceu_train.parquet (rid, oid, ceu)      test -> DN/ceu_test.parquet
"""
import math
import os
import sys
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import ce
from dense_addon import D, DN, log

SRC = "D:/amazon_er/models_ce/ce_v10_fold{f}"
OUTM = "D:/amazon_er/models_ce/ceu_fold{f}"
LO, HI = 0.02, 0.98
EPOCHS = float(os.environ.get("CEU_EPOCHS", 1))
LR = float(os.environ.get("CEU_LR", 2e-5))
BS = int(os.environ.get("CEU_BS", 64))


def band(split):
    f = "train_oof_final_v12_twin" if split == "train" else "test_final_v12_ce_cectx_twin"
    K = pl.read_parquet(f"{D}{f}.parquet", columns=["rid", "oid", "pf"]).rename({"pf": "p"})
    N = pl.read_parquet(f"{DN}{split}_pn.parquet", columns=["rid", "oid", "pn"]).rename({"pn": "p"})
    b = pl.concat([K, N]).filter((pl.col("p") > LO) & (pl.col("p") < HI))
    if split == "test":  # India / US only (France is decided by its own recipe)
        s1c = pl.read_parquet(f"{D}test_s1_norm.parquet", columns=["rid", "country"])
        b = b.join(s1c.filter(pl.col("country").is_in(["India", "US"])).select("rid"), on="rid", how="semi")
    b = b.with_columns(fold=(pl.col("rid").hash(seed=11) % 2).cast(pl.Int8))
    if split == "train":
        gt = pl.read_parquet(f"{D}train_gt_rid.parquet").select("rid", "oid").with_columns(y=pl.lit(1, pl.Int8))
        b = b.join(gt, on=["rid", "oid"], how="left").with_columns(pl.col("y").fill_null(0))
    s1, o = ce.texts(split)
    return b.join(s1, on="rid").join(o, on="oid")


def finetune(df, f):
    tok = AutoTokenizer.from_pretrained(SRC.format(f=f))
    model = AutoModelForSequenceClassification.from_pretrained(SRC.format(f=f)).to(ce.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    n_steps = int(EPOCHS * math.ceil(df.height / BS))
    warm = max(1, int(0.05 * n_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (n_steps - s) / max(1, n_steps - warm)))
    t1, t2, y = df["t1"].to_list(), df["t2"].to_list(), df["y"].to_numpy().astype(np.float32)
    model.train()
    step, t0, run = 0, time.time(), 0.0
    idx = np.random.default_rng(f).permutation(df.height)
    idx = np.concatenate([idx] * math.ceil(EPOCHS))[: n_steps * BS]
    for i in range(0, len(idx), BS):
        bi = idx[i:i + BS]
        enc = tok([t1[j] for j in bi], [t2[j] for j in bi], padding=True, truncation=True, max_length=ce.MAXLEN,
                  return_tensors="pt").to(ce.DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(**enc).logits.squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logit.float(), torch.from_numpy(y[bi]).to(ce.DEV))
        if not torch.isfinite(loss):
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run += loss.item()
        step += 1
        if step % 1000 == 0:
            log(f"  fold {f} step {step}/{n_steps} loss {run / 1000:.4f} ETA {(time.time() - t0) / step * (n_steps - step) / 60:.1f} min")
            run = 0.0
    os.makedirs(OUTM.format(f=f), exist_ok=True)
    model.save_pretrained(OUTM.format(f=f))
    tok.save_pretrained(OUTM.format(f=f))
    return model, tok


def main(split):
    from sklearn.metrics import roc_auc_score
    b = band(split)
    log(f"{split}: union band {b.height:,} pairs")
    out = np.zeros(b.height, dtype=np.float32)
    for f in (0, 1):
        if split == "train":
            if os.path.exists(OUTM.format(f=f) + "/config.json"):
                tok = AutoTokenizer.from_pretrained(OUTM.format(f=f))
                model = AutoModelForSequenceClassification.from_pretrained(OUTM.format(f=f)).to(ce.DEV)
            else:
                tr = b.filter(pl.col("fold") == f)
                log(f"fold {f}: fine-tuning on {tr.height:,} pairs (pos {tr['y'].mean():.3f})")
                model, tok = finetune(tr, f)
            m = (b["fold"] != f).to_numpy()
            out[m] = ce.predict(model, tok, b.filter(pl.Series(m)))
            log(f"fold {f} -> other fold: AUC {roc_auc_score(b.filter(pl.Series(m))['y'], out[m]):.4f} "
                f"(band probability AUC {roc_auc_score(b.filter(pl.Series(m))['y'], b.filter(pl.Series(m))['p']):.4f})")
        else:
            tok = AutoTokenizer.from_pretrained(OUTM.format(f=f))
            model = AutoModelForSequenceClassification.from_pretrained(OUTM.format(f=f)).to(ce.DEV)
            out += ce.predict(model, tok, b) / 2
            log(f"model {f} scored")
        del model
        torch.cuda.empty_cache()
    b.select("rid", "oid").with_columns(ceu=pl.Series(out)).write_parquet(f"{DN}ceu_{split}.parquet")
    log(f"wrote {DN}ceu_{split}.parquet")


if __name__ == "__main__":
    main(sys.argv[1])
