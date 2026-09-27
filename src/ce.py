"""Cross-encoder re-scoring of the uncertain band (honest 2-fold), local GPU.

band = candidate pairs whose stage-2 probability is in (LO, HI). A multilingual-e5-small (MIT) cross-encoder reads
"S1 name | address" [SEP] "record name | address" and predicts match / no match.
  train: CE model f is trained on band pairs of fold-f S1 entities and scores fold 1-f band pairs (out-of-fold);
  test : average of both CE models.
Outputs data/ce_{split}{tag}.parquet with (rid, oid, ce).
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

D = "C:/ML/Challenges/amazon/work/data/"
MD = "C:/ML/Challenges/amazon/work/models_ce/"
BASE = os.environ.get("CE_BASE", "intfloat/multilingual-e5-small")
LO, HI = 0.02, 0.98
BAND_COL = os.environ.get("CE_BAND_COL", "p2")      # p2 = first CE round; p3 = second round on the hard band
CE_NAME = os.environ.get("CE_NAME", "ce")           # model / output name prefix
SRC = os.environ.get("CE_SRC", "train_oof{tag}")    # OOF source table (p3 lives in train_oof_ce{tag})
TEST_SRC = os.environ.get("CE_TEST_SRC", "test_pred{mname}")
MAXLEN = int(os.environ.get("CE_MAXLEN", 128))
DEV = torch.device("cuda")
MTAG = os.environ.get("CE_MTAG")                    # reuse the CE models of another version (same folds -> still OOF)
CACHE = os.environ.get("CE_CACHE")                  # parquet with already computed (rid, oid, CE_NAME) scores


def cached_predict(model, tok, df):
    """Scores for df; pairs found in the CE_CACHE table are taken from there (pairwise model -> same score)."""
    if not CACHE:
        return predict(model, tok, df)
    c = df.select("rid", "oid").with_row_index("_i").join(pl.read_parquet(f"{D}{CACHE}.parquet"), on=["rid", "oid"],
                                                          how="left").sort("_i")
    out = c[CE_NAME].to_numpy().astype(np.float32).copy()
    miss = np.flatnonzero(c[CE_NAME].is_null().to_numpy())
    log(f"  cache hit {1 - len(miss) / max(1, len(out)):.3f}; scoring {len(miss):,} pairs")
    if len(miss):
        out[miss] = predict(model, tok, df[miss])
    return out


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def texts(split):
    if os.environ.get("CE_TEXT") == "raw":  # original strings (scripts, casing, punctuation kept)
        s1 = pl.read_parquet(f"{D}{split}_source1.parquet", columns=["business_name", "business_address"]).with_row_index(
            "rid").select("rid", t1=pl.col("business_name") + " | " + pl.col("business_address"))
        o = pl.concat([pl.read_parquet(f"{D}{split}_source{k}.parquet", columns=["business_name", "business_address"])
                       for k in (2, 3)]).with_row_index("oid").select(
            "oid", t2=pl.col("business_name") + " | " + pl.col("business_address"))
        return s1, o
    s1 = pl.read_parquet(f"{D}{split}_s1_norm.parquet", columns=["rid", "nf", "af"]).select(
        "rid", t1=pl.col("nf") + " | " + pl.col("af"))
    o = pl.concat([pl.read_parquet(f"{D}{split}_s{k}_norm.parquet", columns=["nf", "af"]) for k in (2, 3)]
                  ).with_row_index("oid").select("oid", t2=pl.col("nf") + " | " + pl.col("af"))
    return s1, o


def band(pred, s1, o):
    b = pred.filter((pl.col(BAND_COL) > LO) & (pl.col(BAND_COL) < HI))
    b = b.join(s1, on="rid").join(o, on="oid")
    if os.environ.get("CE_CTX") == "1":
        # listwise context: the S1's 3 most likely OTHER candidates (by out-of-fold p3), appended to the record text,
        # so the model can see whether this record is the odd one out in the S1's cluster
        ctx_n = int(os.environ.get("CE_CTX_N", 3))
        c = (pred.select("rid", "oid", "p3").join(o, on="oid")
             .sort(["rid", "p3"], descending=[False, True])
             .group_by("rid", maintain_order=True).agg(c_oid=pl.col("oid").head(ctx_n + 1), c_t=pl.col("t2").head(ctx_n + 1)))
        b = b.join(c, on="rid", how="left").with_row_index("_i")
        x = (b.select("_i", "oid", "c_oid", "c_t").explode("c_oid", "c_t")
             .filter(pl.col("c_oid") != pl.col("oid")).group_by("_i").agg(ctx=pl.col("c_t").head(ctx_n).str.join(" ; ")))
        b = (b.join(x, on="_i", how="left").sort("_i")
             .with_columns(t2=pl.col("t2") + pl.lit(" || ") + pl.col("ctx").fill_null(""))
             .drop("_i", "c_oid", "c_t", "ctx"))
    return b


def batches(df, bs, shuffle=False, seed=0):
    idx = np.arange(df.height)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    else:  # length-sorted for fast inference
        idx = np.argsort((df["t1"].str.len_chars() + df["t2"].str.len_chars()).to_numpy())
    for i in range(0, len(idx), bs):
        yield idx[i:i + bs]


def train_model(df, out, epochs=2, bs=64, lr=4e-5):
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1).to(DEV)
    if os.environ.get("CE_FREEZE_EMB") == "1":  # big multilingual vocab: keep word embeddings fixed (memory)
        model.base_model.embeddings.word_embeddings.weight.requires_grad_(False)
    if os.environ.get("CE_GRAD_CKPT") == "1":  # trade compute for activation memory (8 GB card)
        model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(df.height / bs)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    t1, t2, y = df["t1"].to_list(), df["t2"].to_list(), df["y"].to_numpy().astype(np.float32)
    model.train()
    step, t0, run = 0, time.time(), 0.0
    for ep in range(epochs):
        for bi in batches(df, bs, shuffle=True, seed=ep):
            enc = tok([t1[i] for i in bi], [t2[i] for i in bi], padding=True, truncation=True, max_length=MAXLEN,
                      return_tensors="pt").to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(**enc).logits.squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logit.float(), torch.from_numpy(y[bi]).to(DEV))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item()
            step += 1
            if step % 500 == 0:
                log(f"  step {step}/{steps} loss {run / 500:.4f} ({(time.time() - t0) / 60:.1f} min, "
                    f"ETA {(time.time() - t0) / step * (steps - step) / 60:.1f} min)")
                run = 0.0
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    return model, tok


@torch.inference_mode()
def predict(model, tok, df, bs=256):
    model.eval()
    t1, t2 = df["t1"].to_list(), df["t2"].to_list()
    out = np.zeros(df.height, dtype=np.float32)
    for bi in batches(df, bs):
        enc = tok([t1[i] for i in bi], [t2[i] for i in bi], padding=True, truncation=True, max_length=MAXLEN,
                  return_tensors="pt").to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out[bi] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
    return out


def run_train(tag, epochs):
    s1, o = texts("train")
    cols = ["rid", "oid", "y", "fold", "p2"] + ([BAND_COL] if BAND_COL != "p2" else []) + (
        ["p3"] if os.environ.get("CE_CTX") == "1" and BAND_COL != "p3" else [])
    pred = pl.read_parquet(f"{D}{SRC.format(tag=tag)}.parquet", columns=cols)
    b = band(pred, s1, o)
    log(f"train band: {b.height:,} pairs, pos rate {b['y'].mean():.3f}")
    outs = []
    for f in (0, 1):
        tr, te = b.filter(pl.col("fold") == f), b.filter(pl.col("fold") != f)
        mdir = f"{MD}{CE_NAME}{MTAG or tag}_fold{f}"
        if os.path.exists(f"{mdir}/config.json"):
            tok = AutoTokenizer.from_pretrained(mdir)
            model = AutoModelForSequenceClassification.from_pretrained(mdir).to(DEV)
        else:
            log(f"fold {f}: training on {tr.height:,} pairs")
            model, tok = train_model(tr, mdir, epochs=epochs, bs=int(os.environ.get("CE_BS", 64)),
                                     lr=float(os.environ.get("CE_LR", 4e-5)))
        te = te.with_columns(ce=pl.Series(cached_predict(model, tok, te)))
        outs.append(te.select("rid", "oid", "ce"))
        from sklearn.metrics import roc_auc_score
        log(f"fold {f} -> fold {1 - f}: CE AUC {roc_auc_score(te['y'], te['ce']):.4f} "
            f"vs stage-2 AUC {roc_auc_score(te['y'], te['p2']):.4f}")
        del model
        torch.cuda.empty_cache()
    pl.concat(outs).rename({"ce": CE_NAME}).write_parquet(f"{D}{CE_NAME}_train{tag}.parquet")


def run_test(tag, mname):
    s1, o = texts("test")
    pred = pl.read_parquet(f"{D}{TEST_SRC.format(mname=mname, tag=tag)}.parquet")
    b = band(pred, s1, o)
    log(f"test band: {b.height:,} pairs")
    out, todo = np.full(b.height, np.nan, np.float32), np.arange(b.height)
    if CACHE:  # test cache already holds the mean of both fold models
        c = b.select("rid", "oid").with_row_index("_i").join(pl.read_parquet(f"{D}{CACHE}.parquet"), on=["rid", "oid"],
                                                             how="left").sort("_i")
        out = c[CE_NAME].to_numpy().astype(np.float32).copy()
        todo = np.flatnonzero(c[CE_NAME].is_null().to_numpy())
        log(f"  cache hit {1 - len(todo) / max(1, b.height):.3f}; scoring {len(todo):,} pairs")
    ps = []
    for f in (0, 1):
        if not len(todo):
            break
        mdir = f"{MD}{CE_NAME}{MTAG or tag}_fold{f}"
        tok = AutoTokenizer.from_pretrained(mdir)
        model = AutoModelForSequenceClassification.from_pretrained(mdir).to(DEV)
        ps.append(predict(model, tok, b[todo]))
        del model
        torch.cuda.empty_cache()
    if ps:
        out[todo] = np.mean(ps, axis=0)
    b.select("rid", "oid").with_columns(pl.Series(CE_NAME, out)).write_parquet(f"{D}{CE_NAME}_test{tag}.parquet")
    log("test done")


if __name__ == "__main__":
    mode, tag = sys.argv[1], sys.argv[2]
    if mode == "train":
        run_train(tag, int(sys.argv[3]) if len(sys.argv) > 3 else 2)
    else:
        run_test(tag, sys.argv[3] if len(sys.argv) > 3 else tag)
