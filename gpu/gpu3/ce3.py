"""Stronger cross-encoders on the uncertain band (A100). Pair input (entity text, record text) - the format that works
locally (e5-small pair CE), not the single long rival-context string that failed in E2a.

  --mode inc     India/US, 2-fold by entity (band_train.fold): model f trains on fold f, scores fold 1-f (out-of-fold);
                 test India/US band rows = mean of both models.          -> out/ce3_<tag>_inc_{train,test}.parquet
  --mode loco    unseen-country proxy for France: train on one country's LOCO band, score the other country.
                                                                          -> out/ce3_<tag>_loco.parquet
  --mode france  train on the LOCO band of both countries, score the France test band rows.
                                                                          -> out/ce3_<tag>_france_test.parquet
Encoders (e5-large): full fine-tune, fp32 weights + bf16 autocast. Decoder LLMs (--lora, e.g. Qwen2.5-1.5B): bf16 +
LoRA, the pair is written as one prompt. Every run logs AUC of the new score vs the band's current probability.
"""
import argparse
import math
import os
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import DEV, log, save_json

HERE = os.path.dirname(os.path.abspath(__file__))


def auc(y, s):
    y = np.asarray(y)
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))


def texts(bundle, split):
    rec = pl.read_parquet(f"{bundle}/records_{split}.parquet", columns=["side", "idx", "name", "addr"])
    mk = lambda d: d.sort("idx").select(t=pl.col("name") + pl.lit(" | ") + pl.col("addr").str.slice(0, 160))["t"].to_list()
    return mk(rec.filter(pl.col("side") == 1)), mk(rec.filter(pl.col("side") == 2))


class Pairs:
    """Tokenizes (entity, record) pairs: encoder -> text pair; decoder -> one prompt."""

    def __init__(self, a, tok, t1, t2, df):
        self.a, self.tok = a, tok
        r, o = df["rid"].to_numpy(), df["oid"].to_numpy()
        self.x1 = [t1[i] for i in r]
        self.x2 = [t2[i] for i in o]
        self.len = np.fromiter((len(p) + len(q) for p, q in zip(self.x1, self.x2)), dtype=np.int32, count=len(r))

    def enc(self, b):
        if self.a.lora:
            txt = [f"Business A: {self.x1[i]}\nBusiness B: {self.x2[i]}\nSame business?" for i in b]
            return self.tok(txt, padding=True, truncation=True, max_length=self.a.max_len, return_tensors="pt").to(DEV)
        return self.tok([self.x1[i] for i in b], [self.x2[i] for i in b], padding=True, truncation="longest_first",
                        max_length=self.a.max_len, return_tensors="pt").to(DEV)


def load_model(a, path=None):
    tok = AutoTokenizer.from_pretrained(path or a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    if a.lora:
        from peft import LoraConfig, PeftModel, get_peft_model
        model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=1, torch_dtype=torch.bfloat16)
        model.config.pad_token_id = tok.pad_token_id
        if path:
            model = PeftModel.from_pretrained(model, path)
        else:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
            model = get_peft_model(model, LoraConfig(task_type="SEQ_CLS", r=16, lora_alpha=32, lora_dropout=0.05,
                                                     target_modules="all-linear"))
    else:
        model = AutoModelForSequenceClassification.from_pretrained(path or a.model, num_labels=1, torch_dtype=torch.float32)
        model.config.pad_token_id = tok.pad_token_id
    return tok, model.to(DEV)


def batches(lengths, bs, shuffle, seed=0):
    idx = np.arange(len(lengths))
    if not shuffle:
        o = np.argsort(lengths)
        return [o[i:i + bs] for i in range(0, len(o), bs)]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(idx)
    out = []
    for i in range(0, len(idx), bs * 64):  # length-sorted mega-batches: little padding, still random
        c = idx[i:i + bs * 64]
        c = c[np.argsort(lengths[c])]
        out += [c[j:j + bs] for j in range(0, len(c), bs)]
    return [out[i] for i in rng.permutation(len(out))]


def train(a, tok, model, P, y, seed):
    bl = batches(P.len, a.bs, True, seed) * a.epochs
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    steps = math.ceil(len(bl) / a.grad_accum)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    base = -(y.mean() * math.log(max(y.mean(), 1e-6)) + (1 - y.mean()) * math.log(max(1 - y.mean(), 1e-6)))
    model.train()
    t0, run, seen, bad = time.time(), 0.0, 0, 0
    for i, b in enumerate(bl):
        enc = P.enc(b)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(**enc).logits.squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logit.float(), torch.from_numpy(y[b]).float().to(DEV)) / a.grad_accum
        if not math.isfinite(loss.item()):
            bad += 1
            opt.zero_grad(set_to_none=True)
            if bad > 20:
                raise RuntimeError("loss not finite 20 times - training diverged")
            continue
        loss.backward()
        run += loss.item() * a.grad_accum
        seen += len(b)
        if (i + 1) % a.grad_accum == 0 or i == len(bl) - 1:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            flag = "  <-- NOT LEARNING (loss ~ base rate)" if i + 1 >= 2000 and run / 200 > 0.9 * base else ""
            log(f"  batch {i + 1}/{len(bl)} loss {run / 200:.4f} (base {base:.3f}) {seen / el:.0f} pairs/s, "
                f"ETA {el / (i + 1) * (len(bl) - i - 1) / 60:.0f} min{flag}")
            run = 0.0


@torch.inference_mode()
def predict(a, model, P):
    model.eval()
    out = np.zeros(len(P.len), dtype=np.float32)
    t0 = time.time()
    for i, b in enumerate(batches(P.len, a.pred_bs, False)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out[b] = torch.sigmoid(model(**P.enc(b)).logits.squeeze(-1).float()).cpu().numpy()
        if (i + 1) % 500 == 0:
            log(f"  predicted {min(len(out), (i + 1) * a.pred_bs):,}/{len(out):,} ({(i + 1) * a.pred_bs / (time.time() - t0):.0f}/s)")
    return out


def sample(df, n, seed):
    return df if not n or df.height <= n else df.sample(n, seed=seed)


def fit_score(a, name, tr, scores, t1, t2, seed):
    """Train (or reload) model `name` on tr; returns predictions for each (df, texts) in scores."""
    mdir = f"{a.out}/models/{name}"
    if os.path.exists(f"{mdir}/.done"):
        tok, model = load_model(a, mdir)
        log(f"[{name}] loaded")
    else:
        tok, model = load_model(a)
        tr = sample(tr, a.train_limit, seed)
        log(f"[{name}] training {a.model} on {tr.height:,} pairs (pos {tr['y'].mean():.3f})")
        train(a, tok, model, Pairs(a, tok, t1[0], t1[1], tr), tr["y"].to_numpy().astype(np.float32), seed)
        os.makedirs(mdir, exist_ok=True)
        model.save_pretrained(mdir)
        tok.save_pretrained(mdir)
        open(f"{mdir}/.done", "w").close()
    outs = [predict(a, model, Pairs(a, tok, tx[0], tx[1], df)) for df, tx in scores]
    del model
    torch.cuda.empty_cache()
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["inc", "loco", "france"])
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", default=f"{HERE}/out")
    ap.add_argument("--data", default=f"{HERE}/data")
    ap.add_argument("--model", default="intfloat/multilingual-e5-large")
    ap.add_argument("--tag", default="e5l")
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--pred_bs", type=int, default=512)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--train_limit", type=int, default=0, help="subsample training pairs per model (0 = all)")
    ap.add_argument("--score_limit", type=int, default=0, help="inc: score only this many out-of-fold pairs per fold (diagnostic)")
    ap.add_argument("--no_test", action="store_true", help="inc: skip scoring the test band (diagnostic runs)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    os.makedirs(f"{a.out}/models", exist_ok=True)
    tr_t, te_t = texts(a.bundle, "train"), None
    res = {}
    if a.mode == "inc":
        b = pl.read_parquet(f"{a.data}/band_train.parquet")
        te = pl.read_parquet(f"{a.data}/band_test.parquet").filter(pl.col("country") != "France")
        if a.smoke:
            b, te = b.sample(4000, seed=0), te.sample(2000, seed=0)
        te_t = texts(a.bundle, "test")
        if a.score_limit:  # diagnostic: a fixed random subsample of the band is scored out-of-fold
            b = b.join(pl.concat([b.filter(pl.col("fold") == f).sample(min(a.score_limit, b.filter(pl.col("fold") == f).height), seed=5)
                                  for f in (0, 1)]).select("rid", "oid").with_columns(_ev=pl.lit(True)), on=["rid", "oid"], how="left")
        else:
            b = b.with_columns(_ev=pl.lit(True))
        oof = np.full(b.height, np.nan, np.float32)
        tst = np.zeros(te.height, np.float32)
        for f in (0, 1):
            m = (b["fold"] == f).to_numpy()
            ev = (~m) & b["_ev"].fill_null(False).to_numpy()
            sc = [(b.filter(pl.Series(ev)), tr_t)] + ([] if a.no_test else [(te, te_t)])
            outs = fit_score(a, f"{a.tag}_inc_f{f}", b.filter(pl.Series(m)).drop("_ev"), sc, tr_t, te_t, f)
            oof[ev] = outs[0]
            if not a.no_test:
                tst = tst + outs[1] / 2
            s_oof = outs[0]
            y, p = b.filter(pl.Series(ev))["y"].to_numpy(), b.filter(pl.Series(ev))["p"].to_numpy()
            res[f"fold{f}"] = dict(auc_new=auc(y, s_oof), auc_current=auc(y, p))
            log(f"[{a.tag} inc] fold {f} -> {1 - f}: AUC new {res[f'fold{f}']['auc_new']:.4f} vs current p {res[f'fold{f}']['auc_current']:.4f}")
        b.select("rid", "oid").with_columns(s=pl.Series(oof)).filter(pl.col("s").is_not_nan()).write_parquet(f"{a.out}/ce3_{a.tag}_inc_train.parquet")
        if not a.no_test:
            te.select("rid", "oid").with_columns(s=pl.Series(tst)).write_parquet(f"{a.out}/ce3_{a.tag}_inc_test.parquet")
    elif a.mode == "loco":
        b = pl.read_parquet(f"{a.data}/band_loco.parquet")
        if a.smoke:
            b = b.sample(4000, seed=0)
        out = np.zeros(b.height, np.float32)
        for i, c in enumerate(("India", "US")):
            m = (b["country"] == c).to_numpy()
            (s,) = fit_score(a, f"{a.tag}_loco_no{c}", b.filter(pl.Series(~m)), [(b.filter(pl.Series(m)), tr_t)], tr_t, None, 10 + i)
            out[m] = s
            y, p = b.filter(pl.Series(m))["y"].to_numpy(), b.filter(pl.Series(m))["p"].to_numpy()
            res[c] = dict(auc_new=auc(y, s), auc_current=auc(y, p))
            log(f"[{a.tag} loco] trained without {c}, scored {c}: AUC new {res[c]['auc_new']:.4f} vs LOCO p {res[c]['auc_current']:.4f}")
        b.select("rid", "oid").with_columns(s=pl.Series(out)).write_parquet(f"{a.out}/ce3_{a.tag}_loco.parquet")
    else:
        b = pl.read_parquet(f"{a.data}/band_loco.parquet")
        te = pl.read_parquet(f"{a.data}/band_test.parquet").filter(pl.col("country") == "France")
        if a.smoke:
            b, te = b.sample(4000, seed=0), te.sample(2000, seed=0)
        te_t = texts(a.bundle, "test")
        (s,) = fit_score(a, f"{a.tag}_france", b, [(te, te_t)], tr_t, te_t, 20)
        te.select("rid", "oid").with_columns(s=pl.Series(s)).write_parquet(f"{a.out}/ce3_{a.tag}_france_test.parquet")
        res["france_scored"] = te.height
    save_json(dict(model=a.model, mode=a.mode, results=res), f"{a.out}/ce3_{a.tag}_{a.mode}.json")
    log(f"[{a.tag} {a.mode}] DONE")


if __name__ == "__main__":
    main()
