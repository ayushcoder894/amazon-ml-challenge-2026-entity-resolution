"""E2 - stronger reranker with rival context on the hard band.

The dominant remaining error: records that several same-name entities compete for. The current cross-encoders
see (entity, record) or (entity, record, entity's other records) - never the RIVAL entities. Here every hard pair
(current final probability pf in (lo, hi)) is read as
  ENTITY: name | address || RECORD: name | address || ENTITY'S OTHER RECORDS: ... || RIVALS: rival entity
  [its best record] ; ...
by a stronger model: an encoder fine-tuned fully (e.g. BAAI/bge-m3, MIT) or a decoder LLM with LoRA (e.g.
Qwen/Qwen2.5-1.5B / 7B, Apache-2.0). Two-fold by entity: model f trains on fold-f pairs and scores fold 1-f; test =
mean of both. Evaluation: a 3-parameter logistic combiner of pf and the new score, fitted on the other fold, then
the pipeline's decision rule; honest macro F0.5 per country on the harder split vs. the current pipeline.
"""
import argparse
import contextlib
import math
import os
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import DEV, Evaluator, auc, log, save_json


def texts(bundle, split):
    rec = pl.read_parquet(f"{bundle}/records_{split}.parquet", columns=["side", "idx", "name", "addr"])
    mk = lambda d: (d.sort("idx").select(t=pl.col("name") + pl.lit(" | ") + pl.col("addr").str.slice(0, 140))["t"].to_list())
    return mk(rec.filter(pl.col("side") == 1)), mk(rec.filter(pl.col("side") == 2))


def with_context(cand, band, t1, t2, n_others, n_rivals):
    """Adds column `text` to the band pairs (rid, oid) from the full candidate table (rid, oid, pf)."""
    c = cand.select("rid", "oid", "pf")
    by_rid = c.sort(["rid", "pf"], descending=[False, True]).group_by("rid", maintain_order=True).agg(
        ol=pl.col("oid").head(n_others + 1))
    by_oid = c.sort(["oid", "pf"], descending=[False, True]).group_by("oid", maintain_order=True).agg(
        rl=pl.col("rid").head(n_rivals + 1))
    best = dict(zip(by_rid["rid"].to_list(), by_rid["ol"].to_list()))
    x = band.join(by_rid, on="rid", how="left").join(by_oid, on="oid", how="left")
    out = []
    for rid, oid, ol, rl in x.select("rid", "oid", "ol", "rl").iter_rows():
        others = [t2[o] for o in (ol or []) if o != oid][:n_others]
        riv = []
        for r in [r for r in (rl or []) if r != rid][:n_rivals]:
            b = [o for o in best.get(r, []) if o != oid]
            riv.append(t1[r] + (f" [its best record: {t2[b[0]]}]" if b else ""))
        out.append(f"ENTITY: {t1[rid]} || RECORD: {t2[oid]} || ENTITY'S OTHER RECORDS: {' ; '.join(others) or 'none'}"
                   f" || RIVAL ENTITIES FOR THIS RECORD: {' ; '.join(riv) or 'none'}")
    return x.drop("ol", "rl").with_columns(text=pl.Series(out))


def load_model(a, path=None):
    """Encoder: full fine-tune (fp32 weights, bf16 autocast). Decoder LLM (--lora): bf16 weights + LoRA adapters.
    path = a saved fold model (adapter directory for LoRA)."""
    tok = AutoTokenizer.from_pretrained(path or a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # decoder seq-cls reads the last non-pad token
    if a.lora:
        from peft import LoraConfig, PeftModel, get_peft_model
        model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=1, torch_dtype=torch.bfloat16)
        model.config.pad_token_id = tok.pad_token_id
        if path:
            model = PeftModel.from_pretrained(model, path)
        else:
            if a.grad_ckpt:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                model.enable_input_require_grads()
            model = get_peft_model(model, LoraConfig(task_type="SEQ_CLS", r=a.lora_r, lora_alpha=2 * a.lora_r,
                                                     lora_dropout=0.05, target_modules="all-linear"))
            model.print_trainable_parameters()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(path or a.model, num_labels=1, torch_dtype=torch.float32)
        model.config.pad_token_id = tok.pad_token_id
        if a.grad_ckpt and not path:
            model.gradient_checkpointing_enable()
    return tok, model.to(DEV)


def batches_by_length(lengths, bs, shuffle, seed):
    idx = np.arange(len(lengths))
    if not shuffle:
        return [a for a in np.array_split(np.argsort(lengths), max(1, math.ceil(len(idx) / bs)))]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(idx)
    out = []
    for i in range(0, len(idx), bs * 64):  # sort by length inside mega-batches -> little padding, still random
        chunk = idx[i:i + bs * 64]
        chunk = chunk[np.argsort(lengths[chunk])]
        out += [chunk[j:j + bs] for j in range(0, len(chunk), bs)]
    return [out[i] for i in rng.permutation(len(out))]


def amp():
    return torch.autocast("cuda", dtype=torch.bfloat16) if DEV.type == "cuda" else contextlib.nullcontext()


def train(a, tok, model, txt, y, seed):
    lengths = np.fromiter((len(t) for t in txt), dtype=np.int32, count=len(txt))
    bl = batches_by_length(lengths, a.bs, True, seed) * a.epochs
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    steps = math.ceil(len(bl) / a.grad_accum)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    model.train()
    t0, run, seen = time.time(), 0.0, 0
    for i, b in enumerate(bl):
        enc = tok([txt[j] for j in b], padding=True, truncation=True, max_length=a.max_len, return_tensors="pt").to(DEV)
        with amp():
            logit = model(**enc).logits.squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logit.float(), torch.from_numpy(y[b]).float().to(DEV)) / a.grad_accum
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"loss is {loss.item()} at batch {i + 1} - training diverged, stopping")
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
            log(f"  batch {i + 1}/{len(bl)} loss {run / 200:.4f} ({seen / el:.0f} pairs/s, ETA {el / (i + 1) * (len(bl) - i - 1) / 60:.0f} min)")
            run = 0.0


@torch.inference_mode()
def predict(a, tok, model, txt):
    model.eval()
    lengths = np.fromiter((len(t) for t in txt), dtype=np.int32, count=len(txt))
    out = np.zeros(len(txt), dtype=np.float32)
    t0 = time.time()
    for i, b in enumerate(batches_by_length(lengths, a.pred_bs, False, 0)):
        enc = tok([txt[j] for j in b], padding=True, truncation=True, max_length=a.max_len, return_tensors="pt").to(DEV)
        with amp():
            out[b] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
        if (i + 1) % 500 == 0:
            log(f"  predicted {min(len(txt), (i + 1) * a.pred_bs):,}/{len(txt):,} ({(i + 1) * a.pred_bs / (time.time() - t0):.0f}/s)")
    return out


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_combiner(pf, s, y):
    """3-parameter logistic combiner on logit(pf), logit(s), their product (torch LBFGS)."""
    X = torch.tensor(np.stack([lg(pf), lg(s), lg(pf) * lg(s)], 1), dtype=torch.float64)
    t = torch.tensor(y, dtype=torch.float64)
    w = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(X @ w[:3] + w[3], t) + 1e-4 * (w[:3] ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach().numpy()


def apply_combiner(w, pf, s):
    z = np.stack([lg(pf), lg(s), lg(pf) * lg(s)], 1) @ w[:3] + w[3]
    return 1 / (1 + np.exp(-z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--lo", type=float, default=0.02)
    ap.add_argument("--hi", type=float, default=0.98)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--pred_bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--train_limit", type=int, default=0, help="max training pairs per fold (0 = all)")
    ap.add_argument("--others", type=int, default=3)
    ap.add_argument("--rivals", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_test", action="store_true")
    ap.add_argument("--limit_eval", type=int, default=0, help="smoke test: use only this many band pairs per fold")
    a = ap.parse_args()
    os.makedirs(f"{a.out}/reports", exist_ok=True)
    t1, t2 = texts(a.bundle, "train")
    cand = pl.read_parquet(f"{a.bundle}/cand_train.parquet")
    gt = pl.read_parquet(f"{a.bundle}/train_gt.parquet", columns=["rid", "oid"]).with_columns(y=pl.lit(1, pl.Int8))
    cand = cand.join(gt, on=["rid", "oid"], how="left").with_columns(pl.col("y").fill_null(0))
    band = cand.filter((pl.col("pf") > a.lo) & (pl.col("pf") < a.hi)).select("rid", "oid", "fold", "pf", "y")
    if a.limit_eval:
        band = pl.concat([band.filter(pl.col("fold") == f).head(a.limit_eval) for f in (0, 1)])
    band = with_context(cand, band, t1, t2, a.others, a.rivals)
    log(f"[{a.tag}] band ({a.lo}, {a.hi}): {band.height:,} pairs, positive rate {band['y'].mean():.3f}; example:\n  {band['text'][0][:500]}")
    oof_path = f"{a.out}/rerank_{a.tag}_oof.parquet"
    if not os.path.exists(oof_path):
        parts = []
        for f in (0, 1):
            mdir = f"{a.out}/models/rerank_{a.tag}_fold{f}"
            tr, te = band.filter(pl.col("fold") == f), band.filter(pl.col("fold") != f)
            if a.train_limit and tr.height > a.train_limit:
                tr = tr.sample(a.train_limit, seed=a.seed + f)
            if os.path.exists(f"{mdir}/.done"):
                tok, model = load_model(a, mdir)
            else:
                log(f"[{a.tag}] fold {f}: training on {tr.height:,} pairs")
                tok, model = load_model(a)
                train(a, tok, model, tr["text"].to_list(), tr["y"].to_numpy(), a.seed + f)
                model.save_pretrained(mdir)
                tok.save_pretrained(mdir)
                open(f"{mdir}/.done", "w").close()
            log(f"[{a.tag}] fold {f}: scoring {te.height:,} pairs of the other fold")
            parts.append(te.select("rid", "oid").with_columns(s=pl.Series(predict(a, tok, model, te["text"].to_list()))))
            del model
            torch.cuda.empty_cache()
        pl.concat(parts).write_parquet(oof_path)
    oof = band.drop("text").join(pl.read_parquet(oof_path), on=["rid", "oid"])
    # evaluation: combiner fitted on one fold, applied to the other
    comb = []
    for f in (0, 1):
        fit, app = oof.filter(pl.col("fold") == f), oof.filter(pl.col("fold") != f)
        w = fit_combiner(fit["pf"].to_numpy(), fit["s"].to_numpy(), fit["y"].to_numpy())
        comb.append(app.select("rid", "oid").with_columns(pc=pl.Series(apply_combiner(w, app["pf"].to_numpy(), app["s"].to_numpy()))))
    comb = pl.concat(comb)
    full = (cand.select("rid", "oid", "pf").join(comb, on=["rid", "oid"], how="left").join(oof.select("rid", "oid", "s"), on=["rid", "oid"], how="left")
            .with_columns(pc=pl.coalesce("pc", "pf"), ps=pl.coalesce("s", "pf")))
    ev = Evaluator(a.bundle)
    res = dict(model=a.model, tag=a.tag, band=[a.lo, a.hi], band_pairs=oof.height,
               auc_band_current=round(auc(oof["y"], oof["pf"]), 5), auc_band_new=round(auc(oof["y"], oof["s"]), 5),
               auc_band_combined=round(auc(full.join(oof.select("rid", "oid", "y"), on=["rid", "oid"])["y"],
                                           full.join(oof.select("rid", "oid", "y"), on=["rid", "oid"])["pc"]), 5),
               f05_current=ev.score(full, "pf"), f05_combined=ev.score(full, "pc"), f05_new_replaces_band=ev.score(full, "ps"))
    log(f"[{a.tag}] RESULT band AUC current {res['auc_band_current']:.4f} new {res['auc_band_new']:.4f} "
        f"combined {res['auc_band_combined']:.4f}")
    for k in ("f05_current", "f05_combined", "f05_new_replaces_band"):
        log(f"[{a.tag}] RESULT {k:<24s} {res[k]}")
    save_json(res, f"{a.out}/reports/rerank_{a.tag}.json")
    if not a.skip_test and not os.path.exists(f"{a.out}/rerank_{a.tag}_test.parquet"):
        t1, t2 = texts(a.bundle, "test")
        ct = pl.read_parquet(f"{a.bundle}/cand_test.parquet")
        tb = ct.filter((pl.col("pf") > a.lo) & (pl.col("pf") < a.hi)).select("rid", "oid", "pf")
        if a.limit_eval:
            tb = tb.head(a.limit_eval)
        bt = with_context(ct, tb, t1, t2, a.others, a.rivals)
        log(f"[{a.tag}] test band: {bt.height:,} pairs")
        ss = []
        for f in (0, 1):
            tok, model = load_model(a, f"{a.out}/models/rerank_{a.tag}_fold{f}")
            ss.append(predict(a, tok, model, bt["text"].to_list()))
            del model
            torch.cuda.empty_cache()
        bt.select("rid", "oid").with_columns(s=pl.Series(np.mean(ss, 0))).write_parquet(f"{a.out}/rerank_{a.tag}_test.parquet")
    log(f"[{a.tag}] done")


if __name__ == "__main__":
    main()
