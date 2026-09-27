"""E1 - dense retrieval: can a fine-tuned multilingual bi-encoder find the true pairs the key-based blocker misses?

Current candidates cap a perfect matcher at India 0.990 / US 0.994 (harder split). Fold-honest protocol:
the model of fold f is trained on the ground-truth pairs of fold-f Source-1 entities only and retrieves candidates for
the kept fold-(1-f) entities; the reverse index holds ALL kept entities (competition as on test).
  forward : entity -> top-k records          reverse : record -> top-r entities
Report (per country): recall of true pairs and perfect-matcher ceiling with the current candidates vs. the current
candidates + dense forward/reverse candidates, and the number of added candidates per entity.
Outputs: dense_train_new.parquet / dense_test_new.parquet (new pairs with cos + ranks), dense_*_cos.parquet (cos of the
current candidate pairs, a feature for the matcher), reports/dense.json.
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
from transformers import AutoModel, AutoTokenizer

from common import DEV, Evaluator, log, save_json

PREFIX = "query: "  # e5 convention for symmetric matching
AMP = DEV.type == "cuda"


class Encoder(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name)

    def forward(self, texts, max_len):
        b = self.tok([PREFIX + t for t in texts], padding=True, truncation=True, max_length=max_len,
                     return_tensors="pt").to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16) if AMP else contextlib.nullcontext():
            h = self.model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        return F.normalize(((h * m).sum(1) / m.sum(1).clamp(min=1)).float(), dim=-1)

    def save(self, path):
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)


@torch.inference_mode()
def encode(enc, texts, max_len, bs):
    enc.eval()
    n = len(texts)
    order = np.argsort(np.fromiter((len(t) for t in texts), dtype=np.int32, count=n), kind="stable")[::-1].copy()
    out = torch.empty((n, enc.model.config.hidden_size), dtype=torch.float16, device=DEV)
    for i in range(0, n, bs):
        idx = order[i:i + bs]
        out[torch.from_numpy(idx).to(DEV)] = enc([texts[j] for j in idx], max_len).half()
    return out


@torch.inference_mode()
def knn(Q, X, k, q_chunk=4096, x_chunk=500_000):
    """Exact top-k inner products of every row of Q against X (both on DEV) -> numpy (scores, indices)."""
    k = min(k, X.shape[0])
    S = np.empty((Q.shape[0], k), dtype=np.float32)
    I = np.empty((Q.shape[0], k), dtype=np.int64)
    for i in range(0, Q.shape[0], q_chunk):
        q = Q[i:i + q_chunk]
        bs = bi = None
        for j in range(0, X.shape[0], x_chunk):
            s = q @ X[j:j + x_chunk].T
            ts, ti = s.topk(min(k, s.shape[1]), dim=1)
            ti = ti + j
            if bs is None:
                bs, bi = ts, ti
            else:
                cs, ci = torch.cat([bs, ts], 1), torch.cat([bi, ti], 1)
                bs, o = cs.topk(k, dim=1)
                bi = ci.gather(1, o)
        S[i:i + q.shape[0]] = bs.float().cpu().numpy()
        I[i:i + q.shape[0]] = bi.cpu().numpy()
    return S, I


@torch.inference_mode()
def pair_cos(A, B, ia, ib, chunk=200_000):
    out = np.empty(len(ia), dtype=np.float32)
    for i in range(0, len(ia), chunk):
        a = A[torch.from_numpy(ia[i:i + chunk]).to(DEV)].float()
        b = B[torch.from_numpy(ib[i:i + chunk]).to(DEV)].float()
        out[i:i + chunk] = (a * b).sum(1).cpu().numpy()
    return out


VIEW = "both"  # "both" = name | address (default); "name" = name only (set by --view)


def texts(bundle, split):
    rec = pl.read_parquet(f"{bundle}/records_{split}.parquet", columns=["side", "idx", "country", "text", "name"])
    if VIEW == "name":  # name-only view: raw name, lowercased, legal forms / order / native script kept
        rec = rec.with_columns(text=pl.col("name").str.normalize("NFKC").str.to_lowercase())
    t1 = rec.filter(pl.col("side") == 1).sort("idx")
    t2 = rec.filter(pl.col("side") == 2).sort("idx")
    assert t1["idx"][-1] == t1.height - 1 and t2["idx"][-1] == t2.height - 1
    return t1["text"].to_list(), t1["country"].to_numpy(), t2["text"].to_list(), t2["country"].to_numpy()


def train_fold(a, fold, out_dir):
    t1, c1, t2, c2 = texts(a.bundle, "train")
    folds = pl.read_parquet(f"{a.bundle}/folds.parquet")
    gt = (pl.read_parquet(f"{a.bundle}/train_gt.parquet")
          .join(folds.filter(pl.col("fold") == fold).select("rid"), on="rid", how="semi")
          .sample(fraction=1.0, shuffle=True, seed=a.seed + fold))
    if a.max_pairs:
        gt = gt.head(a.max_pairs)
    if a.countries:
        gt = gt.filter(pl.col("country").is_in(a.countries))
    rid, oid, cty = gt["rid"].to_numpy(), gt["oid"].to_numpy(), gt["country"].to_numpy()
    rng = np.random.default_rng(a.seed + fold)
    batches = []
    for c in np.unique(cty):  # one country per batch -> harder in-batch negatives
        idx = np.flatnonzero(cty == c)
        batches += [idx[i:i + a.bs] for i in range(0, len(idx) - a.bs + 1, a.bs)]
    batches = [batches[i] for i in rng.permutation(len(batches))] * a.epochs
    enc = Encoder(a.model).to(DEV)
    opt = torch.optim.AdamW(enc.parameters(), lr=a.lr, weight_decay=0.01)
    steps, warm = len(batches), max(1, len(batches) // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / (steps - warm)))
    log(f"[fold {fold}] training {a.model} on {len(gt):,} pairs, {steps:,} steps of {a.bs}")
    enc.train()
    t0, run, acc = time.time(), 0.0, 0.0
    for s, b in enumerate(batches):
        q = enc([t1[i] for i in rid[b]], a.max_len)
        d = enc([t2[i] for i in oid[b]], a.max_len)
        logits = q @ d.T / a.tau
        r = torch.from_numpy(rid[b]).to(DEV)
        same = (r[:, None] == r[None, :]) & ~torch.eye(len(b), dtype=torch.bool, device=DEV)
        if a.mask_same_text:
            h2 = torch.tensor([hash(t2[i]) for i in oid[b]], device=DEV)
            same |= (h2[:, None] == h2[None, :]) & ~torch.eye(len(b), dtype=torch.bool, device=DEV)
        logits = logits.masked_fill(same, -1e4)  # other matches of the same entity are not negatives
        tgt = torch.arange(len(b), device=DEV)
        loss = (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.T, tgt)) / 2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        opt.step()
        sched.step()
        run += loss.item()
        acc += (logits.argmax(1) == tgt).float().mean().item()
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"loss is {loss.item()} at step {s + 1} - training diverged, stopping")
        if (s + 1) % 200 == 0:
            log(f"  step {s + 1}/{steps} loss {run / 200:.4f} in-batch acc {acc / 200:.3f} "
                f"({(time.time() - t0) / 60:.1f} min, ETA {(time.time() - t0) / (s + 1) * (steps - s - 1) / 60:.1f} min)")
            run = acc = 0.0
    os.makedirs(out_dir, exist_ok=True)
    enc.save(out_dir)


def retrieve(a, split, model_dirs):
    """model f serves the (kept) entities of fold 1-f; on test the folds are rid parity, so each entity gets one model as in train."""
    t1, c1, t2, c2 = texts(a.bundle, split)
    if split == "train":
        ent = pl.read_parquet(f"{a.bundle}/folds.parquet").join(pl.read_parquet(f"{a.bundle}/keep.parquet"), on="rid", how="semi")
        cand = pl.read_parquet(f"{a.bundle}/cand_train.parquet", columns=["rid", "oid"])
    else:
        ent = pl.DataFrame({"rid": np.arange(len(t1), dtype=np.uint32), "country": c1,
                            "fold": (np.arange(len(t1)) % 2).astype(np.int8)})  # half per model, as in train
        cand = pl.read_parquet(f"{a.bundle}/cand_test.parquet", columns=["rid", "oid"])
    if a.smoke:  # local smoke test only: a few entities of one country, their candidates and a record sample
        ent = ent.filter(pl.col("country") == "India").sample(2000, seed=0)
        keep_o = cand.join(ent.select("rid"), on="rid", how="semi")["oid"].to_numpy()
        rnd = np.random.default_rng(0).choice(np.flatnonzero(c2 == "India"), 50_000, replace=False)
        mask = np.zeros(len(c2), bool)
        mask[keep_o] = True
        mask[rnd] = True
        c2 = np.where(mask, c2, "__skip__")
    if a.countries:
        ent = ent.filter(pl.col("country").is_in(a.countries))
    new_parts, cos_parts = [], []
    for f, mdir in enumerate(model_dirs):
        enc = Encoder(mdir).to(DEV)
        for c in sorted(ent["country"].unique().to_list()):
            e_c = ent.filter(pl.col("country") == c)
            ridx = e_c["rid"].to_numpy().astype(np.int64)                    # reverse index: all (kept) entities
            serve = e_c.filter(pl.col("fold") == 1 - f)["rid"].to_numpy().astype(np.int64)
            oidx = np.flatnonzero(c2 == c)
            t = time.time()
            E1 = encode(enc, [t1[i] for i in ridx], a.max_len, a.enc_bs)
            E2 = encode(enc, [t2[i] for i in oidx], a.max_len, a.enc_bs)
            pos = {r: i for i, r in enumerate(ridx)}
            sp = np.array([pos[r] for r in serve])
            Sf, If = knn(E1[torch.from_numpy(sp).to(DEV)], E2, a.fwd_k)
            fwd = pl.DataFrame({"rid": np.repeat(serve, If.shape[1]), "oid": oidx[If.ravel()], "cos": Sf.ravel(),
                                "fwd_rank": np.tile(np.arange(1, If.shape[1] + 1), len(serve))})
            Sr, Ir = knn(E2, E1, a.rev_k)
            rev = pl.DataFrame({"oid": np.repeat(oidx, Ir.shape[1]), "rid": ridx[Ir.ravel()], "cos": Sr.ravel(),
                                "rev_rank": np.tile(np.arange(1, Ir.shape[1] + 1), len(oidx))})
            rev = rev.filter(pl.col("rid").is_in(pl.Series(serve).implode()))
            pairs = (fwd.join(rev, on=["rid", "oid"], how="full", coalesce=True, suffix="_r")
                     .with_columns(cos=pl.coalesce("cos", "cos_r"), fwd_rank=pl.col("fwd_rank").fill_null(999),
                                   rev_rank=pl.col("rev_rank").fill_null(999)).drop("cos_r")
                     .with_columns(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32), model=pl.lit(f, pl.Int8)))
            new_parts.append(pairs)
            # cosine of the current candidate pairs of the served entities (feature for the matcher)
            cc = cand.join(pl.DataFrame({"rid": serve.astype(np.uint32)}), on="rid", how="semi")
            opos = {o: i for i, o in enumerate(oidx)}
            cc = cc.filter(pl.col("oid").is_in(pl.Series(oidx).implode()))
            ia = np.array([pos[r] for r in cc["rid"].to_numpy()], dtype=np.int64)
            ib = np.array([opos[o] for o in cc["oid"].to_numpy()], dtype=np.int64)
            cos_parts.append(cc.with_columns(cos=pl.Series(pair_cos(E1, E2, ia, ib)), model=pl.lit(f, pl.Int8)))
            log(f"  [{split} model {f}] {c}: {len(serve):,} entities, {len(oidx):,} records, {pairs.height:,} dense pairs "
                f"({time.time() - t:.0f}s)")
            del E1, E2
            torch.cuda.empty_cache()
        del enc
        torch.cuda.empty_cache()
    dense = (pl.concat(new_parts).group_by("rid", "oid")
             .agg(cos=pl.col("cos").max(), fwd_rank=pl.col("fwd_rank").min(), rev_rank=pl.col("rev_rank").min()))
    cos = pl.concat(cos_parts).group_by("rid", "oid").agg(dense_cos=pl.col("cos").mean())
    return dense, cos, cand


def report(a, dense, cand):
    ev = Evaluator(a.bundle)
    gt = ev.gt.select("rid", "oid", "country")
    n_s1 = ev.s1.group_by("country").len()
    rows = {}
    configs = {"current": None}
    for k in (5, 10, 20):
        for r in (1, 3, 5):
            if k <= a.fwd_k and r <= a.rev_k:
                configs[f"current + fwd{k} + rev{r}"] = (k, r)
    for name, kr in configs.items():
        if kr is None:
            u = cand
        else:
            add = dense.filter((pl.col("fwd_rank") <= kr[0]) | (pl.col("rev_rank") <= kr[1])).select("rid", "oid")
            u = pl.concat([cand, add]).unique()
        hit = gt.join(u, on=["rid", "oid"], how="semi").group_by("country").len()
        added = u.join(cand, on=["rid", "oid"], how="anti").join(ev.s1.select("rid", "country"), on="rid").group_by("country").len()
        ceil = ev.ceiling(u)
        rows[name] = {}
        for c in sorted(n_s1["country"].to_list()):
            g = gt.filter(pl.col("country") == c).height
            h = hit.filter(pl.col("country") == c)["len"].sum()
            ad = added.filter(pl.col("country") == c)["len"].sum()
            rows[name][c] = dict(recall=round(h / g, 5), ceiling_f05=ceil[c],
                                 added_per_s1=round(ad / n_s1.filter(pl.col("country") == c)["len"].item(), 3))
        log(f"  {name:<28s} " + "  ".join(f"{c}: recall {v['recall']:.4f} ceiling {v['ceiling_f05']:.4f} "
                                             f"+{v['added_per_s1']:.2f}/S1" for c, v in rows[name].items()))
    # how the missed-but-recovered true pairs rank
    miss = gt.join(cand, on=["rid", "oid"], how="anti").join(dense, on=["rid", "oid"])
    rows["recovered_missed_by_rev_rank"] = miss.group_by(pl.col("rev_rank").clip(1, 6)).len().sort("rev_rank").to_dicts()
    rows["recovered_missed_by_fwd_rank"] = miss.group_by(pl.col("fwd_rank").clip(1, 21)).len().sort("fwd_rank").to_dicts()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--enc_bs", type=int, default=2048)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_pairs", type=int, default=3_000_000)
    ap.add_argument("--fwd_k", type=int, default=20)
    ap.add_argument("--rev_k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_test", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="local smoke test on a tiny slice")
    ap.add_argument("--view", default="both", choices=["both", "name"])
    ap.add_argument("--countries", nargs="*", default=None)
    ap.add_argument("--mask_same_text", action="store_true")
    a = ap.parse_args()
    global VIEW
    VIEW = a.view
    os.makedirs(f"{a.out}/reports", exist_ok=True)
    mdirs = [f"{a.out}/models/dense_fold{f}" for f in (0, 1)]
    for f, d in enumerate(mdirs):
        if not os.path.exists(f"{d}/.done"):
            train_fold(a, f, d)
            open(f"{d}/.done", "w").close()
    if not os.path.exists(f"{a.out}/dense_train_new.parquet"):
        dense, cos, cand = retrieve(a, "train", mdirs)
        dense = dense.join(cand, on=["rid", "oid"], how="anti")
        gt = pl.read_parquet(f"{a.bundle}/train_gt.parquet", columns=["rid", "oid"]).with_columns(y=pl.lit(1, pl.Int8))
        dense.join(gt, on=["rid", "oid"], how="left").with_columns(pl.col("y").fill_null(0)).write_parquet(f"{a.out}/dense_train_new.parquet")
        (dense.filter((pl.col("fwd_rank") <= 10) | (pl.col("rev_rank") <= 3)).join(gt, on=["rid", "oid"], how="left")
         .with_columns(pl.col("y").fill_null(0)).write_parquet(f"{a.out}/dense_train_new_small.parquet"))
        cos.write_parquet(f"{a.out}/dense_train_cos.parquet")
    dense = pl.read_parquet(f"{a.out}/dense_train_new.parquet")
    cand = pl.read_parquet(f"{a.bundle}/cand_train.parquet", columns=["rid", "oid"])
    log("E1 dense retrieval - harder split, out-of-fold:")
    rep = report(a, dense, cand)
    save_json(dict(model=a.model, results=rep), f"{a.out}/reports/dense.json")
    if not a.skip_test and not os.path.exists(f"{a.out}/dense_test_new.parquet"):
        dense_t, cos_t, cand_t = retrieve(a, "test", mdirs)
        (dense_t.join(cand_t, on=["rid", "oid"], how="anti").filter((pl.col("fwd_rank") <= 10) | (pl.col("rev_rank") <= 3))
         .write_parquet(f"{a.out}/dense_test_new.parquet"))
        cos_t.write_parquet(f"{a.out}/dense_test_cos.parquet")
    log("E1 done")


if __name__ == "__main__":
    main()
