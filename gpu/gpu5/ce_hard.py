"""A100 experiment: hard-negative, edit-aware e5-large cross-encoder.

Uses only supplied US/India labels.  Mode loco trains US->India and India->US,
scoring the exact 10% entity sample expected by src/q7_check.py.  Mode france
trains on both countries and scores only France pairs with base p >= 0.2.
No submission is created here.
"""
import argparse
import gc
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from ce5 import load_model, predict, train, texts


HERE = Path(__file__).resolve().parent
TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
NUM = re.compile(r"\d+[a-z]?", re.IGNORECASE)


def log(*items):
    print(time.strftime("%H:%M:%S"), *items, flush=True)


def fields(t):
    name, sep, addr = t.partition(" | ")
    return name, addr if sep else ""


def relation(a, b):
    na, aa = fields(a)
    nb, ab = fields(b)
    ta, tb = set(TOKEN.findall(na.lower())), set(TOKEN.findall(nb.lower()))
    wa, wb = set(TOKEN.findall(aa.lower())), set(TOKEN.findall(ab.lower()))
    da, db = NUM.findall(aa.lower()), NUM.findall(ab.lower())
    only_a, only_b = ta - tb, tb - ta
    if not only_a and not only_b:
        name = "same name tokens"
    elif len(only_a) == len(only_b) == 1:
        name = "one name word substituted"
    elif not only_a or not only_b:
        name = "name shortened or extended"
    else:
        name = "several name words differ"
    if not aa or not ab:
        addr = "address missing"
    elif aa == ab:
        addr = "same address"
    elif da and db and da[0] == db[0]:
        addr = "same house number"
    elif da and db:
        addr = "different house number"
    else:
        addr = "house number missing"
    street = "street words overlap" if wa & wb else "no street word overlap"
    return f"{name}; {addr}; {street}; shared name words {len(ta & tb)}"


class HardPairs:
    def __init__(self, a, tok, t1, t2, df):
        self.a, self.tok = a, tok
        r, o = df["rid"].to_numpy(), df["oid"].to_numpy()
        self.x1 = [t1[int(i)] for i in r]
        self.x2 = [t2[int(i)] for i in o]
        self.rel = [relation(x, y) for x, y in zip(self.x1, self.x2)]
        self.len = np.fromiter((len(x) + len(y) for x, y in zip(self.x1, self.x2)), dtype=np.int32, count=len(r))

    def enc(self, batch):
        # Relation goes first, so truncation cannot silently delete the new evidence.
        second = [f"Edit pattern: {self.rel[i]}. Record: {self.x2[i]}" for i in batch]
        return self.tok([self.x1[i] for i in batch], second, padding=True,
                        truncation="longest_first", max_length=self.a.max_len,
                        return_tensors="pt").to("cuda")


def take(df, n, seed):
    return df.sample(min(n, df.height), seed=seed) if n and df.height else df.head(0)


def training_sample(df, t1, t2, limit, seed):
    """Balanced real labels; explicitly enrich same-address rival negatives."""
    pos = df.filter(pl.col("y") == 1)
    neg = df.filter(pl.col("y") == 0)
    hard_pos = take(pos.filter(pl.col("p") < 0.8), limit // 4, seed)
    easy_pos = take(pos.filter(pl.col("p") >= 0.8), limit // 4, seed + 1)
    difficult = neg.filter(pl.col("p") >= 0.02)
    # Identify same-address and one-word-swap negatives from actual labels.
    ri, oi = difficult["rid"].to_numpy(), difficult["oid"].to_numpy()
    focused = []
    for k, (r, o) in enumerate(zip(ri, oi)):
        _, aa = fields(t1[int(r)])
        _, ba = fields(t2[int(o)])
        if not aa or aa != ba:
            continue
        focused.append(k)
    same_addr = take(difficult[focused], limit // 4, seed + 2) if focused else difficult.head(0)
    general_neg = take(neg.filter(pl.col("p") >= 0.2), limit // 4, seed + 3)
    chosen = pl.concat([hard_pos, easy_pos, same_addr, general_neg]).unique(["rid", "oid"])
    # Fill sparse bins from real positives/negatives, keeping approximate balance.
    if chosen.height < limit:
        rest = df.join(chosen.select("rid", "oid"), on=["rid", "oid"], how="anti")
        fill = take(rest, limit - chosen.height, seed + 4)
        chosen = pl.concat([chosen, fill])
    chosen = chosen.sample(fraction=1, seed=seed + 5)
    log(f"sample: {chosen.height:,} rows, positive={chosen['y'].mean():.3f}, "
        f"same-address rival pool={len(focused):,}")
    return chosen


def model_scores(a, name, training, scored, train_texts, score_texts):
    model_dir = Path(a.out) / "models" / name
    if (model_dir / ".done").exists():
        tok, model = load_model(a, str(model_dir))
        log("loaded", model_dir)
    else:
        tok, model = load_model(a)
        pair = HardPairs(a, tok, *train_texts, training)
        log("training", name, training.height, "pairs")
        train(a, tok, model, pair, training["y"].to_numpy().astype(np.float32), a.seed)
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(model_dir))
        tok.save_pretrained(str(model_dir))
        (model_dir / ".done").write_text("ok\n")
    log("scoring", name, scored.height, "pairs")
    score = predict(a, model, HardPairs(a, tok, *score_texts, scored))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return scored.select("rid", "oid").with_columns(s=pl.Series(score))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["loco", "france"])
    p.add_argument("--bundle", default="/workspace/1_Amazon_Challenge/gpu2/bundle")
    p.add_argument("--data", default=str(HERE / "data"))
    p.add_argument("--out", default=str(HERE / "out"))
    p.add_argument("--model", default="intfloat/multilingual-e5-large")
    p.add_argument("--tag", default="hard_e5l")
    p.add_argument("--train_limit", type=int, default=60000)
    p.add_argument("--bs", type=int, default=48)
    p.add_argument("--pred_bs", type=int, default=256)
    p.add_argument("--max_len", type=int, default=160)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--seed", type=int, default=73)
    p.add_argument("--selftest", action="store_true", help="check sampling/relations without loading a model")
    a = p.parse_args()
    if a.selftest:
        d = pl.DataFrame({"rid": [1, 2, 3, 4], "oid": [1, 2, 3, 4],
                          "p": [0.9, 0.1, 0.3, 0.01], "y": [1, 1, 0, 0]})
        t1 = ["", "A | 12 Rue Victor", "B | 12 Rue Victor", "C | 12 Rue Victor", "D | 12 Rue Victor"]
        t2 = ["", "A | 12 Rue Victor", "B | 12 Rue Victor", "D | 12 Rue Victor", "D | 12 Rue Victor"]
        assert training_sample(d, t1, t2, 4, 1).height == 4
        assert "one name word substituted" in relation(t1[3], t2[3])
        log("SELFTEST PASS")
        return
    if not torch.cuda.is_available():
        raise SystemExit("A100 CUDA device required")
    for f in (f"{a.data}/band_loco.parquet", f"{a.bundle}/records_train.parquet"):
        if not Path(f).exists():
            raise SystemExit(f"Missing required input: {f}")
    a.lora = False
    a.no_ckpt = True
    exp_tag = f"{a.tag}_n{a.train_limit}"
    Path(a.out).mkdir(parents=True, exist_ok=True)
    band = pl.read_parquet(f"{a.data}/band_loco.parquet")
    tr_texts = texts(a.bundle, "train")
    start = time.time()
    if a.mode == "loco":
        parts = []
        for i, country in enumerate(("India", "US")):
            target = band.filter((pl.col("country") == country) & (pl.col("rid").hash(seed=21) % 1000 < 100))
            source = band.filter(pl.col("country") != country)
            sample = training_sample(source, *tr_texts, a.train_limit, a.seed + i)
            part = model_scores(a, f"{exp_tag}_no{country}", sample, target, tr_texts, tr_texts)
            part.write_parquet(f"{a.out}/cehard_{exp_tag}_loco_{country}.parquet")
            parts.append(part)
        out = Path(a.out) / f"cehard_{exp_tag}_loco.parquet"
        pl.concat(parts).write_parquet(out)
    else:
        target = pl.read_parquet(f"{a.data}/band_france.parquet").filter(pl.col("p") >= 0.2)
        sample = pl.concat([
            training_sample(band.filter(pl.col("country") == country), *tr_texts,
                            a.train_limit // 2, a.seed + i)
            for i, country in enumerate(("India", "US"))
        ])
        te_texts = texts(a.bundle, "test")
        part = model_scores(a, f"{exp_tag}_france", sample, target, tr_texts, te_texts)
        out = Path(a.out) / f"cehard_{exp_tag}_france_test.parquet"
        part.write_parquet(out)
    meta = {"mode": a.mode, "model": a.model, "train_limit": a.train_limit,
            "output": str(out), "rows": pl.read_parquet(out).height,
            "elapsed_minutes": round((time.time() - start) / 60, 1)}
    (Path(a.out) / f"cehard_{exp_tag}_{a.mode}.json").write_text(json.dumps(meta, indent=2))
    log("DONE", json.dumps(meta))


if __name__ == "__main__":
    main()
