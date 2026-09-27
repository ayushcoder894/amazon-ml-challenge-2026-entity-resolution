"""Shared helpers for the A100 experiments (identical decision rule and metric as the local pipeline)."""
import json
import os
import time

import numpy as np
import polars as pl
import torch

# NVIDIA containers route attention to cuDNN SDPA, whose backward returns NaN with padded batches on some builds
if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
    torch.backends.cuda.enable_cudnn_sdp(False)

DEV = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ef_predict(df, pcol, mu, c, floor=0.05):
    """Per-record argmax (one owner per record), then per S1 the top-k records that maximise expected F0.5;
    predict nothing when the probability of having no match outweighs it (c = empty-set weight).
    Copied from the local pipeline (decide.ef_predict)."""
    x = df.with_columns(win=pl.col(pcol).rank("ordinal", descending=True).over("oid") == 1)
    tot = x.group_by("rid").agg(psum=pl.col(pcol).sum())
    w = (x.filter(pl.col("win") & (pl.col(pcol) >= floor))
         .join(tot, on="rid")
         .sort(["rid", pcol], descending=[False, True])
         .with_columns(k=pl.int_range(1, pl.len() + 1).over("rid"),
                       cs=pl.col(pcol).cum_sum().over("rid"),
                       lq=(1 - pl.col(pcol)).clip(1e-6).log().sum().over("rid")))
    w = w.with_columns(ef=1.25 * pl.col("cs") / (0.25 * (pl.col("psum") + mu) + pl.col("k")))
    w = w.with_columns(best=pl.col("ef").max().over("rid"), e0=c * pl.col("lq").exp())
    kstar = w.filter(pl.col("ef") == pl.col("best")).group_by("rid").agg(kstar=pl.col("k").min())
    w = w.join(kstar, on="rid")
    return w.filter((pl.col("k") <= pl.col("kstar")) & (pl.col("best") > pl.col("e0"))).select("rid", "oid")


def macro_f05(pred, gt, s1_ids):
    """Per-S1 F0.5 averaged over s1_ids (singletons included: empty/empty = 1, any prediction on a singleton = 0).
    Copied from the local pipeline (evaluate.macro_f05)."""
    base = pl.DataFrame({"rid": s1_ids.cast(pl.UInt32)})
    tp = pred.join(gt, on=["rid", "oid"]).group_by("rid").agg(tp=pl.len())
    npred = pred.group_by("rid").agg(np_=pl.len())
    ngt = gt.group_by("rid").agg(ng=pl.len())
    m = base.join(tp, on="rid", how="left").join(npred, on="rid", how="left").join(ngt, on="rid", how="left").fill_null(0)
    p = pl.when(pl.col("np_") > 0).then(pl.col("tp") / pl.col("np_")).otherwise(0.0)
    r = pl.when(pl.col("ng") > 0).then(pl.col("tp") / pl.col("ng")).otherwise(0.0)
    m = m.with_columns(p=p, r=r)
    f = (pl.when((pl.col("ng") == 0) & (pl.col("np_") == 0)).then(1.0)
         .when(pl.col("tp") == 0).then(0.0)
         .otherwise(1.25 * pl.col("p") * pl.col("r") / (0.25 * pl.col("p") + pl.col("r"))))
    return m.select(f.mean()).item()


class Evaluator:
    """Honest evaluation on the harder (test-like) split: only the kept Source-1 entities exist."""

    def __init__(self, bundle):
        self.s1 = (pl.read_parquet(f"{bundle}/folds.parquet").join(pl.read_parquet(f"{bundle}/keep.parquet"), on="rid", how="semi"))
        self.gt = pl.read_parquet(f"{bundle}/train_gt.parquet").join(self.s1.select("rid"), on="rid", how="semi")
        self.rule = json.load(open(f"{bundle}/rule.json"))

    def score(self, cand, pcol, rule=None):
        """cand: (rid, oid, pcol) over all candidate pairs. Returns {country: F0.5, 'all': F0.5}."""
        rule = rule or self.rule
        pred = ef_predict(cand.select("rid", "oid", pl.col(pcol)), pcol, rule["mu"], rule["c"])
        out = {}
        for c in sorted(self.s1["country"].unique().to_list()):
            rids = self.s1.filter(pl.col("country") == c)["rid"]
            g = self.gt.filter(pl.col("country") == c).select("rid", "oid")
            out[c] = round(macro_f05(pred.filter(pl.col("rid").is_in(rids.implode())), g, rids), 5)
        out["all"] = round(macro_f05(pred, self.gt.select("rid", "oid"), self.s1["rid"]), 5)
        return out

    def ceiling(self, cand):
        """Macro F0.5 of a perfect matcher restricted to the candidate set (cand: rid, oid)."""
        pred = cand.select("rid", "oid").join(self.gt.select("rid", "oid"), on=["rid", "oid"], how="semi")
        out = {}
        for c in sorted(self.s1["country"].unique().to_list()):
            rids = self.s1.filter(pl.col("country") == c)["rid"]
            g = self.gt.filter(pl.col("country") == c).select("rid", "oid")
            out[c] = round(macro_f05(pred.filter(pl.col("rid").is_in(rids.implode())), g, rids), 5)
        return out


def auc(y, s):
    """Rank AUC without sklearn."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1)
