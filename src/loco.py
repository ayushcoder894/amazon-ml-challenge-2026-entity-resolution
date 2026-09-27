"""Leave-one-country-out: how much does a model lose on a country it never saw (a proxy for France),
and does self-training on confident pseudo-labels of the unseen country recover it?

  1. train stage-1 on SOURCE country only -> predict TARGET country -> score (true labels, never used for training)
  2. pseudo-label TARGET: positives = per-record argmax with p >= hi, negatives = p <= lo
  3. retrain on SOURCE + pseudo-labelled TARGET -> predict TARGET again -> score
Compare with the in-country OOF stage-1 score of TARGET from train3.
"""
import json
import sys
import time

import cupy as cp
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import xgboost as xgb

import evaluate as ev
from train3 import BATCH, D, NON_FEAT, XP

# features whose scale depends on the country's vocabulary / corpus size
ABS = ["score", "sn", "sa", "sx", "sz", "gap1", "gap2", "cis_gap2"] + [
    f"{c}_{s}" for c in ("core", "nums", "words") for s in ("isum", "imax", "miss1", "miss2")]
QN = {}  # country -> {feature: quantile grid}, fitted per country without labels


def fit_qnorm(tag, feats, n=400_000):
    """Per-country quantile grids of the corpus-dependent features (unsupervised)."""
    for f in (0, 1):
        for b in pq.ParquetFile(f"{D}train_feat{tag}_fold{f}.parquet").iter_batches(
                batch_size=BATCH, columns=[c for c in feats if c in ABS] + ["country"]):
            b = pl.from_arrow(b)
            for c in b["country"].unique().to_list():
                x = b.filter(pl.col("country") == c)
                QN.setdefault(c, {k: [] for k in ABS if k in feats})
                for k in QN[c]:
                    QN[c][k].append(x[k].sample(min(x.height, n // 10), seed=0).to_numpy())
    for c in QN:
        for k in QN[c]:
            QN[c][k] = np.quantile(np.concatenate(QN[c][k]).astype(np.float64), np.linspace(0, 1, 1001))


def transform(b, feats):
    if not QN:
        return b.select(pl.col(feats).cast(pl.Float32)).to_numpy()
    parts = []
    for c in b["country"].unique(maintain_order=True).to_list():
        x = b.filter(pl.col("country") == c)
        cols = []
        for k in feats:
            v = x[k].cast(pl.Float64).to_numpy()
            if k in QN.get(c, {}):
                g = QN[c][k]
                v = np.searchsorted(g, v, side="right") / len(g)
            cols.append(v.astype(np.float32))
        parts.append(np.column_stack(cols))
    return np.vstack(parts)


class FilterIter(xgb.DataIter):
    """Batches from both fold files, filtered to some countries; optional pseudo-label override for target rows."""

    def __init__(self, tag, feats, countries, pseudo=None):
        self.tag, self.feats, self.countries, self.pseudo = tag, feats, countries, pseudo
        super().__init__()
        self.reset()

    def _gen(self):
        for f in (0, 1):
            for b in pq.ParquetFile(f"{D}train_feat{self.tag}_fold{f}.parquet").iter_batches(
                    batch_size=BATCH, columns=self.feats + ["rid", "oid", "y", "country"]):
                b = pl.from_arrow(b).filter(pl.col("country").is_in(self.countries))
                if self.pseudo is not None:
                    # target-country rows keep only pseudo-labelled pairs, with the pseudo label
                    src = b.filter(pl.col("country") != self.pseudo["country"])
                    tgt = (b.filter(pl.col("country") == self.pseudo["country"]).drop("y")
                           .join(self.pseudo["labels"], on=["rid", "oid"], how="inner"))
                    b = pl.concat([src, tgt.select(src.columns)])
                if b.height:
                    yield b

    def reset(self):
        self.gen = self._gen()

    def next(self, input_data):
        b = next(self.gen, None)
        if b is None:
            return 0
        b = b.sort("country", maintain_order=True)
        input_data(data=transform(b, self.feats), label=b["y"].to_numpy())
        return 1


def predict_country(bst, tag, feats, country):
    keys, preds = [], []
    for f in (0, 1):
        for b in pq.ParquetFile(f"{D}train_feat{tag}_fold{f}.parquet").iter_batches(
                batch_size=BATCH, columns=feats + ["rid", "oid", "country"]):
            b = pl.from_arrow(b).filter(pl.col("country") == country)
            if b.height:
                X = cp.asarray(transform(b, feats))
                preds.append(cp.asnumpy(bst.inplace_predict(X)))
                keys.append(b.select("rid", "oid"))
    return pl.concat(keys).with_columns(p=pl.Series(np.concatenate(preds).astype(np.float32)))


def score(pred, gt, rids, t):
    best = pred.filter((pl.col("p").rank("ordinal", descending=True).over("oid") == 1) & (pl.col("p") >= t))
    return ev.macro_f05(best.select("rid", "oid"), gt, rids)


def main(tag="_v3", source="US", target="India", mode="all", rounds=600, hi=0.95, lo=0.05, pseudo_rounds=2):
    t0 = time.time()
    cols = pq.read_schema(f"{D}train_feat{tag}_fold0.parquet").names
    feats = [c for c in cols if c not in NON_FEAT]
    if mode == "drop":
        feats = [c for c in feats if c not in ABS]
    elif mode == "qnorm":
        fit_qnorm(tag, feats)
    print(f"mode={mode}: {len(feats)} features", flush=True)
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").filter(pl.col("country") == target).select("rid", "oid")
    rids = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(
        pl.col("country") == target)["rid"]
    res = {}
    bst = xgb.train(XP, xgb.QuantileDMatrix(FilterIter(tag, feats, [source]), max_bin=XP["max_bin"]), rounds)
    pred = predict_country(bst, tag, feats, target)
    res["source_only"] = {t: score(pred, gt, rids, t) for t in (0.5, 0.65, 0.8)}
    print(f"{source}->{target} source-only: {res['source_only']} ({time.time() - t0:.0f}s)", flush=True)
    for it in range(pseudo_rounds):
        win = pred.filter(pl.col("p").rank("ordinal", descending=True).over("oid") == 1)
        pos = win.filter(pl.col("p") >= hi).select("rid", "oid", y=pl.lit(1, pl.UInt8))
        neg = pred.filter(pl.col("p") <= lo).select("rid", "oid", y=pl.lit(0, pl.UInt8))
        labels = pl.concat([pos, neg])
        prec = pos.join(gt, on=["rid", "oid"], how="semi").height / max(1, pos.height)
        print(f"  round {it + 1}: pseudo pos {pos.height:,} (true precision {prec:.4f}), neg {neg.height:,}", flush=True)
        it_ = FilterIter(tag, feats, [source, target], pseudo={"country": target, "labels": labels})
        bst = xgb.train(XP, xgb.QuantileDMatrix(it_, max_bin=XP["max_bin"]), rounds)
        pred = predict_country(bst, tag, feats, target)
        res[f"pseudo_{it + 1}"] = {t: score(pred, gt, rids, t) for t in (0.5, 0.65, 0.8)}
        print(f"{source}->{target} pseudo round {it + 1}: {res[f'pseudo_{it + 1}']} ({time.time() - t0:.0f}s)", flush=True)
    json.dump(res, open(f"{D}loco_{source}_{target}{tag}_{mode}.json", "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2], a[3] if len(a) > 3 else "all", pseudo_rounds=int(a[4]) if len(a) > 4 else 2)
