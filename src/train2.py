"""v2 trainer: XGBoost on the local GPU, same honest protocol as train.py.

2-fold by S1 entity (hash of rid). Stage-1 OOF probabilities -> stage-2 context features -> stage-2 OOF.
Assignment: each S2/S3 record goes to its best S1; then either a global threshold or the per-entity
expected-F0.5 rule. All tuning happens on one fold and is reported on the other.
"""
import gc
import json
import os
import sys
import time

import numpy as np
import polars as pl
import xgboost as xgb

import decide
import evaluate as ev
from train import add_label_fold, stage2_features

D = "C:/ML/Challenges/amazon/work/data/"
M = "C:/ML/Challenges/amazon/work/models2/"
NON_FEAT = {"rid", "oid", "y", "fold", "country"}
XP = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda",
          grow_policy="lossguide", max_leaves=255, max_depth=0, learning_rate=0.08, subsample=0.8,
          colsample_bytree=0.8, min_child_weight=20, reg_lambda=1.0, max_bin=128)
STAGE2_EXTRA = ["p1", "p_rank2", "p_max2", "p_rank1", "p_max1", "p_sum1", "p_cnt1_5", "p_rank1src",
                "p_max1src", "p_second2", "p_gap2", "p_gap1", "p_margin2"]


class Batches(xgb.DataIter):
    def __init__(self, df, feats, label=None, step=2_000_000):
        self.df, self.feats, self.label, self.step, self.i = df, feats, label, step, 0
        super().__init__()

    def next(self, input_data):
        if self.i >= self.df.height:
            return 0
        part = self.df.slice(self.i, self.step)
        kw = {"data": part.select(pl.col(self.feats).cast(pl.Float32)).to_numpy()}
        if self.label:
            kw["label"] = part[self.label].to_numpy()
        input_data(**kw)
        self.i += self.step
        return 1

    def reset(self):
        self.i = 0


def predict(bst, df, feats, step=2_000_000):
    out = []
    for i in range(0, df.height, step):
        X = df.slice(i, step).select(pl.col(feats).cast(pl.Float32)).to_numpy()
        out.append(bst.inplace_predict(X))
    return np.concatenate(out).astype(np.float32)


def oof(df, feats, params, rounds, tag):
    pred = np.zeros(df.height, dtype=np.float32)
    boosters = []
    for f in (0, 1):
        t = time.time()
        tr = df.filter(pl.col("fold") == f)
        dm = xgb.QuantileDMatrix(Batches(tr, feats, "y"), max_bin=params["max_bin"])
        n = tr.height
        del tr
        gc.collect()
        bst = xgb.train(params, dm, rounds)
        del dm
        gc.collect()
        mask = (df["fold"] != f).to_numpy()
        pred[mask] = predict(bst, df.filter(pl.Series(mask)), feats)
        bst.save_model(f"{M}{tag}_fold{f}.json")
        boosters.append(bst)
        print(f"  {tag} fold {f}: trained on {n:,} rows ({time.time() - t:.0f}s)", flush=True)
    return pred, boosters


def report(df, gt, s1_all, pcol):
    """Honest: tune on one fold, report on the other. Global threshold and expected-F rule."""
    fold_of = s1_all.select("rid", fold=(pl.col("rid").hash(seed=11) % 2).cast(pl.UInt8))
    folds = [fold_of.filter(pl.col("fold") == f)["rid"] for f in (0, 1)]

    def score(pred, f):
        r = folds[f]
        return ev.macro_f05(pred.filter(pl.col("rid").is_in(r.implode())),
                            gt.filter(pl.col("rid").is_in(r.implode())), r)
    best = df.filter(pl.col(pcol).rank("ordinal", descending=True).over("oid") == 1).select("rid", "oid", pcol)
    res = {}
    for t in [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
        p = best.filter(pl.col(pcol) >= t).select("rid", "oid")
        res[("t", t)] = (score(p, 0), score(p, 1))
    oo = df.select("rid", "oid", pcol).rename({pcol: "p2"})
    for mu, c in [(0.0, 1.0), (0.2, 1.0), (0.0, 1.5)]:
        p = decide.ef_predict(oo, "p2", mu, c, 0.05)
        res[("ef", mu, c)] = (score(p, 0), score(p, 1))
    honest = []
    for f in (0, 1):
        b = max(res, key=lambda k: res[k][f])
        honest.append(res[b][1 - f])
        print(f"  [{pcol}] tuned on fold {f}: {b} -> held-out fold {1 - f}: {res[b][1 - f]:.5f}")
    tbest = max((k for k in res if k[0] == "t"), key=lambda k: sum(res[k]))
    print(f"  [{pcol}] best global threshold {tbest}: " + " ".join(f"{v:.5f}" for v in res[tbest]))
    print(f"  [{pcol}] HONEST macro F0.5 = {np.mean(honest):.5f}", flush=True)
    return float(np.mean(honest)), max(res, key=lambda k: sum(res[k]))


def per_country(df, gt, s1_all, pcol, rule):
    """Per-country score of the chosen rule (in-sample rule choice; for diagnosis only)."""
    oo = df.select("rid", "oid", pcol).rename({pcol: "p2"})
    if rule[0] == "t":
        pred = oo.filter((pl.col("p2").rank("ordinal", descending=True).over("oid") == 1) & (pl.col("p2") >= rule[1]))
    else:
        pred = decide.ef_predict(oo, "p2", rule[1], rule[2], 0.05)
    for c in sorted(s1_all["country"].unique().to_list()):
        r = s1_all.filter(pl.col("country") == c)["rid"]
        f = ev.macro_f05(pred.select("rid", "oid").filter(pl.col("rid").is_in(r.implode())),
                         gt.filter(pl.col("rid").is_in(r.implode())), r)
        print(f"  [{pcol}] {c}: {f:.5f}")


def main(tag="_v2", countries=("India", "US"), rounds1=800, rounds2=400):
    t = time.time()
    os.makedirs(M, exist_ok=True)
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").filter(pl.col("country").is_in(list(countries)))
    s1_all = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(
        pl.col("country").is_in(list(countries)))
    df = pl.concat([pl.read_parquet(f"{D}train_feat{tag}_{c}.parquet").with_columns(country=pl.lit(c))
                    for c in countries])
    df = add_label_fold(df, gt)
    feats1 = [c for c in df.columns if c not in NON_FEAT]
    print(f"pairs {df.height:,}, pos {df['y'].sum():,} / GT {gt.height:,} (ceiling {df['y'].sum() / gt.height:.4f}); "
          f"{len(feats1)} feats ({time.time() - t:.0f}s)", flush=True)
    gts = gt.select("rid", "oid")
    df = df.with_columns(p1=pl.Series(oof(df, feats1, XP, rounds1, "s1")[0]))
    s1_score, _ = report(df, gts, s1_all, "p1")
    df = stage2_features(df)
    feats2 = feats1 + STAGE2_EXTRA
    df = df.with_columns(p2=pl.Series(oof(df, feats2, dict(XP, max_leaves=63, learning_rate=0.05), rounds2, "s2")[0]))
    s2_score, rule = report(df, gts, s1_all, "p2")
    per_country(df, gts, s1_all, "p2", rule)
    df.select("rid", "oid", "y", "fold", "country", "p1", "p2").write_parquet(f"{D}train_oof{tag}.parquet")
    json.dump(dict(feats1=feats1, feats2=feats2, stage1=s1_score, stage2=s2_score, rule=list(rule), tag=tag),
              open(f"{M}meta{tag}.json", "w"), indent=1)
    print(f"done ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "_v2")
