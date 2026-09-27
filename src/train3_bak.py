"""Streaming trainer (memory-light): per-fold feature files on disk -> XGBoost (GPU) via batched DataIter.

Honest protocol, identical to train.py / train2.py:
  fold = hash(rid, seed=11) % 2 by S1 entity; model trained on fold f predicts fold 1-f (OOF for everything);
  stage-2 context features from OOF stage-1 probabilities; tuning on one fold, report on the other.
Only a key table (rid, oid, y, src, p1, p2, stage-2 extras) is held in memory.
"""
import gc
import json
import os
import sys
import time

import cupy as cp
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import xgboost as xgb

import decide
import evaluate as ev
from train import stage2_features

D = "C:/ML/Challenges/amazon/work/data/"
M = "C:/ML/Challenges/amazon/work/models3/"
NON_FEAT = {"rid", "oid", "y", "fold", "country"}
XP = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda",
          grow_policy="lossguide", max_leaves=255, max_depth=0, learning_rate=0.08, subsample=0.8,
          colsample_bytree=0.8, min_child_weight=20, reg_lambda=1.0, max_bin=128)
XP2 = dict(XP, max_leaves=63, learning_rate=0.05)
EXTRA = ["p1", "p_rank2", "p_max2", "p_rank1", "p_max1", "p_sum1", "p_cnt1_5", "p_rank1src",
         "p_max1src", "p_second2", "p_gap2", "p_gap1", "p_margin2"]
BATCH = 2_000_000


def fold_expr():
    return (pl.col("rid").hash(seed=11) % 2).cast(pl.UInt8)


def split_folds(tag, countries, gt):
    """Write data/train_feat{tag}_fold{f}.parquet (features + y), streaming."""
    lf = pl.concat([pl.scan_parquet(f"{D}train_feat{tag}_{c}.parquet").with_columns(country=pl.lit(c))
                    for c in countries], how="diagonal")
    lf = (lf.join(gt.lazy().select("rid", "oid", y=pl.lit(1, pl.UInt8)), on=["rid", "oid"], how="left")
          .with_columns(pl.col("y").fill_null(0), fold=fold_expr()))
    for f in (0, 1):
        lf.filter(pl.col("fold") == f).sink_parquet(f"{D}train_feat{tag}_fold{f}.parquet", row_group_size=500_000)


def batches(path, cols):
    for b in pq.ParquetFile(path).iter_batches(batch_size=BATCH, columns=cols):
        yield pl.from_arrow(b)


class Iter(xgb.DataIter):
    def __init__(self, path, feats, extra=None):
        self.path, self.feats, self.extra = path, feats, extra
        super().__init__()
        self.reset()

    def reset(self):
        self.gen, self.off = batches(self.path, self.feats + ["y"]), 0

    def next(self, input_data):
        b = next(self.gen, None)
        if b is None:
            return 0
        X = b.select(pl.col(self.feats).cast(pl.Float32)).to_numpy()
        if self.extra is not None:
            X = np.hstack([X, self.extra[self.off:self.off + b.height]])
        self.off += b.height
        input_data(data=X, label=b["y"].to_numpy())
        return 1


def predict_file(bst, path, feats, extra=None):
    out, off = [], 0
    for b in batches(path, feats):
        X = b.select(pl.col(feats).cast(pl.Float32)).to_numpy()
        if extra is not None:
            X = np.hstack([X, extra[off:off + b.height]])
        off += b.height
        out.append(cp.asnumpy(bst.inplace_predict(cp.asarray(X))))
    cp.get_default_memory_pool().free_all_blocks()  # give GPU memory back before the next training
    return np.concatenate(out).astype(np.float32)


def stage(tag, feats, params, rounds, name, extras=None, mname=None):
    """Train on each fold file, predict the other. Returns OOF preds per fold file (in file row order)."""
    preds = {}
    for f in (0, 1):
        t = time.time()
        ex_tr = extras[f] if extras else None
        mpath = f"{M}{name}{mname or tag}_fold{f}.json"
        if os.path.exists(mpath) and os.environ.get("REUSE_MODELS") == "1":
            bst = xgb.Booster()
            bst.load_model(mpath)
            bst.set_param({"device": "cuda"})
        else:
            dm = xgb.QuantileDMatrix(Iter(f"{D}train_feat{tag}_fold{f}.parquet", feats, ex_tr), max_bin=params["max_bin"])
            bst = xgb.train(params, dm, rounds)
            del dm
            gc.collect()
        g = 1 - f
        preds[g] = predict_file(bst, f"{D}train_feat{tag}_fold{g}.parquet", feats, extras[g] if extras else None)
        bst.save_model(mpath)
        del bst
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        print(f"  {name} fold {f} -> fold {g} ({time.time() - t:.0f}s)", flush=True)
    return preds


def main(tag="_v3", countries=("India", "US"), rounds1=800, rounds2=400, resplit=True, drop=(), mname=None):
    mname = mname or tag
    t = time.time()
    os.makedirs(M, exist_ok=True)
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").filter(pl.col("country").is_in(list(countries)))
    s1_all = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(
        pl.col("country").is_in(list(countries)))
    if resplit or not os.path.exists(f"{D}train_feat{tag}_fold1.parquet"):
        split_folds(tag, countries, gt)
    cols = pq.read_schema(f"{D}train_feat{tag}_fold0.parquet").names
    feats1 = [c for c in cols if c not in NON_FEAT and c not in set(drop)]
    keys = {f: pl.read_parquet(f"{D}train_feat{tag}_fold{f}.parquet", columns=["rid", "oid", "y", "src", "country"])
            for f in (0, 1)}
    npos = sum(k["y"].sum() for k in keys.values())
    print(f"pairs {sum(k.height for k in keys.values()):,}, pos {npos:,} / GT {gt.height:,} "
          f"(ceiling {npos / gt.height:.4f}); {len(feats1)} feats ({time.time() - t:.0f}s)", flush=True)

    p1 = stage(tag, feats1, XP, rounds1, "s1", mname=mname)
    K = pl.concat([keys[f].with_columns(fold=pl.lit(f, pl.UInt8), row=pl.int_range(keys[f].height, dtype=pl.UInt32),
                                        p1=pl.Series(p1[f])) for f in (0, 1)])
    gts = gt.select("rid", "oid")
    from train2 import report
    s1_score, _ = report(K, gts, s1_all, "p1")

    K = stage2_features(K)
    extras = {f: K.filter(pl.col("fold") == f).sort("row").select(pl.col(EXTRA).cast(pl.Float32)).to_numpy()
              for f in (0, 1)}
    p2 = stage(tag, feats1, XP2, rounds2, "s2", extras, mname=mname)
    K = K.join(pl.concat([pl.DataFrame({"fold": pl.Series([f] * len(p2[f]), dtype=pl.UInt8),
                                        "row": np.arange(len(p2[f]), dtype=np.uint32), "p2": p2[f]}) for f in (0, 1)]),
               on=["fold", "row"])
    s2_score, rule = report(K, gts, s1_all, "p2")
    from train2 import per_country
    per_country(K, gts, s1_all, "p2", rule)
    K.select("rid", "oid", "y", "fold", "country", "p1", "p2").write_parquet(f"{D}train_oof{mname}.parquet")
    json.dump(dict(feats1=feats1, extra=EXTRA, stage1=s1_score, stage2=s2_score, rule=list(rule), tag=tag,
                   rounds1=rounds1, rounds2=rounds2, mname=mname), open(f"{M}meta{mname}.json", "w"), indent=1)
    print(f"done ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "drop":
        from loco import ABS
        main(sys.argv[1], resplit=False, drop=ABS, mname=sys.argv[1] + "drop")
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "_v3")
