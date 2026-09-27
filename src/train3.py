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
    """Batches of a fold file; optional row mask (cascade: only pairs passing the stage-1 filter) and extra columns
    (aligned with the kept rows, in file order)."""

    def __init__(self, path, feats, extra=None, mask=None):
        self.path, self.feats, self.extra, self.mask = path, feats, extra, mask
        super().__init__()
        self.reset()

    def reset(self):
        self.gen, self.off, self.eoff = batches(self.path, self.feats + ["y"]), 0, 0

    def next(self, input_data):
        while True:
            b = next(self.gen, None)
            if b is None:
                return 0
            X, y = b.select(pl.col(self.feats).cast(pl.Float32)).to_numpy(), b["y"].to_numpy()
            if self.mask is not None:
                m = self.mask[self.off:self.off + b.height]
                X, y = X[m], y[m]
            self.off += b.height
            if len(y) == 0:
                continue
            if self.extra is not None:
                X = np.hstack([X, self.extra[self.eoff:self.eoff + len(y)]])
                self.eoff += len(y)
            input_data(data=X, label=y)
            return 1


def predict_file(bst, path, feats, extra=None, mask=None):
    out, off, eoff = [], 0, 0
    for b in batches(path, feats):
        X = b.select(pl.col(feats).cast(pl.Float32)).to_numpy()
        if mask is not None:
            X = X[mask[off:off + b.height]]
        off += b.height
        if extra is not None:
            X = np.hstack([X, extra[eoff:eoff + len(X)]])
            eoff += len(X)
        if len(X):
            out.append(cp.asnumpy(bst.inplace_predict(cp.asarray(X))))
    cp.get_default_memory_pool().free_all_blocks()  # give GPU memory back before the next training
    return np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)


def stage(tag, feats, params, rounds, name, extras=None, mname=None, masks=None):
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
            dm = xgb.QuantileDMatrix(Iter(f"{D}train_feat{tag}_fold{f}.parquet", feats, ex_tr, masks[f] if masks else None),
                                     max_bin=params["max_bin"])
            bst = xgb.train(params, dm, rounds)
            del dm
            gc.collect()
        g = 1 - f
        preds[g] = predict_file(bst, f"{D}train_feat{tag}_fold{g}.parquet", feats, extras[g] if extras else None,
                                masks[g] if masks else None)
        bst.save_model(mpath)
        del bst
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        print(f"  {name} fold {f} -> fold {g} ({time.time() - t:.0f}s)", flush=True)
    return preds


def sibling_step(K, split, mname):
    """Sibling features computed in a separate process so its memory is fully released afterwards."""
    import subprocess
    tmp_in, tmp_out = f"{D}sib_in_{split}{mname}.parquet", f"{D}sib_{split}{mname}.parquet"
    cache = os.environ.get("SIB_CACHE")
    if cache and os.path.exists(f"{D}sib_{split}{cache}.parquet"):
        return pl.read_parquet(f"{D}sib_{split}{cache}.parquet")
    K.select("rid", "oid", "p1", "country").write_parquet(tmp_in)
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "siblings.py"),
                    split, tmp_in, tmp_out], check=True)
    os.remove(tmp_in)
    return pl.read_parquet(tmp_out)


def orphan_step(K, split, mname):
    """Orphan-cluster features (orphan.py) in a separate process, like the sibling features."""
    import subprocess
    tmp_in, tmp_out = f"{D}orph_in_{split}{mname}.parquet", f"{D}orph_{split}{mname}.parquet"
    K.select("rid", "oid", "p1", "country").write_parquet(tmp_in)
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "orphan.py"),
                    split, tmp_in, tmp_out], check=True)
    os.remove(tmp_in)
    return pl.read_parquet(tmp_out)


def main(tag="_v3", countries=("India", "US"), rounds1=800, rounds2=400, resplit=True, drop=(), mname=None):
    mname = mname or tag
    t = time.time()
    os.makedirs(M, exist_ok=True)
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").filter(pl.col("country").is_in(list(countries)))
    s1_all = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(
        pl.col("country").is_in(list(countries)))
    if os.environ.get("KEEP"):  # test-like condition: only these S1 entities exist (see make_drop_folds.py)
        kr = pl.read_parquet(f"{D}keep{os.environ['KEEP']}.parquet")
        gt, s1_all = gt.join(kr, on="rid", how="semi"), s1_all.join(kr, on="rid", how="semi")
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
    prune = float(os.environ.get("PRUNE", "0"))
    use_sib = os.environ.get("SIBLINGS") == "1"
    extra_cols = list(EXTRA)
    masks = None
    if prune > 0:  # cascade: the final matcher only sees pairs passing the stage-1 filter
        masks = {f: (K.filter(pl.col("fold") == f).sort("row")["p1"] >= prune).to_numpy() for f in (0, 1)}
        K = K.filter(pl.col("p1") >= prune)
        del keys
        gc.collect()
        print(f"  cascade filter p1>={prune}: {K.height:,} pairs kept, {K['y'].sum():,} positives "
              f"({K.height / s1_all.height:.2f} per S1)", flush=True)
    # (pairs below the cascade threshold are never predicted, so the report on the filtered set is identical)
    s1_score, _ = report(K, gts, s1_all, "p1")
    K = stage2_features(K)
    if use_sib:
        from siblings import SIB_COLS, sibling_features
        sib = sibling_step(K, "train", mname)
        K = K.join(sib, on=["rid", "oid"], how="left").with_columns(pl.col(SIB_COLS).fill_null(-1))
        extra_cols += SIB_COLS
        print(f"  sibling features added ({time.time() - t:.0f}s)", flush=True)
    use_orph = os.environ.get("ORPHAN") == "1"
    if use_orph:
        from orphan import ORPH_COLS
        K = K.join(orphan_step(K, "train", mname), on=["rid", "oid"], how="left").with_columns(
            pl.col(ORPH_COLS).fill_null(-1))
        extra_cols += ORPH_COLS
        print(f"  orphan-cluster features added ({time.time() - t:.0f}s)", flush=True)
    extras = {f: K.filter(pl.col("fold") == f).sort("row").select(pl.col(extra_cols).cast(pl.Float32)).to_numpy()
              for f in (0, 1)}
    xp2 = dict(XP2, max_leaves=int(os.environ.get("S2_LEAVES", XP2["max_leaves"])),
               learning_rate=float(os.environ.get("S2_LR", XP2["learning_rate"])))
    rounds2 = int(os.environ.get("S2_ROUNDS", rounds2))
    p2 = stage(tag, feats1, xp2, rounds2, "s2", extras, mname=mname, masks=masks)
    K = K.join(pl.concat([K.filter(pl.col("fold") == f).sort("row").select("fold", "row").with_columns(p2=pl.Series(p2[f]))
                          for f in (0, 1)]), on=["fold", "row"])
    if os.environ.get("ENS_LGB") == "1":
        # a differently-built second stage-2 model (LightGBM, CPU) on the same inputs; averaged with XGBoost
        import lightgbm as lgb
        pl_ = {}
        LP = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=127, num_threads=18, verbose=-1)
        def load_x(f):
            xs, ys = [], []
            it = Iter(f"{D}train_feat{tag}_fold{f}.parquet", feats1, extras[f], masks[f] if masks else None)
            it.reset()
            while it.next(lambda data, label: (xs.append(data), ys.append(label))):
                pass
            return np.vstack(xs), np.concatenate(ys)
        for f in (0, 1):
            t0 = time.time()
            X, y = load_x(f)
            mpath = f"{M}s2lgb{mname}_fold{f}.txt"
            if os.path.exists(mpath) and os.environ.get("REUSE_MODELS") == "1":
                b = lgb.Booster(model_file=mpath)
            else:
                b = lgb.train(LP, lgb.Dataset(X, y, free_raw_data=True), int(os.environ.get("LGB_ROUNDS", 600)))
                b.save_model(mpath)
            del X, y
            Xg, _ = load_x(1 - f)
            pl_[1 - f] = b.predict(Xg, num_threads=18).astype(np.float32)
            del Xg
            gc.collect()
            print(f"  s2lgb fold {f} -> fold {1 - f} ({time.time() - t0:.0f}s)", flush=True)
        K = K.join(pl.concat([K.filter(pl.col("fold") == f).sort("row").select("fold", "row").with_columns(
            p2l=pl.Series(pl_[f])) for f in (0, 1)]), on=["fold", "row"])
        report(K, gts, s1_all, "p2l")
        K = K.with_columns(p2x=pl.col("p2"), p2=(pl.col("p2") + pl.col("p2l")) / 2)
    s2_score, rule = report(K, gts, s1_all, "p2")
    if os.environ.get("STAGE3") == "1":
        # stage 3: the same context features, recomputed from stage-2 probabilities
        ctx = [c for c in EXTRA if c != "p1"]
        c3 = [f"q_{c}" for c in EXTRA]
        K3 = stage2_features(K.select("fold", "row", "rid", "oid", "src", pl.col("p2").alias("p1")))
        K3 = K3.rename({c: f"q_{c}" for c in EXTRA})
        K = K.join(K3.select("fold", "row", *c3), on=["fold", "row"])
        ex3 = {f: K.filter(pl.col("fold") == f).sort("row").select(pl.col(extra_cols + c3).cast(pl.Float32)).to_numpy()
               for f in (0, 1)}
        p3 = stage(tag, feats1, xp2, rounds2, "s3", ex3, mname=mname, masks=masks)
        K = K.join(pl.concat([K.filter(pl.col("fold") == f).sort("row").select("fold", "row").with_columns(
            p3=pl.Series(p3[f])) for f in (0, 1)]), on=["fold", "row"])
        s3_score, rule3 = report(K, gts, s1_all, "p3")
        print(f"  stage 3: {s3_score:.5f} vs stage 2 {s2_score:.5f}", flush=True)
    from train2 import per_country
    per_country(K, gts, s1_all, "p2", rule)
    K.select("rid", "oid", "y", "fold", "country", "p1", "p2").write_parquet(f"{D}train_oof{mname}.parquet")
    json.dump(dict(feats1=feats1, extra=extra_cols, stage1=s1_score, stage2=s2_score, rule=list(rule), tag=tag,
                   rounds1=rounds1, rounds2=rounds2, mname=mname, prune=prune, siblings=use_sib, orphan=use_orph,
                   ens_lgb=os.environ.get("ENS_LGB") == "1"),
              open(f"{M}meta{mname}.json", "w"), indent=1)
    print(f"done ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "split":  # only write the per-fold feature files
        g = pl.read_parquet(f"{D}train_gt_rid.parquet").filter(pl.col("country").is_in(["India", "US"]))
        split_folds(sys.argv[1], ("India", "US"), g)
    elif len(sys.argv) > 2 and sys.argv[2] == "noresplit":  # train3.py <tag> noresplit <model name>
        main(sys.argv[1], resplit=False, mname=sys.argv[3])
    elif len(sys.argv) > 2 and sys.argv[2] == "drop":
        from loco import ABS
        main(sys.argv[1], resplit=False, drop=ABS, mname=sys.argv[1] + "drop")
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "_v3")
