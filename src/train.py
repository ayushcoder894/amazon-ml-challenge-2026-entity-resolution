"""Honest local evaluation: 2-fold by S1 entity, OOF stage-1 -> stage-2 context model -> assignment -> macro F0.5.

Fold assignment is by S1 entity (hash of rid). Every candidate pair belongs to its S1's fold.
Stage-1 model trained on fold A predicts fold B and vice versa, so all pairs get out-of-fold probabilities.
Stage-2 uses those OOF probabilities in context (competition between S1s for the same S2/S3 record, etc.),
again trained/predicted out-of-fold. Thresholds are tuned on fold A and reported on fold B (and vice versa).
"""
import gc
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import evaluate as ev

D = "C:/ML/Challenges/amazon/work/data/"
M = "C:/ML/Challenges/amazon/work/models/"
NON_FEAT = {"rid", "oid", "y", "fold", "country"}
P1 = dict(objective="binary", learning_rate=0.08, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.8,
          bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, max_bin=127, num_threads=18, verbose=-1)
P2 = dict(P1, num_leaves=63, learning_rate=0.05)


def load_feats(split, countries):
    df = pl.concat([pl.read_parquet(f"{D}{split}_feat_{c}.parquet").with_columns(country=pl.lit(c))
                    for c in countries])
    return df


def add_label_fold(df, gt):
    df = df.join(gt.select("rid", "oid", y=pl.lit(1, pl.UInt8)), on=["rid", "oid"], how="left").with_columns(
        pl.col("y").fill_null(0), fold=(pl.col("rid").hash(seed=11) % 2).cast(pl.UInt8))
    return df


def fit(X, y, params, rounds, Xv=None, yv=None):
    dtr = lgb.Dataset(X, y, free_raw_data=True)
    valid = [lgb.Dataset(Xv, yv, reference=dtr)] if Xv is not None else []
    cb = [lgb.log_evaluation(100)] if valid else []
    return lgb.train(params, dtr, rounds, valid_sets=valid, callbacks=cb)


def to_np(df, feats):
    return df.select(pl.col(feats).cast(pl.Float32)).to_numpy()


def predict_chunked(m, df, feats, step=4_000_000):
    return np.concatenate([m.predict(to_np(df.slice(i, step), feats), num_threads=18)
                           for i in range(0, df.height, step)]).astype(np.float32)


def oof(df, feats, params, rounds, tag, train_frac=0.6):
    """2-fold OOF predictions (fold = S1 hash). Each fold model trains on a train_frac subsample of its S1s."""
    pred = np.zeros(df.height, dtype=np.float32)
    models = []
    sub = (pl.col("rid").hash(seed=23) % 1000) < int(train_frac * 1000)
    for f in (0, 1):
        t = time.time()
        tr = df.filter((pl.col("fold") == f) & sub)
        X, y = to_np(tr, feats), tr["y"].to_numpy()
        del tr
        m = fit(X, y, params, rounds)
        del X
        gc.collect()
        te_mask = (df["fold"] != f).to_numpy()
        pred[te_mask] = predict_chunked(m, df.filter(pl.Series(te_mask)), feats)
        m.save_model(f"{M}{tag}_fold{f}.txt")
        models.append(m)
        print(f"  {tag} fold {f} trained on {len(y):,} rows ({time.time() - t:.0f}s)", flush=True)
    return pred, models


def stage2_features(df, pcol="p1"):
    """Context features built from stage-1 probabilities."""
    p = pl.col(pcol)
    return df.with_columns(
        p_rank2=p.rank("ordinal", descending=True).over("oid").cast(pl.UInt16),     # rank among S1s competing for oid
        p_max2=p.max().over("oid"),
        p_rank1=p.rank("ordinal", descending=True).over("rid").cast(pl.UInt16),     # rank among this S1's candidates
        p_max1=p.max().over("rid"),
        p_sum1=p.sum().over("rid"),
        p_cnt1_5=(p > 0.5).sum().over("rid").cast(pl.UInt16),
        p_rank1src=p.rank("ordinal", descending=True).over("rid", "src").cast(pl.UInt16),
        p_max1src=p.max().over("rid", "src"),
    ).with_columns(
        p_second2=p.filter(pl.col("p_rank2") == 2).first().over("oid").fill_null(0.0),
        p_gap2=pl.col("p_max2") - p,
        p_gap1=pl.col("p_max1") - p,
    ).with_columns(
        # margin over the best competing S1 for this oid
        p_margin2=pl.when(pl.col("p_rank2") == 1).then(p - pl.col("p_second2")).otherwise(p - pl.col("p_max2")),
    )


def decide(df, pcol, t):
    """Assignment: each S2/S3 record goes to its best S1 (over ALL S1s) if prob >= t."""
    return df.filter((pl.col(pcol).rank("ordinal", descending=True).over("oid") == 1) & (pl.col(pcol) >= t)).select(
        "rid", "oid")


def score_fold(best, gt, s1_rids, pcol, t):
    """best: per-oid argmax rows computed globally; restricted to the eval fold's S1s only for scoring."""
    pred = best.filter((pl.col(pcol) >= t) & pl.col("rid").is_in(s1_rids.implode())).select("rid", "oid")
    g = gt.filter(pl.col("rid").is_in(s1_rids.implode()))
    return ev.macro_f05(pred, g, s1_rids)


def tune_and_report(df, gt, s1_all, pcol, grid=None):
    grid = grid or [0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8]
    fold_of = s1_all.select("rid", fold=(pl.col("rid").hash(seed=11) % 2).cast(pl.UInt8))
    best = df.filter(pl.col(pcol).rank("ordinal", descending=True).over("oid") == 1).select("rid", "oid", pcol)
    res = {}
    for f in (0, 1):
        rids = fold_of.filter(pl.col("fold") == f)["rid"]
        res[f] = {t: score_fold(best, gt, rids, pcol, t) for t in grid}
    out = []
    for f in (0, 1):
        tb = max(res[f], key=res[f].get)                 # tuned on fold f
        out.append(res[1 - f][tb])                       # reported on the other fold
        print(f"  [{pcol}] tune on fold {f}: t={tb} (F={res[f][tb]:.5f}) -> held-out fold {1 - f}: F={res[1 - f][tb]:.5f}")
    print(f"  [{pcol}] curve fold0: " + " ".join(f"{t}:{v:.4f}" for t, v in res[0].items()))
    print(f"  [{pcol}] HONEST macro F0.5 = {np.mean(out):.5f}", flush=True)
    return np.mean(out), max(res[0], key=res[0].get)


def main(countries=("India", "US"), rounds1=600, rounds2=300):
    t = time.time()
    import os
    os.makedirs(M, exist_ok=True)
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet")
    s1_all = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(
        pl.col("country").is_in(list(countries)))
    gt = gt.filter(pl.col("country").is_in(list(countries)))
    df = add_label_fold(load_feats("train", countries), gt)
    feats1 = [c for c in df.columns if c not in NON_FEAT]
    print(f"pairs {df.height:,}, pos {df['y'].sum():,} / GT {gt.height:,}; {len(feats1)} feats ({time.time() - t:.0f}s)",
          flush=True)

    df = df.with_columns(p1=pl.Series(oof(df, feats1, P1, rounds1, "s1")[0]))
    s1_score, _ = tune_and_report(df, gt, s1_all, "p1")

    df = stage2_features(df)
    feats2 = feats1 + ["p1", "p_rank2", "p_max2", "p_rank1", "p_max1", "p_sum1", "p_cnt1_5", "p_rank1src",
                       "p_max1src", "p_second2", "p_gap2", "p_gap1", "p_margin2"]
    df = df.with_columns(p2=pl.Series(oof(df, feats2, P2, rounds2, "s2")[0]))
    s2_score, t_best = tune_and_report(df, gt, s1_all, "p2")
    df.select("rid", "oid", "y", "fold", "country", "p1", "p2").write_parquet(f"{D}train_oof.parquet")
    json.dump(dict(feats1=feats1, feats2=feats2, stage1=s1_score, stage2=s2_score, t=t_best),
              open(f"{M}meta.json", "w"), indent=1)
    print(f"done ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
