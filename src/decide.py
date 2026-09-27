"""Per-entity expected-F0.5 decision rule (vs. a global threshold), evaluated honestly on OOF predictions.

F0.5 per entity = 1.25*tp / (0.25*n_true + n_pred). For each S1 we rank the records it wins (per-record argmax)
by probability and pick the prefix k maximising  E[F_k] ~= 1.25*sum_{i<=k} p_i / (0.25*(sum_all p + mu) + k),
and predict nothing if  prod(1-p_i) * c  beats every k (the empty set scores 1.0 only for true singletons).
mu (expected true matches outside the candidate set) and c are tuned on one fold and reported on the other.
"""
import itertools

import numpy as np
import polars as pl

import evaluate as ev

D = ev.D


def ef_predict(oof, pcol, mu, c, floor):
    x = oof.with_columns(win=pl.col(pcol).rank("ordinal", descending=True).over("oid") == 1)
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


def main():
    oof = pl.read_parquet(f"{D}train_oof.parquet", columns=["rid", "oid", "p2"])
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").select("rid", "oid")
    s1 = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid"]).with_columns(
        fold=(pl.col("rid").hash(seed=11) % 2).cast(pl.UInt8))
    folds = [s1.filter(pl.col("fold") == f)["rid"] for f in (0, 1)]

    def score(pred, f):
        r = folds[f]
        return ev.macro_f05(pred.filter(pl.col("rid").is_in(r.implode())), gt.filter(pl.col("rid").is_in(r.implode())), r)

    # baseline: global threshold on per-record argmax
    best = oof.filter(pl.col("p2").rank("ordinal", descending=True).over("oid") == 1)
    base = {f: score(best.filter(pl.col("p2") >= 0.65).select("rid", "oid"), f) for f in (0, 1)}
    print(f"baseline t=0.65: fold0 {base[0]:.5f} fold1 {base[1]:.5f} mean {np.mean(list(base.values())):.5f}")

    grid = list(itertools.product([0.0, 0.2, 0.4], [0.5, 1.0, 2.0], [0.05, 0.2]))
    res = {}
    for mu, c, fl in grid:
        pred = ef_predict(oof, "p2", mu, c, fl)
        res[(mu, c, fl)] = (score(pred, 0), score(pred, 1))
        print(f"mu={mu} c={c} floor={fl}: fold0 {res[(mu, c, fl)][0]:.5f} fold1 {res[(mu, c, fl)][1]:.5f}", flush=True)
    honest = []
    for f in (0, 1):
        b = max(res, key=lambda k: res[k][f])
        honest.append(res[b][1 - f])
        print(f"tuned on fold {f}: {b} -> held-out fold {1 - f}: {res[b][1 - f]:.5f} (baseline {base[1 - f]:.5f})")
    print(f"HONEST expected-F rule: {np.mean(honest):.5f} vs baseline {np.mean(list(base.values())):.5f}")


if __name__ == "__main__":
    main()
