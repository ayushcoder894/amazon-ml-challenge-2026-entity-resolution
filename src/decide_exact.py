"""Exact(er) expected-F0.5 decision per S1 entity (honest: parameters tuned on one fold, reported on the other).

For each S1 the records it wins are sorted by probability p_1 >= ... >= p_n. Predicting the top-k set S_k gives
F0.5 = 1.25 * tp / (0.25 * n_true + k). With independent Bernoulli labels, tp ~ PoissonBinomial(p_1..p_k), and
n_true = tp + (matches among the other candidates) + m, where m (matches outside the candidate set) is modelled as
Poisson(mu). E[F] is computed by Monte-Carlo (common random numbers, vectorised across entities).
Empty prediction scores 1 only if n_true == 0.
"""
import json
import sys

import numpy as np
import polars as pl

import evaluate as ev

D = "C:/ML/Challenges/amazon/work/data/"


def exact_predict(K, pcol, mu=0.1, temp=1.0, sims=256, maxk=12, seed=0):
    x = K.select("rid", "oid", pcol).with_columns(win=pl.col(pcol).rank("ordinal", descending=True).over("oid") == 1)
    if temp != 1.0:
        lg = (pl.col(pcol).clip(1e-6, 1 - 1e-6) / (1 - pl.col(pcol).clip(1e-6, 1 - 1e-6))).log() / temp
        x = x.with_columns(pl.lit(1.0).truediv(1 + (-lg).exp()).alias(pcol))
    # probability mass of candidates this S1 does not win (they could still be its matches)
    rest = x.filter(~pl.col("win")).group_by("rid").agg(prest=pl.col(pcol).sum())
    w = (x.filter(pl.col("win") & (pl.col(pcol) >= 0.02)).sort(["rid", pcol], descending=[False, True])
         .with_columns(k=pl.int_range(1, pl.len() + 1).over("rid")).filter(pl.col("k") <= maxk))
    rids = w["rid"].unique(maintain_order=True)
    P = np.zeros((len(rids), maxk), dtype=np.float32)
    idx = {r: i for i, r in enumerate(rids.to_list())}
    ri = np.array([idx[r] for r in w["rid"].to_list()])
    P[ri, w["k"].to_numpy() - 1] = w[pcol].to_numpy()
    pr = rids.to_frame().join(rest, on="rid", how="left").fill_null(0.0)["prest"].to_numpy()
    rng = np.random.default_rng(seed)
    best_k = np.zeros(len(rids), dtype=np.int32)
    for s in range(0, len(rids), 20_000):
        Ps, prs = P[s:s + 20_000], pr[s:s + 20_000]
        U = rng.random((sims, 1, maxk), dtype=np.float32)
        Y = (U < Ps[None]).astype(np.float32)                       # sims x n x maxk
        other = rng.poisson(prs[None, :] + mu, size=(sims, len(prs)))  # matches outside the predicted list
        ntrue = Y.sum(2) + other
        tpk = np.cumsum(Y, axis=2)                                  # tp for top-k
        k = np.arange(1, maxk + 1)[None, None, :]
        F = 1.25 * tpk / (0.25 * ntrue[:, :, None] + k)
        EF = F.mean(0)                                              # n x maxk
        EF[Ps == 0] = -1
        E0 = (ntrue == 0).mean(0)
        kb = EF.argmax(1) + 1
        best_k[s:s + 20_000] = np.where(EF.max(1) > E0, kb, 0)
    sel = pl.DataFrame({"rid": rids, "kstar": best_k})
    return w.join(sel, on="rid").filter(pl.col("k") <= pl.col("kstar")).select("rid", "oid")


def main(src="train_oof_ce_v10", pcol="p3"):
    K = pl.read_parquet(f"{D}{src}.parquet")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid"])
    s1 = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid"]).with_columns(
        fold=(pl.col("rid").hash(seed=11) % 2).cast(pl.UInt8))
    folds = [s1.filter(pl.col("fold") == f)["rid"] for f in (0, 1)]

    def score(pred, f):
        r = folds[f]
        return ev.macro_f05(pred.filter(pl.col("rid").is_in(r.implode())), gt.filter(pl.col("rid").is_in(r.implode())), r)
    import decide
    base = decide.ef_predict(K.select("rid", "oid", pcol).rename({pcol: "p2"}), "p2", 0.0, 1.5, 0.05)
    b = [score(base, f) for f in (0, 1)]
    print(f"current EF rule: {b[0]:.5f} {b[1]:.5f} mean {np.mean(b):.5f}", flush=True)
    res = {}
    for mu in (0.0, 0.1, 0.25):
        for temp in (1.0, 1.15):
            pred = exact_predict(K, pcol, mu=mu, temp=temp)
            res[(mu, temp)] = (score(pred, 0), score(pred, 1))
            print(f"mu={mu} temp={temp}: {res[(mu, temp)][0]:.5f} {res[(mu, temp)][1]:.5f}", flush=True)
    honest = [res[max(res, key=lambda k: res[k][f])][1 - f] for f in (0, 1)]
    print(f"HONEST exact rule: {np.mean(honest):.5f} vs current {np.mean(b):.5f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
