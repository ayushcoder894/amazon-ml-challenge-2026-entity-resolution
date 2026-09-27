"""Which decision source transfers best to an unseen country? LOCO out-of-fold probabilities (train on US score India,
and reverse): stage 1 (p1), stage 2 (p2) and blends, per country, with the rule tuned on the other country."""
import sys

import numpy as np
import polars as pl

import decide
import evaluate as ev

D = "C:/ML/Challenges/amazon/work/data/"
RULES = [("ef", mu, c) for mu in (0.0, 0.3) for c in (1.0, 1.5, 2.0)] + [("t", t) for t in (0.5, 0.6, 0.7, 0.8)]


def predict(K, col, rule):
    if rule[0] == "t":
        return K.filter((pl.col(col).rank("ordinal", descending=True).over("oid") == 1) & (pl.col(col) >= rule[1])).select("rid", "oid")
    return decide.ef_predict(K.select("rid", "oid", pl.col(col).alias("p2")), "p2", rule[1], rule[2], 0.05).select("rid", "oid")


def main(tag, keep=None):
    K = pl.read_parquet(f"{D}train_oof{tag}.parquet")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").select("rid", "oid", "country")
    s1 = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"])
    if keep:
        kr = pl.read_parquet(f"{D}keep{keep}.parquet")
        gt, s1 = gt.join(kr, on="rid", how="semi"), s1.join(kr, on="rid", how="semi")
    lg = lambda c: (pl.col(c).clip(1e-6, 1 - 1e-6) / (1 - pl.col(c).clip(1e-6, 1 - 1e-6))).log()
    cols = {"p1": pl.col("p1"), "p2": pl.col("p2")}
    for a in (0.25, 0.5, 0.75):
        cols[f"{a}*p1+{1 - a}*p2"] = a * pl.col("p1") + (1 - a) * pl.col("p2")
    cols["logit mean"] = 1 / (1 + (-(lg("p1") + lg("p2")) / 2).exp())
    K = K.with_columns(**{k: v for k, v in cols.items() if k not in ("p1", "p2")})
    print(f"\n=== LOCO {tag} ({'harder' if keep else 'normal'}): F0.5 per unseen country, rule tuned on the other")
    for col in cols:
        sc = {}
        for c in ("India", "US"):
            r = s1.filter(pl.col("country") == c)["rid"]
            g = gt.filter(pl.col("country") == c).select("rid", "oid")
            Kc = K.filter(pl.col("country") == c)
            sc[c] = {rule: ev.macro_f05(predict(Kc, col, rule), g, r) for rule in RULES}
        hon = {c: sc[c][max(sc[o], key=sc[o].get)] for c, o in (("India", "US"), ("US", "India"))}
        print(f"  {col:<18s} India {hon['India']:.4f}  US {hon['US']:.4f}  mean {np.mean(list(hon.values())):.4f}", flush=True)


if __name__ == "__main__":
    main("_loco")
    main("_locoh", "_v11")
