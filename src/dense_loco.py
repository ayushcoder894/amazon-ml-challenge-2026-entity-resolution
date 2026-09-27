"""Does the dense add-on transfer to a country absent from training (France)? Leave-one-country-out on the harder split.

Base = LOCO stage-1 probabilities (model trained on the other country) decided with the France rule ef(0, c=4) - this
is how France is predicted today. Add = dense new pairs (not in the LOCO candidate set), add-on model trained on the
OTHER country's new pairs, context features recomputed from the LOCO stage-1 probabilities.
Caveat: the bi-encoder itself was trained on both countries (optimistic for the retrieval part).
"""
import gc
import os

import numpy as np
import polars as pl
import xgboost as xgb

import decide
import evaluate as ev
from dense_addon import COUNTRIES, D, DN, XP, context, feat_cols, log

CTX_PREFIX = ("u_", "k_", "d_cos")
RULES = [("ef", 0.0, 4.0), ("ef", 0.0, 1.0)]


def main():
    Kl = pl.read_parquet(f"{D}train_oof_locoh.parquet", columns=["rid", "oid", "country", "p1", "p2"])
    Kl = Kl.join(pl.read_parquet(f"{DN}dense_train_cos.parquet"), on=["rid", "oid"], how="left").with_columns(pf=pl.col("p1"))
    N = pl.concat([pl.read_parquet(f"{DN}train_nfeat_{c}.parquet") for c in COUNTRIES], how="diagonal_relaxed")
    N = N.drop([c for c in N.columns if c.startswith(CTX_PREFIX)]).join(Kl.select("rid", "oid"), on=["rid", "oid"], how="anti")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").select("rid", "oid").with_columns(y=pl.lit(1, pl.Int8))
    N = N.drop("y").join(gt, on=["rid", "oid"], how="left").with_columns(pl.col("y").fill_null(0))
    parts = []
    for c in COUNTRIES:
        Nc = N.filter(pl.col("country") == c)
        parts.append(context(Nc, Kl.filter(pl.col("country") == c)))
    N = pl.concat(parts, how="diagonal_relaxed")
    del parts
    gc.collect()
    cols = [c for c in feat_cols(N) if c != "is_india"]
    N = N.select("rid", "oid", "country", "y", *[pl.col(c).cast(pl.Float32) for c in cols])
    log(f"LOCO new pairs {N.height:,} (positives {N['y'].sum():,}), {len(cols)} features")
    pn = np.zeros(N.height, dtype=np.float32)
    for c in COUNTRIES:
        tr = (N["country"] != c).to_numpy()
        dm = xgb.QuantileDMatrix(N.filter(pl.Series(tr)).select(cols).to_numpy(), label=N.filter(pl.Series(tr))["y"].to_numpy(),
                                 max_bin=XP["max_bin"])
        bst = xgb.train(XP, dm, int(os.environ.get("ROUNDS", 600)))
        pn[~tr] = bst.predict(xgb.DMatrix(N.filter(pl.Series(~tr)).select(cols).to_numpy()))
        log(f"trained on {'US' if c == 'India' else 'India'}, scored {c}")
        del dm, bst
        gc.collect()
    N = N.select("rid", "oid", "country", pn=pl.Series(pn))
    N.write_parquet(f"{DN}train_pn_loco.parquet")
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    s1 = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).join(keep, on="rid", how="semi")
    gtk = gt.join(keep, on="rid", how="semi").select("rid", "oid")

    def score(U, rule):
        pred = decide.ef_predict(U.select("rid", "oid", pl.col("p").alias("p2")), "p2", rule[1], rule[2], 0.05)
        return {c: ev.macro_f05(pred.join(s1.filter(pl.col("country") == c), on="rid", how="semi"),
                                gtk.join(s1.filter(pl.col("country") == c), on="rid", how="semi"),
                                s1.filter(pl.col("country") == c)["rid"]) for c in COUNTRIES}
    base = Kl.select("rid", "oid", p="p1")
    for rule in RULES:
        b = score(base, rule)
        u = score(pl.concat([base, N.select("rid", "oid", p="pn")]), rule)
        log(f"rule {rule}: LOCO stage-1 alone  India {b['India']:.5f}  US {b['US']:.5f}")
        log(f"rule {rule}: + dense add-on      India {u['India']:.5f} ({u['India'] - b['India']:+.5f})  "
            f"US {u['US']:.5f} ({u['US'] - b['US']:+.5f})")


if __name__ == "__main__":
    main()
