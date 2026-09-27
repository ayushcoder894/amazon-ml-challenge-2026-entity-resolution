"""Hypothesis test: group structure among an S1's candidates separates twins (records of an entity missing from S1)
from true matches. For pair (X, o): compare o with X's OTHER candidates o' and look at p3(X, o') of its look-alikes.
Honest: p3 are out-of-fold; the check model is trained on one fold and evaluated on the other.
"""
import numpy as np
import polars as pl
import xgboost as xgb
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from sklearn.metrics import log_loss, roc_auc_score

D = "C:/ML/Challenges/amazon/work/data/"


def group_features(K, o_txt):
    """K: (rid, oid, p3) for all candidates. Returns per (rid, oid) look-alike features."""
    a = K.select("rid", "oid")
    b = K.select("rid", o2="oid", p_o2="p3")
    pp = a.join(b, on="rid").filter(pl.col("oid") != pl.col("o2"))
    pp = (pp.join(o_txt, on="oid").join(o_txt.rename({"oid": "o2", "cs": "cs2", "ac": "ac2"}), on="o2"))
    pp = pp.with_columns(
        ns=pl.Series(cpdist(pp["cs"].to_list(), pp["cs2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)),
        as_=pl.Series(cpdist(pp["ac"].to_list(), pp["ac2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)))
    pp = pp.with_columns(sim=(pl.col("ns") + pl.col("as_")) / 2)
    return pp.group_by("rid", "oid").agg(
        tw_best_sim=pl.col("sim").max(),
        tw_best_p=pl.col("p_o2").sort_by("sim", descending=True).first(),        # p of the closest look-alike
        tw_n90=(pl.col("ns") >= 90).sum().cast(pl.UInt8),
        tw_p90_mean=pl.col("p_o2").filter(pl.col("ns") >= 90).mean(),
        tw_sim_hi=pl.col("sim").filter(pl.col("p_o2") >= 0.9).max(),            # closest to a confident match
        tw_sim_lo=pl.col("sim").filter(pl.col("p_o2") <= 0.1).max(),            # closest to a confident non-match
        tw_nlo_sim=((pl.col("p_o2") <= 0.1) & (pl.col("sim") >= 85)).sum().cast(pl.UInt8),
    )


def main():
    K = pl.read_parquet(f"{D}train_oof_ce_v10.parquet", columns=["rid", "oid", "y", "fold", "country", "p3"])
    o = pl.concat([pl.read_parquet(f"{D}train_s{k}_norm.parquet", columns=["core", "at"]) for k in (2, 3)]).with_row_index("oid")
    o = o.join(K.select("oid").unique(), on="oid", how="semi").select(
        "oid", cs=pl.col("core").list.join(" "), ac=pl.col("at").list.join(" "))
    feats = []
    for c in ["India", "US"]:
        feats.append(group_features(K.filter(pl.col("country") == c).select("rid", "oid", "p3"), o))
    G = pl.concat(feats)
    band = K.filter((pl.col("p3") > 0.02) & (pl.col("p3") < 0.98)).join(G, on=["rid", "oid"], how="left")
    cols = ["p3", "tw_best_sim", "tw_best_p", "tw_n90", "tw_p90_mean", "tw_sim_hi", "tw_sim_lo", "tw_nlo_sim"]
    print(f"band pairs {band.height:,}, match rate {band['y'].mean():.3f}")
    P = dict(objective="binary:logistic", tree_method="hist", device="cpu", nthread=16, max_depth=6, learning_rate=0.1)
    for name, cc in [("p3 only", ["p3"]), ("p3 + group features", cols)]:
        r = []
        for f in (0, 1):
            tr, te = band.filter(pl.col("fold") == f), band.filter(pl.col("fold") != f)
            m = xgb.train(P, xgb.DMatrix(tr.select(cc).to_numpy().astype(np.float32), tr["y"].to_numpy()), 300)
            p = m.predict(xgb.DMatrix(te.select(cc).to_numpy().astype(np.float32)))
            r.append((roc_auc_score(te["y"], p), log_loss(te["y"], np.clip(p, 1e-6, 1 - 1e-6))))
        print(f"{name:22s} AUC {np.mean([x[0] for x in r]):.4f}  logloss {np.mean([x[1] for x in r]):.4f}", flush=True)
    G.write_parquet(f"{D}twin_train_v10.parquet")


if __name__ == "__main__":
    main()
