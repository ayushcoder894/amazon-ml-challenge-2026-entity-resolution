"""Leaderboard probes: India/US identical to v3 (LB 0.965); only France changes, so LB deltas measure France alone.

A: France from the transfer-robust 'drop' stage-1 model (argmax per record, p >= 0.65)
B: France from v3 stage-2 with a lower bar (argmax per record, p2 >= 0.45)  -> more France recall
C: France from the average of A's and v3's probabilities (argmax, >= 0.55)
"""
import json

import cupy as cp
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import xgboost as xgb

import decide

D = "C:/ML/Challenges/amazon/work/data/"
M = "C:/ML/Challenges/amazon/work/models3/"
OUT = "C:/ML/Challenges/amazon/work/output/"


def drop_probs():
    meta_feats = [c for c in pq.read_schema(f"{D}train_feat_v3_fold0.parquet").names
                  if c not in {"rid", "oid", "y", "fold", "country"}]
    from loco import ABS
    feats = [c for c in meta_feats if c not in ABS]
    bs = []
    for f in (0, 1):
        b = xgb.Booster(); b.load_model(f"{M}s1_v3drop_fold{f}.json"); b.set_param({"device": "cuda"}); bs.append(b)
    keys, ps = [], []
    for b in pq.ParquetFile(f"{D}test_feat_v3_France.parquet").iter_batches(batch_size=1_000_000,
                                                                             columns=feats + ["rid", "oid"]):
        b = pl.from_arrow(b)
        X = cp.asarray(b.select(pl.col(feats).cast(pl.Float32)).to_numpy())
        ps.append(np.mean([cp.asnumpy(m.inplace_predict(X)) for m in bs], axis=0))
        keys.append(b.select("rid", "oid"))
    return pl.concat(keys).with_columns(pd=pl.Series(np.concatenate(ps).astype(np.float32)))


def argmax_t(df, col, t):
    return df.filter((pl.col(col).rank("ordinal", descending=True).over("oid") == 1) & (pl.col(col) >= t)).select("rid", "oid")


def write(pred, name, s1, o):
    ids = pred.join(s1.rename({"entity_id": "s1"}), on="rid").join(o.rename({"entity_id": "m"}), on="oid")
    match = ids.group_by("s1").agg(pl.col("m").sort().str.join(","))
    base = s1.select(source1_entity_id="entity_id")
    path = f"{OUT}matching_results_{name}.tsv"
    (base.join(match.rename({"s1": "source1_entity_id", "m": "matched_entity_ids"}), on="source1_entity_id", how="left")
     .fill_null("").write_csv(path, separator="\t", quote_style="never"))
    return path


def main():
    tp = pl.read_parquet(f"{D}test_pred_v3.parquet")
    rule = json.load(open(f"{M}meta_v3.json"))["rule"]
    iu = decide.ef_predict(tp.filter(pl.col("country") != "France").select("rid", "oid", "p2"), "p2",
                           rule[1], rule[2], 0.05)
    fr = tp.filter(pl.col("country") == "France").select("rid", "oid", "p2")
    fr_v3 = decide.ef_predict(fr, "p2", rule[1], rule[2], 0.05)
    dp = drop_probs()
    fr = fr.join(dp, on=["rid", "oid"]).with_columns(pa=(pl.col("p2") + pl.col("pd")) / 2)
    s1 = pl.read_parquet(f"{D}test_s1_norm.parquet", columns=["rid", "entity_id"])
    o = pl.concat([pl.read_parquet(f"{D}test_s{k}_norm.parquet", columns=["entity_id"]) for k in (2, 3)]
                  ).with_row_index("oid")
    variants = {"A_france_drop": argmax_t(fr, "pd", 0.65),
                "B_france_low": argmax_t(fr, "p2", 0.45),
                "C_france_avg": argmax_t(fr, "pa", 0.55)}
    print(f"v3 France pairs: {fr_v3.height:,}")
    for name, f in variants.items():
        shared = f.join(fr_v3, on=["rid", "oid"]).height
        p = write(pl.concat([iu.select("rid", "oid"), f]), name, s1, o)
        print(f"{name}: France pairs {f.height:,} (shared with v3 {shared:,}) -> {p}")


if __name__ == "__main__":
    main()
