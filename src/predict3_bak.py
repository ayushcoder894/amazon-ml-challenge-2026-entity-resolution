"""Test inference for the streaming (train3) models: stage-1 avg of fold models -> stage-2 context -> assignment."""
import json
import os
import subprocess
import sys
import time

import cupy as cp
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import xgboost as xgb

import decide
from train import stage2_features
from train3 import BATCH, D, EXTRA, M

OUT = "C:/ML/Challenges/amazon/work/output/"
RES = "C:/ML/Challenges/amazon/6ab10eb3b23ba_student_resource/student_resource/"


def load(name, tag):
    bs = []
    for f in (0, 1):
        b = xgb.Booster()
        b.load_model(f"{M}{name}{tag}_fold{f}.json")
        b.set_param({"device": "cuda"})
        bs.append(b)
    return bs


def predict_avg(boosters, path, feats, extra=None):
    out, off = [], 0
    for b in pq.ParquetFile(path).iter_batches(batch_size=BATCH, columns=feats):
        X = pl.from_arrow(b).select(pl.col(feats).cast(pl.Float32)).to_numpy()
        if extra is not None:
            X = np.hstack([X, extra[off:off + len(X)]])
        off += len(X)
        Xg = cp.asarray(X)
        out.append(np.mean([cp.asnumpy(bst.inplace_predict(Xg)) for bst in boosters], axis=0))
    return np.concatenate(out).astype(np.float32)


def main(tag="_v3", rule=None, mname=None, only=None, write=True):
    t0 = time.time()
    mname = mname or tag
    meta = json.load(open(f"{M}meta{mname}.json"))
    rule = rule or meta["rule"]
    feats1 = meta["feats1"]
    countries = sorted(pl.scan_parquet(f"{D}test_s1_norm.parquet").select(pl.col("country").unique()).collect()
                       ["country"].to_list())
    if only:
        countries = [c for c in countries if c in only]
    s1b, s2b = load("s1", mname), load("s2", mname)
    parts = []
    for c in countries:
        path = f"{D}test_feat{tag}_{c}.parquet"
        k = pl.read_parquet(path, columns=["rid", "oid", "src"]).with_columns(
            country=pl.lit(c), row=pl.int_range(pl.len(), dtype=pl.UInt32))
        parts.append(k.with_columns(p1=pl.Series(predict_avg(s1b, path, feats1))))
        print(f"[{c}] stage-1 on {k.height:,} pairs ({time.time() - t0:.0f}s)", flush=True)
    K = stage2_features(pl.concat(parts))
    p2 = []
    for c in countries:
        path = f"{D}test_feat{tag}_{c}.parquet"
        ex = K.filter(pl.col("country") == c).sort("row").select(pl.col(EXTRA).cast(pl.Float32)).to_numpy()
        p2.append(pl.DataFrame({"country": [c] * len(ex), "row": np.arange(len(ex), dtype=np.uint32),
                                "p2": predict_avg(s2b, path, feats1, ex)}))
        print(f"[{c}] stage-2 ({time.time() - t0:.0f}s)", flush=True)
    K = K.join(pl.concat(p2), on=["country", "row"])
    K.select("rid", "oid", "country", "p1", "p2").write_parquet(f"{D}test_pred{mname}.parquet")
    if not write:
        return K

    oo = K.select("rid", "oid", "p2")
    if rule[0] == "t":
        best = oo.filter((pl.col("p2").rank("ordinal", descending=True).over("oid") == 1) & (pl.col("p2") >= rule[1]))
    else:
        best = decide.ef_predict(oo, "p2", rule[1], rule[2], 0.05)
    s1 = pl.read_parquet(f"{D}test_s1_norm.parquet", columns=["rid", "entity_id"])
    o = pl.concat([pl.read_parquet(f"{D}test_s{k}_norm.parquet", columns=["entity_id"]) for k in (2, 3)]
                  ).with_row_index("oid")
    ids = lambda x: (x.join(s1.rename({"entity_id": "s1"}), on="rid").join(o.rename({"entity_id": "m"}), on="oid"))
    match = ids(best.select("rid", "oid")).group_by("s1").agg(pl.col("m").sort().str.join(","))
    cand = ids(K.select("rid", "oid")).group_by("s1").agg(pl.col("m").unique().sort().str.join(","))
    os.makedirs(OUT, exist_ok=True)
    base = s1.select(source1_entity_id="entity_id")
    (base.join(match.rename({"s1": "source1_entity_id", "m": "matched_entity_ids"}), on="source1_entity_id", how="left")
     .fill_null("").write_csv(f"{OUT}matching_results.tsv", separator="\t", quote_style="never"))
    (base.join(cand.rename({"s1": "source1_entity_id", "m": "candidate_entity_ids"}), on="source1_entity_id", how="left")
     .fill_null("").write_csv(f"{OUT}candidate_pairs.tsv", separator="\t", quote_style="never"))
    print(f"wrote submission: {best.height:,} matched pairs, rule={rule} ({time.time() - t0:.0f}s)", flush=True)
    stats = (best.join(K.select("rid", "oid", "country"), on=["rid", "oid"]).group_by("country").agg(pairs=pl.len())
             .join(s1.join(K.select("rid", "country").unique(), on="rid").group_by("country").agg(n_s1=pl.len()),
                   on="country"))
    print(stats.with_columns(matches_per_s1=pl.col("pairs") / pl.col("n_s1")).sort("country"))
    subprocess.run([sys.executable, f"{RES}utils/validate_submission.py", "--matching", f"{OUT}matching_results.tsv",
                    "--candidate", f"{OUT}candidate_pairs.tsv", "--test-dir", f"{RES}dataset/test"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "_v3")
