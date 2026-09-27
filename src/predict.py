"""Test inference: stage-1 (avg of fold models) -> stage-2 context (avg) -> assignment -> submission TSVs."""
import json
import os
import subprocess
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from train import D, M, stage2_features

OUT = "C:/ML/Challenges/amazon/work/output/"
RES = "C:/ML/Challenges/amazon/6ab10eb3b23ba_student_resource/student_resource/"


def predict_avg(tag, X):
    ps = [lgb.Booster(model_file=f"{M}{tag}_fold{f}.txt").predict(X, num_threads=18) for f in (0, 1)]
    return np.mean(ps, axis=0).astype(np.float32)


def main(t=None):
    t0 = time.time()
    meta = json.load(open(f"{M}meta.json"))
    t = meta["t"] if t is None else t
    countries = sorted(pl.scan_parquet(f"{D}test_s1_norm.parquet").select(pl.col("country").unique()).collect()
                       ["country"].to_list())
    df = pl.concat([pl.read_parquet(f"{D}test_feat_{c}.parquet") for c in countries])
    print(f"test pairs {df.height:,} ({time.time() - t0:.0f}s)", flush=True)
    df = df.with_columns(p1=pl.Series(predict_avg("s1", df.select(meta["feats1"]).to_numpy().astype(np.float32))))
    df = stage2_features(df)
    df = df.with_columns(p2=pl.Series(predict_avg("s2", df.select(meta["feats2"]).to_numpy().astype(np.float32))))
    print(f"scored ({time.time() - t0:.0f}s)", flush=True)

    s1 = pl.read_parquet(f"{D}test_s1_norm.parquet", columns=["rid", "entity_id"])
    o = pl.concat([pl.read_parquet(f"{D}test_s{k}_norm.parquet", columns=["entity_id"]) for k in (2, 3)]
                  ).with_row_index("oid")
    ids = lambda x: (x.join(s1.rename({"entity_id": "s1"}), on="rid").join(o.rename({"entity_id": "m"}), on="oid"))

    best = df.filter((pl.col("p2").rank("ordinal", descending=True).over("oid") == 1) & (pl.col("p2") >= t))
    match = ids(best.select("rid", "oid")).group_by("s1").agg(pl.col("m").sort().str.join(","))
    cand = ids(df.select("rid", "oid")).group_by("s1").agg(pl.col("m").unique().sort().str.join(","))

    os.makedirs(OUT, exist_ok=True)
    base = s1.select(source1_entity_id="entity_id")
    (base.join(match.rename({"s1": "source1_entity_id", "m": "matched_entity_ids"}), on="source1_entity_id",
               how="left").fill_null("")
     .write_csv(f"{OUT}matching_results.tsv", separator="\t", quote_style="never"))
    (base.join(cand.rename({"s1": "source1_entity_id", "m": "candidate_entity_ids"}), on="source1_entity_id",
               how="left").fill_null("")
     .write_csv(f"{OUT}candidate_pairs.tsv", separator="\t", quote_style="never"))
    n_match = best.height
    print(f"wrote submission: {n_match:,} matched pairs, {match.height:,}/{s1.height:,} S1 with >=1 match, "
          f"t={t} ({time.time() - t0:.0f}s)", flush=True)
    subprocess.run([sys.executable, f"{RES}utils/validate_submission.py", "--matching", f"{OUT}matching_results.tsv",
                    "--candidate", f"{OUT}candidate_pairs.tsv", "--test-dir", f"{RES}dataset/test"])


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else None)
