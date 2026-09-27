"""Test submission with the cross-encoder stage: stage-2 predictions -> CE on the uncertain band -> combiner -> TSVs.

Uses data/test_pred{mname}.parquet (from predict3) and data/ce_test{tag}.parquet (from `ce.py test`).
The candidate file is the cascade set scored by the final matcher (unchanged by the CE stage).
"""
import json
import os
import subprocess
import sys

import numpy as np
import polars as pl
import xgboost as xgb

import decide
from ce_combine import FEATS, ce_features

D = "C:/ML/Challenges/amazon/work/data/"
MC = "C:/ML/Challenges/amazon/work/models_ce/"
OUT = "C:/ML/Challenges/amazon/work/output/"
RES = "C:/ML/Challenges/amazon/6ab10eb3b23ba_student_resource/student_resource/"


def main(tag="_v9", mname="_v9", name="ce"):
    meta = json.load(open(f"{MC}comb_meta{tag}.json"))
    K = pl.read_parquet(f"{D}test_pred{mname}.parquet").join(
        pl.read_parquet(f"{D}ce_test{tag}.parquet"), on=["rid", "oid"], how="left")
    b = ce_features(K)
    X = xgb.DMatrix(b.select(FEATS).to_numpy().astype(np.float32))
    pc = np.mean([xgb.Booster(model_file=f"{MC}comb{tag}_fold{f}.json").predict(X) for f in (0, 1)], axis=0)
    K = K.join(b.select("rid", "oid").with_columns(pc=pl.Series(pc)), on=["rid", "oid"], how="left").with_columns(
        p3=pl.coalesce("pc", "p2"))
    K.select("rid", "oid", "country", "p1", "p2", "ce", "p3").write_parquet(f"{D}test_pred_ce{tag}.parquet")
    rule = meta["rule"]
    oo = K.select("rid", "oid", pl.col("p3").alias("p2"))
    best = (oo.filter((pl.col("p2").rank("ordinal", descending=True).over("oid") == 1) & (pl.col("p2") >= rule[1]))
            if rule[0] == "t" else decide.ef_predict(oo, "p2", rule[1], rule[2], 0.05))
    s1 = pl.read_parquet(f"{D}test_s1_norm.parquet", columns=["rid", "entity_id"])
    o = pl.concat([pl.read_parquet(f"{D}test_s{k}_norm.parquet", columns=["entity_id"]) for k in (2, 3)]
                  ).with_row_index("oid")
    ids = lambda x: x.join(s1.rename({"entity_id": "s1"}), on="rid").join(o.rename({"entity_id": "m"}), on="oid")
    match = ids(best.select("rid", "oid")).group_by("s1").agg(pl.col("m").sort().str.join(","))
    cand = ids(K.select("rid", "oid")).group_by("s1").agg(pl.col("m").unique().sort().str.join(","))
    base = s1.select(source1_entity_id="entity_id")
    mp, cpth = f"{OUT}matching_results_{name}.tsv", f"{OUT}candidate_pairs_{name}.tsv"
    (base.join(match.rename({"s1": "source1_entity_id", "m": "matched_entity_ids"}), on="source1_entity_id", how="left")
     .fill_null("").write_csv(mp, separator="\t", quote_style="never"))
    (base.join(cand.rename({"s1": "source1_entity_id", "m": "candidate_entity_ids"}), on="source1_entity_id", how="left")
     .fill_null("").write_csv(cpth, separator="\t", quote_style="never"))
    changed = best.join(K.filter(pl.col("pc").is_not_null()).select("rid", "oid"), on=["rid", "oid"]).height
    print(f"wrote {mp}: {best.height:,} matched pairs ({changed:,} decided inside the CE band), "
          f"{K.height / s1.height:.2f} candidates per S1")
    subprocess.run([sys.executable, f"{RES}utils/validate_submission.py", "--matching", mp, "--candidate", cpth,
                    "--test-dir", f"{RES}dataset/test"])


if __name__ == "__main__":
    main(*sys.argv[1:4])
