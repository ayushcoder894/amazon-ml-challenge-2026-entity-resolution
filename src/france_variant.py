"""Submission variant: India / US from a finished submission, France (or any country not in train) decided by the
stage-1 probability p1 only - the decision source that transfers best to an unseen country in the leave-one-country-
out test (loco_blend.py). The candidate file is unchanged (same cascade set)."""
import subprocess
import sys

import polars as pl

import decide

D = "C:/ML/Challenges/amazon/work/data/"
OUT = "C:/ML/Challenges/amazon/work/output/"
RES = "C:/ML/Challenges/amazon/6ab10eb3b23ba_student_resource/student_resource/"


def main(base="v12final", pred="test_pred_v12", name="v12_FRp1", rule=("ef", 0.0, 4.0)):
    train_c = set(pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["country"])["country"].unique().to_list())
    s1 = pl.read_parquet(f"{D}test_s1_norm.parquet", columns=["rid", "entity_id", "country"])
    o = pl.concat([pl.read_parquet(f"{D}test_s{k}_norm.parquet", columns=["entity_id"]) for k in (2, 3)]).with_row_index("oid")
    unseen = sorted(set(s1["country"].unique().to_list()) - train_c)
    K = pl.read_parquet(f"{D}{pred}.parquet", columns=["rid", "oid", "country", "p1"]).filter(pl.col("country").is_in(unseen))
    best = decide.ef_predict(K.select("rid", "oid", pl.col("p1").alias("p2")), "p2", rule[1], rule[2], 0.05)
    m = (best.join(s1.select("rid", s="entity_id"), on="rid").join(o.select("oid", m="entity_id"), on="oid")
         .group_by("s").agg(pl.col("m").sort().str.join(",")))
    base_df = pl.read_csv(f"{OUT}matching_results_{base}.tsv", separator="\t", quote_char=None, infer_schema=False,
                          missing_utf8_is_empty_string=True)
    ids = s1.filter(pl.col("country").is_in(unseen))["entity_id"]
    new = (base_df.join(m.rename({"s": "source1_entity_id", "m": "new"}), on="source1_entity_id", how="left")
           .with_columns(matched_entity_ids=pl.when(pl.col("source1_entity_id").is_in(ids.implode()))
                         .then(pl.col("new").fill_null("")).otherwise(pl.col("matched_entity_ids"))).drop("new"))
    mp = f"{OUT}matching_results_{name}.tsv"
    new.write_csv(mp, separator="\t", quote_style="never")
    cp = f"{OUT}candidate_pairs_{name}.tsv"
    pl.read_csv(f"{OUT}candidate_pairs_{base}.tsv", separator="\t", quote_char=None, infer_schema=False,
                missing_utf8_is_empty_string=True).write_csv(cp, separator="\t", quote_style="never")
    print(f"unseen countries {unseen}: {best.height:,} matched pairs ({best.height / ids.len():.3f} per S1) -> {mp}")
    subprocess.run([sys.executable, f"{RES}utils/validate_submission.py", "--matching", mp, "--candidate", cp,
                    "--test-dir", f"{RES}dataset/test"])


if __name__ == "__main__":
    main(*sys.argv[1:4])
