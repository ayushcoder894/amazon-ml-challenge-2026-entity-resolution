"""Build a submission from a reranker experiment (gpu2/exp_rerank.py outputs).

Train: the 3-parameter combiner of the current final probability pf and the reranker score s is fitted on ALL
out-of-fold band pairs (both folds). Test: applied to the test band pairs that have a reranker score (mean of the two
fold models); other pairs keep pf. Decisions use the pipeline's rule. Countries absent from training (France) keep the
stage-1 decisions of the base submission unless --rerank_unseen is given.
Usage: python integrate_rerank.py <exp_out_dir> <tag> <name> [--rerank_unseen]
"""
import subprocess
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "C:/ML/Challenges/amazon/work/gpu2")
from exp_rerank import apply_combiner, fit_combiner  # noqa: E402

import decide  # noqa: E402

D = "C:/ML/Challenges/amazon/work/data/"
OUT = "C:/ML/Challenges/amazon/work/output/"
A = "C:/ML/Challenges/amazon/work/gpu2/aux/"
RES = "C:/ML/Challenges/amazon/6ab10eb3b23ba_student_resource/student_resource/"


def main(exp, tag, name, rerank_unseen=False, base="v12_FRp1"):
    import json
    rule = json.load(open(A + "rule.json"))
    tr = pl.read_parquet(A + "cand_train.parquet", columns=["rid", "oid", "pf"])
    gt = pl.read_parquet(D + "train_gt_rid.parquet", columns=["rid", "oid"]).with_columns(y=pl.lit(1, pl.Int8))
    oof = (pl.read_parquet(f"{exp}/rerank_{tag}_oof.parquet").join(tr, on=["rid", "oid"])
           .join(gt, on=["rid", "oid"], how="left").with_columns(pl.col("y").fill_null(0)))
    w = fit_combiner(oof["pf"].to_numpy(), oof["s"].to_numpy(), oof["y"].to_numpy())
    print(f"combiner weights (logit pf, logit s, product, bias): {np.round(w, 3)} on {oof.height:,} train pairs")
    te = pl.read_parquet(A + "cand_test.parquet", columns=["rid", "oid", "country", "pf"])
    st = pl.read_parquet(f"{exp}/rerank_{tag}_test.parquet")
    te = te.join(st, on=["rid", "oid"], how="left")
    has = te["s"].is_not_null().to_numpy()
    pc = te["pf"].to_numpy().copy()
    pc[has] = apply_combiner(w, te["pf"].to_numpy()[has], te["s"].to_numpy()[has])
    te = te.with_columns(pc=pl.Series(pc))
    print(f"test pairs re-scored: {has.sum():,} of {te.height:,}")
    train_c = set(pl.read_parquet(D + "train_s1_norm.parquet", columns=["country"])["country"].unique().to_list())
    keep_c = te["country"].unique().to_list() if rerank_unseen else [c for c in te["country"].unique().to_list() if c in train_c]
    best = decide.ef_predict(te.filter(pl.col("country").is_in(keep_c)).select("rid", "oid", pl.col("pc").alias("p2")),
                             "p2", rule["mu"], rule["c"], 0.05)
    s1 = pl.read_parquet(D + "test_s1_norm.parquet", columns=["rid", "entity_id", "country"])
    o = pl.concat([pl.read_parquet(D + f"test_s{k}_norm.parquet", columns=["entity_id"]) for k in (2, 3)]).with_row_index("oid")
    m = (best.join(s1.select("rid", s="entity_id"), on="rid").join(o.select("oid", m="entity_id"), on="oid")
         .group_by("s").agg(pl.col("m").sort().str.join(",")))
    ids = s1.filter(pl.col("country").is_in(keep_c))["entity_id"]
    bdf = pl.read_csv(f"{OUT}matching_results_{base}.tsv", separator="\t", quote_char=None, infer_schema=False,
                      missing_utf8_is_empty_string=True)
    new = (bdf.join(m.rename({"s": "source1_entity_id", "m": "new"}), on="source1_entity_id", how="left")
           .with_columns(matched_entity_ids=pl.when(pl.col("source1_entity_id").is_in(ids.implode()))
                         .then(pl.col("new").fill_null("")).otherwise(pl.col("matched_entity_ids"))).drop("new"))
    mp, cp = f"{OUT}matching_results_{name}.tsv", f"{OUT}candidate_pairs_{name}.tsv"
    new.write_csv(mp, separator="\t", quote_style="never")
    pl.read_csv(f"{OUT}candidate_pairs_{base}.tsv", separator="\t", quote_char=None, infer_schema=False,
                missing_utf8_is_empty_string=True).write_csv(cp, separator="\t", quote_style="never")
    changed = new.join(bdf, on="source1_entity_id").filter(pl.col("matched_entity_ids") != pl.col("matched_entity_ids_right")).height
    print(f"re-decided countries {keep_c}; S1 rows changed vs {base}: {changed:,}")
    subprocess.run([sys.executable, f"{RES}utils/validate_submission.py", "--matching", mp, "--candidate", cp,
                    "--test-dir", f"{RES}dataset/test"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], "--rerank_unseen" in sys.argv)
