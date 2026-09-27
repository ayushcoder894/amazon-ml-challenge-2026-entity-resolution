"""Submission v15 = India / US from v13_ceu (v12 + dense add-on + joint + union CE) + France v15
(validated France recipe: blend of stage-1/dense and a joint pass trained on the unseen-country unions, ef(0, 4);
+ R2/R3 own-address rules; - near-number veto on blend additions). Every predicted pair is also a listed candidate.
"""
import subprocess
import sys

import polars as pl

D = "C:/ML/Challenges/amazon/work/data/"
OUT = "C:/ML/Challenges/amazon/work/output/"
NIGHT = "D:/amazon_er/france_night/"
RES = "C:/ML/Challenges/amazon/6ab10eb3b23ba_student_resource/student_resource/"


def main(base="v13_ceu", name="v15"):
    rd = dict(separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
    s1 = pl.read_parquet(f"{D}test_s1_norm.parquet", columns=["rid", "entity_id", "country"])
    o = pl.concat([pl.read_parquet(f"{D}test_s{k}_norm.parquet", columns=["entity_id"]) for k in (2, 3)]).with_row_index("oid").with_columns(pl.col("oid").cast(pl.UInt32))
    import os
    P = pl.read_parquet(f"{NIGHT}data/" + os.environ.get("FR_PAIRS", "france_v15_pairs") + ".parquet").select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32))
    fr = s1.filter(pl.col("country") == "France")
    m = (P.join(fr.select(pl.col("rid").cast(pl.UInt32), s="entity_id"), on="rid").join(o.select("oid", m="entity_id"), on="oid")
         .group_by("s").agg(pl.col("m").sort().str.join(",")).rename({"s": "source1_entity_id", "m": "new"}))
    ids = fr["entity_id"].implode()
    mr = pl.read_csv(f"{OUT}matching_results_{base}.tsv", **rd)
    mr = (mr.join(m, on="source1_entity_id", how="left")
          .with_columns(matched_entity_ids=pl.when(pl.col("source1_entity_id").is_in(ids)).then(pl.col("new").fill_null(""))
                        .otherwise(pl.col("matched_entity_ids"))).drop("new"))
    # Candidate files describe the actual scoring pool, not just accepted additions.
    fu = pl.read_parquet(f"{NIGHT}data/france_union.parquet", columns=["rid", "oid"])
    fc = pl.concat([fu, P]).unique().join(fr.select("rid"), on="rid", how="semi")
    if base in ("v13_e9", "v13_set"):
        iu = (pl.read_parquet("D:/amazon_er/dense/test_union_p.parquet", columns=["rid", "oid"])
              .join(fr.select("rid"), on="rid", how="anti"))
        pairs = pl.concat([iu, fc]).unique()
        cc = (pairs.join(s1.select("rid", source1_entity_id="entity_id"), on="rid")
              .join(o.select("oid", record_id="entity_id"), on="oid").group_by("source1_entity_id")
              .agg(candidate_entity_ids=pl.col("record_id").sort().str.join(",")))
        cp = s1.select(source1_entity_id="entity_id").join(cc, on="source1_entity_id", how="left").fill_null("")
        print(f"Exact scored candidate pool: India/US {iu.height:,}; France {fc.height:,}; total {pairs.height:,}", flush=True)
    else:
        cp = pl.read_csv(f"{OUT}candidate_pairs_{base}.tsv", **rd)
        cc = (fc.join(fr.select("rid", source1_entity_id="entity_id"), on="rid")
              .join(o.select("oid", record_id="entity_id"), on="oid").group_by("source1_entity_id")
              .agg(fc=pl.col("record_id").sort().str.join(",")))
        cp = (cp.join(cc,on="source1_entity_id",how="left")
              .with_columns(candidate_entity_ids=pl.when(pl.col("source1_entity_id").is_in(ids)).then(pl.col("fc").fill_null(""))
                            .otherwise(pl.col("candidate_entity_ids"))).drop("fc"))
    assert mr.height == cp.height == 1732544
    outdir = os.environ.get("SUBMISSION_OUT", OUT).rstrip("/\\") + "/"
    os.makedirs(outdir, exist_ok=True)
    mp, cpp = f"{outdir}matching_results_{name}.tsv", f"{outdir}candidate_pairs_{name}.tsv"
    mr.write_csv(mp, separator="\t", quote_style="never")
    cp.write_csv(cpp, separator="\t", quote_style="never")
    k = mr.join(s1.rename({"entity_id": "source1_entity_id"}), on="source1_entity_id").with_columns(
        n=pl.when(pl.col("matched_entity_ids") == "").then(0).otherwise(pl.col("matched_entity_ids").str.count_matches(",") + 1))
    print(k.group_by("country").agg(per_S1=pl.col("n").mean(), empty=(pl.col("n") == 0).mean()).sort("country"))
    subprocess.run([sys.executable, f"{RES}utils/validate_submission.py", "--matching", mp, "--candidate", cpp,
                    "--test-dir", f"{RES}dataset/test"], check=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
