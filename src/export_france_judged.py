"""Export the hand-judged France samples (night of 2026-09-27) as a labelled TSV for the validation kit.
Re-creates exactly the samples shown in france_night/audit/delta_*.txt (same seed) and attaches the verdicts given while
reading them: Y = same business, N = different, ? = cannot tell. These are JUDGEMENTS, not ground truth.
"""
import polars as pl

import decide

D = "C:/ML/Challenges/amazon/work/data/"
NIGHT = "D:/amazon_er/france_night/"
OUT = "C:/ML/Challenges/amazon/work/france_validation_kit/"
VERDICTS = {
    "blend_removed": dict(Y=[0, 1, 2, 3, 5, 8, 9, 11, 13, 20, 22, 24], N=[4, 6, 7, 10, 16, 18, 21, 23], Q=[12, 14, 15, 17, 19]),
    "blend_added": dict(Y=[0, 1, 2, 4, 5, 9, 12, 13, 17, 18, 20, 21], N=[6, 7, 10, 14, 15, 22, 24], Q=[3, 8, 11, 16, 19, 23]),
    "rule_invented_pseudo_word": dict(Y=[i for i in range(25) if i != 17], N=[17], Q=[]),
    "rule_truncation_single_prefix": dict(Y=list(range(25)), N=[], Q=[]),
    "rule_acronym": dict(Y=list(range(25)), N=[], Q=[]),
}


def main(n=25, seed=0):
    F = pl.read_parquet(f"{NIGHT}data/france_union.parquet")
    v13 = decide.ef_predict(F.select("rid", "oid", pl.col("p").alias("p2")), "p2", 0.0, 4.0, 0.05).select("rid", "oid")
    v14 = pl.read_parquet(f"{NIGHT}data/france_v14_pairs.parquet")
    pool = pl.read_parquet(f"{NIGHT}data/pool_France_blend.parquet").select("rid", "oid", "name")
    kinds = {"blend_removed": v13.join(v14.select("rid", "oid"), on=["rid", "oid"], how="anti"),
             "blend_added": v14.filter(pl.col("rule") == 0).select("rid", "oid").join(v13, on=["rid", "oid"], how="anti")}
    ar = v14.filter(pl.col("rule") == 1).select("rid", "oid").join(pool, on=["rid", "oid"])
    for nm in ar["name"].unique().sort().to_list():
        kinds[f"rule_{nm}"] = ar.filter(pl.col("name") == nm).select("rid", "oid")
    s1 = pl.read_parquet(f"{D}test_source1.parquet", columns=["entity_id", "business_name", "business_address"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
    o = pl.concat([pl.read_parquet(f"{D}test_source{k}.parquet", columns=["entity_id", "business_name", "business_address"]) for k in (2, 3)]).with_row_index("oid").with_columns(pl.col("oid").cast(pl.UInt32))
    rows = []
    for kind, v in VERDICTS.items():
        smp = kinds[kind].sample(min(n, kinds[kind].height), seed=seed).with_row_index("i")
        lab = {i: "Y" for i in v["Y"]} | {i: "N" for i in v["N"]} | {i: "?" for i in v["Q"]}
        x = (smp.join(s1, on="rid").join(o, on="oid", suffix="_r").sort("i")
             .select(pl.lit(kind).alias("sample"), "i", source1_entity_id="entity_id", s1_name="business_name",
                     s1_address="business_address", record_entity_id="entity_id_r", record_name="business_name_r",
                     record_address="business_address_r")
             .with_columns(verdict=pl.col("i").replace_strict(lab, default="?", return_dtype=pl.String)))
        rows.append(x)
    t = pl.concat(rows)
    with open(f"{OUT}france_hand_judged_pairs.tsv", "w", encoding="utf-8-sig", newline="") as f:
        f.write(t.write_csv(separator="\t"))
    print(t.group_by("sample", "verdict").len().sort("sample", "verdict"))


if __name__ == "__main__":
    main()
