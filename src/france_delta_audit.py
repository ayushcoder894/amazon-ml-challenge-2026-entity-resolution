"""Hand audit of the France changes v13 -> v14 (the France labels the organisers now allow, disclosed):
random samples of (a) pairs the blend REMOVES, (b) pairs the blend ADDS, (c) pairs each rule ADDS - each shown with the
entity (raw name | address), the record, v13 probability p, joint pj, blend pb, the entity's other accepted records
(for context) and the record's best rival entity. Output: france_night/audit/delta_<kind>.txt
"""
import sys

import polars as pl

import decide

D = "C:/ML/Challenges/amazon/work/data/"
NIGHT = "D:/amazon_er/france_night/"


def main(n=25, seed=0):
    F = pl.read_parquet(f"{NIGHT}data/france_union.parquet")
    v13 = decide.ef_predict(F.select("rid", "oid", pl.col("p").alias("p2")), "p2", 0.0, 4.0, 0.05).select("rid", "oid")
    v14 = pl.read_parquet(f"{NIGHT}data/france_v14_pairs.parquet")
    pool = pl.read_parquet(f"{NIGHT}data/pool_France_blend.parquet").select("rid", "oid", "name", "addr", "sh")
    s1 = pl.read_parquet(f"{D}test_source1.parquet", columns=["entity_id", "business_name", "business_address"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
    o = pl.concat([pl.read_parquet(f"{D}test_source{k}.parquet", columns=["entity_id", "business_name", "business_address"]) for k in (2, 3)]).with_row_index("oid").with_columns(pl.col("oid").cast(pl.UInt32))
    kinds = {
        "blend_removed": v13.join(v14.select("rid", "oid"), on=["rid", "oid"], how="anti"),
        "blend_added": v14.filter(pl.col("rule") == 0).select("rid", "oid").join(v13, on=["rid", "oid"], how="anti"),
    }
    added_rules = v14.filter(pl.col("rule") == 1).select("rid", "oid").join(pool, on=["rid", "oid"])
    for nm in added_rules["name"].unique().sort().to_list():
        kinds[f"rule_{nm}"] = added_rules.filter(pl.col("name") == nm).select("rid", "oid")
    acc14 = v14.select("rid", "oid")
    for kind, X in kinds.items():
        smp = X.sample(min(n, X.height), seed=seed)
        lines = [f"# {kind}: {X.height:,} pairs in France; {smp.height} sampled. Label each: Y (same business) / N / ? "]
        for i, (r, oo) in enumerate(smp.iter_rows()):
            e = s1.filter(pl.col("rid") == r).row(0, named=True)
            rec = o.filter(pl.col("oid") == oo).row(0, named=True)
            pr = F.filter((pl.col("rid") == r) & (pl.col("oid") == oo)).row(0, named=True)
            lines.append(f"\n[{i:02d}] ENTITY {e['business_name']} | {e['business_address']}")
            lines.append(f"     RECORD {rec['business_name']} | {rec['business_address']}   (p={pr['p']:.2f} pj={pr['pj']:.2f} pb={pr['pb']:.2f})")
            others = acc14.filter((pl.col("rid") == r) & (pl.col("oid") != oo)).join(o, on="oid").head(4)
            for q in others.iter_rows(named=True):
                lines.append(f"        entity's other v14 record: {q['business_name']} | {q['business_address']}")
            riv = F.filter((pl.col("oid") == oo) & (pl.col("rid") != r)).sort("pb", descending=True).head(1).join(s1, on="rid")
            for q in riv.iter_rows(named=True):
                lines.append(f"        best rival entity (pb={q['pb']:.2f}): {q['business_name']} | {q['business_address']}")
        open(f"{NIGHT}audit/delta_{kind}.txt", "w", encoding="utf-8").write("\n".join(lines))
        print(f"{kind}: {X.height:,} pairs -> audit/delta_{kind}.txt")


if __name__ == "__main__":
    main(*[int(x) for x in sys.argv[1:3]])
