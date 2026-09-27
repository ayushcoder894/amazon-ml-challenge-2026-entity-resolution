"""India/US matching rows from a test probability file (rid, oid, p) with the v13 rule ef(0, 1.0, 0.05); every other row copied
from output/matching_results_v13_e9.tsv (France is replaced later by build_v15.py).
  python make_iu_matching.py <prob parquet> <name>     -> output/matching_results_<name>.tsv"""
import sys
import polars as pl
import decide
D = "C:/ML/Challenges/amazon/work/data/"
OUT = "C:/ML/Challenges/amazon/work/output/"
rd = dict(separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
T = pl.read_parquet(sys.argv[1]).select("rid", "oid", "p")
best = decide.ef_predict(T.select("rid", "oid", p2="p"), "p2", 0.0, 1.0, 0.05)
s1 = pl.read_parquet(f"{D}test_s1_norm.parquet", columns=["rid", "entity_id", "country"])
o = pl.concat([pl.read_parquet(f"{D}test_s{k}_norm.parquet", columns=["entity_id"]) for k in (2, 3)]).with_row_index("oid")
m = (best.join(s1.select("rid", s="entity_id"), on="rid").join(o.select(pl.col("oid").cast(best["oid"].dtype), m="entity_id"), on="oid")
     .group_by("s").agg(pl.col("m").sort().str.join(",")).rename({"s": "source1_entity_id", "m": "new"}))
ids = s1.filter(pl.col("country").is_in(["India", "US"]))["entity_id"].implode()
base = pl.read_csv(f"{OUT}matching_results_v13_e9.tsv", **rd)
new = (base.join(m, on="source1_entity_id", how="left")
       .with_columns(matched_entity_ids=pl.when(pl.col("source1_entity_id").is_in(ids)).then(pl.col("new").fill_null(""))
                     .otherwise(pl.col("matched_entity_ids"))).drop("new"))
new.write_csv(f"{OUT}matching_results_{sys.argv[2]}.tsv", separator="\t", quote_style="never")
old = base.join(s1.rename({"entity_id": "source1_entity_id"}), on="source1_entity_id")
x = old.join(new.rename({"matched_entity_ids": "nm"}), on="source1_entity_id")
print(x.group_by("country").agg(changed=(pl.col("matched_entity_ids") != pl.col("nm")).sum(), n=pl.len()).sort("country"))
print(f"India/US pairs: {best.height:,}")
