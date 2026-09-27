"""A8 = A7 minus further changed-house-number pairs (US, India) whose ensemble false fraction f (A100 v9 dr_ens) clears
both the null noise floor and the per-entity F0.5 break-even.  python build_a8.py <dir with f_US.parquet/f_India.parquet> [fmin]"""
import sys
import polars as pl
from dratio import pairs_of

OUT = "C:/ML/Challenges/amazon/work/output/"
D = "C:/ML/Challenges/amazon/work/data/"
N = "D:/amazon_er/france_night/data/"
rd = dict(separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
src, fmin = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
NC = ~((pl.col("ae") == 1) | (pl.col("addr_eq") == 1) | ((pl.col("ninter") == pl.col("nn1")) & (pl.col("nn1") == pl.col("nn2")))
       | (pl.col("nn2") == 0) | (pl.col("ninter") == pl.col("nn1")))
base = pl.read_csv(f"{OUT}matching_results_probeA7.tsv", **rd)
ex = base.filter(pl.col("matched_entity_ids") != "").with_columns(m=pl.col("matched_entity_ids").str.split(",")).explode("m").select(s="source1_entity_id", m="m")
drops, gain = [], 0.0
for c in ("US", "India"):
    F = pl.read_parquet(f"{src}/f_{c}.parquet").join(pl.read_parquet(f"{N}dratio_{c}.parquet", columns=["rid", "oid", "ae", "addr_eq", "ninter", "nn1", "nn2"]), on=["rid", "oid"])
    cur, _ = pairs_of("probeA7", c)
    k = cur.group_by("rid").len("k")
    R = F.join(cur, on=["rid", "oid"], how="semi").filter(NC & (pl.col("f_ens") > fmin)).join(k, on="rid").with_columns(kk=pl.col("k").cast(pl.Float64), f=pl.col("f_ens"))
    R = R.with_columns(g=pl.when(pl.col("k") == 1).then(2 * pl.col("f") - 1).otherwise(
        pl.col("f") * (1 - 1.25 * (pl.col("kk") - 1) / (1.25 * (pl.col("kk") - 1) + 1)) - (1 - pl.col("f")) * (1 - 1.25 * (pl.col("kk") - 1) / (1.25 * (pl.col("kk") - 1) + 0.25)))).filter(pl.col("g") > 0)
    print(f"{c}: remove {R.height:,} pairs (mean f {R['f'].mean() if R.height else 0:.3f}, xgb {R['f_xgb'].mean() if R.height else 0:.3f}, mlp {R['f_mlp'].mean() if R.height else 0:.3f}), expected entity-F gain {R['g'].sum():.1f}")
    gain += R["g"].sum()
    s1t = pl.read_parquet(f"{D}p1_canon_test_{c}_s1.parquet", columns=["idx", "entity_id"]).select(pl.col("idx").cast(pl.UInt32).alias("rid"), s="entity_id")
    ot = pl.read_parquet(f"{D}p1_canon_test_{c}_o.parquet", columns=["idx", "entity_id"]).select(pl.col("idx").cast(pl.UInt32).alias("oid"), m="entity_id")
    drops.append(R.select("rid", "oid").join(s1t, on="rid").join(ot, on="oid").select("s", "m"))
print(f"expected LB gain vs A7 ~ {0.85025 * gain / (663106 + 809986):+.5f}")
keep = ex.join(pl.concat(drops), on=["s", "m"], how="anti")
new = base.select("source1_entity_id").join(keep.group_by("s").agg(pl.col("m").sort().str.join(",")).rename({"s": "source1_entity_id", "m": "matched_entity_ids"}), on="source1_entity_id", how="left").with_columns(pl.col("matched_entity_ids").fill_null(""))
new.write_csv(f"{OUT}matching_results_probeA8.tsv", separator="\t", quote_style="never")
print("A8 pairs", keep.height, "vs A7", ex.height)
