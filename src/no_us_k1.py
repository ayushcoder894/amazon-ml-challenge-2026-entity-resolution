import polars as pl
import decide
from no_shift import groups, D, DN, OUT, rd
pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(70); pl.Config.set_tbl_rows(40)
c = "US"
s1t = pl.read_parquet(f"{D}p1_canon_test_{c}_s1.parquet", columns=["idx", "entity_id"])
ot = pl.read_parquet(f"{D}p1_canon_test_{c}_o.parquet", columns=["idx", "entity_id"])
fin = (pl.read_csv(f"{OUT}matching_results_v19.tsv", **rd).filter(pl.col("matched_entity_ids") != "").with_columns(m=pl.col("matched_entity_ids").str.split(",")).explode("m"))
v19 = fin.join(s1t.rename({"entity_id": "source1_entity_id", "idx": "rid"}), on="source1_entity_id").join(ot.rename({"entity_id": "m", "idx": "oid"}), on="m").select("rid", "oid")
T = pl.read_parquet(f"{DN}test_union_p.parquet").join(s1t.select(rid="idx"), on="rid", how="semi").select("rid", "oid", "p")
te = groups("test", c, s1t.select(rid="idx"), T, v19).filter((pl.col("kb") == "k=1") & (pl.col("pb") == "0.5-0.9"))
raw1 = pl.read_parquet(f"{D}test_source1.parquet", columns=["business_name", "business_address"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
raw2 = pl.concat([pl.read_parquet(f"{D}test_source{k}.parquet", columns=["entity_id", "business_name"]) for k in (2, 3)]).with_row_index("oid").with_columns(pl.col("oid").cast(pl.UInt32))
nm = v19.group_by("rid").agg(n_matched=pl.len())
x = (te.join(raw2, on="oid").join(raw1.rename({"rid": "top_rid", "business_name": "s1_name", "business_address": "s1_addr"}), on="top_rid")
     .join(nm.rename({"rid": "top_rid"}), on="top_rid", how="left").with_columns(assigned=pl.col("ass_rid").is_not_null()))
print(x.height)
print(x.sample(30, seed=2).select("business_name", "s1_name", "s1_addr", pl.col("top_p").round(3), "assigned", "n_matched"))
# how many candidates per record, p of second candidate
sec = T.join(te.select("oid"), on="oid", how="semi").sort("p", descending=True).group_by("oid", maintain_order=True).agg(n=pl.len(), p2=pl.col("p").slice(1, 1).first())
print(sec.describe())
