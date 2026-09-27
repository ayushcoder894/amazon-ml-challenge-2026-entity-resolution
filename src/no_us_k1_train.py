import polars as pl
import decide
from no_shift import groups, D, DN
pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(60); pl.Config.set_tbl_rows(40)
c = "US"
keep = pl.read_parquet(f"{D}keep_v11.parquet")
s1c = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == c).select("rid").join(keep, on="rid", how="semi")
gt = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid"]).rename({"rid": "owner"})
U = pl.read_parquet(f"{DN}train_joint9.parquet", columns=["rid", "oid", "pj", "country"]).filter(pl.col("country") == c).select("rid", "oid", p="pj")
pred = decide.ef_predict(U.select("rid", "oid", p2="p"), "p2", 0.0, 1.0, 0.05)
tr = groups("train", c, s1c, U, pred).filter((pl.col("kb") == "k=1") & (pl.col("pb") == "0.5-0.9")).join(gt, on="oid", how="left")
kept = s1c.with_columns(kept=pl.lit(True)).rename({"rid": "owner"})
tr = tr.join(kept, on="owner", how="left").with_columns(
    truth=pl.when(pl.col("owner").is_null()).then(pl.lit("unowned distractor")).when(pl.col("owner") == pl.col("top_rid")).then(pl.lit("owner = top"))
    .when(pl.col("kept").is_null()).then(pl.lit("owner dropped")).otherwise(pl.lit("other kept S1")))
print(tr.group_by("truth").agg(n=pl.len(), assigned=pl.col("ass_rid").is_not_null().mean(), mean_p=pl.col("top_p").mean()).sort("n", descending=True))
raw1 = pl.read_parquet(f"{D}train_source1.parquet", columns=["business_name", "business_address"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
raw2 = pl.concat([pl.read_parquet(f"{D}train_source{k}.parquet", columns=["business_name"]) for k in (2, 3)]).with_row_index("oid").with_columns(pl.col("oid").cast(pl.UInt32))
x = tr.join(raw2, on="oid").join(raw1.rename({"rid": "top_rid", "business_name": "s1_name", "business_address": "s1_addr"}), on="top_rid")
x = x.join(raw1.rename({"rid": "owner", "business_name": "owner_name", "business_address": "owner_addr"}), on="owner", how="left")
for t in ("unowned distractor", "owner dropped", "other kept S1"):
    print(t)
    print(x.filter(pl.col("truth") == t).sample(min(12, x.filter(pl.col("truth") == t).height), seed=1).select("business_name", "s1_name", pl.col("top_p").round(3), "owner_name", "owner_addr"))
