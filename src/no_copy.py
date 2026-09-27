"""Do name-only records copy the raw name of one of their owner's other records? Oracle check on train (true siblings)."""
import polars as pl
D = "C:/ML/Challenges/amazon/work/data/"
keep = pl.read_parquet(f"{D}keep_v11.parquet")
o = pl.concat([pl.read_parquet(f"{D}train_source{k}.parquet", columns=["business_name"]) for k in (2, 3)]).with_row_index("oid").with_columns(pl.col("oid").cast(pl.UInt32))
s1raw = pl.read_parquet(f"{D}train_source1.parquet", columns=["business_name"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
for c in ("India", "US"):
    s1c = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == c).select("rid").join(keep, on="rid", how="semi")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid"]).join(s1c, on="rid", how="semi")
    s1n = pl.read_parquet(f"{D}p1_canon_train_{c}_s1.parquet", columns=["idx", "cns"]).rename({"idx": "rid"}).join(s1c, on="rid", how="semi")
    oc = pl.read_parquet(f"{D}p1_canon_train_{c}_o.parquet", columns=["idx", "cns", "aempty"]).rename({"idx": "oid"})
    sh = s1n.filter(pl.col("cns") != "").group_by("cns").agg(k=pl.len())
    # ambiguous name-only records that have an owner among the namesakes
    no = oc.filter(pl.col("aempty")).join(sh.filter(pl.col("k") >= 2), on="cns").join(gt, on="oid").join(o, on="oid").rename({"rid": "owner", "business_name": "nm"})
    no = no.join(s1n.rename({"rid": "owner"}), on=["owner", "cns"], how="semi")  # owner is a namesake
    # all records (true siblings) of all namesake entities of the same cns
    recs = gt.join(s1n, on="rid").join(o, on="oid").join(oc.select("oid", "aempty"), on="oid").rename({"business_name": "snm"})
    x = no.join(recs.filter(~pl.col("aempty")).select("cns", "rid", pl.col("oid").alias("oid2"), "snm"), on="cns")
    x = x.with_columns(is_owner=pl.col("rid") == pl.col("owner"), exact=pl.col("nm") == pl.col("snm"), exact_ci=pl.col("nm").str.to_lowercase().str.strip_chars() == pl.col("snm").str.to_lowercase().str.strip_chars())
    e = x.group_by("oid", "rid", "is_owner").agg(exact=pl.col("exact").any(), exact_ci=pl.col("exact_ci").any())
    s1x = no.join(s1raw.rename({"rid": "owner", "business_name": "s1nm"}), on="owner").with_columns(eq_s1=pl.col("nm") == pl.col("s1nm"))
    print(f"\n{c}: ambiguous name-only records owned by a namesake: {no.height:,}; name == owner's S1 name exactly: {s1x['eq_s1'].mean():.1%}")
    print(e.group_by("is_owner").agg(pairs=pl.len(), exact=pl.col("exact").mean(), exact_ci=pl.col("exact_ci").mean()))
    # decision power: records where exactly one namesake has an exact-name sibling
    w = e.filter(pl.col("exact")).group_by("oid").agg(n=pl.len(), hit=pl.col("is_owner").any())
    print(f"  records with >=1 namesake having an exact-name sibling: {w.height:,} ({w.height / no.height:.1%}); of these exactly one namesake: {w.filter(pl.col('n') == 1).height:,}, that one is the owner: {w.filter(pl.col('n') == 1)['hit'].mean():.1%}")
