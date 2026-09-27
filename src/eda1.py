import polars as pl
D = r"C:\ML\Challenges\amazon\work\data\\"
pl.Config.set_tbl_rows(40); pl.Config.set_fmt_str_lengths(120); pl.Config.set_tbl_width_chars(250)
gt = pl.read_parquet(D+"train_ground_truth.parquet")
s1 = pl.read_parquet(D+"train_source1.parquet"); s2 = pl.read_parquet(D+"train_source2.parquet"); s3 = pl.read_parquet(D+"train_source3.parquet")
t1 = pl.read_parquet(D+"test_source1.parquet"); t2 = pl.read_parquet(D+"test_source2.parquet"); t3 = pl.read_parquet(D+"test_source3.parquet")

print("GT ids == S1 ids:", set(gt["source1_entity_id"]) == set(s1["entity_id"]))
pairs = gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids")!="")
pairs = pairs.rename({"source1_entity_id":"s1","matched_entity_ids":"m"}).with_columns(src=pl.col("m").str.slice(0,2))
print("total pairs", pairs.height, pairs["src"].value_counts())
n = gt.with_columns(k=pl.col("matched_entity_ids").str.split(",").list.eval(pl.element().filter(pl.element()!="")).list.len())
print("matches per S1 distribution:\n", n["k"].value_counts().sort("k"))
print("singleton rate", (n["k"]==0).mean())
per = pairs.group_by("s1","src").len().pivot(on="src", index="s1", values="len").fill_null(0)
print("S2 per S1:", per["S2"].value_counts().sort("S2").head(12)); print("S3 per S1:", per["S3"].value_counts().sort("S3").head(12))
# is each S2/S3 record matched to at most one S1?
print("S2/S3 matched multiple S1s:", pairs.group_by("m").len().filter(pl.col("len")>1).height)
print("S2 coverage: matched", pairs.filter(pl.col("src")=="S2")["m"].n_unique(), "of", s2.height)
print("S3 coverage: matched", pairs.filter(pl.col("src")=="S3")["m"].n_unique(), "of", s3.height)
for nm, df in [("tr1",s1),("tr2",s2),("tr3",s3),("te1",t1),("te2",t2),("te3",t3)]:
    print(nm, df["country"].value_counts().sort("country").to_dicts(),
          "empty name", (df["business_name"]=="").sum(), "empty addr", (df["business_address"]=="").sum())
# singleton rate by country
n2 = n.join(s1.select("entity_id","country"), left_on="source1_entity_id", right_on="entity_id")
print(n2.group_by("country").agg(single=(pl.col("k")==0).mean(), mean_k=pl.col("k").mean(), n=pl.len()))
# country of matches vs S1
allo = pl.concat([s2,s3]).select(pl.col("entity_id").alias("m"), pl.col("country").alias("mc"))
pc = pairs.join(s1.select(pl.col("entity_id").alias("s1"), pl.col("country").alias("c1")), on="s1").join(allo, on="m")
print("country agree:", (pc["c1"]==pc["mc"]).mean())
pairs.write_parquet(D+"train_pairs.parquet")
