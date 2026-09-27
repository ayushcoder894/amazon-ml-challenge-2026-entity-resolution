import polars as pl
D = r"C:\ML\Challenges\amazon\work\data\\"
pl.Config.set_fmt_str_lengths(150); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_rows(30)
s1 = pl.read_parquet(D+"train_source1.parquet"); s2 = pl.read_parquet(D+"train_source2.parquet"); s3 = pl.read_parquet(D+"train_source3.parquet")
t1 = pl.read_parquet(D+"test_source1.parquet"); t2 = pl.read_parquet(D+"test_source2.parquet"); t3 = pl.read_parquet(D+"test_source3.parquet")
pairs = pl.read_parquet(D+"train_pairs.parquet")
LEGAL = r"\b(private|pvt|limited|ltd|llc|inc|corp|corporation|co|company|llp|l\.l\.c|incorporated|group|the|services?|center)\b"
def key(c): return pl.col(c).str.to_lowercase().str.replace_all(r"[^a-z0-9 ]"," ").str.replace_all(LEGAL," ").str.replace_all(r"\s+"," ").str.strip_chars()
x = s1.with_columns(k=key("business_name"), city=pl.col("business_address").str.split(", ").list.get(-2, null_on_oob=True))
print("S1 duplicate exact names:", x.height - x["business_name"].n_unique(), " dup core names:", x.height - x["k"].n_unique())
print("S1 dup (core name, city):", x.height - x.select("k","city").unique().height)
print("S1 dup (name, address):", x.height - x.select("business_name","business_address").unique().height)
print(x.group_by("k").len().sort("len", descending=True).head(15))
# hard negatives: do unmatched S2/S3 core names equal some S1 core name?
allo = pl.concat([s2,s3])
un = allo.join(pairs.select(pl.col("m").alias("entity_id")), on="entity_id", how="anti").with_columns(k=key("business_name"))
ma = allo.join(pairs.select(pl.col("m").alias("entity_id")), on="entity_id").with_columns(k=key("business_name"))
ks = x.select("k").unique()
print("unmatched with core name in S1:", un.join(ks,on="k",how="semi").height/un.height, " matched:", ma.join(ks,on="k",how="semi").height/ma.height)
# singletons: are they similar to anything in S2/S3?
gt = pl.read_parquet(D+"train_ground_truth.parquet")
sing = x.join(gt.filter(pl.col("matched_entity_ids")=="").select(pl.col("source1_entity_id").alias("entity_id")), on="entity_id", how="semi")
ak = allo.with_columns(k=key("business_name")).select("k").unique()
print("singletons with core name in S2/S3:", sing.join(ak,on="k",how="semi").height/sing.height,
      " non-singletons:", x.join(sing.select("entity_id"),on="entity_id",how="anti").join(ak,on="k",how="semi").height/(x.height-sing.height))
print("\n== singleton examples"); print(sing.sample(8, seed=5).select("business_name","business_address"))
print("\n== FRANCE test samples")
for df,n in [(t1,"S1"),(t2,"S2"),(t3,"S3")]:
    print(n); print(df.filter(pl.col("country")=="France").sample(8, seed=4).select("business_name","business_address"))
print("\n== India S2 name prefixes/suffix noise sample"); print(s2.filter(pl.col("country")=="India").sample(10,seed=9).select("business_name","business_address"))
print(s3.filter(pl.col("country")=="US").sample(10,seed=9).select("business_name","business_address"))
