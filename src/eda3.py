import polars as pl, re
D = r"C:\ML\Challenges\amazon\work\data\\"
s1 = pl.read_parquet(D+"train_source1.parquet"); s2 = pl.read_parquet(D+"train_source2.parquet"); s3 = pl.read_parquet(D+"train_source3.parquet")
t1 = pl.read_parquet(D+"test_source1.parquet"); t2 = pl.read_parquet(D+"test_source2.parquet"); t3 = pl.read_parquet(D+"test_source3.parquet")
pairs = pl.read_parquet(D+"train_pairs.parquet")
# id overlap train/test
for a,b,n in [(s1,t1,"S1"),(s2,t2,"S2"),(s3,t3,"S3")]:
    print(n, "train/test id overlap:", a.join(b, on="entity_id").height)
# id numeric order leak?
p = pairs.with_columns(a=pl.col("s1").str.slice(3).cast(pl.Int64), b=pl.col("m").str.slice(3).cast(pl.Int64))
print("id corr", p.select(pl.corr("a","b")).item())
# row-order leak: position in file
s1p = s1.with_row_index("r1").select(pl.col("entity_id").alias("s1"),"r1")
op = pl.concat([s2,s3]).with_row_index("r2").select(pl.col("entity_id").alias("m"),"r2")
q = pairs.join(s1p,on="s1").join(op,on="m"); print("row corr", q.select(pl.corr("r1","r2")).item())
# script composition
def script_stats(df, name):
    x = df.with_columns(nonlatin=pl.col("business_name").str.contains(r"[^\x00-\x7F\u00C0-\u024F\u2018-\u201f]"),
                        addr_nonlatin=pl.col("business_address").str.contains(r"[\u0900-\u0DFF]"))
    print(name, x.group_by("country").agg(pl.col("nonlatin").mean(), pl.col("addr_nonlatin").mean()).sort("country").rows())
for df,n in [(s1,"tr1"),(s2,"tr2"),(s3,"tr3"),(t1,"te1"),(t2,"te2"),(t3,"te3")]: script_stats(df,n)
# digit-token overlap between matched pairs
def nums(c): return pl.col(c).str.extract_all(r"\d+")
allo = pl.concat([s2,s3]).rename({"entity_id":"m","business_name":"n2","business_address":"a2"}).drop("country")
q = pairs.join(s1.rename({"entity_id":"s1","business_name":"n1","business_address":"a1"}), on="s1").join(allo, on="m")
q = q.with_columns(d1=nums("a1").list.unique(), d2=nums("a2").list.unique())
q = q.with_columns(inter=pl.col("d1").list.set_intersection("d2").list.len(), l1=pl.col("d1").list.len(), l2=pl.col("d2").list.len())
print("pairs where both have digits:", ((q["l1"]>0)&(q["l2"]>0)).mean(), " of those share >=1 number:", q.filter((pl.col("l1")>0)&(pl.col("l2")>0))["inter"].gt(0).mean())
print("exact name match (lower):", (q["n1"].str.to_lowercase()==q["n2"].str.to_lowercase()).mean())
print("by src/country share-number rate:")
print(q.filter((pl.col("l1")>0)&(pl.col("l2")>0)).group_by("src","country").agg((pl.col("inter")>0).mean()).sort("src","country"))
# PIN / ZIP presence
print("S1 India with 6-digit PIN:", s1.filter(pl.col("country")=="India")["business_address"].str.contains(r"\b\d{6}\b").mean())
print("S1 US with 5-digit ZIP:", s1.filter(pl.col("country")=="US")["business_address"].str.contains(r"\b\d{5}\b").mean())
q.sample(200000, seed=0).write_parquet(D+"pair_sample.parquet")
