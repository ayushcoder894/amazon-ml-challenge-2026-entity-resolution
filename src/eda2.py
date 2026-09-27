import polars as pl
D = r"C:\ML\Challenges\amazon\work\data\\"
s1 = pl.read_parquet(D+"train_source1.parquet"); s2 = pl.read_parquet(D+"train_source2.parquet"); s3 = pl.read_parquet(D+"train_source3.parquet")
pairs = pl.read_parquet(D+"train_pairs.parquet")
allo = pl.concat([s2,s3])
ids = s1.filter(pl.col("country")=="US").sample(4, seed=1)["entity_id"].to_list() + s1.filter(pl.col("country")=="India").sample(5, seed=2)["entity_id"].to_list()
for i in ids:
    r = s1.filter(pl.col("entity_id")==i).row(0)
    print(f"\n### {r[0]} | {r[1]} | {r[2]}")
    for m in pairs.filter(pl.col("s1")==i)["m"].to_list():
        x = allo.filter(pl.col("entity_id")==m).row(0); print(f"   {x[0][:2]} | {x[1]} | {x[2]}")
print("\n\n===== UNMATCHED S2/S3 samples")
un = allo.join(pairs.select(pl.col("m").alias("entity_id")), on="entity_id", how="anti")
for x in un.sample(15, seed=3).rows(): print(f"   {x[0][:2]} | {x[3]} | {x[1]} | {x[2]}")
