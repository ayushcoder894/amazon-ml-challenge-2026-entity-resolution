import numpy as np
import polars as pl
D = "C:/ML/Challenges/amazon/work/data/"; DN = "D:/amazon_er/dense/"; OUT = "C:/ML/Challenges/amazon/work/output/"
rd = dict(separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
s1 = pl.read_parquet(f"{D}test_source1.parquet", columns=["entity_id", "business_name", "business_address", "country"]).with_row_index("rid")
o = pl.concat([pl.read_parquet(f"{D}test_source{k}.parquet", columns=["entity_id", "country"]).with_columns(src=pl.lit(f"S{k}")) for k in (2, 3)]).with_row_index("oid")
fin = pl.read_csv(f"{OUT}matching_results_probeA2.tsv", **rd).filter(pl.col("matched_entity_ids") != "").with_columns(m=pl.col("matched_entity_ids").str.split(",")).explode("m")
P = fin.join(s1.select("rid", source1_entity_id="entity_id", country="country"), on="source1_entity_id").join(o.select("oid", m="entity_id", src="src"), on="m")
T = pl.read_parquet(f"{DN}test_union_p.parquet")
conf = P.join(T.filter(pl.col("p") >= 0.995).select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32)), on=["rid", "oid"], how="semi") if False else P
print("== 1. leak check on test (confident pairs p>=0.995)")
x = P.select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32), "source1_entity_id", "m").join(T.select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32), "p"), on=["rid", "oid"]).filter(pl.col("p") >= 0.995)
x = x.with_columns(n1=pl.col("source1_entity_id").str.slice(3).cast(pl.Int64), n2=pl.col("m").str.slice(3).cast(pl.Int64))
print("  corr(S1 id, record id):", round(np.corrcoef(x["n1"], x["n2"])[0, 1], 5), " corr(S1 row, record row):", round(np.corrcoef(x["rid"].cast(pl.Int64), x["oid"].cast(pl.Int64))[0, 1], 5))
own = np.full(o.height, -1, np.int64); own[x["oid"].to_numpy()] = x["rid"].to_numpy()
a, b = own[:-1], own[1:]; m = (a >= 0) & (b >= 0)
print("  consecutive record rows with the same owner:", round(float((a[m] == b[m]).mean()), 6))
print("\n== 2. matches per entity: test (A2 decisions) vs train truth")
gt = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "country"]).join(pl.read_parquet(f"{D}keep_v11.parquet"), on="rid", how="semi")
s1tr = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).join(pl.read_parquet(f"{D}keep_v11.parquet"), on="rid", how="semi")
for c in ("US", "India", "France"):
    te = s1.filter(pl.col("country") == c).select("rid").join(P.filter(pl.col("country") == c).group_by("rid").agg(k=pl.len()), on="rid", how="left").with_columns(pl.col("k").fill_null(0))
    row = {"k": [str(i) for i in range(0, 9)] + ["9+"]}
    dt = te.with_columns(kb=pl.when(pl.col("k") >= 9).then(pl.lit("9+")).otherwise(pl.col("k").cast(pl.String))).group_by("kb").len()
    out = f"  {c:<7s} test : " + " ".join(f"{kb}:{dt.filter(pl.col('kb') == kb)['len'].sum() / te.height:.3f}" for kb in row["k"])
    if c != "France":
        tr = s1tr.filter(pl.col("country") == c).select("rid").join(gt.filter(pl.col("country") == c).group_by("rid").agg(k=pl.len()), on="rid", how="left").with_columns(pl.col("k").fill_null(0))
        dr = tr.with_columns(kb=pl.when(pl.col("k") >= 9).then(pl.lit("9+")).otherwise(pl.col("k").cast(pl.String))).group_by("kb").len()
        out += f"\n  {c:<7s} train: " + " ".join(f"{kb}:{dr.filter(pl.col('kb') == kb)['len'].sum() / tr.height:.3f}" for kb in row["k"])
    print(out)
print("\n== 3. source mix per entity (share of entities by #S2/#S3 matches), test predicted")
for c in ("US", "India", "France"):
    g = P.filter(pl.col("country") == c).group_by("rid").agg(s2=(pl.col("src") == "S2").sum(), s3=(pl.col("src") == "S3").sum())
    print(f"  {c:<7s} mean S2 {g['s2'].mean():.2f}  S3 {g['s3'].mean():.2f}  max S2 {g['s2'].max()}  S3 {g['s3'].max()}  share with 0 S2: {(g['s2'] == 0).mean():.3f}, 0 S3: {(g['s3'] == 0).mean():.3f}")
tro = pl.concat([pl.read_parquet(f"{D}train_source{k}.parquet", columns=["entity_id"]).with_columns(src=pl.lit(f"S{k}")) for k in (2, 3)]).with_row_index("oid").with_columns(pl.col("oid").cast(pl.UInt32))
gg = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid", "country"]).join(tro.select("oid", "src"), on="oid").group_by("rid", "country").agg(s2=(pl.col("src") == "S2").sum(), s3=(pl.col("src") == "S3").sum())
for c in ("US", "India"):
    g = gg.filter(pl.col("country") == c)
    print(f"  {c:<7s} TRAIN truth mean S2 {g['s2'].mean():.2f}  S3 {g['s3'].mean():.2f}  max S2 {g['s2'].max()}  S3 {g['s3'].max()}  share with 0 S2: {(g['s2'] == 0).mean():.3f}, 0 S3: {(g['s3'] == 0).mean():.3f}")
print("\n== 4. duplicate S1 entities (same name + same address) in test")
d = s1.group_by("country", pl.col("business_name").str.to_lowercase(), pl.col("business_address").str.to_lowercase()).len().filter(pl.col("len") > 1)
print(d.group_by("country").agg(groups=pl.len(), entities=pl.col("len").sum()))
