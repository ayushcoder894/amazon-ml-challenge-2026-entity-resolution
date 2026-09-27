import polars as pl
import features as F
D = F.D
x = pl.read_parquet(f"{D}gt_signal.parquet")
full = pl.read_parquet(f"{D}train_cand.parquet", columns=["rid", "oid"]).with_columns(in40=pl.lit(1, pl.UInt8))
x = x.join(full, on=["rid", "oid"], how="left").with_columns(pl.col("in40").fill_null(0))
m = x.filter((pl.col("inc") == 0) & pl.col("name_ok") & pl.col("addr_ok"))
print(f"missed name+addr pairs: {m.height:,}; of which present in full top-40 blocking: {m['in40'].mean():.3f}")
# name ambiguity for name-only / empty-address pairs: how many S1s share the true S1's core name (sorted tokens)?
coll = []
for c in ["India", "US"]:
    s1 = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country", "core"]).filter(pl.col("country") == c)
    s1 = s1.with_columns(k=pl.col("core").list.sort().list.join(" "))
    s1 = s1.with_columns(n_same=pl.len().over("k"))
    coll.append(s1.select("rid", "n_same"))
coll = pl.concat(coll)
x = x.join(coll, on="rid")
e = x.filter(~pl.col("addr_ok"))
print("\nname-only pairs (no usable address): S1 core-name collision count")
print(e.with_columns(b=pl.when(pl.col("n_same") == 1).then(pl.lit("1 unique")).when(pl.col("n_same") <= 3)
                     .then(pl.lit("2-3")).when(pl.col("n_same") <= 10).then(pl.lit("4-10")).otherwise(pl.lit(">10")))
      .group_by("b").agg(pairs=pl.len(), in_cands=pl.col("inc").mean()).sort("b"))
# optimistic ceiling: pair resolvable if it has address signal, or name signal with a unique S1 name (else 1/n chance)
x = x.with_columns(res=pl.when(pl.col("addr_ok")).then(1.0).when(pl.col("name_ok")).then(1.0 / pl.col("n_same")).otherwise(0.0))
s1all = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid"])
agg = x.group_by("rid").agg(ng=pl.len(), r=pl.col("res").sum())
m2 = s1all.join(agg, on="rid", how="left").fill_null(0).with_columns(
    F=pl.when(pl.col("ng") == 0).then(1.0).when(pl.col("r") == 0).then(0.0)
    .otherwise(1.25 * (pl.col("r") / pl.col("ng")) / (0.25 + pl.col("r") / pl.col("ng"))))
print(f"\nApprox. ceiling (perfect blocking + perfect matcher, name-only pairs need a unique name): {m2['F'].mean():.4f}")
