"""Per-entity match rates by pattern: India/US train TRUTH (harder split, union candidates) vs France v21 ACCEPTED (test).
If the generator uses the same per-entity rates in every country, big gaps flag France misses (+) or false positives (-)."""
import polars as pl
from france_cats import canon, category

D = "C:/ML/Challenges/amazon/work/data/"
DN = "D:/amazon_er/dense/"
N = "D:/amazon_er/france_night/data/"


def rates(P, split, c, n_s1, n_sample, col):
    s1, o, vocab = canon(split, c)
    x = P.sample(min(n_sample, P.height), seed=0)
    f = P.height / x.height / n_s1
    x = (x.join(s1.select(pl.col("idx").cast(pl.UInt32).alias("i1"), cn1="cn", cns1="cns", cak1="cak", na="na"), on="i1")
         .join(o.select(pl.col("idx").cast(pl.UInt32).alias("i2"), cn2="cn", cns2="cns", cak2="cak", ae2="aempty", csize="csize"), on="i2"))
    cats = [category(r, vocab) for r in x.iter_rows(named=True)]
    x = x.with_columns(name=pl.Series([k[0] for k in cats]), addr=pl.Series([k[1] for k in cats]), clus=pl.Series([k[2] for k in cats]))
    return x.group_by("name", "addr", "clus").agg(**{f"cand_{col}": pl.len() * f, f"pos_{col}": pl.col(col).sum() * f})


keep = pl.read_parquet(f"{D}keep_v11.parquet")
tabs = []
for c in ("India", "US"):
    U = pl.read_parquet(f"{DN}train_joint9.parquet", columns=["rid", "oid", "y", "country"]).filter(pl.col("country") == c)
    n = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == c).join(keep, on="rid", how="semi").height
    tabs.append(rates(U.select(i1=pl.col("rid").cast(pl.UInt32), i2=pl.col("oid").cast(pl.UInt32), y=pl.col("y").cast(pl.Float64)), "train", c, n, 400000, "y").rename(
        {"cand_y": f"cand_{c}", "pos_y": f"true_{c}"}))
    print(c, "done", flush=True)
F = pl.read_parquet(f"{N}france_union.parquet").select(i1=pl.col("rid").cast(pl.UInt32), i2=pl.col("oid").cast(pl.UInt32))
A = pl.read_parquet(f"{N}france_v21_pairs.parquet").select(i1=pl.col("rid").cast(pl.UInt32), i2=pl.col("oid").cast(pl.UInt32), acc=pl.lit(1.0))
F = pl.concat([F, A.select("i1", "i2")]).unique().join(A, on=["i1", "i2"], how="left").with_columns(pl.col("acc").fill_null(0.0))
tf = rates(F, "test", "France", 259452, 800000, "acc").rename({"cand_acc": "cand_FR", "pos_acc": "acc_FR"})
k = ["name", "addr", "clus"]
t = tf.join(tabs[0], on=k, how="full", coalesce=True).join(tabs[1], on=k, how="full", coalesce=True).fill_null(0.0)
t = t.with_columns(true_avg=(pl.col("true_India") + pl.col("true_US")) / 2).with_columns(gap=pl.col("true_avg") - pl.col("acc_FR"))
t.write_parquet(f"{N}fr_rates_v21.parquet")
cols = ["name", "addr", "clus", "cand_FR", "acc_FR", "true_India", "true_US", "cand_India", "cand_US", "gap"]
with pl.Config(tbl_rows=30, tbl_width_chars=260, float_precision=4, tbl_hide_column_data_types=True, fmt_str_lengths=34):
    print("France ACCEPTS FEWER than the India/US truth rate (possible misses):")
    print(t.sort("gap", descending=True).select(cols).head(22))
    print("France ACCEPTS MORE than the India/US truth rate (possible false positives):")
    print(t.sort("gap").select(cols).head(12))
print(f"total per entity: France accepted {t['acc_FR'].sum():.3f}; India true {t['true_India'].sum():.3f}; US true {t['true_US'].sum():.3f}")
