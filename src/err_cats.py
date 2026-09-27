"""Addressed FP / FN (in candidates) on the harder split by name x address category, with totals per category
(candidates, true, accepted) so the fixable buckets show."""
import polars as pl
import decide
from france_rules import categorise
D = "C:/ML/Challenges/amazon/work/data/"
DN = "D:/amazon_er/dense/"
pl.Config.set_tbl_rows(30); pl.Config.set_tbl_width_chars(220); pl.Config.set_fmt_str_lengths(45)
keep = pl.read_parquet(f"{D}keep_v11.parquet")
for c in ("India", "US"):
    s1c = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == c).select("rid").join(keep, on="rid", how="semi")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid"]).join(s1c, on="rid", how="semi").with_columns(y=pl.lit(1, pl.Int8))
    U = pl.read_parquet(f"{DN}train_joint9.parquet", columns=["rid", "oid", "pj", "country"]).filter(pl.col("country") == c).select("rid", "oid", p="pj")
    pred = decide.ef_predict(U.select("rid", "oid", p2="p"), "p2", 0.0, 1.0, 0.05).with_columns(acc=pl.lit(1, pl.Int8))
    ae = pl.read_parquet(f"{D}p1_canon_train_{c}_o.parquet", columns=["idx", "aempty"]).rename({"idx": "oid"})
    X = U.join(ae.filter(~pl.col("aempty")).select("oid"), on="oid", how="semi").join(gt, on=["rid", "oid"], how="left").join(pred, on=["rid", "oid"], how="left").with_columns(pl.col("y", "acc").fill_null(0))
    err = X.filter((pl.col("y") != pl.col("acc")))
    # categorise errors fully, plus a sample of all candidates with p >= 0.02 for totals
    E = categorise(err, "train", c).drop("y", "p", strict=False).join(err, on=["rid", "oid"])
    A = X.filter(pl.col("p") >= 0.02)
    Asm = A.sample(min(600000, A.height), seed=0)
    As = categorise(Asm, "train", c).drop("y", "p", strict=False).join(Asm, on=["rid", "oid"])
    f = A.height / As.height
    tot = As.group_by("name", "addr").agg(cands=pl.len() * f, true=pl.col("y").sum() * f, acc=pl.col("acc").sum() * f)
    e = E.group_by("name", "addr").agg(FP=((pl.col("acc") == 1) & (pl.col("y") == 0)).sum(), FN=((pl.col("acc") == 0) & (pl.col("y") == 1)).sum(),
                                       mean_p_FP=pl.col("p").filter(pl.col("acc") == 1).mean(), mean_p_FN=pl.col("p").filter(pl.col("acc") == 0).mean())
    t = e.join(tot, on=["name", "addr"], how="left").with_columns(err=pl.col("FP") + pl.col("FN")).sort("err", descending=True)
    print(f"\n==== {c}: addressed errors FP {E.filter(pl.col('acc') == 1).height:,} FN {E.filter(pl.col('acc') == 0).height:,}")
    print(t.head(22).with_columns(pl.col("cands", "true", "acc").round(0), pl.col("mean_p_FP", "mean_p_FN").round(3)))
    E.write_parquet(f"D:/amazon_er/france_night/data/err_addressed_{c}.parquet")
