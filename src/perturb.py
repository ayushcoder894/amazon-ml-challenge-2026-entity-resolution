"""How do planted distractors differ from true matches? Perturbation signatures on uncertain pairs (v10 OOF)."""
import polars as pl

D = "C:/ML/Challenges/amazon/work/data/"
K = pl.read_parquet(f"{D}train_oof_ce_v10.parquet").filter((pl.col("p2") > 0.02) & (pl.col("p2") < 0.98))
s1 = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "core", "nums"])
o = pl.concat([pl.read_parquet(f"{D}train_s{k}_norm.parquet", columns=["core", "nums"]) for k in (2, 3)]).with_row_index("oid")
vocab = (s1.select(t=pl.col("core")).explode("t").drop_nulls().group_by("t").agg(n=pl.len()).filter(pl.col("n") >= 20))["t"]
x = K.join(s1, on="rid").join(o.rename({"core": "core2", "nums": "nums2"}), on="oid")
x = x.with_columns(
    only1=pl.col("core").list.set_difference("core2"), only2=pl.col("core2").list.set_difference("core"),
    n1=pl.col("nums").list.first(), n2=pl.col("nums2").list.first())
x = x.with_columns(
    # a word of the record that is absent from the S1 name but is itself a common real word (substitution)
    sub_realword=(pl.col("only2").list.eval(pl.element().is_in(vocab.implode())).list.any()
                  & (pl.col("only1").list.len() > 0)),
    sub_nonword=(pl.col("only2").list.eval(~pl.element().is_in(vocab.implode())).list.any()
                 & (pl.col("only1").list.len() > 0)),
    num_rel=pl.when(pl.col("n1").is_null() | pl.col("n2").is_null()).then(pl.lit("missing"))
    .when(pl.col("n1") == pl.col("n2")).then(pl.lit("equal"))
    .when(pl.col("n1").str.starts_with(pl.col("n2")) | pl.col("n2").str.starts_with(pl.col("n1"))).then(pl.lit("prefix/truncated"))
    .when((pl.col("n1").cast(pl.Int64, strict=False) - pl.col("n2").cast(pl.Int64, strict=False)).abs() <= 10).then(pl.lit("off by <=10"))
    .otherwise(pl.lit("other")))
print(f"uncertain pairs: {x.height:,}, match rate {x['y'].mean():.3f}")
for col in ["sub_realword", "sub_nonword", "num_rel"]:
    print(x.group_by(col).agg(n=pl.len(), match_rate=pl.col("y").mean(), mean_p3=pl.col("p3").mean()).sort("n", descending=True))
print(x.group_by("sub_realword", "num_rel").agg(n=pl.len(), match_rate=pl.col("y").mean(), mean_p3=pl.col("p3").mean())
      .sort("n", descending=True).head(12))
