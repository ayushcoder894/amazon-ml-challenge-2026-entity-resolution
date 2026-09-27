"""Phase 1.1: ground-truth structure per country (train).

#S1, singleton rate, distribution of #matches per S1, S2 vs S3 share of matches, and whether any S2/S3 record appears
in more than one S1's match list (one-owner constraint). Also: share of S2/S3 records owned by no S1.
"""
import polars as pl

D = "C:/ML/Challenges/amazon/work/data/"
RES = "C:/ML/Challenges/amazon/6ab10eb3b23ba_student_resource/student_resource/dataset/"


def load_gt():
    gt = pl.read_csv(f"{RES}train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False,
                     missing_utf8_is_empty_string=True)
    return gt


def main():
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(200)
    gt = load_gt()
    s1 = pl.read_parquet(f"{D}train_source1.parquet", columns=["entity_id", "country"])
    o = pl.concat([pl.read_parquet(f"{D}train_source{k}.parquet", columns=["entity_id", "country"]) for k in (2, 3)])
    print(f"ground-truth rows {gt.height:,}; S1 rows {s1.height:,}; S1 ids missing from GT: "
          f"{s1.join(gt, left_on='entity_id', right_on='source1_entity_id', how='anti').height:,}")
    pairs = (gt.with_columns(m=pl.col("matched_entity_ids").str.split(",")).explode("m")
             .filter(pl.col("m") != "").select(s1="source1_entity_id", m="m"))
    n = (s1.join(pairs.group_by("s1").agg(k=pl.len()), left_on="entity_id", right_on="s1", how="left")
         .with_columns(pl.col("k").fill_null(0)))
    print("\n#S1, singleton rate, mean / median / max matches per S1")
    print(n.group_by("country").agg(n_s1=pl.len(), singleton=(pl.col("k") == 0).mean().round(4),
                                    mean=pl.col("k").mean().round(3), median=pl.col("k").median(),
                                    p95=pl.col("k").quantile(0.95), max=pl.col("k").max()).sort("country"))
    print("\ndistribution of #matches per S1 (share of S1)")
    dist = n.with_columns(b=pl.col("k").clip(0, 10)).group_by("country", "b").len().with_columns(
        share=(pl.col("len") / pl.col("len").sum().over("country")).round(4))
    print(dist.pivot(on="country", index="b", values="share").sort("b"))
    pc = pairs.join(s1, left_on="s1", right_on="entity_id")
    print("\nS2 vs S3 share of true matches")
    print(pc.with_columns(src=pl.col("m").str.slice(0, 2)).group_by("country", "src").len()
          .with_columns(share=(pl.col("len") / pl.col("len").sum().over("country")).round(4)).sort("country", "src"))
    multi = pairs.group_by("m").agg(n=pl.col("s1").n_unique()).filter(pl.col("n") > 1)
    print(f"\none-owner check: S2/S3 ids listed under more than one S1: {multi.height:,} "
          f"(of {pairs['m'].n_unique():,} matched ids)")
    cross = pc.join(o.rename({"country": "c2"}), left_on="m", right_on="entity_id").filter(pl.col("country") != pl.col("c2"))
    print(f"matches whose record country differs from the S1 country: {cross.height:,}")
    print("\nS2/S3 records owned by no S1 (decoys / other businesses)")
    print(o.with_columns(owned=pl.col("entity_id").is_in(pairs["m"].implode()), src=pl.col("entity_id").str.slice(0, 2))
          .group_by("country", "src").agg(n=pl.len(), unowned=(~pl.col("owned")).mean().round(4)).sort("country", "src"))


if __name__ == "__main__":
    main()
