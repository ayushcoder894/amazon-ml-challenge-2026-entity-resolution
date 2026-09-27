"""Do decoys come in clusters? For near-number candidates (same canonical name + street words, different number):
how many OTHER records carry the same canonical name, street words AND the candidate's number - for true matches
(typos) vs decoys vs other entities' records."""
import sys

import polars as pl

from near_number import tag

D = "C:/ML/Challenges/amazon/work/data/"


def main(c="US"):
    K = pl.read_parquet(f"{D}train_oof_final_v12_twin.parquet", columns=["rid", "oid", "y", "pf", "country"]).filter(pl.col("country") == c)
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").filter(pl.col("country") == c).select(owner="rid", oid="oid")
    x = tag(K.select("rid", "oid", "y", "pf"), "train", c).filter(pl.col("near"))
    o = pl.read_parquet(f"{D}p1_canon_train_{c}_o.parquet", columns=["idx", "cnk", "cak"])
    w = lambda e: e.str.split(" ").list.eval(pl.element().filter(~pl.element().str.contains(r"\d"))).list.join(" ")
    n = lambda e: e.str.split(" ").list.eval(pl.element().filter(pl.element().str.contains(r"\d"))).list.join(" ")
    o = o.with_columns(w=w(pl.col("cak")), n=n(pl.col("cak")))
    grp = o.group_by("cnk", "w", "n").agg(g=pl.len())           # records with identical name, street words, numbers
    x = (x.join(o.select(oid="idx", cnk="cnk", w="w", n="n"), on="oid").join(grp, on=["cnk", "w", "n"], how="left")
         .with_columns(mates=pl.col("g") - 1).join(gt, on="oid", how="left")
         .with_columns(kind=pl.when(pl.col("y") == 1).then(pl.lit("true match (typo)"))
                       .when(pl.col("owner").is_null()).then(pl.lit("unowned decoy")).otherwise(pl.lit("other entity's record"))))
    print(f"\n=== {c}: near-number candidates by number of cluster-mates (other records with same name+street+number)")
    print(x.group_by("kind", pl.col("mates").clip(0, 3).alias("mates")).agg(n=pl.len(), mean_p=pl.col("pf").mean().round(3))
          .with_columns(share=(pl.col("n") / pl.col("n").sum().over("kind")).round(3)).sort("kind", "mates"))
    print(x.group_by(pl.col("mates").clip(0, 3).alias("mates")).agg(n=pl.len(), true_rate=pl.col("y").mean().round(3),
                                                                    mean_p=pl.col("pf").mean().round(3)).sort("mates"))


if __name__ == "__main__":
    main(*sys.argv[1:2])
