"""Group ('twin') features for every candidate pair: compare the record with the S1's other candidates."""
import sys

import polars as pl

from twin_check import group_features

D = "C:/ML/Challenges/amazon/work/data/"


def main(split, tag):
    src = f"{D}train_oof_ce{tag}.parquet" if split == "train" else f"{D}test_pred_ce{tag}.parquet"
    K = pl.read_parquet(src, columns=["rid", "oid", "country", "p3"])
    o = pl.concat([pl.read_parquet(f"{D}{split}_s{k}_norm.parquet", columns=["core", "at"]) for k in (2, 3)]).with_row_index("oid")
    o = o.join(K.select("oid").unique(), on="oid", how="semi").select(
        "oid", cs=pl.col("core").list.join(" "), ac=pl.col("at").list.join(" "))
    G = pl.concat([group_features(K.filter(pl.col("country") == c).select("rid", "oid", "p3"), o)
                   for c in sorted(K["country"].unique().to_list())])
    G.write_parquet(f"{D}twin_{split}{tag}.parquet")
    print(split, G.height, "rows")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
