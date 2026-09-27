"""Ground-truth pair tables from the raw training files.

train_pairs.parquet   (s1 entity_id, m entity_id, src)
train_gt_rid.parquet  (rid, oid, src, country) with rid = row in train_source1, oid = row in concat(source2, source3)
"""
import polars as pl

D = "C:/ML/Challenges/amazon/work/data/"


def main():
    gt = pl.read_parquet(f"{D}train_ground_truth.parquet")
    pairs = (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
             .filter(pl.col("matched_entity_ids") != "")
             .select(s1="source1_entity_id", m="matched_entity_ids").with_columns(src=pl.col("m").str.slice(0, 2)))
    pairs.write_parquet(f"{D}train_pairs.parquet")
    s1 = pl.read_parquet(f"{D}train_source1.parquet", columns=["entity_id", "country"]).with_row_index("rid")
    o = pl.concat([pl.read_parquet(f"{D}train_source{k}.parquet", columns=["entity_id"]) for k in (2, 3)]
                  ).with_row_index("oid")
    g = (pairs.join(s1.rename({"entity_id": "s1"}), on="s1").join(o.rename({"entity_id": "m"}), on="m")
         .select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32), "src", "country"))
    g.write_parquet(f"{D}train_gt_rid.parquet")
    print(f"ground truth: {g.height:,} pairs")


if __name__ == "__main__":
    main()
