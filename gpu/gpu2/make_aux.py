"""Local side: export the files the A100 experiments need (run on the machine with the local pipeline)."""
import json
import shutil

import polars as pl

W = "C:/ML/Challenges/amazon/work/"
D = W + "data/"
A = W + "gpu2/aux/"
s1 = pl.read_parquet(D + "train_s1_norm.parquet", columns=["rid", "country"])
folds = s1.select(pl.col("rid").cast(pl.UInt32), fold=(pl.col("rid").hash(seed=11) % 2).cast(pl.Int8), country="country")
folds.write_parquet(A + "folds.parquet")
pl.read_parquet(D + "keep_v11.parquet").select(pl.col("rid").cast(pl.UInt32)).write_parquet(A + "keep.parquet")
tr = pl.read_parquet(D + "train_oof_final_v12_twin.parquet")
chk = tr.select("rid", "fold").unique().join(folds, on="rid")
assert (chk["fold"] == chk["fold_right"]).all(), "fold mismatch"
tr.select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32), pl.col("fold").cast(pl.Int8), "country",
          pl.col("p1").cast(pl.Float32), pl.col("p2").cast(pl.Float32), pl.col("pf").cast(pl.Float32)).write_parquet(A + "cand_train.parquet")
te = pl.read_parquet(D + "test_final_v12_ce_cectx_twin.parquet")
te.select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32), "country", pl.col("p1").cast(pl.Float32),
          pl.col("p2").cast(pl.Float32), pl.col("pf").cast(pl.Float32)).write_parquet(A + "cand_test.parquet")
rule = json.load(open(W + "models_ce/multi_meta_v12_ce_cectx_twin.json"))["rule"]
json.dump({"mu": rule[1], "c": rule[2]}, open(A + "rule.json", "w"))
shutil.copy(W + "gpu/aux/expected_ids.json", A + "expected_ids.json")
print("aux ready:", {k: pl.read_parquet(A + k).height for k in ("folds.parquet", "keep.parquet", "cand_train.parquet", "cand_test.parquet")}, rule)
