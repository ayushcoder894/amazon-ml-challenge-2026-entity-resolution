"""Build the upload bundle for the GPU box (dense retrieval). Everything is keyed by the same integer ids the
local pipeline uses: rid = row in *_source1, oid = row in concat(*_source2, *_source3)."""
import os
import sys

import polars as pl

D = "C:/ML/Challenges/amazon/work/data/"
OUT = sys.argv[1] if len(sys.argv) > 1 else "C:/ML/Challenges/amazon/work/gpu_bundle/"


def text_expr():
    # raw name + address (native scripts kept: the model learns transliteration), lowercased, whitespace collapsed
    return (pl.concat_str([pl.col("business_name"), pl.lit(" | "), pl.col("business_address")])
            .str.normalize("NFKC").str.to_lowercase().str.replace_all(r"\s+", " ").str.strip_chars())


def records(split):
    s1 = pl.read_parquet(f"{D}{split}_source1.parquet").with_row_index("idx").select(
        side=pl.lit(1, pl.UInt8), idx="idx", country="country", text=text_expr())
    o = pl.concat([pl.read_parquet(f"{D}{split}_source{k}.parquet") for k in (2, 3)]).with_row_index("idx").select(
        side=pl.lit(2, pl.UInt8), idx="idx", country="country", text=text_expr())
    return pl.concat([s1, o])


def main():
    os.makedirs(OUT, exist_ok=True)
    for split in ("train", "test"):
        r = records(split)
        r.write_parquet(f"{OUT}records_{split}.parquet")
        print(split, r.height, flush=True)
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").select("rid", "oid", "country")
    gt.write_parquet(f"{OUT}train_gt.parquet")
    s1 = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid"])
    s1.select("rid", fold=(pl.col("rid").hash(seed=11) % 2).cast(pl.UInt8)).write_parquet(f"{OUT}train_folds.parquet")
    # hard negatives: the 3 highest-scoring non-matching blocking candidates of each S1
    cand = pl.read_parquet(f"{D}train_cand.parquet", columns=["rid", "oid", "score"])
    hn = (cand.join(gt.select("rid", "oid"), on=["rid", "oid"], how="anti")
          .filter(pl.col("score").rank("ordinal", descending=True).over("rid") <= 3).select("rid", "oid"))
    hn.write_parquet(f"{OUT}train_hardneg.parquet")
    print("hardneg", hn.height, flush=True)
    del cand, hn
    # the exact candidate pairs the current model scores: dense cosine is computed for all of them
    for split in ("train", "test"):
        c = (pl.scan_parquet(f"{D}{split}_cand.parquet").filter((pl.col("r1") <= 10) | (pl.col("r2") <= 3))
             .select("rid", "oid").collect())
        c.write_parquet(f"{OUT}cand_{split}.parquet")
        print("cand", split, c.height, flush=True)


if __name__ == "__main__":
    main()
