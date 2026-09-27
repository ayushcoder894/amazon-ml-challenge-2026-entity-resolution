"""Build the experiment bundle on the GPU box from the raw challenge dataset + the shipped aux files.

Ids must line up exactly with the local pipeline: rid = row in *_source1.tsv, oid = row in
concat(*_source2.tsv, *_source3.tsv). Verified against md5 checksums of the entity_id sequences.
records_{split}.parquet: side (1 = Source 1, 2 = Source 2/3), idx, country, name, addr (raw, whitespace collapsed),
text (lowercased NFKC "name | address", for the bi-encoder).
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

import polars as pl


def read_tsv(p):
    return pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)


def clean(c):
    return pl.col(c).str.replace_all(r"\s+", " ").str.strip_chars()


def md5(ids):
    return hashlib.md5("\n".join(ids).encode()).hexdigest()


def find_split_dir(data, split):
    for cand in (os.path.join(data, split), os.path.join(data, "dataset", split), data):
        if os.path.exists(os.path.join(cand, f"{split}_source1.tsv")):
            return cand
    sys.exit(f"cannot find {split}_source1.tsv under {data} (pass --data pointing at the folder with train/ and test/)")


def records(df):
    return df.with_row_index("idx").select(
        "idx", "country", name=clean("business_name"), addr=clean("business_address"),
        text=(pl.concat_str([pl.col("business_name"), pl.lit(" | "), pl.col("business_address")])
              .str.normalize("NFKC").str.to_lowercase().str.replace_all(r"\s+", " ").str.strip_chars()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset folder containing train/ and test/")
    ap.add_argument("--aux", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "aux"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    exp = json.load(open(os.path.join(a.aux, "expected_ids.json")))
    for split in ("train", "test"):
        d = find_split_dir(a.data, split)
        s1 = read_tsv(os.path.join(d, f"{split}_source1.tsv"))
        o = pl.concat([read_tsv(os.path.join(d, f"{split}_source{k}.tsv")) for k in (2, 3)])
        e = exp[split]
        got = dict(n_s1=s1.height, n_other=o.height, md5_s1=md5(s1["entity_id"].to_list()),
                   md5_other=md5(o["entity_id"].to_list()))
        if got != e:
            sys.exit(f"[{split}] dataset does not match the local pipeline ids!\n expected {e}\n got      {got}")
        rec = pl.concat([records(s1).with_columns(side=pl.lit(1, pl.UInt8)), records(o).with_columns(side=pl.lit(2, pl.UInt8))])
        rec.select("side", "idx", "country", "name", "addr", "text").write_parquet(os.path.join(a.out, f"records_{split}.parquet"))
        print(f"[{split}] ids verified; {rec.height:,} records", flush=True)
        if split == "train":
            gt = read_tsv(os.path.join(d, "train_ground_truth.tsv"))
            pairs = (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
                     .filter(pl.col("matched_entity_ids") != "")
                     .rename({"source1_entity_id": "s1", "matched_entity_ids": "m"}))
            pairs = (pairs.join(s1.with_row_index("rid").select(s1="entity_id", rid="rid", country="country"), on="s1")
                     .join(o.with_row_index("oid").select(m="entity_id", oid="oid"), on="m")
                     .select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32), "country"))
            if pairs.height != exp["train_gt_pairs"]:
                sys.exit(f"ground truth pairs {pairs.height} != expected {exp['train_gt_pairs']}")
            pairs.write_parquet(os.path.join(a.out, "train_gt.parquet"))
            print(f"[train] ground truth: {pairs.height:,} pairs", flush=True)
    for f in ("folds.parquet", "keep.parquet", "cand_train.parquet", "cand_test.parquet", "rule.json"):
        shutil.copy(os.path.join(a.aux, f), os.path.join(a.out, f))
    print(f"bundle ready in {a.out}")


if __name__ == "__main__":
    main()
