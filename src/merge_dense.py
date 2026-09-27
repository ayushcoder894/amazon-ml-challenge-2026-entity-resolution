"""Merge A100 dense-retrieval output into the candidate set.

candidates = blocking (r1<=K | r2<=k)  U  dense (dr1<=Kd | dr2<=kd)
Every pair keeps its blocking scores when the pair exists anywhere in the blocking top-40 file (else zeros /
rank 999) and its dense cosine / ranks when the dense output has it (else null -> handled natively by the GBDT).
Output: data/<split>_cand_merged.parquet with an `insel` column (1 = in the model's candidate set).
"""
import sys

import polars as pl

D = "C:/ML/Challenges/amazon/work/data/"
DENSE = "C:/ML/Challenges/amazon/work/gpu_out/"


def merge(split, K=10, k=3, Kd=10, kd=3):
    blk = pl.scan_parquet(f"{D}{split}_cand.parquet")
    den = pl.scan_parquet(f"{DENSE}dense_{split}.parquet")
    sel = pl.concat([
        blk.filter((pl.col("r1") <= K) | (pl.col("r2") <= k)).select("rid", "oid"),
        den.filter((pl.col("dr1") <= Kd).fill_null(False) | (pl.col("dr2") <= kd).fill_null(False)).select("rid", "oid"),
    ]).unique()
    m = (sel.join(blk, on=["rid", "oid"], how="left")
         .join(den.select("rid", "oid", "cos", "dr1", "dr2"), on=["rid", "oid"], how="left")
         .with_columns([pl.col(c).fill_null(0.0) for c in ("score", "sn", "sa", "sx", "sz")]
                       + [pl.col("nk").fill_null(0), pl.col("r1").fill_null(999), pl.col("r2").fill_null(999),
                          pl.lit(1, pl.UInt8).alias("insel")])
         .collect())
    m.write_parquet(f"{D}{split}_cand_merged.parquet")
    n_blk = blk.filter((pl.col("r1") <= K) | (pl.col("r2") <= k)).select(pl.len()).collect().item()
    print(f"{split}: blocking set {n_blk:,} -> merged {m.height:,} pairs; cos missing for {m['cos'].null_count():,}")
    return m


if __name__ == "__main__":
    for s in sys.argv[1:] or ["train", "test"]:
        merge(s)
