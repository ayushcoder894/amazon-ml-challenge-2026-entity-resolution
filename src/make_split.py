"""Alternative validation splits from existing per-fold feature files (same rows, new fold assignment).

  loco : fold 0 = US, fold 1 = India          (train on one country, score the other; both directions)
  8020 : fold 0 = 80% of S1 entities, fold 1 = 20% (hash of the S1 index, seed 11)
src '_v10' = normal training data, '_v11' = harder (18.6% of S1 entities removed, see make_drop_folds.py).
Writes data/train_feat{out}_fold{0,1}.parquet.
"""
import sys
import time

import polars as pl

D = "C:/ML/Challenges/amazon/work/data/"


def main(kind, src, out):
    t = time.time()
    lf = pl.concat([pl.scan_parquet(f"{D}train_feat{src}_fold{f}.parquet") for f in (0, 1)])
    if kind == "loco":
        cond = pl.col("country") == "US"
    elif kind == "8020":
        cond = (pl.col("rid").hash(seed=11) % 5) != 0
    else:
        raise ValueError(kind)
    for f, c in ((0, cond), (1, ~cond)):
        lf.filter(c).with_columns(fold=pl.lit(f, pl.UInt8)).sink_parquet(
            f"{D}train_feat{out}_fold{f}.parquet", row_group_size=500_000)
        n = pl.scan_parquet(f"{D}train_feat{out}_fold{f}.parquet").select(pl.len()).collect().item()
        print(f"{out} fold {f}: {n:,} rows ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
