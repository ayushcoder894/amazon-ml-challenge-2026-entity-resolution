"""Canonical keys for every Source-1 entity and Source-2/3 record, per split and country (Phase 1 reuse).

data/p1_canon_{split}_{country}_{s1|o}.parquet: idx (rid for S1 / oid for records, as in train_gt_rid), entity_id,
cn (name tokens), cnk (sorted name key), cns (no-space name), cak (address key), aempty.
"""
import sys
import time

import polars as pl

import normalize as N

D = "C:/ML/Challenges/amazon/work/data/"


def main(split):
    t = time.time()
    s1 = pl.read_parquet(f"{D}{split}_source1.parquet").with_row_index("idx")
    o = pl.concat([pl.read_parquet(f"{D}{split}_source{k}.parquet") for k in (2, 3)]).with_row_index("idx")
    for c in sorted(s1["country"].unique().to_list()):
        a1 = s1.filter(pl.col("country") == c)
        admin = N.admin_for(c, a1["business_address"])
        for name, df in (("s1", a1), ("o", o.filter(pl.col("country") == c))):
            out = N.add_canon(df, c, admin).select("idx", "entity_id", "cn", "cnk", "cns", "cak", "aempty")
            out.write_parquet(f"{D}p1_canon_{split}_{c}_{name}.parquet")
            print(f"{split} {c} {name}: {out.height:,} rows, empty name key {(out['cnk'] == '').mean():.4f}, "
                  f"empty addr key {(out['cak'] == '').mean():.4f} ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    for sp in sys.argv[1:] or ["train", "test"]:
        main(sp)
