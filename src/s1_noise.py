"""Is S1 itself noisy? For entities with >= 3 true records: how often do the records agree on a house number / name
token / street token that S1 does not carry (S1 is the odd one out), and what does the current model do there?"""
import re
import sys
from collections import Counter

import polars as pl

D = "C:/ML/Challenges/amazon/work/data/"


def first_num(a):
    m = re.search(r"\b(\d+)\b", re.sub(r"(?i)\b(unit|apt|apartment|suite|ste|pmb|po box|box|fl|floor)\b[^,]*", " ", a))
    return m.group(1).lstrip("0") if m else None


def main(c="US"):
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").filter(pl.col("country") == c).join(keep, on="rid", how="semi")
    K = pl.read_parquet(f"{D}train_oof_final_v12_twin.parquet", columns=["rid", "oid", "pf"])
    s1 = pl.read_parquet(f"{D}train_source1.parquet").with_row_index("rid").select("rid", a1="business_address")
    o = pl.concat([pl.read_parquet(f"{D}train_source{k}.parquet") for k in (2, 3)]).with_row_index("oid").select("oid", a2="business_address")
    x = gt.join(s1, on="rid").join(o, on="oid").join(K, on=["rid", "oid"], how="left")
    st = Counter()
    for rid, g in x.group_by("rid"):
        rows = g.to_dicts()
        if len(rows) < 3:
            continue
        n1 = first_num(rows[0]["a1"])
        nums = [first_num(r["a2"]) for r in rows]
        cnt = Counter(n for n in nums if n)
        if not n1 or not cnt:
            continue
        top, k = cnt.most_common(1)[0]
        st["entities"] += 1
        if top != n1 and k >= 2 and k > cnt.get(n1, 0):
            st["records agree on a number S1 does not have (S1 odd one out)"] += 1
            for r, n in zip(rows, nums):
                if n == top:
                    st["  their true records"] += 1
                    p = r["pf"]
                    st["  ... predicted (pf>=0.5)"] += p is not None and p >= 0.5
                    st["  ... not even a candidate"] += p is None
        if cnt.get(n1, 0) >= 1:
            st["S1 number carried by >=1 record"] += 1
    n = st["entities"]
    print(f"\n=== {c}: {n:,} entities with >= 3 true records and a house number")
    for k, v in st.items():
        print(f"  {k:<64s} {v:>9,}" + (f"  ({v / n:.4f} of entities)" if not k.startswith("  ") else ""))


if __name__ == "__main__":
    main(*sys.argv[1:2])
