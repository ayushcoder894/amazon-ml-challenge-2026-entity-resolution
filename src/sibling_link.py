"""Potential of sibling expansion: for true pairs missing from the candidate set, does the record share a RARE token
(name or address, not present in the S1 record) with another true record of the same entity that IS a candidate?
Control: the same statistic for decoy / other records paired with that entity's candidate siblings."""
import re
import sys
from collections import Counter

import polars as pl

D = "C:/ML/Challenges/amazon/work/data/"


def toks(s):
    return {t for t in re.split(r"[^a-z0-9]+", s.lower()) if len(t) >= 3}


def main(c, rare=30, n=40000):
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").filter(pl.col("country") == c).join(keep, on="rid", how="semi").select("rid", "oid")
    K = pl.read_parquet(f"{D}train_oof_final_v12_twin.parquet", columns=["rid", "oid", "pf"])
    miss = gt.join(K, on=["rid", "oid"], how="anti").sample(n, seed=0)
    found = gt.join(K.filter(pl.col("pf") >= 0.5), on=["rid", "oid"], how="semi")       # confidently found siblings
    s1 = pl.read_parquet(f"{D}train_source1.parquet").with_row_index("rid").filter(pl.col("country") == c).select("rid", n1="business_name", a1="business_address")
    o = pl.concat([pl.read_parquet(f"{D}train_source{k}.parquet") for k in (2, 3)]).with_row_index("oid").filter(pl.col("country") == c).select("oid", n2="business_name", a2="business_address")
    df = Counter(t for a, b in zip(o["n2"].to_list(), o["a2"].to_list()) for t in toks(a) | toks(b))
    txt = {r: toks(a) | toks(b) for r, a, b in s1.iter_rows()}
    rec = {r: toks(a) | toks(b) for r, a, b in o.iter_rows()}
    sib = {}
    for rid, oid in found.iter_rows():
        sib.setdefault(rid, []).append(oid)
    st = Counter()
    for rid, oid in miss.iter_rows():
        extra = {t for t in rec[oid] - txt[rid] if df[t] <= rare}
        link = any(extra & (rec[s] - txt[rid]) for s in sib.get(rid, []))
        st["missed"] += 1
        st["has rare non-S1 token"] += bool(extra)
        st["linked to a found sibling by a rare non-S1 token"] += link
    # control: random other records (not of this entity) vs the entity's found siblings
    others = o["oid"].sample(n, seed=1).to_list()
    for (rid, _), oid in zip(miss.iter_rows(), others):
        extra = {t for t in rec[oid] - txt[rid] if df[t] <= rare}
        st["control: random record linked"] += any(extra & (rec[s] - txt[rid]) for s in sib.get(rid, []))
    print(f"\n=== {c}: {st['missed']:,} sampled missed true pairs (rare = token in <= {rare} records)")
    for k, v in st.items():
        print(f"  {k:<52s} {v / st['missed']:.4f}")


if __name__ == "__main__":
    for c in sys.argv[1:] or ["India", "US"]:
        main(c)
