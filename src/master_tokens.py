"""Do true records share "master" tokens (address / name tokens absent from the S1 record) with the entity's OTHER
true records, and do decoys / other entities' records not? Candidate set of v12 (harder split)."""
import re
import sys
from collections import Counter

import polars as pl

D = "C:/ML/Challenges/amazon/work/data/"


def toks(s):
    return {t for t in re.split(r"[^a-z0-9]+", s.lower()) if len(t) >= 2}


def main(c="US", n_s1=40000):
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    K = pl.read_parquet(f"{D}train_oof_final_v12_twin.parquet").filter(pl.col("country") == c)
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").filter(pl.col("country") == c).select(owner="rid", oid="oid")
    K = K.join(gt, on="oid", how="left")
    rids = K.select("rid").unique().sample(n_s1, seed=0)
    K = K.join(rids, on="rid", how="semi")
    s1 = pl.read_parquet(f"{D}train_source1.parquet").with_row_index("rid").select("rid", n1="business_name", a1="business_address")
    o = pl.concat([pl.read_parquet(f"{D}train_source{k}.parquet") for k in (2, 3)]).with_row_index("oid").select(
        "oid", n2="business_name", a2="business_address")
    tru = gt.join(rids.rename({"rid": "owner"}), on="owner", how="semi").join(o, on="oid")  # all true records of sampled S1
    K = K.join(s1, on="rid").join(o, on="oid")
    sib = {}
    for owner, oid, n2, a2 in tru.select("owner", "oid", "n2", "a2").iter_rows():
        sib.setdefault(owner, []).append((oid, toks(a2), toks(n2)))
    st = Counter()
    for r in K.iter_rows(named=True):
        own_a, own_n = toks(r["a1"]), toks(r["n1"])
        ra, rn = toks(r["a2"]) - own_a, toks(r["n2"]) - own_n       # record tokens not explained by the S1
        others = [(sa - own_a, sn - own_n) for oid, sa, sn in sib.get(r["rid"], []) if oid != r["oid"]]
        if not others:
            continue
        sh_a = any(ra & sa for sa, sn in others)
        sh_n = any(rn & sn for sa, sn in others)
        kind = "true match" if r["owner"] == r["rid"] else ("other entity's record" if r["owner"] is not None else "unowned (decoy)")
        band = "uncertain (0.02<p<0.98)" if 0.02 < r["pf"] < 0.98 else "confident"
        for k in (kind, f"{kind} | {band}"):
            st[(k, "n")] += 1
            st[(k, "a")] += sh_a
            st[(k, "n_")] += sh_n
            st[(k, "any")] += sh_a or sh_n
            st[(k, "has_extra")] += bool(ra or rn)
    print(f"\n=== {c}: candidate pairs of {n_s1:,} S1 - share of a non-S1 token with the entity's OTHER true records")
    print(f"{'candidate kind':<58s} {'pairs':>8s} {'has non-S1 tok':>14s} {'addr shared':>11s} {'name shared':>11s} {'any':>6s}")
    for k in sorted({k for k, _ in st}):
        n = st[(k, "n")]
        print(f"{k:<58s} {n:>8,} {st[(k, 'has_extra')] / n:14.3f} {st[(k, 'a')] / n:11.3f} {st[(k, 'n_')] / n:11.3f} {st[(k, 'any')] / n:6.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
