"""Name-only records (empty address) whose canonical name is carried by 2+ candidate S1 entities: can the raw surface
(legal form, exact spelling) tell the owner apart? Train, harder split, candidate set of v12."""
import re
import sys
from collections import Counter

import polars as pl

import normalize as N

D = "C:/ML/Challenges/amazon/work/data/"


def legal(s):
    toks = re.split(r"[^a-z0-9]+", re.sub(r"(?<=\b[a-z])\.", "", s.lower()))
    return frozenset(t for t in toks if t in N.LEGAL)


def surf(s):
    return " ".join(re.split(r"[^a-z0-9&]+", s.lower())).strip()


def main(c):
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").filter(pl.col("country") == c).select(owner="rid", oid="oid")
    K = pl.read_parquet(f"{D}train_oof_final_v12_twin.parquet", columns=["rid", "oid", "y", "pf", "country"]).filter(pl.col("country") == c)
    s1 = pl.read_parquet(f"{D}p1_canon_train_{c}_s1.parquet", columns=["idx", "cnk"])
    o = pl.read_parquet(f"{D}p1_canon_train_{c}_o.parquet", columns=["idx", "cnk", "aempty"])
    raw1 = pl.read_parquet(f"{D}train_source1.parquet").with_row_index("rid").select("rid", n1="business_name")
    raw2 = pl.concat([pl.read_parquet(f"{D}train_source{k}.parquet") for k in (2, 3)]).with_row_index("oid").select("oid", n2="business_name")
    x = (K.join(s1.select(rid="idx", k1="cnk"), on="rid").join(o.select(oid="idx", k2="cnk", ae="aempty"), on="oid")
         .filter(pl.col("ae") & (pl.col("k1") == pl.col("k2")) & (pl.col("k1") != "")))
    x = x.with_columns(nc=pl.len().over("oid")).filter(pl.col("nc") >= 2).join(raw1, on="rid").join(raw2, on="oid")
    stats = Counter()
    for oid, g in x.group_by("oid"):
        rows = g.to_dicts()
        rec = rows[0]["n2"]
        lr, sr = legal(rec), surf(rec)
        owner = [r["rid"] for r in rows if r["y"] == 1]
        stats["records"] += 1
        stats["owner among candidates"] += bool(owner)
        # legal form: record's legal set equals exactly one candidate's legal set
        eq = [r["rid"] for r in rows if legal(r["n1"]) == lr]
        if lr and len(eq) == 1:
            stats["legal picks one S1"] += 1
            stats["legal picks the owner"] += bool(owner and eq[0] == owner[0])
        if len({legal(r["n1"]) for r in rows}) == 1:
            stats["all candidate S1 share one legal form"] += 1
        eqs = [r["rid"] for r in rows if surf(r["n1"]) == sr]
        if len(eqs) == 1:
            stats["exact surface picks one S1"] += 1
            stats["exact surface picks the owner"] += bool(owner and eqs[0] == owner[0])
        if len({surf(r["n1"]) for r in rows}) == 1:
            stats["all candidate S1 names identical (surface)"] += 1
    n = stats["records"]
    print(f"\n=== {c}: {n:,} name-only records with 2+ namesake candidate S1")
    for k, v in stats.items():
        print(f"  {k:<46s} {v:>8,}  ({v / n:.3f})")


if __name__ == "__main__":
    for c in sys.argv[1:] or ["India", "US"]:
        main(c)
