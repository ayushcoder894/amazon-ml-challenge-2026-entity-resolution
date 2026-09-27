"""Astra hypothesis #3: do records of the same entity in the same source share a surface style that records of a
same-name OTHER entity (namesake) do not? Raw strings, no normalisation.

For entities that have >= 2 records in one source (S2 or S3), compare a pair of their records with a pair
(that record, a record of a namesake entity - same canonical name - in the same source).
Style features of a name string: case pattern, double spaces, brackets/parentheses, legal form position (first/last
token), trailing punctuation, accents present, leetspeak, web handle.
Output: P(same feature value) for same-entity pairs vs namesake pairs. A real ownership signal needs a clear gap.
"""
import random
import re
import sys
from collections import Counter

import polars as pl

D = "C:/ML/Challenges/amazon/work/data/"
LEGAL = {"llc", "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "pvt", "private", "plc",
         "llp", "lp", "pc", "pllc", "l.l.c.", "p.c."}


def case(s):
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return "none"
    if all(c.isupper() for c in letters):
        return "UPPER"
    if all(c.islower() for c in letters):
        return "lower"
    return "Mixed"


def legal_pos(s):
    t = [x.strip(".,()[]") for x in s.lower().split()]
    if not t:
        return "none"
    if t[0] in LEGAL:
        return "first"
    if t[-1] in LEGAL:
        return "last"
    return "middle" if any(x in LEGAL for x in t) else "none"


def feats(s):
    return {"case": case(s), "double_space": "  " in s, "brackets": bool(re.search(r"[\[\]()]", s)),
            "legal_pos": legal_pos(s), "trail_punct": bool(re.search(r"[.,;:]$", s.strip())),
            "accent": bool(re.search(r"[À-ÿ]", s)), "leet": bool(re.search(r"[a-z][0-9][a-z]", s.lower())),
            "handle": bool(re.search(r"@|#|www|\.com", s.lower()))}


def main(c="US", n=40000, seed=0):
    random.seed(seed)
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").filter(pl.col("country") == c)
    o = pl.concat([pl.read_parquet(f"{D}train_source{k}.parquet", columns=["business_name", "business_address"])
                   .with_columns(src=pl.lit(k)) for k in (2, 3)]).with_row_index("oid")
    s1k = pl.read_parquet(f"{D}p1_canon_train_{c}_s1.parquet", columns=["idx", "cnk"]).rename({"idx": "rid"})
    x = gt.join(o, on="oid").join(s1k, on="rid").filter(pl.col("cnk") != "")
    # namesake entities: canonical-name groups with >= 2 entities
    grp = x.group_by("cnk").agg(nent=pl.col("rid").n_unique()).filter(pl.col("nent") >= 2)
    x = x.join(grp, on="cnk")
    by_ent = {}
    for rid, src, name, addr, cnk in x.select("rid", "src", "business_name", "business_address", "cnk").iter_rows():
        by_ent.setdefault((rid, src), []).append((name, addr))
    by_name = {}
    for (rid, src), recs in by_ent.items():
        by_name.setdefault(cnk_of := None, None)
    ents_by_cnk_src = {}
    for rid, src, cnk in x.select("rid", "src", "cnk").unique().iter_rows():
        ents_by_cnk_src.setdefault((cnk, src), []).append(rid)
    cnk_of = dict(x.select("rid", "cnk").unique().iter_rows())
    keys = [k for k, v in by_ent.items() if len(v) >= 2]
    random.shuffle(keys)
    same, diff, used = Counter(), Counter(), 0
    also_empty = Counter()
    for rid, src in keys:
        rivals = [r for r in ents_by_cnk_src.get((cnk_of[rid], src), []) if r != rid and len(by_ent.get((r, src), [])) >= 1]
        if not rivals:
            continue
        a, b = random.sample(by_ent[(rid, src)], 2)
        r = random.choice(by_ent[(random.choice(rivals), src)])
        fa, fb, fr = feats(a[0]), feats(b[0]), feats(r[0])
        for k in fa:
            same[k] += fa[k] == fb[k]
            diff[k] += fa[k] == fr[k]
        used += 1
        if used >= n:
            break
    print(f"\n=== {c} source-matched pairs: {used:,} entities with a namesake in the same source")
    print(f"{'style feature':<14s} {'same entity':>12s} {'namesake':>10s} {'gap':>7s}")
    for k in same:
        print(f"{k:<14s} {same[k] / used:12.3f} {diff[k] / used:10.3f} {(same[k] - diff[k]) / used:+7.3f}")


if __name__ == "__main__":
    for c in sys.argv[1:] or ["US", "India"]:
        main(c)
