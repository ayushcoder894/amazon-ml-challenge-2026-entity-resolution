"""Do records of the same entity share generator artifacts (long numbers, pseudo-words, garbled tokens) more often
than chance? If so, siblings link a record to its entity beyond its own name / address."""
import random
import re
import sys
from collections import Counter, defaultdict

import polars as pl

D = "C:/ML/Challenges/amazon/work/data/"


def toks(s):
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]


def main(c="US", n_ent=60000, seed=0):
    random.seed(seed)
    s1 = pl.read_parquet(f"{D}train_source1.parquet").with_row_index("rid").filter(pl.col("country") == c)
    o = pl.concat([pl.read_parquet(f"{D}train_source{k}.parquet") for k in (2, 3)]).with_row_index("oid").filter(pl.col("country") == c)
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").filter(pl.col("country") == c)
    vocab = Counter(t for n in s1["business_name"].to_list() for t in set(toks(n)))
    addr_vocab = Counter(t for a in s1["business_address"].to_list() for t in set(toks(a)))
    names = dict(zip(o["oid"].to_list(), o["business_name"].to_list()))
    addrs = dict(zip(o["oid"].to_list(), o["business_address"].to_list()))
    s1n = dict(zip(s1["rid"].to_list(), s1["business_name"].to_list()))
    s1a = dict(zip(s1["rid"].to_list(), s1["business_address"].to_list()))
    groups = gt.group_by("rid").agg(pl.col("oid")).filter(pl.col("oid").list.len() >= 2).sample(n_ent, seed=seed)

    def arts(oid, rid):
        n, a = names[oid], addrs[oid]
        own = set(toks(s1n[rid])) | set(toks(s1a[rid]))
        nt, at = toks(n), toks(a)
        return {
            "long number in name": {t for t in nt if t.isdigit() and len(t) >= 4},
            "pseudo-word in name (not in any S1 name)": {t for t in nt if t.isalpha() and vocab.get(t, 0) == 0 and len(t) >= 4},
            "garbled name token (not in own S1 name, in vocab)": {t for t in nt if t.isalpha() and t not in own and 0 < vocab.get(t, 0) <= 50},
            "address token not in own S1 address": {t for t in at if t not in own and len(t) >= 3},
            "PO box / PMB number": set(re.findall(r"(?:box|pmb)\s*#?\s*(\d+)", a.lower())),
        }
    both, same, rnd_same = Counter(), Counter(), Counter()
    all_oids = o["oid"].to_list()
    for rid, oids in groups.iter_rows():
        a, b = random.sample(oids, 2)
        A, B = arts(a, rid), arts(b, rid)
        r = random.choice(all_oids)
        R = arts(r, rid)
        for k in A:
            if A[k] and B[k]:
                both[k] += 1
                same[k] += bool(A[k] & B[k])
            if A[k] and R[k]:
                rnd_same[(k, "n")] += 1
                rnd_same[(k, "s")] += bool(A[k] & R[k])
    print(f"\n=== {c}: {groups.height:,} entities, one random sibling pair each (and a random record as control)")
    print(f"{'artifact':<52s} {'both have':>9s} {'shared':>8s} {'random pair shared':>19s}")
    for k in both:
        print(f"{k:<52s} {both[k]:>9,} {same[k] / both[k]:8.3f} {rnd_same[(k, 's')] / max(1, rnd_same[(k, 'n')]):19.4f}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
