"""Phase 1.3 (refined): the three open generator questions, answered on train with evidence.

For every (S1, record) pair that shares the canonical address key:
  invented name  -> pseudo-word (no record token occurs in any S1 name of the country) vs real other name
  word swap      -> swapped-in token is a real S1-vocabulary word vs a garbled string
  truncation     -> single-token prefix of the S1 name
  acronym        -> initials of the S1 name
each split by whether the address key belongs to one S1 only or is shared by several S1 entities.
Reported: pairs, share that are true matches of that S1, of another S1, or of nobody; plus examples.
"""
import random
import sys
from collections import Counter, defaultdict

import polars as pl

from p1_catalog import D, load_side, name_rel, pairs_frame


def refine(rel, A, B, vocab):
    if rel == "invented":
        return "invented_pseudo_word" if all(t not in vocab for t in B) else "invented_real_words"
    if rel.startswith("word_swap"):
        new = [t for t in B if t not in A]
        return "word_swap_real_word" if new and all(t in vocab for t in new) else "word_swap_garbled"
    if rel.startswith("truncation_single"):
        return "truncation_single_prefix" if A and B and B[0] == A[0] else "truncation_single_other"
    return rel


def main(c, max_s1=30, max_o=300, n_sample=400000, seed=1):
    random.seed(seed)
    cols = ["cn", "cns", "cak"]
    s1, o = load_side(c, "s1", cols), load_side(c, "o", cols)
    vocab = set(s1.select(pl.col("cn").explode()).to_series().drop_nulls().to_list())
    own = pl.read_parquet(f"{D}train_gt_rid.parquet").filter(pl.col("country") == c).select(owner="rid", oid="oid")
    k1 = s1.filter(pl.col("cak") != "").select("idx", "cak")
    k2 = o.filter(pl.col("cak") != "").select("idx", "cak")
    sz = k1.group_by("cak").agg(n1=pl.len()).join(k2.group_by("cak").agg(n2=pl.len()), on="cak").filter(
        (pl.col("n1") <= max_s1) & (pl.col("n2") <= max_o))
    P = (k1.join(sz, on="cak").rename({"idx": "rid"}).join(k2.rename({"idx": "oid"}), on="cak")
         .select("rid", "oid", shared=pl.col("n1") > 1).join(own, on="oid", how="left")
         .with_columns(status=pl.when(pl.col("owner") == pl.col("rid")).then(pl.lit("match"))
                       .when(pl.col("owner").is_not_null()).then(pl.lit("other_s1")).otherwise(pl.lit("unowned"))))
    x = pairs_frame(c, P.sample(min(n_sample, P.height), seed=seed), s1, o)
    stat, ex = defaultdict(Counter), defaultdict(list)
    for r in x.iter_rows(named=True):
        A, B = list(r["cn_1"]), list(r["cn_2"])
        rel = refine(name_rel(A, B, r["cns_1"], r["cns_2"]), A, B, vocab)
        if rel.startswith("canon_equal") or rel in ("extra_word", "truncation", "glued", "typo"):
            continue
        key = (rel, "shared addr" if r["shared"] else "unique addr")
        stat[key][r["status"]] += 1
        if len(ex[(key, r["status"])]) < 3 and random.random() < 0.3:
            ex[(key, r["status"])].append(f"{r['business_name_1']!r} -> {r['business_name_2']!r}")
    print(f"\n===== {c}: records at the S1's canonical address (sample {x.height:,} of {P.height:,} pairs) =====")
    print(f"{'name relation':<30s} {'address':<12s} {'pairs':>7s} {'match':>7s} {'otherS1':>8s} {'unowned':>8s}")
    for key in sorted(stat, key=lambda k: (k[0], k[1])):
        cnt = stat[key]
        n = sum(cnt.values())
        print(f"{key[0]:<30s} {key[1]:<12s} {n:>7,} {cnt['match'] / n:7.3f} {cnt['other_s1'] / n:8.3f} {cnt['unowned'] / n:8.3f}")
        for st in ("match", "other_s1", "unowned"):
            for e in ex[(key, st)][:2]:
                print(f"        [{st}] {e}")


if __name__ == "__main__":
    for c in sys.argv[1:] or ["US", "India"]:
        main(c)
