"""Phase 1.3: real-word swaps at the S1's address - is the swapped-in word a generator filler (noise) or a distinctive
word (decoy)? Split by the new word's S2/S3-vs-S1 over-representation ratio (p1_filler) and by position."""
import sys
from collections import Counter, defaultdict

import polars as pl

from p1_catalog import D, load_side, name_rel, pairs_frame
from p1_decoy2 import refine


def main(c, n_sample=600000, seed=2):
    s1, o = load_side(c, "s1", ["cn", "cns", "cak"]), load_side(c, "o", ["cn", "cns", "cak"])
    vocab = set(s1.select(pl.col("cn").explode()).to_series().drop_nulls().to_list())
    ratio = dict(pl.read_parquet(f"{D}p1_filler_business_name_{c}.parquet").select("t", "ratio").iter_rows())
    df1 = Counter(s1.select(pl.col("cn").list.unique().explode()).to_series().drop_nulls().to_list())
    own = pl.read_parquet(f"{D}train_gt_rid.parquet").filter(pl.col("country") == c).select(owner="rid", oid="oid")
    k1, k2 = s1.filter(pl.col("cak") != "").select("idx", "cak"), o.filter(pl.col("cak") != "").select("idx", "cak")
    sz = k1.group_by("cak").agg(n1=pl.len()).filter(pl.col("n1") == 1)
    P = (k1.join(sz, on="cak").rename({"idx": "rid"}).join(k2.rename({"idx": "oid"}), on="cak").select("rid", "oid")
         .join(own, on="oid", how="left").with_columns(m=(pl.col("owner") == pl.col("rid")).fill_null(False)))
    x = pairs_frame(c, P.sample(min(n_sample, P.height), seed=seed), s1, o)
    st, ex = defaultdict(Counter), defaultdict(list)
    for r in x.iter_rows(named=True):
        A, B = list(r["cn_1"]), list(r["cn_2"])
        if refine(name_rel(A, B, r["cns_1"], r["cns_2"]), A, B, vocab) != "word_swap_real_word":
            continue
        news, olds = [t for t in B if t not in A], [t for t in A if t not in B]
        if not news or not olds:
            continue
        new, old = news[0], olds[0]
        filler = "new word = filler (ratio>=1.5)" if ratio.get(new, 0) >= 1.5 else "new word = distinctive"
        pos = "old word first" if A.index(old) == 0 else ("old word last" if A.index(old) == len(A) - 1 else "old word middle")
        freq = "new word common in S1 (df>=0.1%)" if df1.get(new, 0) >= 0.001 * s1.height else "new word rare in S1"
        rare = min(A, key=lambda t: df1.get(t, 0))
        rar = "old word = rarest name word" if old == rare else "old word = not the rarest"
        for k in (filler, pos, freq, f"{filler} & {pos}", rar, f"{rar} & {pos}", f"{rar} & {filler}"):
            st[k]["m" if r["m"] else "n"] += 1
            if len(ex[(k, r["m"])]) < 3:
                ex[(k, r["m"])].append(f"{r['business_name_1']!r} -> {r['business_name_2']!r}")
    print(f"\n===== {c}: real-word swaps at a single-S1 address =====")
    for k in sorted(st):
        n = st[k]["m"] + st[k]["n"]
        print(f"{k:<60s} {n:>6,}  match {st[k]['m'] / n:.3f}")
        for m in (True, False):
            for e in ex[(k, m)][:2]:
                print(f"      [{'match' if m else 'decoy'}] {e}")


if __name__ == "__main__":
    for c in sys.argv[1:] or ["US", "India"]:
        main(c)
