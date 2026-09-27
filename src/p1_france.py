"""Phase 1 sanity check (test, no labels): records at an S1's own (single-S1) canonical address, by name relation,
with the share the current submission (v12) predicts as a match. Compare France with India / US, whose train truth
rates are known (p1_decoy2.txt, p1_swap.txt)."""
import sys
from collections import Counter, defaultdict

import polars as pl

import normalize as N
from p1_catalog import name_rel
from p1_decoy2 import refine

D = "C:/ML/Challenges/amazon/work/data/"


def main(c, n_sample=300000, seed=3):
    s1 = pl.read_parquet(f"{D}p1_canon_test_{c}_s1.parquet")
    o = pl.read_parquet(f"{D}p1_canon_test_{c}_o.parquet")
    vocab = set(s1.select(pl.col("cn").explode()).to_series().drop_nulls().to_list())
    res = pl.read_csv("C:/ML/Challenges/amazon/work/output/matching_results_v12final.tsv", separator="\t",
                      quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
    pred = (res.filter(pl.col("matched_entity_ids") != "").with_columns(m=pl.col("matched_entity_ids").str.split(","))
            .explode("m").select(e1="source1_entity_id", e2="m").with_columns(pred=pl.lit(True)))
    k1, k2 = s1.filter(pl.col("cak") != ""), o.filter(pl.col("cak") != "")
    one = k1.group_by("cak").agg(n1=pl.len()).filter(pl.col("n1") == 1).select("cak")
    P = (k1.join(one, on="cak").select("cak", e1="entity_id", A="cn", cnsA="cns")
         .join(k2.select("cak", e2="entity_id", B="cn", cnsB="cns"), on="cak"))
    P = P.join(pred, on=["e1", "e2"], how="left").with_columns(pl.col("pred").fill_null(False))
    x = P.sample(min(n_sample, P.height), seed=seed)
    st, ex = defaultdict(Counter), defaultdict(list)
    for r in x.iter_rows(named=True):
        A, B = list(r["A"]), list(r["B"])
        rel = refine(name_rel(A, B, r["cnsA"], r["cnsB"]), A, B, vocab)
        if rel == "word_swap_real_word":
            news, olds = [t for t in B if t not in A], [t for t in A if t not in B]
            if olds:
                rel += " (old word " + ("first)" if A.index(olds[0]) == 0 else "last)" if A.index(olds[0]) == len(A) - 1 else "middle)")
        st[rel]["p" if r["pred"] else "n"] += 1
        if len(ex[(rel, r["pred"])]) < 3:
            ex[(rel, r["pred"])].append(f"{' '.join(A)!r} -> {' '.join(B)!r}")
    n_s1 = s1.height
    print(f"\n===== test {c}: records at a single-S1 canonical address (sample {x.height:,} of {P.height:,}) =====")
    print(f"{'name relation':<44s} {'per S1 (est.)':>13s} {'predicted match':>16s}")
    for k in sorted(st, key=lambda k: -(st[k]['p'] + st[k]['n'])):
        n = st[k]["p"] + st[k]["n"]
        print(f"{k:<44s} {n * P.height / x.height / n_s1:13.4f} {st[k]['p'] / n:16.3f}")
        for pr in (True, False):
            for e in ex[(k, pr)][:2]:
                print(f"      [{'pred' if pr else 'rej '}] {e}")


if __name__ == "__main__":
    for c in sys.argv[1:] or ["France", "US", "India"]:
        main(c)
