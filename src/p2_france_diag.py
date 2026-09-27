"""B2: label-free France diagnosis - compare the France predictions with the generator's train rates.

Datasets (all on the harder / test-like condition where it applies):
  truth  : train true pairs of the kept S1 entities (keep_v11), per country
  tpred  : train out-of-fold decisions of the current model (v12 round-1 combiner p3, expected-F0.5 rule)
  test   : current submission (v12final) for France / India / US
Per dataset: matches per S1, empty-prediction rate, match-count distribution, and the per-S1 rate of every
generator category = name relation (p1_catalog, swap position) x address relation (canonical) x shared address.
Ranking: France deviation from the train truth, minus the model's own bias on train (tpred - truth), per S1.
"""
import sys
from collections import Counter

import polars as pl

import decide
from p1_catalog import addr_rel, name_rel
from p1_decoy2 import refine

D = "C:/ML/Challenges/amazon/work/data/"
OUT = "C:/ML/Challenges/amazon/work/output/"


def addr_cls(ar):
    if ar in ("canon_equal", "empty"):
        return ar
    n, w = ar.split("|")
    n, w = n[4:], w[6:]
    if n == "same":
        return "num same, words differ"
    if n == "missing":
        return "num missing"
    if n in ("added", "truncated"):
        return f"num {n}"
    return "num changed/typo" + (", words same" if w == "same" else ", words differ")


def categorise(pairs, s1, o, vocab, n_sample, seed=0):
    """pairs: (s1 idx, o idx). Returns Counter of categories (sample) and the sampling factor."""
    x = pairs.sample(min(n_sample, pairs.height), seed=seed)
    x = (x.join(s1.select(i1="idx", cn1="cn", cns1="cns", cak1="cak", na="na"), on="i1")
         .join(o.select(i2="idx", cn2="cn", cns2="cns", cak2="cak", ae2="aempty"), on="i2"))
    cnt = Counter()
    for r in x.iter_rows(named=True):
        A, B = list(r["cn1"]), list(r["cn2"])
        nr = refine(name_rel(A, B, r["cns1"], r["cns2"]), A, B, vocab)
        if nr == "word_swap_real_word":
            olds = [t for t in A if t not in B]
            if olds:
                i = A.index(olds[0])
                nr += " (first)" if i == 0 else " (last)" if i == len(A) - 1 else " (middle)"
        ar = addr_cls(addr_rel(r["cak1"].split(" ") if r["cak1"] else [], r["cak2"].split(" ") if r["cak2"] else [],
                               r["ae2"]))
        if ar == "canon_equal":
            ar += " (shared)" if (r["na"] or 0) > 1 else " (unique)"
        cnt[(nr, ar)] += 1
    return cnt, pairs.height / x.height


def canon(split, c):
    s1 = pl.read_parquet(f"{D}p1_canon_{split}_{c}_s1.parquet")
    o = pl.read_parquet(f"{D}p1_canon_{split}_{c}_o.parquet")
    na = s1.filter(pl.col("cak") != "").group_by("cak").agg(na=pl.len())
    s1 = s1.join(na, on="cak", how="left")
    vocab = set(s1.select(pl.col("cn").explode()).to_series().drop_nulls().to_list())
    return s1, o, vocab


def count_stats(pairs, n_s1):
    k = pairs.group_by("i1").agg(n=pl.len())
    dist = {i: 0 for i in range(10)}
    for n, m in k.group_by(pl.col("n").clip(0, 9)).len().iter_rows():
        dist[n] = m
    dist[0] = n_s1 - k.height
    return pairs.height / n_s1, {i: dist[i] / n_s1 for i in dist}


def main(n_sample=150000):
    """Candidate-level view: per generator category, candidates per S1 and the share predicted (and, on train, the
    true share). France-specific deviation = France candidates/S1 x (train predicted share - France predicted share)."""
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").select(i1=pl.col("rid").cast(pl.UInt32), i2=pl.col("oid").cast(pl.UInt32), y=pl.lit(1))
    oof = pl.read_parquet(f"{D}train_oof_ce_v12.parquet", columns=["rid", "oid", "country", "p3"])
    tp = decide.ef_predict(oof.select("rid", "oid", "p3"), "p3", 0.0, 1.5, 0.05).select(
        i1=pl.col("rid").cast(pl.UInt32), i2=pl.col("oid").cast(pl.UInt32), pred=pl.lit(1))
    res = pl.read_csv(f"{OUT}matching_results_v12final.tsv", separator="	", quote_char=None, infer_schema=False,
                      missing_utf8_is_empty_string=True)
    tpairs = (res.filter(pl.col("matched_entity_ids") != "").with_columns(m=pl.col("matched_entity_ids").str.split(","))
              .explode("m").select(e1="source1_entity_id", e2="m"))
    tcand = pl.read_parquet(f"{D}test_pred_v12.parquet", columns=["rid", "oid", "country"])
    stats = {}
    for split, c in (("train", "US"), ("train", "India"), ("test", "France"), ("test", "India"), ("test", "US")):
        s1, o, vocab = canon(split, c)
        if split == "train":
            s1 = s1.join(keep.rename({"rid": "idx"}), on="idx", how="semi")
            C = oof.filter(pl.col("country") == c).select(i1=pl.col("rid").cast(pl.UInt32), i2=pl.col("oid").cast(pl.UInt32))
            C = C.join(gt, on=["i1", "i2"], how="left").join(tp, on=["i1", "i2"], how="left")
        else:
            C = tcand.filter(pl.col("country") == c).select(i1=pl.col("rid").cast(pl.UInt32), i2=pl.col("oid").cast(pl.UInt32))
            pr = (tpairs.join(s1.select(e1="entity_id", i1="idx"), on="e1").join(o.select(e2="entity_id", i2="idx"), on="e2")
                  .select("i1", "i2", pred=pl.lit(1)))
            C = C.join(pr, on=["i1", "i2"], how="left").with_columns(y=pl.lit(None, pl.Int32))
        C = C.with_columns(pl.col("pred").fill_null(0), pl.col("y").fill_null(0) if split == "train" else pl.col("y"))
        x = C.sample(min(n_sample, C.height), seed=0)
        x = (x.join(s1.select(i1="idx", cn1="cn", cns1="cns", cak1="cak", na="na"), on="i1")
             .join(o.select(i2="idx", cn2="cn", cns2="cns", cak2="cak", ae2="aempty"), on="i2"))
        f = C.height / x.height / s1.height
        st = {}
        for r in x.iter_rows(named=True):
            A, B = list(r["cn1"]), list(r["cn2"])
            nr = refine(name_rel(A, B, r["cns1"], r["cns2"]), A, B, vocab)
            if nr == "word_swap_real_word":
                olds = [t for t in A if t not in B]
                if olds:
                    i = A.index(olds[0])
                    nr += " (first)" if i == 0 else " (last)" if i == len(A) - 1 else " (middle)"
            ar = addr_cls(addr_rel(r["cak1"].split(" ") if r["cak1"] else [], r["cak2"].split(" ") if r["cak2"] else [],
                                   r["ae2"]))
            if ar == "canon_equal":
                ar += " (shared)" if (r["na"] or 0) > 1 else " (unique)"
            a = st.setdefault((nr, ar), [0, 0, 0])
            a[0] += 1
            a[1] += r["pred"]
            a[2] += r["y"] or 0
        stats[(split, c)] = {k: (v[0] * f, v[1] / v[0], v[2] / v[0]) for k, v in st.items()}
        print(f"{split} {c}: {C.height / s1.height:.2f} candidates per S1", flush=True)
    cats = set().union(*[set(v) for v in stats.values()])
    g = lambda d, k, j: stats[d].get(k, (0, 0, 0))[j]
    tr_pred = {k: (g(("train", "US"), k, 1) + g(("train", "India"), k, 1)) / 2 for k in cats}
    tr_true = {k: (g(("train", "US"), k, 2) + g(("train", "India"), k, 2)) / 2 for k in cats}
    dev = {k: g(("test", "France"), k, 0) * (tr_pred[k] - g(("test", "France"), k, 1)) for k in cats}
    print("\ncandidate pairs by generator category (per S1) and share predicted; train true share = generator's rate")
    print(f"{'name relation':<30s} {'address':<30s} {'trCand':>6s} {'trTrue':>6s} {'trPred':>6s} {'FRcand':>6s} "
          f"{'FRpred':>6s} {'INpred':>6s} {'USpred':>6s} {'FRdev/S1':>8s}")
    for k in sorted(cats, key=lambda k: -abs(dev[k]))[:32]:
        trc = (g(("train", "US"), k, 0) + g(("train", "India"), k, 0)) / 2
        print(f"{k[0]:<30s} {k[1]:<30s} {trc:6.3f} {tr_true[k]:6.3f} {tr_pred[k]:6.3f} {g(('test', 'France'), k, 0):6.3f} "
              f"{g(('test', 'France'), k, 1):6.3f} {g(('test', 'India'), k, 1):6.3f} {g(('test', 'US'), k, 1):6.3f} {dev[k]:+8.4f}")
    pos = sum(v for v in dev.values() if v > 0)
    neg = sum(v for v in dev.values() if v < 0)
    print(f"France predicts {pos:.3f} fewer matches per S1 than train behaviour in under-predicted categories and "
          f"{-neg:.3f} more in over-predicted ones")


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:2]])
