"""Phase 1.6: how far do rule-based decisions go, given perfect candidates? (train, per country, S1 sample)

Candidates for each sampled S1 = its true matches (perfect recall) + every record sharing its canonical address key
or canonical name key. Each pair gets the refined name relation and the canonical address relation from the
catalogue; decision rules are derived only from the Phase-1 match rates (p1_catalog / p1_decoy2 / p1_swap):

  name OK     : canon_equal(_reordered), truncation*, extra_word*, typo, glued, acronym, word_swap_garbled,
                word_swap_real (old word last / middle), invented_pseudo_word, name_empty
  name strong : canon_equal(_reordered), typo, extra_word, truncation (multi-token), glued
  address OK  : canonical equal, or number kept with street words equal / typo / missing / added, or number missing /
                added ("1/2", ranges) / truncated with the street words kept
R0  name strong AND address canonical-equal
R1  R0 + (name OK AND address canonical-equal AND the address belongs to this S1 only)
R2  R1 + (name strong AND address OK)
R3  R2 + (name canon-equal AND record address empty AND no other S1 carries the same name key)
Every rule set is followed by the one-owner constraint (a record claimed by several sampled S1 goes to none).
The macro F0.5 scorer is per S1, singletons included (empty/empty = 1, any prediction on a singleton = 0).
"""
import sys
from collections import Counter

import polars as pl

from p1_catalog import D, addr_rel, load_side, name_rel, pairs_frame
from p1_decoy2 import refine

NAME_STRONG = {"canon_equal", "canon_equal_reordered", "typo", "extra_word", "extra_word_typo", "truncation",
               "truncation_typo", "glued"}
NAME_OK = NAME_STRONG | {"truncation_single_prefix", "truncation_single_other", "truncation_single_typo", "acronym",
                         "word_swap_garbled", "word_swap_real_word (old word last)",
                         "word_swap_real_word (old word middle)", "invented_pseudo_word", "name_empty"}
ADDR_OK = {"canon_equal", "num:same|words:typo", "num:same|words:missing", "num:same|words:added",
           "num:missing|words:same", "num:missing|words:typo", "num:missing|words:missing", "num:added|words:same",
           "num:added|words:typo", "num:truncated|words:same", "num:missing|words:added", "num:added|words:missing"}


def macro_f05(pred, truth, rids):
    """pred / truth: dict rid -> set of oids. Per-S1 F0.5, singletons included, averaged over rids."""
    tot = 0.0
    for r in rids:
        p, t = pred.get(r, set()), truth.get(r, set())
        if not p and not t:
            tot += 1.0
            continue
        tp = len(p & t)
        if tp == 0:
            continue
        pr, rc = tp / len(p), tp / len(t)
        tot += 1.25 * pr * rc / (0.25 * pr + rc)
    return tot / len(rids)


def main(c, n_s1=50000, seed=0):
    cols = ["cn", "cns", "cnk", "cak", "aempty"]
    s1, o = load_side(c, "s1", cols), load_side(c, "o", cols)
    vocab = set(s1.select(pl.col("cn").explode()).to_series().drop_nulls().to_list())
    n_addr = s1.filter(pl.col("cak") != "").group_by("cak").agg(na=pl.len())
    n_name = s1.filter(pl.col("cnk") != "").group_by("cnk").agg(nn=pl.len())
    samp = s1.sample(n_s1, seed=seed).select(rid="idx", cak="cak", cnk="cnk")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").filter(pl.col("country") == c).select("rid", "oid")
    gts = gt.join(samp, on="rid", how="semi")
    cand = pl.concat([
        gts,
        samp.filter(pl.col("cak") != "").join(o.filter(pl.col("cak") != "").select(oid="idx", cak="cak"), on="cak").select("rid", "oid"),
        samp.filter(pl.col("cnk") != "").join(o.filter(pl.col("cnk") != "").select(oid="idx", cnk="cnk"), on="cnk").select("rid", "oid"),
    ]).unique()
    x = pairs_frame(c, cand, s1, o).join(n_addr.rename({"cak": "cak_1"}), on="cak_1", how="left").join(
        n_name.rename({"cnk": "cnk_1"}), on="cnk_1", how="left")
    truth = {}
    for r, oo in gts.iter_rows():
        truth.setdefault(r, set()).add(oo)
    rules = {k: {} for k in ("R0", "R1", "R2", "R3")}
    cls = Counter()
    ambiguous = {}
    for r in x.iter_rows(named=True):
        A, B = list(r["cn_1"]), list(r["cn_2"])
        nr = refine(name_rel(A, B, r["cns_1"], r["cns_2"]), A, B, vocab)
        if nr == "word_swap_real_word":
            olds = [t for t in A if t not in B]
            if olds:
                i = A.index(olds[0])
                nr += " (old word " + ("first)" if i == 0 else "last)" if i == len(A) - 1 else "middle)")
        ar = addr_rel(r["cak_1"].split(" ") if r["cak_1"] else [], r["cak_2"].split(" ") if r["cak_2"] else [],
                      r["aempty_2"])
        single_addr = (r["na"] or 0) <= 1
        unique_name = (r["nn"] or 0) <= 1
        ok = {"R0": nr in NAME_STRONG and ar == "canon_equal"}
        ok["R1"] = ok["R0"] or (nr in NAME_OK and ar == "canon_equal" and single_addr)
        ok["R2"] = ok["R1"] or (nr in NAME_STRONG and ar in ADDR_OK)
        ok["R3"] = ok["R2"] or (nr in ("canon_equal", "canon_equal_reordered") and ar == "empty" and unique_name)
        y = r["oid"] in truth.get(r["rid"], set())
        if y and ((ar == "empty" and not unique_name and nr in ("canon_equal", "canon_equal_reordered"))
                  or (ar == "canon_equal" and not single_addr
                      and nr in ("invented_pseudo_word", "acronym", "name_empty", "truncation_single_other"))):
            ambiguous.setdefault(r["rid"], set()).add(r["oid"])
        cls[(nr if ar == "canon_equal" else "*", ar if ar in ("canon_equal", "empty") else "other", y, ok["R3"])] += 1
        for k, v in ok.items():
            if v:
                rules[k].setdefault(r["oid"], []).append(r["rid"])
    rids = samp["rid"].to_list()
    print(f"\n=== {c}: rule-based oracle on {n_s1:,} random S1, {x.height:,} candidate pairs "
          f"(true pairs {gts.height:,}, all included) ===")
    for k, claims in rules.items():
        pred, conflicts = {}, 0
        for oid, rs in claims.items():
            if len(set(rs)) > 1:
                conflicts += 1
                continue
            pred.setdefault(rs[0], set()).add(oid)
        n_pred = sum(len(v) for v in pred.values())
        tp = sum(len(v & truth.get(r, set())) for r, v in pred.items())
        print(f"{k}: macro F0.5 = {macro_f05(pred, truth, rids):.4f}   predicted pairs {n_pred:,}, precision "
              f"{tp / max(1, n_pred):.4f}, pair recall {tp / gts.height:.4f}, records claimed by 2+ S1 {conflicts:,}")
    missed = Counter({k: v for k, v in cls.items() if k[2] and not k[3]})
    print("largest buckets of true pairs that R3 misses (name relation at equal address | address relation):")
    tot_miss = sum(missed.values())
    for (nr, ar, _, _), n in missed.most_common(12):
        print(f"   {nr:<40s} {ar:<12s} {n:>7,}  ({n / gts.height:.4f} of true pairs)")
    print(f"   total missed true pairs {tot_miss:,} ({tot_miss / gts.height:.4f})")
    fp = Counter({k: v for k, v in cls.items() if not k[2] and k[3]})
    print("false positives of R3 by bucket:", ", ".join(f"{k[0]}|{k[1]}: {v}" for k, v in fp.most_common(6)))
    # information ceiling: abstain on true pairs that canonical evidence cannot attribute to one S1, predict every
    # other true pair and nothing else
    amb_pred = {r: {oo for oo in s if oo not in ambiguous.get(r, set())} for r, s in truth.items()}
    n_amb = sum(len(v) for v in ambiguous.values())
    print(f"information ceiling: {n_amb:,} true pairs ({n_amb / gts.height:.4f}) are ambiguous between S1 entities "
          f"(name-only record whose name 2+ S1 carry, or alias / acronym / pseudo-word record at an address 2+ S1 "
          f"share); perfect decisions elsewhere + abstaining there -> macro F0.5 = {macro_f05(amb_pred, truth, rids):.4f}")


if __name__ == "__main__":
    for c in sys.argv[1:] or ["US", "India"]:
        main(c)
