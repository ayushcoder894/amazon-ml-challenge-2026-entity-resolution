"""Follow-up: US/India/France test v19 vs v19_USnear per nearby-number bucket; expected gain of removing the
accepted pairs of the buckets where US test exceeds train (siblings odd/mixed/none, |diff| <= 50) under two scenarios."""
import polars as pl

from nearby_number import D, OUT, BINS, SIBS, pattern, sib_agree

rd = dict(separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)


def load_pred(name, s1, o):
    fin = (pl.read_csv(f"{OUT}matching_results_{name}.tsv", **rd).filter(pl.col("matched_entity_ids") != "")
           .with_columns(m=pl.col("matched_entity_ids").str.split(",")).explode("m"))
    return (fin.join(s1.rename({"entity_id": "source1_entity_id", "idx": "rid"}), on="source1_entity_id")
            .join(o.rename({"entity_id": "m", "idx": "oid"}), on="m").select("rid", "oid"))


def f05(tp, npred, ntrue):
    if npred == 0 and ntrue == 0:
        return 1.0
    if tp == 0:
        return 0.0
    p, r = tp / npred, tp / ntrue
    return 1.25 * p * r / (0.25 * p + r)


for c in ("US", "India", "France"):
    s1 = pl.read_parquet(f"{D}p1_canon_test_{c}_s1.parquet", columns=["idx", "entity_id"])
    o = pl.read_parquet(f"{D}p1_canon_test_{c}_o.parquet", columns=["idx", "entity_id"])
    v19 = load_pred("v19", s1, o)
    usn = load_pred("v19_USnear", s1, o)
    P, oc = pattern("test", c)
    P = P.join(v19.with_columns(acc=pl.lit(1, pl.Int8)), on=["rid", "oid"], how="left").join(
        usn.with_columns(accn=pl.lit(1, pl.Int8)), on=["rid", "oid"], how="left").with_columns(pl.col("acc", "accn").fill_null(0))
    P = sib_agree(P, v19, oc, "sib")
    n = s1.height
    print(f"\n{c}: accepted pattern pairs/S1  v19 -> v19_USnear, per bucket")
    for s in SIBS:
        print(f"   {s:<20s} " + "  ".join(f"{b}: {P.filter((pl.col('sib') == s) & (pl.col('dbin') == b))['acc'].sum() / n:.4f}->"
                                        f"{P.filter((pl.col('sib') == s) & (pl.col('dbin') == b))['accn'].sum() / n:.4f}" for b in BINS))
    if c != "US":
        continue
    tgt = P.filter((pl.col("acc") == 1) & pl.col("sib").is_in(["odd one out", "mixed", "none"]) & pl.col("dbin").is_in(["<=10", "11-50"]))
    k = v19.group_by("rid").agg(k=pl.len())
    t = tgt.group_by("rid").agg(m=pl.len()).join(k, on="rid")
    # per entity: k predicted, m of them in the target set. If a target pair is a FP, removing it helps; if TP, hurts.
    # expected gain with each target pair independently FP with prob q, other predictions assumed correct
    import itertools
    rows = t.select("k", "m").to_numpy()
    for q, lab in ((0.025, "test precision = train precision (97.5%)"), (1 - 0.0095 / (tgt.height / n), "test true count per S1 = train (0.0095/S1)")):
        g = 0.0
        for kk, mm in rows:
            for j in range(mm + 1):  # j false positives among the m target pairs
                from math import comb
                pr = comb(mm, j) * q ** j * (1 - q) ** (mm - j)
                ntrue = kk - j
                before = f05(kk - j, kk, ntrue)
                after = f05(kk - mm, kk - mm, ntrue)
                g += pr * (after - before)
        print(f"   remove {tgt.height:,} target pairs ({tgt.height / n:.4f}/S1) if {lab}: US {g / n:+.5f}, LB {g / n * 0.85025 * 663106 / (663106 + 809986):+.5f}")
