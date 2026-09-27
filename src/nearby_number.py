"""Nearby-number pattern: (S1, record) with the same canonical name, the same non-numeric address words and a different,
non-truncated number (at least one number on each side that the other lacks, neither contained in the other).
Found by a direct join on (name, street words), so the universe does not depend on blocking.

  sibling agreement (the entity's OTHER matched records that have numbers):
    odd one out       : every such sibling carries the S1's differing number, none the record's
    siblings share rec: at least one sibling carries the record's differing number
    mixed             : otherwise;   none: no sibling with numbers
  |number difference| = closest pair of differing numbers, buckets <=10, 11-50, 51-500, >500

TRAIN harder split (keep_v11 entities, current India/US model = joint variant 9 out-of-fold + v13 rule ef(0,1)):
  siblings from the MODEL's predictions (what a test rule can see) and from the TRUTH (oracle, for reference).
TRAIN normal split (all entities): truth only.  TEST: v19 predictions.
  python nearby_number.py
"""
import polars as pl

import decide
import evaluate as ev

D = "C:/ML/Challenges/amazon/work/data/"
DN = "D:/amazon_er/dense/"
OUT = "C:/ML/Challenges/amazon/work/output/"
BINS = ["<=10", "11-50", "51-500", ">500"]
SIBS = ["odd one out", "siblings share rec", "mixed", "none"]


def canon2(split, c, side):
    x = pl.read_parquet(f"{D}p1_canon_{split}_{c}_{side}.parquet", columns=["idx", "entity_id", "cnk", "cak", "aempty"])
    t = pl.col("cak").str.split(" ")
    return x.with_columns(
        nums=t.list.eval(pl.element().filter(pl.element().str.contains(r"\d"))).list.unique().list.sort(),
        w=t.list.eval(pl.element().filter(~pl.element().str.contains(r"\d") & (pl.element() != ""))).list.unique().list.sort().list.join(" "))


def digits(t):
    s = "".join(ch for ch in t if ch.isdigit())
    return int(s[:12]) if s else 0


def pattern(split, c):
    s1, o = canon2(split, c, "s1"), canon2(split, c, "o")
    ok = (pl.col("cnk") != "") & (pl.col("w") != "") & (pl.col("nums").list.len() > 0)
    P = (s1.filter(ok).select(pl.col("idx").alias("rid"), "cnk", "w", n1="nums")
         .join(o.filter(ok & ~pl.col("aempty")).select(pl.col("idx").alias("oid"), "cnk", "w", n2="nums"), on=["cnk", "w"])
         .filter(pl.col("n1").list.join(" ") != pl.col("n2").list.join(" ")))
    da, db, dd, keep = [], [], [], []
    for a, b in zip(P["n1"].to_list(), P["n2"].to_list()):
        x, y = sorted(set(a) - set(b)), sorted(set(b) - set(a))
        k = bool(x) and bool(y) and not any(u in v or v in u for u in x for v in y)
        keep.append(k)
        da.append(x)
        db.append(y)
        dd.append(min(abs(digits(u) - digits(v)) for u in x for v in y) if k else None)
    P = P.with_columns(da=pl.Series(da, dtype=pl.List(pl.String)), db=pl.Series(db, dtype=pl.List(pl.String)),
                       diff=pl.Series(dd, dtype=pl.Int64), k=pl.Series(keep)).filter(pl.col("k"))
    P = P.with_columns(dbin=pl.when(pl.col("diff") <= 10).then(pl.lit(BINS[0])).when(pl.col("diff") <= 50).then(pl.lit(BINS[1]))
                       .when(pl.col("diff") <= 500).then(pl.lit(BINS[2])).otherwise(pl.lit(BINS[3])))
    return P.select("rid", "oid", "da", "db", "diff", "dbin"), o.select("idx", "nums", "aempty")


def sib_agree(P, M, o, col):
    """Sibling agreement of each pattern pair given the entity's matched set M (rid, oid)."""
    S = M.join(o.select(pl.col("idx").alias("soid"), ns=pl.when(pl.col("aempty")).then(pl.lit([], pl.List(pl.String))).otherwise(pl.col("nums"))),
               left_on="oid", right_on="soid").select("rid", pl.col("oid").alias("soid"), "ns")
    x = P.select("rid", "oid", "da", "db").join(S, on="rid").filter(pl.col("soid") != pl.col("oid"))
    x = x.with_columns(info=pl.col("ns").list.len() > 0, rec=pl.col("ns").list.set_intersection(pl.col("db")).list.len() > 0,
                       sown=pl.col("ns").list.set_intersection(pl.col("da")).list.len() > 0)
    g = x.group_by("rid", "oid").agg(ni=pl.col("info").sum(), rec=pl.col("rec").sum(), own=(pl.col("sown") & ~pl.col("rec")).sum())
    g = g.with_columns(pl.when(pl.col("ni") == 0).then(pl.lit("none")).when(pl.col("rec") > 0).then(pl.lit("siblings share rec"))
                       .when(pl.col("own") == pl.col("ni")).then(pl.lit("odd one out")).otherwise(pl.lit("mixed")).alias(col))
    return P.join(g.select("rid", "oid", col), on=["rid", "oid"], how="left").with_columns(pl.col(col).fill_null("none"))


def table(x, sib, n_s1, with_pred):
    print(f"   {'siblings (' + sib + ')':<34s} {'|diff|':<7s} {'pairs/S1':>9s} {'true rate':>9s}" + (f" {'pred/S1':>8s} {'pred precision':>14s}" if with_pred else ""))
    for s in SIBS:
        for b in BINS + ["all"]:
            y = x.filter(pl.col(sib) == s) if b == "all" else x.filter((pl.col(sib) == s) & (pl.col("dbin") == b))
            if y.height == 0:
                continue
            line = f"   {s if b == BINS[0] or (b == 'all' and y.height == x.filter(pl.col(sib) == s).height and False) else '':<34s} {b:<7s} {y.height / n_s1:9.4f} {y['y'].mean():9.3f}"
            if with_pred:
                yp = y.filter(pl.col("acc") == 1)
                line += f" {yp.height / n_s1:8.4f} {yp['y'].mean() if yp.height else float('nan'):14.3f}"
            print(line)


def train(c, keep, gt):
    print(f"\n==================== {c} (train)")
    s1c = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == c).select("rid")
    P, o = pattern("train", c)
    gtc = gt.join(s1c, on="rid", how="semi")
    P = P.join(gtc.with_columns(y=pl.lit(1, pl.Int8)), on=["rid", "oid"], how="left").with_columns(pl.col("y").fill_null(0))
    # normal split: all entities, true siblings
    n_all = s1c.height
    Pn = sib_agree(P, gtc, o, "sib_true")
    print(f"-- normal split (all {n_all:,} entities): {Pn.height / n_all:.4f} pattern pairs/S1, true rate {Pn['y'].mean():.3f}; "
          f"entities whose TRUE set contains one: {Pn.filter(pl.col('y') == 1)['rid'].n_unique() / n_all:.2%}")
    table(Pn, "sib_true", n_all, False)
    # harder split: kept entities, model predictions of joint variant 9
    sk = s1c.join(keep, on="rid", how="semi")
    n_k = sk.height
    U = pl.read_parquet(f"{DN}train_joint9.parquet", columns=["rid", "oid", "pj", "country"]).filter(pl.col("country") == c).drop("country")
    pred = decide.ef_predict(U.select("rid", "oid", pl.col("pj").alias("p2")), "p2", 0.0, 1.0, 0.05)
    del U
    gtk = gtc.join(sk, on="rid", how="semi")
    Ph = P.join(sk, on="rid", how="semi").join(pred.with_columns(acc=pl.lit(1, pl.Int8)), on=["rid", "oid"], how="left").with_columns(pl.col("acc").fill_null(0))
    Ph = sib_agree(Ph, pred, o, "sib_model")
    Ph = sib_agree(Ph, gtk, o, "sib_true")
    print(f"-- harder split ({n_k:,} kept entities): {Ph.height / n_k:.4f} pattern pairs/S1, true rate {Ph['y'].mean():.3f}, "
          f"predicted {Ph['acc'].sum() / n_k:.4f}/S1 (precision {Ph.filter(pl.col('acc') == 1)['y'].mean():.3f}, recall {Ph.filter(pl.col('y') == 1)['acc'].mean():.3f})")
    print(f"   entities whose TRUE set contains one: {Ph.filter(pl.col('y') == 1)['rid'].n_unique() / n_k:.2%};  "
          f"whose PREDICTED set does: {Ph.filter(pl.col('acc') == 1)['rid'].n_unique() / n_k:.2%}")
    print("   siblings from the MODEL's predictions (usable on test):")
    table(Ph, "sib_model", n_k, True)
    print("   siblings from the TRUTH (oracle):")
    table(Ph, "sib_true", n_k, True)
    base = ev.macro_f05(pred, gtk, sk["rid"])
    print(f"   macro F0.5 harder split {base:.5f}; removing PREDICTED odd-one-out pairs (model siblings):")
    for b in BINS + ["all"]:
        r = Ph.filter((pl.col("acc") == 1) & (pl.col("sib_model") == "odd one out") & ((pl.col("dbin") == b) if b != "all" else pl.lit(True)))
        f = ev.macro_f05(pred.join(r.select("rid", "oid"), on=["rid", "oid"], how="anti"), gtk, sk["rid"])
        print(f"      |diff| {b:<7s} removes {r.height:>6,} ({r.height / n_k:.4f}/S1, {r['y'].mean() if r.height else 0:.3f} of them true) -> {f:.5f} ({f - base:+.5f})")
    return Ph


def test(c):
    s1 = pl.read_parquet(f"{D}p1_canon_test_{c}_s1.parquet", columns=["idx", "entity_id"])
    o = pl.read_parquet(f"{D}p1_canon_test_{c}_o.parquet", columns=["idx", "entity_id"])
    rd = dict(separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
    fin = (pl.read_csv(f"{OUT}matching_results_v19.tsv", **rd).filter(pl.col("matched_entity_ids") != "")
           .with_columns(m=pl.col("matched_entity_ids").str.split(",")).explode("m"))
    pred = (fin.join(s1.rename({"entity_id": "source1_entity_id", "idx": "rid"}), on="source1_entity_id")
            .join(o.rename({"entity_id": "m", "idx": "oid"}), on="m").select("rid", "oid"))
    P, oc = pattern("test", c)
    n = s1.height
    P = P.join(pred.with_columns(acc=pl.lit(1, pl.Int8)), on=["rid", "oid"], how="left").with_columns(pl.col("acc").fill_null(0), y=pl.lit(None, pl.Float64))
    P = sib_agree(P, pred, oc, "sib_model")
    print(f"\n-- TEST {c} ({n:,} S1): {P.height / n:.4f} pattern pairs/S1, v19 accepts {P['acc'].sum() / n:.4f}/S1; "
          f"S1s whose v19 set contains one: {P.filter(pl.col('acc') == 1)['rid'].n_unique() / n:.2%}")
    print(f"   {'siblings (v19)':<22s} " + " ".join(f"{b:>15s}" for b in BINS) + "   (pairs/S1 accepted by v19 of candidates/S1)")
    for s in SIBS:
        row = []
        for b in BINS:
            y = P.filter((pl.col("sib_model") == s) & (pl.col("dbin") == b))
            row.append(f"{y['acc'].sum() / n:.4f}/{y.height / n:.4f}")
        print(f"   {s:<22s} " + " ".join(f"{r:>15s}" for r in row))
    odd = P.filter((pl.col("acc") == 1) & (pl.col("sib_model") == "odd one out"))
    print(f"   v19 odd-one-out pairs: {odd.height:,} ({odd.height / n:.4f}/S1) on {odd['rid'].n_unique():,} S1s")


def main():
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid"])
    for c in ("US", "India"):
        train(c, keep, gt)
    for c in ("US", "India", "France"):
        test(c)


if __name__ == "__main__":
    main()
