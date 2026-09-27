"""B3: French normalisation audit (test, no labels).

A: the 500 France candidate pairs with the highest name similarity but low address similarity,
B: the 500 with the highest address similarity but low name similarity (pipeline features n_tset / a_tset).
For both: the token differences left AFTER the pipeline's normalisation (norm.py tokens `at` / `nt`), aggregated,
plus the model's current probability. Then a pattern scan over all France records for the variant families of the
brief (St-/St. -> Saint, bis/ter, CEDEX, dotted / leet legal forms, N° / Nº / NO. / # / (n) / leading zeros,
region vs department, née) showing how the normaliser renders them.
"""
import re
from collections import Counter

import polars as pl

D = "C:/ML/Challenges/amazon/work/data/"


def diff_report(x, col1, col2, title, k=25):
    only1, only2, subs = Counter(), Counter(), Counter()
    for a, b in zip(x[col1].to_list(), x[col2].to_list()):
        a, b = set(a or []), set(b or [])
        d1, d2 = sorted(a - b), sorted(b - a)
        only1.update(d1)
        only2.update(d2)
        if len(d1) == 1 and len(d2) == 1:
            subs[(d1[0], d2[0])] += 1
    print(f"\n  {title}: tokens only in S1 (top {k}): " + ", ".join(f"{t}({n})" for t, n in only1.most_common(k)))
    print(f"  {title}: tokens only in record (top {k}): " + ", ".join(f"{t}({n})" for t, n in only2.most_common(k)))
    print(f"  {title}: single-token substitutions S1 -> record (top {k}): "
          + ", ".join(f"{a}->{b}({n})" for (a, b), n in subs.most_common(k)))


def main():
    pl.Config.set_fmt_str_lengths(80)
    K = pl.read_parquet(f"{D}test_pred_v12.parquet").filter(pl.col("country") == "France").select("rid", "oid", "p2")
    F = pl.scan_parquet(f"{D}test_feat_v10_France.parquet").select("rid", "oid", "n_tset", "a_tset", "c_tset").join(
        K.lazy(), on=["rid", "oid"]).collect()
    s1 = pl.read_parquet(f"{D}test_s1_norm.parquet").filter(pl.col("country") == "France").select(
        "rid", "nt", "at", "af")
    o = pl.concat([pl.read_parquet(f"{D}test_s{k}_norm.parquet") for k in (2, 3)]).drop("rid").with_row_index("oid").filter(
        pl.col("country") == "France").select("oid", nt2="nt", at2="at", af2="af")
    raw1 = pl.read_parquet(f"{D}test_source1.parquet").with_row_index("rid").select("rid", n1="business_name", a1="business_address")
    raw2 = pl.concat([pl.read_parquet(f"{D}test_source{k}.parquet") for k in (2, 3)]).with_row_index("oid").select(
        "oid", n2="business_name", a2="business_address")
    print(f"France candidate pairs: {F.height:,}")
    for title, cond, key in (
            ("A: high name / low address similarity", (pl.col("n_tset") >= 90) & (pl.col("a_tset") <= 60),
             pl.col("n_tset") - pl.col("a_tset")),
            ("B: high address / low name similarity", (pl.col("a_tset") >= 90) & (pl.col("n_tset") <= 60),
             pl.col("a_tset") - pl.col("n_tset"))):
        x = (F.filter(cond).with_columns(k=key).sort(["k", "p2"], descending=True).head(500)
             .join(s1, on="rid").join(o, on="oid").join(raw1, on="rid").join(raw2, on="oid"))
        print(f"\n=== {title}: {F.filter(cond).height:,} pairs qualify; top 500, mean p2 {x['p2'].mean():.3f}, "
              f"predicted (p2>=0.5) {(x['p2'] >= 0.5).mean():.3f}")
        diff_report(x, "at", "at2", "address tokens")
        diff_report(x, "nt", "nt2", "name tokens")
        for r in x.sample(8, seed=1).iter_rows(named=True):
            print(f"    p2={r['p2']:.2f} | {r['n1']} | {r['a1']}\n              | {r['n2']} | {r['a2']}")
    # pattern scan: how does the normaliser render each variant family?
    allr = pl.concat([pl.read_parquet(f"{D}test_source{k}.parquet") for k in (2, 3)]).with_row_index("oid").filter(
        pl.col("country") == "France").join(o.select("oid", "af2", "nt2"), on="oid")
    pats = {"St-/St. (Saint)": ("business_address", r"\bSt[\.\-]|\bST[\.\-]|\bSte[\.\-]"),
            "bis / ter": ("business_address", r"(?i)\b\d+\s*(bis|ter)\b"),
            "CEDEX": ("business_address", r"(?i)cedex"),
            "N°/Nº/NO./No/#/(n) prefix": ("business_address", r"(?i)(n°|nº|\bno\.?\s*\d|#\s*\d|\(\d+\))"),
            "leading zeros": ("business_address", r"(^|[\s,])0\d"),
            "department name": ("business_address", r"(?i)\b(nord|gironde|loire-atlantique|pas-de-calais)\b"),
            "5-digit postcode": ("business_address", r"\b\d{5}\b"),
            "dotted legal form": ("business_name", r"(?i)\b(s\.a\.r\.l|e\.u\.r\.l|s\.a\.s\.u|s\.a\.s|s\.a\.|s\.c\.i|s\.n\.c)"),
            "leet legal form": ("business_name", r"(?i)\b(5arl|5as|5asu|eur1|5a)\b"),
            "née / nee": ("business_name", r"(?i)\bn[ée]e\b"),
            "(Cie) / & Cie": ("business_name", r"(?i)\bcie\b"),
            "Et Fils / & Fils": ("business_name", r"(?i)\bfils\b")}
    print(f"\n=== variant families in France S2/S3 ({allr.height:,} records): share, and normalised examples")
    for name, (col, pat) in pats.items():
        m = allr.filter(pl.col(col).str.contains(pat))
        print(f"  {name:<28s} {m.height / allr.height:7.4f}")
        norm_col = "af2" if col == "business_address" else "nt2"
        for r in m.sample(min(3, m.height), seed=2).iter_rows(named=True):
            print(f"        {r[col]!r:70s} -> {r[norm_col]!r}")


if __name__ == "__main__":
    main()
