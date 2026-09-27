"""Matcher errors inside the candidate set (harder split, final v12 out-of-fold decisions): false positives and false
negatives by generator category, with the macro-F0.5 cost of each bucket (score gained if that bucket were fixed)."""
import sys
from collections import Counter, defaultdict

import polars as pl

import decide
import evaluate as ev
from p1_catalog import addr_rel, name_rel
from p1_decoy2 import refine
from p2_france_diag import addr_cls, canon

D = "C:/ML/Challenges/amazon/work/data/"


def main(c):
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").filter(pl.col("country") == c).join(keep, on="rid", how="semi").select("rid", "oid")
    rids = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == c).join(keep, on="rid", how="semi")["rid"]
    K = pl.read_parquet(f"{D}train_oof_final_v12_twin.parquet").filter(pl.col("country") == c)
    pred = decide.ef_predict(K.select("rid", "oid", pl.col("pf").alias("p2")), "p2", 0.0, 1.5, 0.05).select("rid", "oid")
    base = ev.macro_f05(pred, gt, rids)
    gtc = gt.join(K.select("rid", "oid"), on=["rid", "oid"], how="semi")  # true pairs inside the candidate set
    fp = pred.join(gt, on=["rid", "oid"], how="anti")
    fn = gtc.join(pred, on=["rid", "oid"], how="anti")
    s1, o, vocab = canon("train", c)
    nn = s1.filter(pl.col("cnk") != "").group_by("cnk").agg(nn=pl.len())
    na = s1.filter(pl.col("cak") != "").group_by("cak").agg(na=pl.len())

    def cats(P):
        x = (P.join(s1.select(rid="idx", cn1="cn", cns1="cns", cak1="cak", cnk1="cnk"), on="rid")
             .join(o.select(oid="idx", cn2="cn", cns2="cns", cak2="cak", ae2="aempty"), on="oid")
             .join(nn.rename({"cnk": "cnk1"}), on="cnk1", how="left").join(na.rename({"cak": "cak1"}), on="cak1", how="left"))
        out = {}
        for r in x.iter_rows(named=True):
            A, B = list(r["cn1"]), list(r["cn2"])
            nr = refine(name_rel(A, B, r["cns1"], r["cns2"]), A, B, vocab)
            ar = addr_cls(addr_rel(r["cak1"].split(" ") if r["cak1"] else [], r["cak2"].split(" ") if r["cak2"] else [], r["ae2"]))
            if ar == "canon_equal":
                ar += " (shared)" if (r["na"] or 0) > 1 else " (unique)"
            if ar == "empty":
                ar += " (name shared)" if (r["nn"] or 0) > 1 else " (name unique)"
            out[(r["rid"], r["oid"])] = (nr, ar)
        return out
    cf, cn = cats(fp), cats(fn)
    # cost of a bucket = macro F0.5 gain if its errors were fixed (FP removed / FN added)
    rows = []
    for kind, errs, P in (("FP", cf, fp), ("FN", cn, fn)):
        groups = defaultdict(list)
        for k, v in errs.items():
            groups[v].append(k)
        for g, ks in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:10]:
            fix = pl.DataFrame(ks, schema=["rid", "oid"], orient="row").with_columns(pl.all().cast(pl.UInt32))
            new = pred.join(fix, on=["rid", "oid"], how="anti") if kind == "FP" else pl.concat([pred, fix])
            rows.append((kind, g[0], g[1], len(ks), ev.macro_f05(new, gt, rids) - base))
    print(f"\n=== {c}: macro F0.5 {base:.5f}; FP {fp.height:,}, FN inside candidates {fn.height:,} "
          f"(missing from candidates {gt.height - gtc.height:,})")
    print(f"{'':3s} {'name relation':<30s} {'address':<30s} {'pairs':>7s} {'F0.5 gain if fixed':>18s}")
    for r in sorted(rows, key=lambda r: -r[4]):
        print(f"{r[0]:3s} {r[1]:<30s} {r[2]:<30s} {r[3]:>7,} {r[4]:+18.5f}")
    tot_fp = ev.macro_f05(pred.join(fp, on=["rid", "oid"], how="anti"), gt, rids) - base
    tot_fn = ev.macro_f05(pl.concat([pred, fn]), gt, rids) - base
    print(f"all FP fixed: {tot_fp:+.5f}; all FN (inside candidates) fixed: {tot_fn:+.5f}")


if __name__ == "__main__":
    for c in sys.argv[1:] or ["India", "US"]:
        main(c)
