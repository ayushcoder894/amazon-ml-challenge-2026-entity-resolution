"""Bayes re-weighting of name-only namesake candidates by legal form (and script for India), evaluated on TRAIN harder split.
p'_i = p_i * LR_i / (sum_j p_j LR_j + (1 - sum_j p_j)), LR = P(record form | S1 form) / P(record form).
Tables estimated on one fold of entities, applied to the other (honest)."""
import re
import sys
import polars as pl
import decide
import evaluate as ev
from no_signal import legal, script_expr

D = "C:/ML/Challenges/amazon/work/data/"
DN = "D:/amazon_er/dense/"


def legal_us(n):
    t = re.findall(r"[a-z]+", (n or "").lower().replace("l.l.c", "llc").replace("l.l.p", "llp"))
    for w in reversed(t):
        if w in ("llc",): return "llc"
        if w in ("inc", "incorporated"): return "inc"
        if w in ("corp", "corporation"): return "corp"
        if w in ("ltd", "limited"): return "ltd"
        if w in ("co", "company"): return "co"
        if w in ("llp",): return "llp"
        if w in ("lp",): return "lp"
        if w in ("pllc", "pc", "pa", "plc"): return "prof"
    return "none"


def forms(split, c):
    lf = legal if c == "India" else legal_us
    r1 = pl.read_parquet(f"{D}{split}_source1.parquet", columns=["business_name", "country"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32)).filter(pl.col("country") == c)
    r1 = r1.select("rid", f1=pl.Series([lf(n) for n in r1["business_name"].to_list()]))
    oc = pl.read_parquet(f"{D}p1_canon_{split}_{c}_o.parquet", columns=["idx", "aempty"]).rename({"idx": "oid"})
    r2 = pl.concat([pl.read_parquet(f"{D}{split}_source{k}.parquet", columns=["business_name"]) for k in (2, 3)]).with_row_index("oid").with_columns(pl.col("oid").cast(pl.UInt32)).join(oc, on="oid")
    r2 = r2.select("oid", "aempty", f2=pl.Series([lf(n) for n in r2["business_name"].to_list()]))
    return r1, r2


def lr_table(T):
    """T: true pairs with f1, f2 -> LR(f2 | f1) = P(f2|f1)/P(f2)."""
    j = T.group_by("f1", "f2").agg(n=pl.len())
    j = j.with_columns(n1=pl.col("n").sum().over("f1"), n2=pl.col("n").sum().over("f2"), N=pl.col("n").sum())
    return j.select("f1", "f2", lr=((pl.col("n") + 0.5) / pl.col("n1")) / (pl.col("n2") / pl.col("N")))


def reweight(U, F, S, lr):
    """U: (rid, oid, p) union; F: name-only ambiguous oids; S: rid->f1; returns U with p replaced for those records."""
    x = U.join(F, on="oid").join(S, on="rid").join(lr, on=["f1", "f2"], how="left").with_columns(pl.col("lr").fill_null(0.01))
    x = x.with_columns(ps=pl.col("p").sum().over("oid").clip(0, 1), z=(pl.col("p") * pl.col("lr")).sum().over("oid"))
    x = x.with_columns(pn=pl.col("p") * pl.col("lr") / (pl.col("z") + 1 - pl.col("ps")))
    return U.join(x.select("rid", "oid", "pn"), on=["rid", "oid"], how="left").with_columns(p=pl.coalesce("pn", "p")).drop("pn")


if __name__ == "__main__":
    c = sys.argv[1]
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    s1c = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == c).select("rid").join(keep, on="rid", how="semi")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid"]).join(s1c, on="rid", how="semi")
    S, O = forms("train", c)
    T = gt.join(S, on="rid").join(O.select("oid", "f2"), on="oid")
    print(T.group_by("f1", "f2").len().pivot(on="f2", index="f1", values="len").fill_null(0))
    U = pl.read_parquet(f"{DN}train_joint9.parquet", columns=["rid", "oid", "pj", "country"]).filter(pl.col("country") == c).select("rid", "oid", p="pj")
    # ambiguous name-only records: name-only and >= 2 candidate S1 sharing the canonical name
    s1n = pl.read_parquet(f"{D}p1_canon_train_{c}_s1.parquet", columns=["idx", "cns"]).rename({"idx": "rid"}).join(s1c, on="rid", how="semi")
    sh = s1n.filter(pl.col("cns") != "").group_by("cns").agg(nsh=pl.len())
    on = pl.read_parquet(f"{D}p1_canon_train_{c}_o.parquet", columns=["idx", "cns", "aempty"]).rename({"idx": "oid"}).filter(pl.col("aempty")).join(sh, on="cns", how="left")
    fold = (pl.col("rid").hash(seed=11) % 2)
    base = decide.ef_predict(U.select("rid", "oid", p2="p"), "p2", 0.0, 1.0, 0.05)
    f0 = ev.macro_f05(base, gt, s1c["rid"])
    for lab, Fsel in (("ambiguous (name shared by 2+)", on.filter(pl.col("nsh") >= 2)), ("all name-only", on)):
        F = Fsel.select("oid").join(O.select("oid", "f2"), on="oid")
        Us = []
        for f in (0, 1):   # LR from truth of entities in fold f, applied to candidate pairs of entities... (records compete across folds; table is population-level)
            lr = lr_table(T.filter(fold == f))
            Us.append(lr)
        # honest: re-weight with the table from fold 1-f for records whose candidates are in fold f -> use record-level fold by oid hash
        lr0, lr1 = Us
        Uf = [reweight(U, F.filter(pl.col("oid").hash(seed=5) % 2 == g), S, (lr1 if g == 0 else lr0)) for g in (0, 1)]
        # merge: take reweighted p for each record group
        Un = U.join(pl.concat([Uf[0].join(F.filter(pl.col("oid").hash(seed=5) % 2 == 0).select("oid"), on="oid", how="semi"),
                               Uf[1].join(F.filter(pl.col("oid").hash(seed=5) % 2 == 1).select("oid"), on="oid", how="semi")]).rename({"p": "pn"}),
                    on=["rid", "oid"], how="left").with_columns(p=pl.coalesce("pn", "p")).drop("pn")
        for cc in (1.0,):
            new = decide.ef_predict(Un.select("rid", "oid", p2="p"), "p2", 0.0, cc, 0.05)
            f1 = ev.macro_f05(new, gt, s1c["rid"])
            add = new.join(base, on=["rid", "oid"], how="anti"); rem = base.join(new, on=["rid", "oid"], how="anti")
            pa = add.join(gt, on=["rid", "oid"], how="semi").height / max(add.height, 1)
            pr = rem.join(gt, on=["rid", "oid"], how="semi").height / max(rem.height, 1)
            print(f"{c} {lab:<30s} base {f0:.5f} -> {f1:.5f} ({f1 - f0:+.5f})  added {add.height:,} (true {pa:.1%})  removed {rem.height:,} (true {pr:.1%})", flush=True)
