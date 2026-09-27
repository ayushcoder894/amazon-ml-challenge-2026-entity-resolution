"""Friend's 5-tier rules for unassigned name-only records, measured on top of v21.
Proxy (India/US, France recipe v18 + rule S + v21 name-only rule): precision and F0.5 change per tier, true labels.
France test: volumes per tier (no labels). Tier 2 (city in record name) counted on France only.
  python fr_friend_tiers.py India|US|France"""
import importlib
import os
import re
import sys
import unicodedata

import polars as pl

import evaluate as ev
from no_legal import legal_us
from no_signal import legal

D = "C:/ML/Challenges/amazon/work/data/"
N = "D:/amazon_er/france_night/data/"
FR_FORMS = {"SASU", "SAS", "EURL", "SARL", "SCI", "SA", "EI", "SNC", "SELARL", "SELAS", "SCM", "SCP", "GIE", "EIRL", "EARL", "GAEC"}


def fr_legal(n):
    s = unicodedata.normalize("NFD", (n or "").upper()).encode("ascii", "ignore").decode()
    s = re.sub(r"\b([A-Z])\.(?=[A-Z]\.)", r"\1", s).replace(".", "")
    t = re.sub(r"[^A-Z0-9 ]", " ", s).split()
    f = [x for x in t if x in FR_FORMS]
    return f[-1] if f else "none"


def main(c):
    lf = fr_legal if c == "France" else (legal if c == "India" else legal_us)
    split = "test" if c == "France" else "train"
    if c == "France":
        s1 = pl.read_parquet(f"{D}test_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == c).select("rid")
        P = pl.read_parquet(f"{N}france_v21_pairs.parquet").select("rid", "oid")
        U = pl.read_parquet(f"{N}france_union.parquet").select("rid", "oid", "pb")
        gt = None
    else:
        keep = pl.read_parquet(f"{D}keep_v11.parquet")
        s1 = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == c).join(keep, on="rid", how="semi").select("rid")
        gt = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid"]).join(s1, on="rid", how="semi")
        import france_v18 as v18
        from france_v19 import rule_s
        _, _, C, _ = v18.v18(c)
        P = pl.concat([C, rule_s(c, "train", C)]).unique().join(s1, on="rid", how="semi")
        os.environ["W_S"] = "0"
        import france_v15 as f0
        importlib.reload(f0)
        U, _ = f0.frames(c)
        U = U.select("rid", "oid", "pb").join(s1, on="rid", how="semi")
    P = P.select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32))
    U = U.with_columns(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32))
    cs1 = pl.read_parquet(f"{D}p1_canon_{split}_{c}_s1.parquet", columns=["idx", "cns"]).rename({"idx": "rid"}).join(s1, on="rid", how="semi")
    co = pl.read_parquet(f"{D}p1_canon_{split}_{c}_o.parquet", columns=["idx", "cns", "aempty"]).rename({"idx": "oid"})
    ae = co.filter(pl.col("aempty")).select("oid")

    def top2(P):
        return (U.join(ae, on="oid", how="semi").join(P.select("oid"), on="oid", how="anti")
                .sort("pb", descending=True).group_by("oid", maintain_order=True)
                .agg(rid=pl.col("rid").first(), p1=pl.col("pb").first(), p2=pl.col("pb").slice(1, 1).first(), cands=pl.col("rid"), ps=pl.col("pb"))
                .with_columns(pl.col("p2").fill_null(0.0)))

    if c != "France":  # proxy: apply the v21 rule first so everything below is on top of v21
        X0 = top2(P)
        P = pl.concat([P, X0.filter((pl.col("p1") >= 0.5) & (pl.col("p2") < 0.1)).select("rid", "oid")]).unique()
    X = top2(P)
    k = cs1.filter(pl.col("cns") != "").group_by("cns").agg(shared=pl.len())
    me = P.group_by("rid").agg(me=pl.len())
    X = (X.join(co.select("oid", rcns="cns"), on="oid").join(k, left_on="rcns", right_on="cns", how="left")
         .join(cs1.rename({"cns": "ccns"}), on="rid", how="left").join(me, on="rid", how="left")
         .with_columns(pl.col("shared", "me").fill_null(0), stem_eq=(pl.col("rcns") == pl.col("ccns")) & (pl.col("rcns") != ""),
                       dp=pl.col("p1") - pl.col("p2"), odds=pl.col("p1") / pl.col("p2").clip(1e-4, None)))
    raw1 = (pl.read_parquet(f"{D}{split}_source1.parquet", columns=["business_name", "business_address"]).with_row_index("rid")
            .with_columns(pl.col("rid").cast(pl.UInt32)).join(s1, on="rid", how="semi"))
    raw2 = (pl.concat([pl.read_parquet(f"{D}{split}_source{j}.parquet", columns=["business_name"]) for j in (2, 3)]).with_row_index("oid")
            .with_columns(pl.col("oid").cast(pl.UInt32)).join(X.select("oid"), on="oid", how="semi"))
    L1 = raw1.select("rid", L1=pl.Series([lf(n) for n in raw1["business_name"].to_list()]))
    X = X.join(raw2.select("oid", rname="business_name", Lq=pl.Series([lf(n) for n in raw2["business_name"].to_list()])), on="oid")
    # Tier 3: record has a legal form, exactly one candidate shares it, that candidate p >= 0.5
    cand = X.select("oid", "cands", "ps", "Lq").explode(["cands", "ps"]).rename({"cands": "rid", "ps": "p"}).join(L1, on="rid", how="left")
    t3 = (cand.filter((pl.col("Lq") != "none") & (pl.col("L1") == pl.col("Lq"))).group_by("oid")
          .agg(n=pl.len(), rid3=pl.col("rid").first(), p3=pl.col("p").first()).filter((pl.col("n") == 1) & (pl.col("p3") >= 0.5)))
    X = X.join(t3.select("oid", "rid3"), on="oid", how="left")
    X = X.with_columns(tier=pl.when(((pl.col("shared") == 1) | pl.col("stem_eq")) & (pl.col("p1") >= 0.45) & (pl.col("dp") >= 0.4)).then(pl.lit("T1"))
                       .when(pl.col("rid3").is_not_null()).then(pl.lit("T3"))
                       .when((pl.col("me") == 0) & (pl.col("p1") >= 0.5) & (pl.col("dp") >= 0.25) & (pl.col("odds") >= 2)).then(pl.lit("T4A"))
                       .when((pl.col("me") >= 1) & (pl.col("p1") >= 0.727) & (pl.col("dp") >= 0.35) & (pl.col("odds") >= 3)).then(pl.lit("T4B"))
                       .otherwise(pl.lit("T5")))
    X = X.with_columns(pick=pl.when(pl.col("tier") == "T3").then(pl.col("rid3")).otherwise(pl.col("rid")))
    print(f"\n==== {c}: unassigned name-only records on top of v21: {X.height:,} ({X.height / s1.height:.4f}/S1)", flush=True)
    if gt is not None:
        X = X.join(gt.rename({"rid": "pick"}).with_columns(y=pl.lit(1, pl.Int8)), on=["pick", "oid"], how="left").with_columns(pl.col("y").fill_null(0))
        f0v = ev.macro_f05(P, gt, s1["rid"])
        print(f"  proxy F0.5 with v21 rule: {f0v:.5f}")
        acc = P
        for t in ("T1", "T3", "T4A", "T4B"):
            b = X.filter(pl.col("tier") == t)
            f1 = ev.macro_f05(pl.concat([P, b.select(rid="pick", oid="oid")]), gt, s1["rid"])
            acc = pl.concat([acc, b.select(rid="pick", oid="oid")])
            print(f"  {t:<4s} adds {b.height:>6,} ({b.height / s1.height:.4f}/S1)  correct {b['y'].mean() if b.height else float('nan'):6.1%}  alone {f1 - f0v:+.5f}")
        f_all = ev.macro_f05(acc, gt, s1["rid"])
        print(f"  all tiers together: {f_all:.5f} ({f_all - f0v:+.5f})")
    else:
        print(X.group_by("tier").agg(n=pl.len(), per_S1=pl.len() / s1.height).sort("tier"))
        cities = (raw1.select(city=pl.col("business_address").str.split(",").list.eval(pl.element().str.strip_chars().str.to_uppercase()))
                  .explode("city").group_by("city").len().filter(pl.col("len") >= 500)["city"].to_list())
        cities = [x for x in cities if x and not any(ch.isdigit() for ch in x) and x not in ("HAUTS-DE-FRANCE", "NOUVELLE-AQUITAINE", "PAYS DE LA LOIRE")]
        hit = X.filter(pl.col("rname").str.to_uppercase().str.contains(r"\b(" + "|".join(re.escape(x) for x in cities) + r")\b"))
        print(f"  Tier 2: records whose name contains one of the {len(cities)} France cities ({', '.join(cities[:8])}...): {hit.height:,}")
        print(hit.select("rname").head(10))
        X.filter(pl.col("tier") != "T5").select("oid", "tier", rid="pick").write_parquet(f"{N}france_friend_tier_adds.parquet")


if __name__ == "__main__":
    main(sys.argv[1])
