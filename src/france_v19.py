"""France v19 = v18 + rule S: exact canonical name + same first house number + street words similar (token-set >= 75)
+ record taken by nobody -> accept. Proxy (India / US with the France recipe, TRUE labels): precision 0.89-1.00.
  python france_v19.py proxy | france
"""
import re
import sys

import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

import evaluate as ev
from france_cats import canon

D = "C:/ML/Challenges/amazon/work/data/"
N = "D:/amazon_er/france_night/data/"


def words(s):
    return " ".join(t for t in (s or "").split(" ") if not re.search(r"\d", t))


def rule_s(c, split, C):
    P = pl.read_parquet(f"{N}pool_{c}_blend.parquet").filter((pl.col("name") == "canon_equal") & (pl.col("addr") == "num same, words differ"))
    s1, o, _ = canon(split, c)
    P = P.join(s1.select(rid="idx", a1="cak"), on="rid").join(o.select(oid="idx", a2="cak"), on="oid")
    st = cpdist([words(x) for x in P["a1"].to_list()], [words(x) for x in P["a2"].to_list()], scorer=fuzz.token_set_ratio, workers=-1)
    P = P.with_columns(st=pl.Series(st)).filter(pl.col("st") >= 75)
    return (P.join(C.select("oid"), on="oid", how="anti").join(C, on=["rid", "oid"], how="anti")
            .sort(["st", "p"], descending=True).unique("oid", keep="first").select("rid", "oid"))


def proxy():
    import france_v18 as v18
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    for c in ("India", "US"):
        s1 = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == c).join(keep, on="rid", how="semi")
        gt = pl.read_parquet(f"{D}train_gt_rid.parquet").join(s1, on="rid", how="semi").select("rid", "oid")
        f = lambda P: ev.macro_f05(P.join(s1, on="rid", how="semi"), gt, s1["rid"])
        _, _, C, _ = v18.v18(c)
        S = rule_s(c, "train", C)
        prec = S.join(gt.with_columns(y=pl.lit(1)), on=["rid", "oid"], how="left")["y"].fill_null(0).mean() if S.height else float("nan")
        print(f"{c:<6s} v18 {f(C):.5f} -> + rule S {f(pl.concat([C, S])):.5f} (added {S.height:,}, precision {prec:.3f})", flush=True)


def france():
    C = pl.read_parquet(f"{N}france_v18_pairs.parquet").select("rid", "oid")
    S = rule_s("France", "test", C)
    out = pl.concat([C, S]).unique()
    out.write_parquet(f"{N}france_v19_pairs.parquet")
    print(f"France v19: {out.height:,} pairs ({out.height / 259452:.3f}/S1); rule S added {S.height:,}")


if __name__ == "__main__":
    {"proxy": proxy, "france": france}[sys.argv[1]]()
