"""France v18 = mix of the France blend with the MEAN of e5-large and bge-m3 (weight 0.3, uncertain pairs) + R2/R3 rules
- near-number VETO + PROTECT (pairs the CE mix removes in categories true >= 0.85 in both proxy countries, plus the
audited France truncation / filler categories). Proxy first (India / US true labels), then France.
  python france_v18.py proxy | france
"""
import os
import sys

os.environ["W_S"] = "0.3"
os.environ["S_MODE"] = "avg"
import importlib

import polars as pl

import evaluate as ev
import france_v15 as f0
from france_rules import categorise

D = "C:/ML/Challenges/amazon/work/data/"
N = "D:/amazon_er/france_night/data/"
PROT = [("canon_equal", "canon_equal"), ("canon_equal", "num same, words differ"), ("canon_equal", "num missing")]
FR_EXTRA = [("truncation_single_prefix", "canon_equal"), ("truncation", "canon_equal")]


def recipe(c, ws):
    os.environ["W_S"] = ws
    importlib.reload(f0)
    b = f0.build(c)
    return pl.concat([b["base"], b["r2"], b["r3"].select("rid", "oid")]).join(b["veto"], on=["rid", "oid"], how="anti").unique()


def v18(c):
    A = recipe(c, "0")        # without the CE mix
    B = recipe(c, "0.3")      # with the averaged CE mix
    U, split = f0.frames(c)   # (W_S=0.3 still set: pb = mixed probability, used only to order re-adds)
    rem = A.join(B, on=["rid", "oid"], how="anti").join(U.select("rid", "oid", p="pb"), on=["rid", "oid"], how="left")
    x = categorise(rem, split, c)
    cats = PROT + (FR_EXTRA if c == "France" else [])
    m = pl.lit(False)
    for nm, ad in cats:
        m = m | ((pl.col("name") == nm) & (pl.col("addr") == ad))
    back = x.filter(m).join(B.select("oid"), on="oid", how="anti").sort("p", descending=True).unique("oid", keep="first").select("rid", "oid")
    return A, B, pl.concat([B, back]).unique(), back


def proxy():
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    for c in ("India", "US"):
        s1 = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == c).join(keep, on="rid", how="semi")
        gt = pl.read_parquet(f"{D}train_gt_rid.parquet").join(s1, on="rid", how="semi").select("rid", "oid")
        f = lambda P: ev.macro_f05(P.join(s1, on="rid", how="semi"), gt, s1["rid"])
        A, B, C, back = v18(c)
        print(f"{c:<6s} no CE mix {f(A):.5f} | avg-CE mix {f(B):.5f} | + protect {f(C):.5f} (re-added {back.height:,})", flush=True)


def france():
    A, B, C, back = v18("France")
    C.write_parquet(f"{N}france_v18_pairs.parquet")
    print(f"France v18: {C.height:,} pairs ({C.height / 259452:.3f}/S1); protected {back.height:,}")


if __name__ == "__main__":
    {"proxy": proxy, "france": france}[sys.argv[1]]()
