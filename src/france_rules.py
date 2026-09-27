"""France rules, validated on the unseen-country proxy (India / US with the France recipe, true labels).

Base = France recipe: stage-1 (LOCO for India/US, v12 stage-1 for France) + dense add-on, rule ef(0, c=4).
Pool = candidate pairs the base REJECTS whose record is taken by NO entity (free record), p >= 0.005.
Each pool pair gets its generator category (name relation x address relation x S1-address shared). For every
category: pool pairs per S1 and true share on India / US (= precision of adding them) and pool pairs per S1 in France.
Rules = categories to add; a free record goes to its highest-p pool entity only (one owner). Their effect on the
India / US F0.5 (true labels) is measured before anything is applied to France.

  python france_rules.py scan            -> D:/amazon_er/france_night/data/pool_{India,US,France}.parquet + table
  python france_rules.py eval <cats...>  -> F0.5 India/US base vs base + rules (and France pairs added per S1)
"""
import os
import sys

import polars as pl

import decide
import evaluate as ev
from france_cats import canon
from p1_catalog import addr_rel, name_rel
from p1_decoy2 import refine
from p2_france_diag import addr_cls

D = "C:/ML/Challenges/amazon/work/data/"
DN = "D:/amazon_er/dense/"
OUT = "D:/amazon_er/france_night/data/"


BASE = os.environ.get("BASE", "p")      # "blend": 50/50 blend of p and the joint pass (the validated France recipe)
SFX = "" if BASE == "p" else "_blend"


def base_union(c):
    if BASE == "blend":
        if c == "France":
            U = pl.read_parquet("D:/amazon_er/france_night/data/france_union.parquet").select("rid", "oid", p=pl.col("pb").cast(pl.Float32))
        else:
            U = (pl.scan_parquet("D:/amazon_er/france_night/data/loco_joint_U.parquet").filter(pl.col("country") == c)
                 .select("rid", "oid", p=(0.5 * pl.col("p") + 0.5 * pl.col("pj")).cast(pl.Float32)).collect())
        acc = decide.ef_predict(U.select("rid", "oid", pl.col("p").alias("p2")), "p2", 0.0, 4.0, 0.05).with_columns(acc=pl.lit(1))
        return U.join(acc, on=["rid", "oid"], how="left").with_columns(pl.col("acc").fill_null(0))
    if c == "France":
        K = pl.read_parquet(f"{D}test_final_v12_ce_cectx_twin.parquet", columns=["rid", "oid", "country", "p1"]).filter(
            pl.col("country") == "France").select("rid", "oid", p="p1")
        N = pl.read_parquet(f"{DN}test_pn_france.parquet").select("rid", "oid", p=pl.col("pn").cast(pl.Float32))
    else:
        K = pl.read_parquet(f"{D}train_oof_locoh.parquet", columns=["rid", "oid", "country", "p1"]).filter(
            pl.col("country") == c).select("rid", "oid", p="p1")
        N = pl.read_parquet(f"{DN}train_pn_loco.parquet", columns=["rid", "oid", "pn", "country"]).filter(
            pl.col("country") == c).select("rid", "oid", p=pl.col("pn").cast(pl.Float32))
    U = pl.concat([K, N.join(K, on=["rid", "oid"], how="anti")])
    acc = decide.ef_predict(U.select("rid", "oid", pl.col("p").alias("p2")), "p2", 0.0, 4.0, 0.05).with_columns(acc=pl.lit(1))
    return U.join(acc, on=["rid", "oid"], how="left").with_columns(pl.col("acc").fill_null(0))


def categorise(P, split, c):
    s1, o, vocab = canon(split, c)
    x = (P.join(s1.select(rid="idx", cn1="cn", cns1="cns", cak1="cak", na="na"), on="rid")
         .join(o.select(oid="idx", cn2="cn", cns2="cns", cak2="cak", ae2="aempty"), on="oid"))
    names, addrs = [], []
    for r in x.select("cn1", "cn2", "cns1", "cns2", "cak1", "cak2", "ae2").iter_rows():
        A, B = list(r[0]), list(r[1])
        nr = refine(name_rel(A, B, r[2], r[3]), A, B, vocab)
        if nr == "word_swap_real_word":
            olds = [t for t in A if t not in B]
            if olds:
                i = A.index(olds[0])
                nr += " (first)" if i == 0 else " (last)" if i == len(A) - 1 else " (middle)"
        names.append(nr)
        addrs.append(addr_cls(addr_rel(r[4].split(" ") if r[4] else [], r[5].split(" ") if r[5] else [], r[6])))
    return x.select("rid", "oid", "p", *([c for c in ("y",) if c in x.columns]), "na").with_columns(
        name=pl.Series(names), addr=pl.Series(addrs), sh=(pl.col("na").fill_null(0) > 1))


def scan():
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").select("rid", "oid").with_columns(y=pl.lit(1, pl.Int8))
    tabs = {}
    for c in ("India", "US", "France"):
        U = base_union(c)
        if c != "France":
            U = U.join(keep, on="rid", how="semi")
        taken = U.filter(pl.col("acc") == 1).select("oid").unique()
        pool = U.filter((pl.col("acc") == 0) & (pl.col("p") >= 0.005)).join(taken, on="oid", how="anti")
        if c != "France":
            pool = pool.join(gt, on=["rid", "oid"], how="left").with_columns(pl.col("y").fill_null(0))
        x = categorise(pool, "test" if c == "France" else "train", c)
        x.write_parquet(f"{OUT}pool_{c}{SFX}.parquet")
        n_s1 = (pl.read_parquet(f"{D}test_s1_norm.parquet", columns=["country"]).filter(pl.col("country") == c).height if c == "France"
                else pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == c).join(keep, on="rid", how="semi").height)
        agg = [pl.len().alias("n")] + ([pl.col("y").mean().alias("true")] if c != "France" else [])
        tabs[c] = x.group_by("name", "addr", "sh").agg(*agg).with_columns((pl.col("n") / n_s1).alias("perS1")).drop("n")
        print(f"{c}: base accepts {U['acc'].sum():,}; free-record rejected pool {x.height:,} ({x.height / n_s1:.3f}/S1)", flush=True)
    key = ["name", "addr", "sh"]
    t = (tabs["France"].rename({"perS1": "FR"})
         .join(tabs["India"].rename({"perS1": "IN", "true": "tIN"}), on=key, how="full", coalesce=True)
         .join(tabs["US"].rename({"perS1": "US", "true": "tUS"}), on=key, how="full", coalesce=True).fill_null(0))
    t.write_parquet(f"{OUT}pool_table{SFX}.parquet")
    print(f"{'name':<28s}{'addr':<26s}{'sh':<3s}{'FR/S1':>7s}{'IN/S1':>7s}{'US/S1':>7s}{'trueIN':>7s}{'trueUS':>7s}")
    for r in t.sort("FR", descending=True).head(40).iter_rows(named=True):
        print(f"{r['name'][:27]:<28s}{r['addr'][:25]:<26s}{'Y' if r['sh'] else '-':<3s}{r['FR']:7.3f}{r['IN']:7.3f}{r['US']:7.3f}{r['tIN']:7.2f}{r['tUS']:7.2f}")


def add_rules(U, pool, cats):
    """cats: list of (name, addr, sh-or-None). Adds each free record to its best matching pool entity."""
    m = pl.lit(False)
    for nm, ad, sh in cats:
        cond = (pl.col("name") == nm) & (pl.col("addr") == ad)
        if sh is not None:
            cond = cond & (pl.col("sh") == sh)
        m = m | cond
    add = pool.filter(m).sort("p", descending=True).unique("oid", keep="first").select("rid", "oid")
    return pl.concat([U.filter(pl.col("acc") == 1).select("rid", "oid"), add]).unique(), add


def evaluate(cats):
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").select("rid", "oid")
    for c in ("India", "US"):
        U = base_union(c).join(keep, on="rid", how="semi")
        pool = pl.read_parquet(f"{OUT}pool_{c}{SFX}.parquet")
        s1 = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == c).join(keep, on="rid", how="semi")
        g = gt.join(s1, on="rid", how="semi")
        b = ev.macro_f05(U.filter(pl.col("acc") == 1).select("rid", "oid"), g, s1["rid"])
        new, add = add_rules(U, pool, cats)
        f = ev.macro_f05(new, g, s1["rid"])
        prec = add.join(g.with_columns(y=pl.lit(1)), on=["rid", "oid"], how="left")["y"].fill_null(0).mean() if add.height else 0
        print(f"{c}: base {b:.5f} -> {f:.5f} ({f - b:+.5f}); added {add.height:,} pairs ({add.height / s1.height:.4f}/S1), precision {prec:.3f}")
    pool = pl.read_parquet(f"{OUT}pool_France{SFX}.parquet")
    Uf = base_union("France")
    _, add = add_rules(Uf, pool, cats)
    n = pl.read_parquet(f"{D}test_s1_norm.parquet", columns=["country"]).filter(pl.col("country") == "France").height
    print(f"France: would add {add.height:,} pairs ({add.height / n:.4f}/S1)")
    return add


if __name__ == "__main__":
    if sys.argv[1] == "scan":
        scan()
