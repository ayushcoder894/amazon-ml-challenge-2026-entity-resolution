"""France v15 = v14 (blend + R2 rules) + two fixes found in the hand audit of the v14 changes:
  R3  raw-name acronym / glued-name / handle / website of the entity at the entity's own address (canonical address equal),
      record taken by nobody. Uses the RAW names: canonical names drop filler words ("Deleves Groupe" -> "deleves"), which
      hides acronyms like "DG" from R2.
  VETO pairs that the blend ADDS (not accepted by the v13 recipe) whose first house numbers differ (both present, neither a
      prefix of the other) -> France's sister-company / near-number decoys.
Validation on the unseen-country proxy (India / US with a model that never saw them, TRUE labels) before France.

  python france_v15.py proxy     -> F0.5 India / US: blend, +R2, +R2+R3, +R2+R3+VETO
  python france_v15.py france    -> data/france_v15_pairs.parquet
"""
import re
import sys
import unicodedata

import polars as pl

import decide
import evaluate as ev
from france_cats import canon

D = "C:/ML/Challenges/amazon/work/data/"
DN = "D:/amazon_er/dense/"
NIGHT = "D:/amazon_er/france_night/"
LEGAL = {"sas", "sarl", "sasu", "eurl", "sa", "sci", "snc", "ei", "llc", "inc", "ltd", "pvt", "private", "limited", "corp",
         "co", "company", "llp", "pc", "pllc", "lp", "plc", "incorporated", "corporation", "l", "c"}
PART = {"de", "des", "du", "la", "le", "les", "d", "of", "the", "and", "et", "a", "l", "en", "au", "aux"}
R2 = [("invented_pseudo_word", "canon_equal", False), ("truncation_single_prefix", "canon_equal", False),
      ("extra_word", "canon_equal", False), ("acronym", "canon_equal", None), ("truncation", "canon_equal", False),
      ("truncation_single_prefix", "canon_equal", True)]


def fold(s):
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower()


def toks(name):
    t = re.findall(r"[a-z0-9]+", fold(name).replace("l.l.c", "llc").replace("s.a.r.l", "sarl").replace("s.a.s", "sas"))
    return t


def raw_form_match(ename, rname):
    """acronym / glued / handle / website of the entity name."""
    r = fold(rname).strip()
    letters = re.sub(r"[^a-z]", "", r)
    et = toks(ename)
    core = [t for t in et if t not in LEGAL]
    core_np = [t for t in core if t not in PART]
    if not core_np:
        return None
    if 2 <= len(letters) <= 6 and re.fullmatch(r"[a-z]+", re.sub(r"[\s\.\-]", "", r) or "x"):
        noparen = [t for t in toks(re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", ename or "")) if t not in LEGAL and t not in PART]
        if letters in ("".join(t[0] for t in core_np), "".join(t[0] for t in core), "".join(t[0] for t in noparen)):
            return "acronym_raw"
    g = re.sub(r"(^www\.|\.com$|\.fr$|^#|^@)", "", r.replace(" ", ""))
    g = re.sub(r"(\.com|\.fr)$", "", g)
    g = re.sub(r"[^a-z0-9]", "", g)
    if len(g) >= 5:
        for cand in ("".join(core), "".join(core_np), "".join(et)):
            if len(cand) >= 5 and (g == cand or (g.startswith(cand) and len(g) - len(cand) <= 5)):
                return "glued_raw"
    return None


def first_num(cak):
    m = re.search(r"\d+", cak or "")
    return m.group(0) if m else None


import os

W_S = float(os.environ.get("W_S", 0))  # > 0: mix the e5-large cross-encoder score into pb on the uncertain pairs


def mix_q7(sc, q, a):
    """Mix only covered pairs; absent Qwen scores retain the current CE mean."""
    if not 0 <= a <= 1:
        raise ValueError("Q7_A must be between zero and one")
    q = q.select(pl.col("rid", "oid").cast(pl.UInt32), pl.col("s").cast(pl.Float64))
    if q.select(pl.struct("rid", "oid").n_unique()).item() != q.height:
        raise ValueError("Duplicate Qwen pair keys")
    if q.filter(pl.col("s").is_null() | ~pl.col("s").is_finite() | ~pl.col("s").is_between(0, 1)).height:
        raise ValueError("Qwen scores must be finite probabilities")
    return (sc.join(q.rename({"s": "q"}), on=["rid", "oid"], how="left")
            .with_columns(s=pl.when(pl.col("q").is_null()).then(pl.col("s"))
                          .otherwise((1 - a) * pl.col("s") + a * pl.col("q")).cast(sc.schema["s"])).drop("q"))


def frames(c):
    """Union with base probability p (v13 recipe) and blend pb for country c (proxy) or France."""
    if c == "France":
        U = pl.read_parquet(f"{NIGHT}data/france_union.parquet").select("rid", "oid", "p", "pb")
        split = "test"
        sfile = "D:/amazon_er/gpu3_results/ce3_e5l_france_test.parquet"
    else:
        U = (pl.scan_parquet(f"{NIGHT}data/loco_joint_U.parquet").filter(pl.col("country") == c)
             .select("rid", "oid", "p", pb=0.5 * pl.col("p") + 0.5 * pl.col("pj")).collect())
        split = "train"
        sfile = "D:/amazon_er/gpu3_results/ce3_e5l_loco.parquet"
    if W_S > 0 and os.environ.get("S_MODE", "e5l") in ("avg", "q7"):  # mean of e5-large and bge-m3
        g = sfile.replace("ce3_e5l_", "ce3_bgem3_")
        sc = (pl.read_parquet(sfile).rename({"s": "s1_"}).join(pl.read_parquet(g).rename({"s": "s2_"}), on=["rid", "oid"], how="inner")
              .select("rid", "oid", s=(pl.col("s1_") + pl.col("s2_")) / 2))
        if os.environ["S_MODE"] == "q7":  # Qwen2.5-7B score where available: s = (1 - Q7_A) * CE mean + Q7_A * q7
            a = float(os.environ.get("Q7_A", 1 / 3))
            sc = mix_q7(sc, pl.read_parquet(os.environ["Q7_FILE"]), a)
        U = U.join(sc, on=["rid", "oid"], how="left").with_columns(
            pb=pl.when(pl.col("s").is_not_null()).then((1 - W_S) * pl.col("pb") + W_S * pl.col("s")).otherwise(pl.col("pb"))).drop("s")
    elif W_S > 0:
        U = U.join(pl.read_parquet(sfile), on=["rid", "oid"], how="left").with_columns(
            pb=pl.when(pl.col("s").is_not_null()).then((1 - W_S) * pl.col("pb") + W_S * pl.col("s")).otherwise(pl.col("pb"))).drop("s")
    return U, split


def build(c):
    U, split = frames(c)
    base = decide.ef_predict(U.select("rid", "oid", pl.col("pb").alias("p2")), "p2", 0.0, 4.0, 0.05).select("rid", "oid")
    v13 = decide.ef_predict(U.select("rid", "oid", pl.col("p").alias("p2")), "p2", 0.0, 4.0, 0.05).select("rid", "oid")
    # R2 on the blend pools
    pool = pl.read_parquet(f"{NIGHT}data/pool_{c}_blend.parquet")
    m = pl.lit(False)
    for nm, ad, sh in R2:
        cond = (pl.col("name") == nm) & (pl.col("addr") == ad)
        m = m | (cond if sh is None else cond & (pl.col("sh") == sh))
    r2 = (pool.filter(m).join(base.select("oid"), on="oid", how="anti").sort("p", descending=True)
          .unique("oid", keep="first").select("rid", "oid"))
    # R3 raw forms, own address, free record, not already accepted
    s1, o, _ = canon(split, c)
    taken = pl.concat([base, r2]).select("oid").unique()
    P = (U.select("rid", "oid", "pb").join(taken, on="oid", how="anti")
         .join(s1.select(rid="idx", cak1="cak"), on="rid").join(o.select(oid="idx", cak2="cak", ae="aempty"), on="oid")
         .filter(~pl.col("ae") & (pl.col("cak1") == pl.col("cak2")) & (pl.col("cak1") != "")))
    raw1 = pl.read_parquet(f"{D}{split}_source1.parquet", columns=["business_name"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
    raw2 = pl.concat([pl.read_parquet(f"{D}{split}_source{k}.parquet", columns=["business_name"]) for k in (2, 3)]).with_row_index("oid").with_columns(pl.col("oid").cast(pl.UInt32))
    P = P.join(raw1.rename({"business_name": "en"}), on="rid").join(raw2.rename({"business_name": "rn"}), on="oid")
    kinds = [raw_form_match(a, b) for a, b in P.select("en", "rn").iter_rows()]
    P = P.with_columns(kind=pl.Series(kinds, dtype=pl.String)).filter(pl.col("kind").is_not_null())
    r3 = P.sort("pb", descending=True).unique("oid", keep="first").select("rid", "oid", "kind")
    # VETO: blend-added pairs whose first house numbers differ
    add = base.join(v13, on=["rid", "oid"], how="anti")
    A = (add.join(s1.select(rid="idx", cak1="cak"), on="rid").join(o.select(oid="idx", cak2="cak"), on="oid"))
    n1 = [first_num(x) for x in A["cak1"].to_list()]
    n2 = [first_num(x) for x in A["cak2"].to_list()]
    bad = [a is not None and b is not None and a != b and not a.startswith(b) and not b.startswith(a) for a, b in zip(n1, n2)]
    veto = A.filter(pl.Series(bad)).select("rid", "oid")
    return dict(base=base, v13=v13, r2=r2, r3=r3, veto=veto)


def proxy():
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    for c in ("India", "US"):
        b = build(c)
        s1 = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == c).join(keep, on="rid", how="semi")
        gt = pl.read_parquet(f"{D}train_gt_rid.parquet").join(s1, on="rid", how="semi").select("rid", "oid")
        f = lambda P: ev.macro_f05(P.join(s1, on="rid", how="semi"), gt, s1["rid"])
        prec = lambda P: P.join(gt.with_columns(y=pl.lit(1)), on=["rid", "oid"], how="left")["y"].fill_null(0).mean() if P.height else float("nan")
        v = {"v13 recipe": b["v13"], "blend": b["base"]}
        v["blend+R2"] = pl.concat([b["base"], b["r2"]])
        v["blend+R2+R3"] = pl.concat([v["blend+R2"], b["r3"].select("rid", "oid")])
        v["blend+R2+R3+VETO"] = v["blend+R2+R3"].join(b["veto"], on=["rid", "oid"], how="anti")
        for k, P in v.items():
            print(f"{c:<6s} {k:<18s} F0.5 {f(P):.5f}  ({P.height:,} pairs)", flush=True)
        print(f"{c:<6s} R3 adds {b['r3'].height:,} (precision {prec(b['r3'].select('rid', 'oid')):.3f}: "
              + ", ".join(f"{k} {n:,}" for k, n in b["r3"].group_by("kind").len().iter_rows()) + f"); VETO removes {b['veto'].height:,} "
              f"(true share {prec(b['veto']):.3f})", flush=True)


def france():
    b = build("France")
    P = pl.concat([b["base"].with_columns(src=pl.lit("blend")), b["r2"].with_columns(src=pl.lit("R2")),
                   b["r3"].select("rid", "oid").with_columns(src=pl.lit("R3"))])
    P = P.join(b["veto"], on=["rid", "oid"], how="anti")
    P.write_parquet(f"{NIGHT}data/france_v15_pairs{os.environ.get('FR_SFX', '')}.parquet")
    print(f"France v15: blend {b['base'].height:,} + R2 {b['r2'].height:,} + R3 {b['r3'].height:,} "
          f"({', '.join(f'{k} {n:,}' for k, n in b['r3'].group_by('kind').len().iter_rows())}) - VETO {b['veto'].height:,} "
          f"= {P.height:,} pairs ({P.height / 259452:.3f} per S1); v13 recipe {b['v13'].height:,}")


if __name__ == "__main__":
    {"proxy": proxy, "france": france}[sys.argv[1]]()
