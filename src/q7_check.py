"""Does the Qwen2.5-7B pair score (A100 v5, trained on one country, scored on the other) help the France recipe?
Proxy = India / US scored as unseen countries with the France recipe (v18 level: blend + CE mix + R2/R3 - veto,
then PROTECT), evaluated only on the entities Qwen scored (10% sample). Qwen is used only where the band p >= 0.2,
as on France (band_france_p20). Variants: current CE mean (baseline) vs Qwen mixed into the CE mean at share a,
at the current mix weight W_S = 0.3 and at 0.4.
  python q7_check.py <path to ce5_q7_loco.parquet>
"""
import importlib
import os
import sys
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

import evaluate as ev
from france_rules import categorise

D = "C:/ML/Challenges/amazon/work/data/"
R = "D:/amazon_er/gpu5_results/"
PROT = [("canon_equal", "canon_equal"), ("canon_equal", "num same, words differ"), ("canon_equal", "num missing")]


def build(c, ws, mode, a=None):
    os.environ["W_S"], os.environ["S_MODE"] = ws, mode
    if a is not None:
        os.environ["Q7_A"] = str(a)
    import france_v15 as f0
    importlib.reload(f0)
    b = f0.build(c)
    P = pl.concat([b["base"], b["r2"], b["r3"].select("rid", "oid")]).join(b["veto"], on=["rid", "oid"], how="anti").unique()
    U, split = f0.frames(c)
    return P, U, split


def protect(A, B, U, split, c, cats=None):
    rem = A.join(B, on=["rid", "oid"], how="anti").join(U.select("rid", "oid", p="pb"), on=["rid", "oid"], how="left")
    x = categorise(rem, split, c)
    m = pl.lit(False)
    for nm, ad in (PROT if cats is None else cats):
        m = m | ((pl.col("name") == nm) & (pl.col("addr") == ad))
    back = x.filter(m).join(B.select("oid"), on="oid", how="anti").sort("p", descending=True).unique("oid", keep="first").select("rid", "oid")
    return pl.concat([B, back]).unique()


def per_entity(P, gt, s1):
    x = (s1.select("rid").join(gt.group_by("rid").agg(nt=pl.len()), on="rid", how="left")
         .join(P.group_by("rid").agg(np=pl.len()), on="rid", how="left")
         .join(P.join(gt, on=["rid", "oid"]).group_by("rid").agg(tp=pl.len()), on="rid", how="left").fill_null(0))
    return x.with_columns(f=pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0)
                          .otherwise(1.25 * pl.col("tp") / (pl.col("np") + 0.25 * pl.col("nt"))))


def main(qfile, allow_fake=False):
    os.makedirs(R, exist_ok=True)
    fake = "fake" in Path(qfile).name.lower() or "smoke" in Path(qfile).name.lower()
    if fake and not allow_fake:
        raise ValueError("Refusing synthetic stand-in scores without --allow-fake; never use these for selection")
    rawq = pl.read_parquet(qfile).select(pl.col("rid", "oid").cast(pl.UInt32), "s")
    import france_v15 as f0
    f0.mix_q7(rawq.head(0), rawq, 0.5)  # key/range checks before any large evaluation
    digest = hashlib.sha256(Path(qfile).read_bytes()).hexdigest()
    band = pl.read_parquet("C:/ML/Challenges/amazon/work/gpu5/data/band_loco.parquet").select("rid", "oid", "country", "p", "y")
    q = rawq.join(band, on=["rid", "oid"], validate="1:1")
    if q.height != rawq.height:
        raise ValueError("Qwen keys outside the supplied LOCO band")
    result = dict(input=str(Path(qfile).resolve()), sha256=digest, synthetic=fake, countries={}, variants=[])
    from sklearn.metrics import roc_auc_score
    for c in ("India", "US"):
        x = q.filter(pl.col("country") == c)
        if x.height:
            print(f"{c}: Qwen scored {x.height:,} band pairs on {x['rid'].n_unique():,} entities; AUC Qwen {roc_auc_score(x['y'], x['s']):.4f} vs LOCO p {roc_auc_score(x['y'], x['p']):.4f}")
    qf = f"{R}{'SMOKE_' if fake else ''}q7_checked_{digest[:12]}_p20.parquet"
    q.filter(pl.col("p") >= 0.2).select("rid", "oid", "s").write_parquet(qf)
    os.environ["Q7_FILE"] = qf
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    for c in ("India", "US"):
        ents = q.filter(pl.col("country") == c).select("rid").unique()
        if ents.height == 0:
            continue
        s1 = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == c).join(keep, on="rid", how="semi")
        n_all = s1.height
        s1 = s1.join(ents, on="rid", how="semi")
        expected = band.filter((pl.col("country") == c) & (pl.col("rid").hash(seed=21) % 1000 < 100))
        missing = expected.join(rawq.select("rid", "oid"), on=["rid", "oid"], how="anti").height
        extra = q.filter(pl.col("country") == c).join(expected.select("rid", "oid"), on=["rid", "oid"], how="anti").height
        result["countries"][c] = dict(entities=s1.height, expected_pairs=expected.height, missing_pairs=missing, unexpected_pairs=extra)
        gt = pl.read_parquet(f"{D}train_gt_rid.parquet").join(s1, on="rid", how="semi").select("rid", "oid")
        f = lambda P: ev.macro_f05(P.join(s1, on="rid", how="semi"), gt, s1["rid"])
        A, _, _ = build(c, "0", "avg")
        print(f"\n{c}: {s1.height:,} scored entities; France recipe without CE mix {f(A):.5f}", flush=True)
        base = None
        base_rows = None
        from france_v19 import rule_s
        for lab, ws, mode, a in (("current: CE mean (e5l+bgem3), W 0.3", "0.3", "avg", None),
                                 ("+ Qwen share 1/3, W 0.3", "0.3", "q7", 1 / 3), ("+ Qwen share 1/2, W 0.3", "0.3", "q7", 0.5),
                                 ("Qwen only, W 0.3", "0.3", "q7", 1.0), ("+ Qwen share 1/2, W 0.4", "0.4", "q7", 0.5),
                                 ("current: CE mean, W 0.4", "0.4", "avg", None)):
            B, U, split = build(c, ws, mode, a)
            C = protect(A, B, U, split, c)
            C = pl.concat([C, rule_s(c, split, C)]).unique()
            fb, fc = f(B), f(C)
            base = fc if base is None else base
            rows = per_entity(C, gt, s1)
            if base_rows is None:
                base_rows = rows.select("rid", f0="f")
            delta = rows.join(base_rows, on="rid").with_columns(delta=pl.col("f") - pl.col("f0"),
                        subset=pl.when(pl.col("rid").hash(seed=927) % 2 == 0).then(pl.lit("select")).otherwise(pl.lit("evaluate")))
            stats = {r["subset"]:r for r in delta.group_by("subset").agg(n=pl.len(), delta=pl.col("delta").mean(),
                          se=pl.col("delta").std()/pl.len().sqrt()).to_dicts()}
            result["variants"].append(dict(country=c, label=lab, ws=float(ws), alpha=a, mode=mode, score=fc, delta=fc-base, subsets=stats))
            print(f"  {lab:<38s} mix {fb:.5f} | + protect + S {fc:.5f} ({fc - base:+.5f}); {stats}", flush=True)
            delta.write_parquet(f"{R}{'SMOKE_' if fake else ''}q7_{digest[:12]}_{c}_w{ws}_a{a}_{mode}_deltas.parquet")
    complete = set(result["countries"]) == {"India", "US"} and all(v["missing_pairs"] == 0 and v["unexpected_pairs"] == 0 for v in result["countries"].values())
    options = []
    for ws, alpha in [(0.3, 1/3), (0.3, 0.5), (0.3, 1.0)]:
        vs = [v for v in result["variants"] if v["ws"] == ws and v["alpha"] == alpha and v["mode"] == "q7"]
        if len(vs) == 2 and all(v["subsets"]["select"]["delta"] > 0 for v in vs):
            options.append((sum(v["subsets"]["select"]["delta"] for v in vs), ws, alpha, vs))
    pick = max(options, key=lambda x:x[0]) if options else None
    passed = bool(complete and not fake and pick and all(v["subsets"]["evaluate"]["delta"] > 0 for v in pick[3]))
    result.update(complete=complete, eligible_for_v20=passed,
                  selected=dict(ws=pick[1], alpha=pick[2]) if pick else None,
                  uncertainty="Entity SE is descriptive; competing entities are dependent. Positive point estimates are not proof of France gain.")
    out = Path(R) / f"{'SMOKE_' if fake else ''}q7_check_{digest[:12]}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"REPORT {out}; complete={complete}, eligible_for_v20={passed}", flush=True)


def smoke():
    from unittest.mock import patch
    import france_v15 as f0
    def df(**x):
        return pl.DataFrame(x).with_columns(pl.col("rid", "oid").cast(pl.UInt32))
    u = df(rid=[1,2,3], oid=[11,12,13], p=[.6,.7,.8], pb=[.6,.7,.8])
    e = df(rid=[1,2], oid=[11,12], s=[.2,.4])
    b = df(rid=[1,2], oid=[11,12], s=[.4,.6])
    q = df(rid=[1], oid=[11], s=[.9])
    def read(path, *args, **kwargs):
        p = str(path)
        return u if "france_union" in p else e if "e5l" in p else b if "bgem3" in p else q
    with patch.dict(os.environ, {"S_MODE":"q7", "Q7_FILE":"SMOKE", "Q7_A":"0.5"}), patch.object(f0, "W_S", .3), patch.object(pl, "read_parquet", side_effect=read):
        got, split = f0.frames("France")
        assert split == "test" and np.allclose(got["pb"], [.6,.64,.8])
    assert f0.mix_q7(e,q,0).equals(e.cast({"s":pl.Float64}))
    assert np.allclose(f0.mix_q7(e,q,1)["s"], [.9,.4])
    e32=e.with_columns(pl.col("s").cast(pl.Float32))
    mixed32=f0.mix_q7(e32,q,.5)
    assert mixed32.schema["s"] == pl.Float32
    assert mixed32.filter(pl.col("rid")==2).equals(e32.filter(pl.col("rid")==2))
    assert f0.mix_q7(e32,q,0).equals(e32)
    for bad in [pl.concat([q,q]), q.with_columns(s=pl.lit(float("nan"))), q.with_columns(s=pl.lit(1.1))]:
        try:
            f0.mix_q7(e,bad,.5)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid Qwen input accepted")
    s1=pl.DataFrame({"rid":pl.Series([1,2,3],dtype=pl.UInt32)})
    gt=df(rid=[1],oid=[11]); pred=df(rid=[1,2],oid=[11,12])
    assert np.isclose(per_entity(pred,gt,s1)["f"].mean(), ev.macro_f05(pred,gt,s1["rid"]))
    print("PASS: Qwen mixing, missing-score fallback, alpha endpoints, invalid-key/probability checks, macro F0.5; no real artifacts overwritten")


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("qfile", nargs="?")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-fake", action="store_true")
    args=ap.parse_args()
    if args.smoke:
        smoke()
    elif args.qfile:
        main(args.qfile,args.allow_fake)
    else:
        ap.error("qfile or --smoke is required")
