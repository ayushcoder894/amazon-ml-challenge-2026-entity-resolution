"""Probe variant: prior correction for "small-shift" near-number candidates (same canonical name, same street words,
first house number differs by <= 20 and is not a prefix) in a given country.

Train (harder split) gives the generator's rate of TRUE small-shift matches per Source-1 entity; on test the country
has ~2x the small-shift candidates per entity. Assuming the true-match rate per entity is unchanged, the odds of those
pairs are scaled by r (solved by bisection) so that their expected matches per entity equal the train rate; decisions
are then re-made for that country with the submission's rule. Everything else is copied from the base submission.
"""
import json
import subprocess
import sys

import numpy as np
import polars as pl

import decide
from near_number import tag

D = "C:/ML/Challenges/amazon/work/data/"
OUT = "C:/ML/Challenges/amazon/work/output/"
M = "C:/ML/Challenges/amazon/work/models_ce/"
RES = "C:/ML/Challenges/amazon/6ab10eb3b23ba_student_resource/student_resource/"


def small_shift(P, split, c):
    x = tag(P, split, c).filter(pl.col("near"))
    num = lambda e: e.str.split(" ").list.eval(pl.element().filter(pl.element().str.contains(r"^\d+$"))).list.first()
    x = x.with_columns(a=num(pl.col("a1")), b=num(pl.col("a2"))).filter(
        pl.col("a").is_not_null() & pl.col("b").is_not_null() & (pl.col("a") != pl.col("b")))
    x = x.with_columns(d=(pl.col("a").cast(pl.Int64, strict=False) - pl.col("b").cast(pl.Int64, strict=False)).abs(),
                       pre=pl.col("a").str.starts_with(pl.col("b")) | pl.col("b").str.starts_with(pl.col("a")))
    return x.filter(~pl.col("pre") & (pl.col("d") <= 20)).select("rid", "oid")


def main(c="US", base="v12_FRp1", name="v12_FRp1_USnear"):
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    Ktr = pl.read_parquet(f"{D}train_oof_final_v12_twin.parquet").filter(pl.col("country") == c)
    n_tr = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == c).join(
        keep, on="rid", how="semi").height
    ss_tr = small_shift(Ktr.select("rid", "oid"), "train", c).join(Ktr.select("rid", "oid", "y", "pf"), on=["rid", "oid"])
    rate = ss_tr["y"].sum() / n_tr
    T = pl.read_parquet(f"{D}test_final_v12_ce_cectx_twin.parquet").filter(pl.col("country") == c)
    n_te = pl.read_parquet(f"{D}test_s1_norm.parquet", columns=["country"]).filter(pl.col("country") == c).height
    ss_te = small_shift(T.select("rid", "oid"), "test", c).with_columns(ss=pl.lit(True))
    T = T.join(ss_te, on=["rid", "oid"], how="left").with_columns(pl.col("ss").fill_null(False))
    p = T.filter(pl.col("ss"))["pf"].to_numpy().astype(np.float64)
    f = lambda r: (r * p / (r * p + 1 - p)).sum() / n_te
    lo, hi = 1e-4, 1.0
    for _ in range(60):
        mid = (lo * hi) ** 0.5
        lo, hi = (mid, hi) if f(mid) < rate else (lo, mid)
    r = (lo * hi) ** 0.5
    print(f"{c}: train small-shift true matches {rate:.4f}/S1 (candidates {ss_tr.height / n_tr:.3f}/S1, mean p "
          f"{ss_tr['pf'].mean():.3f}); test candidates {ss_te.height / n_te:.3f}/S1, expected matches before "
          f"{f(1.0):.4f}/S1 -> odds factor r = {r:.3f}")
    T = T.with_columns(pc=pl.when(pl.col("ss")).then(r * pl.col("pf") / (r * pl.col("pf") + 1 - pl.col("pf"))).otherwise(pl.col("pf")))
    rule = json.load(open(f"{M}multi_meta_v12_ce_cectx_twin.json"))["rule"]
    best = (T.filter((pl.col("pc").rank("ordinal", descending=True).over("oid") == 1) & (pl.col("pc") >= rule[1])).select("rid", "oid")
            if rule[0] == "t" else decide.ef_predict(T.select("rid", "oid", pl.col("pc").alias("p2")), "p2", rule[1], rule[2], 0.05))
    old = decide.ef_predict(T.select("rid", "oid", pl.col("pf").alias("p2")), "p2", rule[1], rule[2], 0.05) if rule[0] == "ef" else None
    if old is not None:
        print(f"{c}: pairs {old.height / n_te:.4f}/S1 -> {best.height / n_te:.4f}/S1 "
              f"(removed {old.join(best, on=['rid', 'oid'], how='anti').height:,}, added {best.join(old, on=['rid', 'oid'], how='anti').height:,})")
    s1 = pl.read_parquet(f"{D}test_s1_norm.parquet", columns=["rid", "entity_id", "country"])
    o = pl.concat([pl.read_parquet(f"{D}test_s{k}_norm.parquet", columns=["entity_id"]) for k in (2, 3)]).with_row_index("oid")
    m = (best.join(s1.select("rid", s="entity_id"), on="rid").join(o.select("oid", m="entity_id"), on="oid")
         .group_by("s").agg(pl.col("m").sort().str.join(",")))
    ids = s1.filter(pl.col("country") == c)["entity_id"]
    bdf = pl.read_csv(f"{OUT}matching_results_{base}.tsv", separator="\t", quote_char=None, infer_schema=False,
                      missing_utf8_is_empty_string=True)
    new = (bdf.join(m.rename({"s": "source1_entity_id", "m": "new"}), on="source1_entity_id", how="left")
           .with_columns(matched_entity_ids=pl.when(pl.col("source1_entity_id").is_in(ids.implode()))
                         .then(pl.col("new").fill_null("")).otherwise(pl.col("matched_entity_ids"))).drop("new"))
    mp, cp = f"{OUT}matching_results_{name}.tsv", f"{OUT}candidate_pairs_{name}.tsv"
    new.write_csv(mp, separator="\t", quote_style="never")
    pl.read_csv(f"{OUT}candidate_pairs_{base}.tsv", separator="\t", quote_char=None, infer_schema=False,
                missing_utf8_is_empty_string=True).write_csv(cp, separator="\t", quote_style="never")
    subprocess.run([sys.executable, f"{RES}utils/validate_submission.py", "--matching", mp, "--candidate", cp,
                    "--test-dir", f"{RES}dataset/test"])


if __name__ == "__main__":
    main(*sys.argv[1:4])
