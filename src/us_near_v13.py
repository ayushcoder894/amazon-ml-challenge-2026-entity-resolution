"""Probe variant on top of v13_dense: US "small-shift" near-number prior correction (see us_near_variant.py).
Train rate of TRUE small-shift matches per kept S1 comes from the v13 union (v12 + dense pairs, harder split); on test
the odds of the US small-shift pairs (final joint probabilities) are scaled so their expected matches per S1 equal that
rate, and US decisions are re-made with the v13 rule. India / France rows are copied from v13_dense.
"""
import subprocess
import sys

import numpy as np
import polars as pl

import decide
from us_near_variant import OUT, RES, D, small_shift

DN = "D:/amazon_er/dense/"


def main(c="US", base="v13_dense", name="v13_dense_USnear"):
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    s1c = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"])
    Ktr = pl.read_parquet(f"{DN}train_joint.parquet", columns=["rid", "oid", "pj", "y"]).join(
        s1c.filter(pl.col("country") == c), on="rid", how="semi").join(keep, on="rid", how="semi")
    n_tr = s1c.filter(pl.col("country") == c).join(keep, on="rid", how="semi").height
    ss_tr = small_shift(Ktr.select("rid", "oid"), "train", c).join(Ktr, on=["rid", "oid"])
    rate = ss_tr["y"].sum() / n_tr
    s1t = pl.read_parquet(f"{D}test_s1_norm.parquet", columns=["rid", "entity_id", "country"])
    T = pl.read_parquet(f"{DN}test_union_p.parquet").join(s1t.filter(pl.col("country") == c).select("rid"), on="rid", how="semi")
    n_te = s1t.filter(pl.col("country") == c).height
    ss_te = small_shift(T.select("rid", "oid"), "test", c).with_columns(ss=pl.lit(True))
    T = T.join(ss_te, on=["rid", "oid"], how="left").with_columns(pl.col("ss").fill_null(False))
    p = T.filter(pl.col("ss"))["p"].to_numpy().astype(np.float64)
    f = lambda r: (r * p / (r * p + 1 - p)).sum() / n_te
    lo, hi = 1e-4, 1.0
    for _ in range(60):
        mid = (lo * hi) ** 0.5
        lo, hi = (mid, hi) if f(mid) < rate else (lo, mid)
    r = (lo * hi) ** 0.5
    print(f"{c}: train small-shift true matches {rate:.4f}/S1 (cands {ss_tr.height / n_tr:.3f}/S1); test cands "
          f"{ss_te.height / n_te:.3f}/S1, expected matches before {f(1.0):.4f}/S1 -> odds factor r = {r:.3f}")
    T = T.with_columns(pc=pl.when(pl.col("ss")).then(r * pl.col("p") / (r * pl.col("p") + 1 - pl.col("p"))).otherwise(pl.col("p")))
    old = decide.ef_predict(T.select("rid", "oid", pl.col("p").alias("p2")), "p2", 0.0, 1.0, 0.05)
    best = decide.ef_predict(T.select("rid", "oid", pl.col("pc").alias("p2")), "p2", 0.0, 1.0, 0.05)
    print(f"{c}: pairs {old.height / n_te:.4f}/S1 -> {best.height / n_te:.4f}/S1 (removed "
          f"{old.join(best, on=['rid', 'oid'], how='anti').height:,}, added {best.join(old, on=['rid', 'oid'], how='anti').height:,})")
    o = pl.concat([pl.read_parquet(f"{D}test_s{k}_norm.parquet", columns=["entity_id"]) for k in (2, 3)]).with_row_index("oid")
    m = (best.join(s1t.select("rid", s="entity_id"), on="rid").join(o.select("oid", m="entity_id"), on="oid")
         .group_by("s").agg(pl.col("m").sort().str.join(",")))
    ids = s1t.filter(pl.col("country") == c)["entity_id"]
    rd = dict(separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
    bdf = pl.read_csv(f"{OUT}matching_results_{base}.tsv", **rd)
    new = (bdf.join(m.rename({"s": "source1_entity_id", "m": "new"}), on="source1_entity_id", how="left")
           .with_columns(matched_entity_ids=pl.when(pl.col("source1_entity_id").is_in(ids.implode()))
                         .then(pl.col("new").fill_null("")).otherwise(pl.col("matched_entity_ids"))).drop("new"))
    mp, cp = f"{OUT}matching_results_{name}.tsv", f"{OUT}candidate_pairs_{name}.tsv"
    new.write_csv(mp, separator="\t", quote_style="never")
    pl.read_csv(f"{OUT}candidate_pairs_{base}.tsv", **rd).write_csv(cp, separator="\t", quote_style="never")
    subprocess.run([sys.executable, f"{RES}utils/validate_submission.py", "--matching", mp, "--candidate", cp,
                    "--test-dir", f"{RES}dataset/test"])


if __name__ == "__main__":
    main(*sys.argv[1:4])
