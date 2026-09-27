"""Do the US test's extra accepted nearby-number pairs (odd one out / mixed / none, |diff| <= 50) look like TRAIN true
matches or TRAIN decoys?  Profile of the differing number: one-digit edit (same length, 1 position), same parity
(same side of the street), |diff|; plus the model probability."""
import polars as pl
from rapidfuzz.distance import Levenshtein

import decide
from nearby_number import D, DN, OUT, pattern, sib_agree
from nn_followup import load_pred

keep = pl.read_parquet(f"{D}keep_v11.parquet")
gt = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid"])


def prof(x):
    a = [u[0] if len(u) == 1 else None for u in x["da"].to_list()]
    b = [u[0] if len(u) == 1 else None for u in x["db"].to_list()]
    lev = [Levenshtein.distance(u, v) if u and v else None for u, v in zip(a, b)]
    x = x.with_columns(lev=pl.Series(lev, dtype=pl.Int64))
    return x.select(n=pl.len(), one_edit=(pl.col("lev") == 1).mean(), even=((pl.col("diff") % 2) == 0).mean(),
                    d_le2=(pl.col("diff") <= 2).mean(), d_med=pl.col("diff").median(), p=pl.col("p").mean())


def show(lab, x):
    r = prof(x).row(0, named=True)
    print(f"   {lab:<44s} n={r['n']:>7,}  one-digit edit {r['one_edit']:.2f}  even diff {r['even']:.2f}  diff<=2 {r['d_le2']:.2f}  median diff {r['d_med']:>4.0f}  mean p {r['p']:.3f}")


for c in ("US", "India"):
    print(f"\n==== {c}")
    s1c = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == c).select("rid").join(keep, on="rid", how="semi")
    U = pl.read_parquet(f"{DN}train_joint9.parquet", columns=["rid", "oid", "pj", "country"]).filter(pl.col("country") == c).drop("country")
    pred = decide.ef_predict(U.select("rid", "oid", pl.col("pj").alias("p2")), "p2", 0.0, 1.0, 0.05)
    P, o = pattern("train", c)
    P = P.join(s1c, on="rid", how="semi").join(gt.with_columns(y=pl.lit(1, pl.Int8)), on=["rid", "oid"], how="left").join(
        pred.with_columns(acc=pl.lit(1, pl.Int8)), on=["rid", "oid"], how="left").join(U.rename({"pj": "p"}), on=["rid", "oid"], how="left").with_columns(
        pl.col("y", "acc").fill_null(0), pl.col("p").fill_null(0.0))
    del U
    P = sib_agree(P, pred, o, "sib")
    T = P.filter(pl.col("sib").is_in(["odd one out", "mixed", "none"]) & (pl.col("diff") <= 50))
    show("train candidates, TRUE", T.filter(pl.col("y") == 1))
    show("train candidates, DECOY", T.filter(pl.col("y") == 0))
    show("train ACCEPTED (joint9)", T.filter(pl.col("acc") == 1))
    s1 = pl.read_parquet(f"{D}p1_canon_test_{c}_s1.parquet", columns=["idx", "entity_id"])
    ot = pl.read_parquet(f"{D}p1_canon_test_{c}_o.parquet", columns=["idx", "entity_id"])
    v19 = load_pred("v19", s1, ot)
    Pt, oc = pattern("test", c)
    tp = pl.read_parquet(f"{DN}test_union_p.parquet").select("rid", "oid", "p")
    Pt = Pt.join(v19.with_columns(acc=pl.lit(1, pl.Int8)), on=["rid", "oid"], how="left").join(tp, on=["rid", "oid"], how="left").with_columns(
        pl.col("acc").fill_null(0), pl.col("p").fill_null(0.0))
    Pt = sib_agree(Pt, v19, oc, "sib")
    Tt = Pt.filter(pl.col("sib").is_in(["odd one out", "mixed", "none"]) & (pl.col("diff") <= 50))
    show("test candidates (all)", Tt)
    show("test ACCEPTED (v19)", Tt.filter(pl.col("acc") == 1))
    show("test ACCEPTED, p >= 0.9", Tt.filter((pl.col("acc") == 1) & (pl.col("p") >= 0.9)))
    show("test ACCEPTED, p < 0.9", Tt.filter((pl.col("acc") == 1) & (pl.col("p") < 0.9)))
    show("train ACCEPTED, p >= 0.9", T.filter((pl.col("acc") == 1) & (pl.col("p") >= 0.9)))
    show("train ACCEPTED, p < 0.9", T.filter((pl.col("acc") == 1) & (pl.col("p") < 0.9)))
    print(f"   accepted share with p >= 0.9: train {T.filter(pl.col('acc') == 1).select((pl.col('p') >= 0.9).mean()).item():.2f}, "
          f"test {Tt.filter(pl.col('acc') == 1).select((pl.col('p') >= 0.9).mean()).item():.2f}")
