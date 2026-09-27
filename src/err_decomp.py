"""Harder-split error decomposition for India/US (joint9 + v13 rule): oracle gain of fixing each error type."""
import polars as pl
import decide
import evaluate as ev

D = "C:/ML/Challenges/amazon/work/data/"
DN = "D:/amazon_er/dense/"
keep = pl.read_parquet(f"{D}keep_v11.parquet")
for c in ("India", "US"):
    s1c = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == c).select("rid").join(keep, on="rid", how="semi")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid"]).join(s1c, on="rid", how="semi")
    U = pl.read_parquet(f"{DN}train_joint9.parquet", columns=["rid", "oid", "pj", "country"]).filter(pl.col("country") == c).select("rid", "oid", p="pj")
    pred = decide.ef_predict(U.select("rid", "oid", p2="p"), "p2", 0.0, 1.0, 0.05)
    ae = pl.read_parquet(f"{D}p1_canon_train_{c}_o.parquet", columns=["idx", "aempty"]).rename({"idx": "oid"})
    f0 = ev.macro_f05(pred, gt, s1c["rid"])
    inU = U.select("rid", "oid").with_columns(inu=pl.lit(True))
    fp = pred.join(gt, on=["rid", "oid"], how="anti").join(ae, on="oid", how="left")
    fn = gt.join(pred, on=["rid", "oid"], how="anti").join(ae, on="oid", how="left").join(inU, on=["rid", "oid"], how="left").with_columns(pl.col("inu").fill_null(False))
    print(f"\n{c}: {f0:.5f}  FP {fp.height:,}  FN {fn.height:,}  (S1 {s1c.height:,})")
    cases = {
        "FP name-only": (fp.filter(pl.col("aempty")), None),
        "FP addressed": (fp.filter(~pl.col("aempty")), None),
        "FN name-only, in candidates": (None, fn.filter(pl.col("aempty") & pl.col("inu"))),
        "FN addressed, in candidates": (None, fn.filter(~pl.col("aempty") & pl.col("inu"))),
        "FN name-only, not in candidates": (None, fn.filter(pl.col("aempty") & ~pl.col("inu"))),
        "FN addressed, not in candidates": (None, fn.filter(~pl.col("aempty") & ~pl.col("inu"))),
    }
    for lab, (rm, ad) in cases.items():
        P = pred
        if rm is not None:
            P = P.join(rm.select("rid", "oid"), on=["rid", "oid"], how="anti"); n = rm.height
        if ad is not None:
            P = pl.concat([P, ad.select("rid", "oid")]); n = ad.height
        f = ev.macro_f05(P, gt, s1c["rid"])
        print(f"  fix {lab:<34s} {n:>7,} pairs -> {f:.5f} (+{f - f0:.5f})")
