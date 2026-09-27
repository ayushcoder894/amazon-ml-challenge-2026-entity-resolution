"""Validate 'add unassigned name-only records to their top candidate' for France on the proxy: India/US scored with the
France recipe (v18 + rule S, blend pb before the CE mix as in the review CSV), true labels. Precision and F0.5 change per
probability band of the top candidate."""
import os
import polars as pl
import evaluate as ev

D = "C:/ML/Challenges/amazon/work/data/"
keep = pl.read_parquet(f"{D}keep_v11.parquet")
_gt_all = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid"])
ORPH = _gt_all.join(keep, on="rid", how="anti").select("oid")
import france_v18 as v18
from france_v19 import rule_s
import importlib
for c in ("India", "US"):
    s1 = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == c).join(keep, on="rid", how="semi").select("rid")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid"]).join(s1, on="rid", how="semi")
    _, _, C, _ = v18.v18(c)
    P = pl.concat([C, rule_s(c, "train", C)]).unique().join(s1, on="rid", how="semi")
    os.environ["W_S"] = "0"
    import france_v15 as f0
    importlib.reload(f0)
    U, _ = f0.frames(c)                       # pb = blend before the CE mix (same column as the review CSV)
    U = U.join(s1, on="rid", how="semi")
    ae = pl.read_parquet(f"{D}p1_canon_train_{c}_o.parquet", columns=["idx", "aempty"]).rename({"idx": "oid"}).filter(pl.col("aempty")).select("oid")
    X = (U.join(ae, on="oid", how="semi").join(P.select("oid"), on="oid", how="anti")          # name-only, assigned to nobody
         .join(ORPH, on="oid", how="anti").sort("pb", descending=True).group_by("oid", maintain_order=True).agg(rid=pl.col("rid").first(), p1=pl.col("pb").first(), p2=pl.col("pb").slice(1, 1).first()))
    X = X.join(gt.with_columns(y=pl.lit(1, pl.Int8)), on=["rid", "oid"], how="left").with_columns(pl.col("y").fill_null(0), pl.col("p2").fill_null(0.0))
    f_base = ev.macro_f05(P, gt, s1["rid"])
    print(f"\n{c} proxy (France recipe v18 + rule S) F0.5 {f_base:.5f} [TEST-LIKE: orphan records removed]; unassigned name-only records with a candidate: {X.height:,}")
    print(f"  {'top p band':<12s} {'records':>8s} {'per S1':>8s} {'correct':>8s} | {'F0.5 if added':>13s}")
    for lo, hi in ((0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.77), (0.77, 0.85), (0.85, 0.9), (0.9, 0.95), (0.95, 1.01)):
        b = X.filter((pl.col("p1") >= lo) & (pl.col("p1") < hi) & (pl.col("p2") < 0.1))
        f = ev.macro_f05(pl.concat([P, b.select("rid", "oid")]), gt, s1["rid"])
        print(f"  [{lo:.2f},{min(hi, 1):.2f})  {b.height:>8,} {b.height / s1.height:8.4f} {b['y'].mean() if b.height else float('nan'):8.1%} | {f:.5f} ({f - f_base:+.5f})")
    for lo in (0.2, 0.3, 0.4, 0.5):
        b = X.filter((pl.col("p1") >= lo) & (pl.col("p2") < 0.1))
        f = ev.macro_f05(pl.concat([P, b.select("rid", "oid")]), gt, s1["rid"])
        print(f"  all with top p >= {lo}: add {b.height:,} ({b['y'].mean():.1%} correct) -> {f:.5f} ({f - f_base:+.5f})")
