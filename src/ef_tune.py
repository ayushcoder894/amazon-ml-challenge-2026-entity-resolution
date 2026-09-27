import polars as pl
import decide
import evaluate as ev
D = "C:/ML/Challenges/amazon/work/data/"
DN = "D:/amazon_er/dense/"
keep = pl.read_parquet(f"{D}keep_v11.parquet")
for c in ("India", "US"):
    s1c = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == c).select("rid").join(keep, on="rid", how="semi")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid"]).join(s1c, on="rid", how="semi")
    U = pl.read_parquet(f"{DN}train_joint9.parquet", columns=["rid", "oid", "pj", "country"]).filter(pl.col("country") == c).select("rid", "oid", p2="pj")
    folds = [s1c.filter((pl.col("rid").hash(seed=11) % 2) == f) for f in (0, 1)]
    res = {}
    for mu in (0.0, 0.1, 0.3):
        for cc in (0.7, 1.0, 1.3, 1.6):
            P = decide.ef_predict(U, "p2", mu, cc, 0.05)
            res[(mu, cc)] = [ev.macro_f05(P.join(s, on="rid", how="semi"), gt.join(s, on="rid", how="semi"), s["rid"]) for s in folds]
    base = res[(0.0, 1.0)]
    print(f"\n{c}: current ef(0, 1.0) folds {base[0]:.5f} / {base[1]:.5f}")
    for fa in (0, 1):
        best = max(res, key=lambda k: res[k][fa])
        print(f"  best on fold {fa}: mu={best[0]} c={best[1]} -> fold {1 - fa}: {res[best][1 - fa]:.5f} ({res[best][1 - fa] - base[1 - fa]:+.5f})")
