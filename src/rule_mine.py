"""Rule mining on unaccepted candidates, harder split: bucket = (name rel, addr rel, S1 address unique?, record free?).
Choose buckets with precision >= T on fold A entities, add their best pair per record on fold B, measure F0.5 on B (and swap)."""
import polars as pl
import decide
import evaluate as ev
from france_rules import categorise
D = "C:/ML/Challenges/amazon/work/data/"
DN = "D:/amazon_er/dense/"
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200); pl.Config.set_fmt_str_lengths(40)
keep = pl.read_parquet(f"{D}keep_v11.parquet")
for c in ("India", "US"):
    s1c = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == c).select("rid").join(keep, on="rid", how="semi")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid"]).join(s1c, on="rid", how="semi")
    U = pl.read_parquet(f"{DN}train_joint9.parquet", columns=["rid", "oid", "pj", "country"]).filter(pl.col("country") == c).select("rid", "oid", p="pj")
    pred = decide.ef_predict(U.select("rid", "oid", p2="p"), "p2", 0.0, 1.0, 0.05)
    ae = pl.read_parquet(f"{D}p1_canon_train_{c}_o.parquet", columns=["idx", "aempty"]).rename({"idx": "oid"}).filter(~pl.col("aempty"))
    # unaccepted addressed candidates of free records (record accepted by nobody), p >= 0.01
    X = (U.filter(pl.col("p") >= 0.01).join(ae.select("oid"), on="oid", how="semi").join(pred.select("oid"), on="oid", how="anti")
         .join(gt.with_columns(y=pl.lit(1, pl.Int8)), on=["rid", "oid"], how="left").with_columns(pl.col("y").fill_null(0)))
    X = categorise(X, "train", c).drop("y", "p", strict=False).join(X, on=["rid", "oid"])
    X = X.with_columns(uniq=pl.col("na").fill_null(1) <= 1, fold=(pl.col("rid").hash(seed=11) % 2).cast(pl.Int8))
    keys = ["name", "addr", "uniq"]
    print(f"\n==== {c}: unaccepted candidates of free addressed records: {X.height:,} (true {X['y'].mean():.3f})")
    b = X.group_by(keys).agg(n=pl.len(), prec=pl.col("y").mean(), mean_p=pl.col("p").mean()).filter(pl.col("n") >= 200).sort("prec", descending=True)
    print(b.head(20))
    f0 = ev.macro_f05(pred, gt, s1c["rid"])
    for T in (0.8, 0.85, 0.9):
        tot = 0.0
        for fa in (0, 1):
            sel = X.filter(pl.col("fold") == fa).group_by(keys).agg(n=pl.len(), prec=pl.col("y").mean()).filter((pl.col("n") >= 100) & (pl.col("prec") >= T))
            add = (X.filter(pl.col("fold") == 1 - fa).join(sel.select(keys), on=keys, how="semi")
                   .sort("p", descending=True).unique("oid", keep="first").select("rid", "oid", "y"))
            sB = s1c.filter((pl.col("rid").hash(seed=11) % 2) == 1 - fa)
            gB = gt.join(sB, on="rid", how="semi")
            pB = pred.join(sB, on="rid", how="semi")
            fb0, fb1 = ev.macro_f05(pB, gB, sB["rid"]), ev.macro_f05(pl.concat([pB, add.select("rid", "oid")]), gB, sB["rid"])
            tot += (fb1 - fb0) * sB.height / s1c.height
            print(f"  T={T} choose on fold {fa} ({sel.height} buckets) -> fold {1 - fa}: add {add.height:,} (precision {add['y'].mean() if add.height else 0:.3f}) F {fb0:.5f} -> {fb1:.5f} ({fb1 - fb0:+.5f})")
        print(f"  T={T}: combined honest gain {tot:+.5f} (base {f0:.5f})")
