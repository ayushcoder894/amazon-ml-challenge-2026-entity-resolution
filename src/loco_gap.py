"""Where does an unseen-country model (France proxy) lose against the in-country model?
India and US on the harder split: in-country = v12 final OOF (rule ef(0,1)); unseen = LOCO stage-1 p1 (rule ef(0,4),
the France recipe). Wrong pairs (FP / FN) of each are bucketed by simple pair properties; the per-bucket difference
in wrong pairs shows which kinds of decisions do not transfer across countries.
"""
import polars as pl

import decide

D = "C:/ML/Challenges/amazon/work/data/"
ARCH = "D:/amazon_er/archive/data/"
COLS = ["rid", "oid", "css_eq", "addr_empty2", "first_num_eq", "n_tset", "a_tset", "legal_conflict", "num_prefix", "indic"]


def main():
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    s1 = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).join(keep, on="rid", how="semi")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").join(keep, on="rid", how="semi").select("rid", "oid")
    Kin = pl.read_parquet(f"{D}train_oof_final_v12_twin.parquet", columns=["rid", "oid", "pf"])
    Klo = pl.read_parquet(f"{D}train_oof_locoh.parquet", columns=["rid", "oid", "p1"])
    pin = decide.ef_predict(Kin.select("rid", "oid", pl.col("pf").alias("p2")), "p2", 0.0, 1.0, 0.05).select("rid", "oid")
    plo = decide.ef_predict(Klo.select("rid", "oid", pl.col("p1").alias("p2")), "p2", 0.0, 4.0, 0.05).select("rid", "oid")
    pairs = pl.concat([pin, plo, gt]).unique()
    F = (pl.concat([pl.scan_parquet(f"{ARCH}train_feat_v11_fold{f}.parquet").select(COLS) for f in (0, 1)])
         .join(pairs.lazy(), on=["rid", "oid"], how="semi").collect())
    b = F.with_columns(
        name=pl.when(pl.col("css_eq") == 1).then(pl.lit("name exact")).when(pl.col("n_tset") >= 90).then(pl.lit("name ~")).otherwise(pl.lit("name differs")),
        addr=pl.when(pl.col("addr_empty2") == 1).then(pl.lit("no address")).when(pl.col("first_num_eq") == 1).then(
            pl.when(pl.col("a_tset") >= 90).then(pl.lit("num= street~")).otherwise(pl.lit("num= street differs")))
        .when(pl.col("num_prefix") == 1).then(pl.lit("num prefix")).otherwise(pl.lit("num differs/missing")),
        legal=pl.when(pl.col("legal_conflict") == 1).then(pl.lit("legal conflict")).otherwise(pl.lit("")),
    ).select("rid", "oid", "name", "addr", "legal")
    rows = []
    for tag, pred in (("in-country", pin), ("unseen", plo)):
        fp = pred.join(gt, on=["rid", "oid"], how="anti").with_columns(err=pl.lit("FP"))
        fn = gt.join(pred, on=["rid", "oid"], how="anti").with_columns(err=pl.lit("FN"))
        rows.append(pl.concat([fp, fn]).join(s1, on="rid").join(b, on=["rid", "oid"], how="left").with_columns(model=pl.lit(tag)))
    E = pl.concat(rows).with_columns(pl.col("name", "addr", "legal").fill_null("?"))
    t = (E.group_by("country", "err", "name", "addr", "legal", "model").len()
         .pivot(on="model", index=["country", "err", "name", "addr", "legal"], values="len").fill_null(0)
         .with_columns(extra=pl.col("unseen") - pl.col("in-country")).sort("extra", descending=True))
    n = s1.group_by("country").len()
    print(n)
    with pl.Config(tbl_rows=30, tbl_width_chars=160):
        for c in ("India", "US"):
            print(f"\n=== {c}: extra wrong pairs of the unseen-country model (top buckets)")
            print(t.filter(pl.col("country") == c).drop("country").head(14))


if __name__ == "__main__":
    main()
