"""Same-country shift audit: per-entity TRUE rate by pattern on train (harder split) vs per-entity ACCEPTED rate on test
(final = v21 + US near-number fix). Test accepting clearly more than train truth -> likely test-only decoys."""
import polars as pl
from fr_rates import rates

D = "C:/ML/Challenges/amazon/work/data/"
DN = "D:/amazon_er/dense/"
OUT = "C:/ML/Challenges/amazon/work/output/"
rd = dict(separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
keep = pl.read_parquet(f"{D}keep_v11.parquet")
fin = pl.read_csv(f"{OUT}matching_results_probeA.tsv", **rd).filter(pl.col("matched_entity_ids") != "").with_columns(m=pl.col("matched_entity_ids").str.split(",")).explode("m")
for c in ("US", "India"):
    U = pl.read_parquet(f"{DN}train_joint9.parquet", columns=["rid", "oid", "y", "country"]).filter(pl.col("country") == c)
    n = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == c).join(keep, on="rid", how="semi").height
    tr = rates(U.select(i1=pl.col("rid").cast(pl.UInt32), i2=pl.col("oid").cast(pl.UInt32), y=pl.col("y").cast(pl.Float64)), "train", c, n, 500000, "y")
    s1 = pl.read_parquet(f"{D}p1_canon_test_{c}_s1.parquet", columns=["idx", "entity_id"])
    o = pl.read_parquet(f"{D}p1_canon_test_{c}_o.parquet", columns=["idx", "entity_id"])
    acc = (fin.join(s1.rename({"entity_id": "source1_entity_id", "idx": "i1"}), on="source1_entity_id")
           .join(o.rename({"entity_id": "m", "idx": "i2"}), on="m").select(pl.col("i1").cast(pl.UInt32), pl.col("i2").cast(pl.UInt32), acc=pl.lit(1.0)))
    T = pl.read_parquet(f"{DN}test_union_p.parquet").select(i1=pl.col("rid").cast(pl.UInt32), i2=pl.col("oid").cast(pl.UInt32)).join(s1.select(pl.col("idx").cast(pl.UInt32).alias("i1")), on="i1", how="semi")
    T = pl.concat([T, acc.select("i1", "i2")]).unique().join(acc, on=["i1", "i2"], how="left").with_columns(pl.col("acc").fill_null(0.0))
    te = rates(T, "test", c, s1.height, 500000, "acc")
    k = ["name", "addr", "clus"]
    t = tr.join(te, on=k, how="full", coalesce=True).fill_null(0.0).with_columns(excess=pl.col("pos_acc") - pl.col("pos_y"), ratio=pl.col("pos_acc") / (pl.col("pos_y") + 1e-4))
    t.write_parquet(f"D:/amazon_er/france_night/data/iu_rates_{c}.parquet")
    with pl.Config(tbl_rows=16, tbl_width_chars=240, float_precision=4, tbl_hide_column_data_types=True, fmt_str_lengths=36):
        print(f"\n==== {c}: test ACCEPTS MORE per entity than train TRUTH (top excess)")
        print(t.sort("excess", descending=True).select("name", "addr", "clus", "cand_y", "pos_y", "cand_acc", "pos_acc", "excess", "ratio").head(14))
        print(f"{c}: test FEWER than train truth (top)")
        print(t.sort("excess").select("name", "addr", "clus", "cand_y", "pos_y", "cand_acc", "pos_acc", "excess").head(6))
    print(f"{c} totals: train true/S1 {t['pos_y'].sum():.3f}  test accepted/S1 {t['pos_acc'].sum():.3f}")
