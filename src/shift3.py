"""Finer same-country shift audit (US and India): same canonical name + non-truncated different numbers, split by
street same/differs x sibling agreement x number gap x model probability band. Test = current best (probeA2)."""
import sys
import polars as pl
import decide
from nearby_number import sib_agree
from us_shift2 import pairs as _pairs_us
import us_shift2 as us2

D = "C:/ML/Challenges/amazon/work/data/"
DN = "D:/amazon_er/dense/"
OUT = "C:/ML/Challenges/amazon/work/output/"
rd = dict(separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
PB = pl.when(pl.col("p") >= 0.97).then(pl.lit("p>=.97")).when(pl.col("p") >= 0.9).then(pl.lit(".90-.97")).when(pl.col("p") >= 0.7).then(pl.lit(".70-.90")).otherwise(pl.lit("<.70"))
res = {}
for c in sys.argv[1:] or ["US", "India"]:
    import nearby_number as nn
    orig = us2.canon2
    us2.canon2 = lambda split, cc, side, _c=c: nn.canon2(split, _c, side)
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    s1c = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == c).select("rid").join(keep, on="rid", how="semi")
    U = pl.read_parquet(f"{DN}train_joint9.parquet", columns=["rid", "oid", "pj", "y", "country"]).filter(pl.col("country") == c).join(s1c, on="rid", how="semi")
    pred = decide.ef_predict(U.select("rid", "oid", p2="pj"), "p2", 0.0, 1.0, 0.05)
    Xtr, otr = _pairs_us("train", U.select("rid", "oid", "y", p="pj").join(pred.with_columns(acc=pl.lit(1, pl.Int8)), on=["rid", "oid"], how="left").with_columns(pl.col("acc").fill_null(0)))
    Xtr = sib_agree(Xtr, pred, otr, "sib").with_columns(pb=PB)
    s1t = pl.read_parquet(f"{D}p1_canon_test_{c}_s1.parquet", columns=["idx", "entity_id"])
    ot = pl.read_parquet(f"{D}p1_canon_test_{c}_o.parquet", columns=["idx", "entity_id"])
    fin = pl.read_csv(f"{OUT}matching_results_probeA2.tsv", **rd).filter(pl.col("matched_entity_ids") != "").with_columns(m=pl.col("matched_entity_ids").str.split(",")).explode("m")
    pt = (fin.join(s1t.rename({"entity_id": "source1_entity_id", "idx": "rid"}), on="source1_entity_id").join(ot.rename({"entity_id": "m", "idx": "oid"}), on="m")
          .select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32)))
    T = pl.read_parquet(f"{DN}test_union_p.parquet").select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32), "p").join(s1t.select(pl.col("idx").cast(pl.UInt32).alias("rid")), on="rid", how="semi")
    Xte, ote = _pairs_us("test", T.join(pt.with_columns(acc=pl.lit(1, pl.Int8)), on=["rid", "oid"], how="left").with_columns(pl.col("acc").fill_null(0)))
    Xte = sib_agree(Xte, pt, ote, "sib").with_columns(pb=PB)
    us2.canon2 = orig
    ntr, nte = s1c.height, s1t.height
    k = ["ws", "sib", "dbin", "pb"]
    a = Xtr.group_by(k).agg(true_tr=pl.col("y").sum() / ntr, acc_tr=pl.col("acc").sum() / ntr, prec_tr=pl.col("y").filter(pl.col("acc") == 1).mean())
    b = Xte.group_by(k).agg(acc_te=pl.col("acc").sum() / nte, n_acc_te=pl.col("acc").sum())
    t = (a.join(b, on=k, how="full", coalesce=True).fill_null(0)
         .with_columns(ratio=pl.col("acc_te") / (pl.col("true_tr") + 1e-6), excess=(pl.col("acc_te") - pl.col("true_tr")) * nte))
    print(f"\n==== {c}: cells with >= 200 accepted test pairs, sorted by excess (test accepted - train-true, in pairs)")
    with pl.Config(tbl_rows=30, tbl_width_chars=230, float_precision=4, tbl_hide_column_data_types=True):
        print(t.filter(pl.col("n_acc_te") >= 200).sort("excess", descending=True).select(k + ["true_tr", "acc_tr", "prec_tr", "acc_te", "n_acc_te", "ratio", "excess"]).head(25))
    Xte.write_parquet(f"D:/amazon_er/france_night/data/shift3_test_{c}.parquet")
    t.write_parquet(f"D:/amazon_er/france_night/data/shift3_cells_{c}.parquet")
