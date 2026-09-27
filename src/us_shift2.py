"""US: same canonical name + non-truncated different numbers (street words same OR different), per sub-group
(words same/differ x sibling agreement x |diff| bucket): accepted per entity on test (probeA) vs TRUE per entity on train."""
import polars as pl
import decide
from nearby_number import canon2, digits, sib_agree, BINS

D = "C:/ML/Challenges/amazon/work/data/"
DN = "D:/amazon_er/dense/"
OUT = "C:/ML/Challenges/amazon/work/output/"
rd = dict(separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)


def pairs(split, P):
    s1, o = canon2(split, "US", "s1"), canon2(split, "US", "o")
    x = (P.join(s1.select(pl.col("idx").cast(pl.UInt32).alias("rid"), c1="cnk", w1="w", n1="nums"), on="rid")
         .join(o.filter(~pl.col("aempty")).select(pl.col("idx").cast(pl.UInt32).alias("oid"), c2="cnk", w2="w", n2="nums"), on="oid")
         .filter((pl.col("c1") == pl.col("c2")) & (pl.col("c1") != "") & (pl.col("n1").list.len() > 0) & (pl.col("n2").list.len() > 0)
                 & (pl.col("n1").list.join(" ") != pl.col("n2").list.join(" "))))
    da, db, dd, keep = [], [], [], []
    for a, b in zip(x["n1"].to_list(), x["n2"].to_list()):
        u, v = sorted(set(a) - set(b)), sorted(set(b) - set(a))
        k = bool(u) and bool(v) and not any(p in q or q in p for p in u for q in v)
        keep.append(k); da.append(u); db.append(v)
        dd.append(min(abs(digits(p) - digits(q)) for p in u for q in v) if k else None)
    x = x.with_columns(da=pl.Series(da, dtype=pl.List(pl.String)), db=pl.Series(db, dtype=pl.List(pl.String)), diff=pl.Series(dd, dtype=pl.Int64), k=pl.Series(keep)).filter(pl.col("k"))
    x = x.with_columns(ws=pl.when(pl.col("w1") == pl.col("w2")).then(pl.lit("street same")).otherwise(pl.lit("street differs")),
                       dbin=pl.when(pl.col("diff") <= 10).then(pl.lit(BINS[0])).when(pl.col("diff") <= 50).then(pl.lit(BINS[1])).when(pl.col("diff") <= 500).then(pl.lit(BINS[2])).otherwise(pl.lit(BINS[3])))
    return x, o.select("idx", "nums", "aempty")


keep = pl.read_parquet(f"{D}keep_v11.parquet")
s1c = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == "US").select("rid").join(keep, on="rid", how="semi")
U = pl.read_parquet(f"{DN}train_joint9.parquet", columns=["rid", "oid", "pj", "y", "country"]).filter(pl.col("country") == "US").join(s1c, on="rid", how="semi")
pred = decide.ef_predict(U.select("rid", "oid", p2="pj"), "p2", 0.0, 1.0, 0.05)
Xtr, otr = pairs("train", U.select("rid", "oid", "y").join(pred.with_columns(acc=pl.lit(1, pl.Int8)), on=["rid", "oid"], how="left").with_columns(pl.col("acc").fill_null(0)))
Xtr = sib_agree(Xtr, pred, otr, "sib")
s1t = pl.read_parquet(f"{D}p1_canon_test_US_s1.parquet", columns=["idx", "entity_id"])
ot = pl.read_parquet(f"{D}p1_canon_test_US_o.parquet", columns=["idx", "entity_id"])
fin = pl.read_csv(f"{OUT}matching_results_probeA.tsv", **rd).filter(pl.col("matched_entity_ids") != "").with_columns(m=pl.col("matched_entity_ids").str.split(",")).explode("m")
pt = (fin.join(s1t.rename({"entity_id": "source1_entity_id", "idx": "rid"}), on="source1_entity_id").join(ot.rename({"entity_id": "m", "idx": "oid"}), on="m")
      .select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32)))
T = pl.read_parquet(f"{DN}test_union_p.parquet").select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32), "p").join(s1t.select(pl.col("idx").cast(pl.UInt32).alias("rid")), on="rid", how="semi")
Xte, ote = pairs("test", T.join(pt.with_columns(acc=pl.lit(1, pl.Int8)), on=["rid", "oid"], how="left").with_columns(pl.col("acc").fill_null(0)))
Xte = sib_agree(Xte, pt, ote, "sib")
ntr, nte = s1c.height, s1t.height
k = ["ws", "sib", "dbin"]
a = Xtr.group_by(k).agg(true_tr=pl.col("y").sum() / ntr, acc_tr=pl.col("acc").sum() / ntr, prec_tr=pl.col("y").filter(pl.col("acc") == 1).mean())
b = Xte.group_by(k).agg(acc_te=pl.col("acc").sum() / nte, n_acc_te=pl.col("acc").sum())
t = a.join(b, on=k, how="full", coalesce=True).fill_null(0).with_columns(ratio=pl.col("acc_te") / (pl.col("true_tr") + 1e-5), excess_pairs=(pl.col("acc_te") - pl.col("true_tr")) * nte)
with pl.Config(tbl_rows=40, tbl_width_chars=220, float_precision=4, tbl_hide_column_data_types=True):
    print(t.sort("excess_pairs", descending=True).head(20))
Xte.write_parquet("D:/amazon_er/france_night/data/us_shift2_test.parquet")
