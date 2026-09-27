"""All-pattern shift scan (US, India): per (probability band x cheap pattern key), TRUE matches per entity on train vs
ACCEPTED per entity on test (current best = probeA2). Pattern key: name equal? x address relation (equal / number
same+words differ / number missing / number added / number changed / name-only) x S1 address shared?"""
import polars as pl
import decide
from nearby_number import canon2

D = "C:/ML/Challenges/amazon/work/data/"
DN = "D:/amazon_er/dense/"
OUT = "C:/ML/Challenges/amazon/work/output/"
rd = dict(separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
PB = (pl.when(pl.col("p") >= 0.97).then(pl.lit("5 p>=.97")).when(pl.col("p") >= 0.9).then(pl.lit("4 .90-.97"))
      .when(pl.col("p") >= 0.7).then(pl.lit("3 .70-.90")).when(pl.col("p") >= 0.5).then(pl.lit("2 .50-.70")).otherwise(pl.lit("1 <.50")))


def keyed(P, split, c):
    s1, o = canon2(split, c, "s1"), canon2(split, c, "o")
    na = s1.filter(pl.col("cak") != "").group_by("cak").agg(na=pl.len())
    s1 = s1.join(na, on="cak", how="left")
    x = (P.join(s1.select(pl.col("idx").cast(pl.UInt32).alias("rid"), n1="cnk", a1="cak", w1="w", u1="nums", na="na"), on="rid")
         .join(o.select(pl.col("idx").cast(pl.UInt32).alias("oid"), n2="cnk", a2="cak", w2="w", u2="nums", ae="aempty"), on="oid"))
    s_eq = pl.col("u1").list.join(" ") == pl.col("u2").list.join(" ")
    inter = pl.col("u1").list.set_intersection(pl.col("u2")).list.len()
    addr = (pl.when(pl.col("ae")).then(pl.lit("name-only")).when(pl.col("a1") == pl.col("a2")).then(pl.lit("addr equal"))
            .when(s_eq).then(pl.lit("num same, words differ"))
            .when(pl.col("u2").list.len() == 0).then(pl.lit("num missing"))
            .when(inter == pl.col("u1").list.len()).then(pl.lit("num added"))
            .otherwise(pl.lit("num changed")))
    return x.with_columns(key=pl.when(pl.col("n1") == pl.col("n2")).then(pl.lit("name eq")).otherwise(pl.lit("name diff")) + " | " + addr
                          + pl.when(pl.col("na").fill_null(1) > 1).then(pl.lit(" | shared addr")).otherwise(pl.lit("")), pb=PB)


keep = pl.read_parquet(f"{D}keep_v11.parquet")
fin = pl.read_csv(f"{OUT}matching_results_probeA2.tsv", **rd).filter(pl.col("matched_entity_ids") != "").with_columns(m=pl.col("matched_entity_ids").str.split(",")).explode("m")
for c in ("US", "India"):
    s1c = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == c).select("rid").join(keep, on="rid", how="semi")
    U = pl.read_parquet(f"{DN}train_joint9.parquet", columns=["rid", "oid", "pj", "y", "country"]).filter(pl.col("country") == c).join(s1c, on="rid", how="semi").select("rid", "oid", "y", p="pj")
    pred = decide.ef_predict(U.select("rid", "oid", p2="p"), "p2", 0.0, 1.0, 0.05)
    tr = keyed(U.join(pred.with_columns(acc=pl.lit(1, pl.Int8)), on=["rid", "oid"], how="left").with_columns(pl.col("acc").fill_null(0)), "train", c)
    s1t = pl.read_parquet(f"{D}p1_canon_test_{c}_s1.parquet", columns=["idx", "entity_id"])
    ot = pl.read_parquet(f"{D}p1_canon_test_{c}_o.parquet", columns=["idx", "entity_id"])
    pt = (fin.join(s1t.rename({"entity_id": "source1_entity_id", "idx": "rid"}), on="source1_entity_id").join(ot.rename({"entity_id": "m", "idx": "oid"}), on="m")
          .select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32)))
    T = pl.read_parquet(f"{DN}test_union_p.parquet").select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32), "p").join(s1t.select(pl.col("idx").cast(pl.UInt32).alias("rid")), on="rid", how="semi")
    te = keyed(T.join(pt.with_columns(acc=pl.lit(1, pl.Int8)), on=["rid", "oid"], how="left").with_columns(pl.col("acc").fill_null(0)), "test", c)
    ntr, nte = s1c.height, s1t.height
    a = tr.group_by("key", "pb").agg(true_tr=pl.col("y").sum() / ntr, acc_tr=pl.col("acc").sum() / ntr, prec_tr=pl.col("y").filter(pl.col("acc") == 1).mean())
    b = te.group_by("key", "pb").agg(acc_te=pl.col("acc").sum() / nte, n_acc_te=pl.col("acc").sum())
    t = a.join(b, on=["key", "pb"], how="full", coalesce=True).fill_null(0).with_columns(
        ratio=pl.col("acc_te") / (pl.col("true_tr") + 1e-6), excess=(pl.col("acc_te") - pl.col("true_tr")) * nte)
    t.write_parquet(f"D:/amazon_er/france_night/data/shift_all_cells_{c}.parquet")
    te.filter(pl.col("acc") == 1).select("rid", "oid", "key", "pb").write_parquet(f"D:/amazon_er/france_night/data/shift_all_test_{c}.parquet")
    print(f"\n==== {c}: per band totals (train true/S1 vs test accepted/S1)")
    with pl.Config(tbl_rows=40, tbl_width_chars=230, float_precision=4, tbl_hide_column_data_types=True, fmt_str_lengths=50):
        print(t.group_by("pb").agg(pl.col("true_tr").sum(), pl.col("acc_tr").sum(), pl.col("acc_te").sum(), pl.col("n_acc_te").sum()).sort("pb"))
        print(t.filter((pl.col("n_acc_te") >= 150) & (pl.col("pb") != "5 p>=.97")).sort("excess", descending=True).select("key", "pb", "true_tr", "prec_tr", "acc_te", "n_acc_te", "ratio", "excess").head(25))
