"""Name-only records left unassigned: find the owner by NAME over ALL S1s (not only blocking candidates).
Test condition: ~98% of name-only records have a present owner (no orphans), so an unassigned name-only record
usually belongs to some S1. Rare-token inverted index over S1 canonical names -> fuzzy scores -> top1/top2 per record.
  python nameonly_fuzzy.py val US      precision of top1 by (score, margin, in-candidates) on the test-like harder split
  python nameonly_fuzzy.py test US     writes D:/amazon_er/france_night/data/nofz_test_US.parquet (record, top1, scores)"""
import sys
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
import decide

D = "C:/ML/Challenges/amazon/work/data/"; DN = "D:/amazon_er/dense/"; OUT = "C:/ML/Challenges/amazon/work/output/"
N = "D:/amazon_er/france_night/data/"
rd = dict(separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
MAXDF = 200


def top2(R, S):
    """R: records (oid, ocn, otok list); S: S1s (rid, cnk, tok list). Returns per record top1 rid, s1, s2."""
    st = S.select("rid", "cnk", t="tok").explode("t").filter(pl.col("t") != "").unique(["rid", "t"])
    df = st.group_by("t").len("df")
    st = st.join(df.filter(pl.col("df") <= MAXDF), on="t")
    rt = R.select("oid", "ocn", t="otok").explode("t").filter(pl.col("t") != "").unique(["oid", "t"])
    P = rt.join(st.select("t", "rid", "cnk"), on="t").unique(["oid", "rid"])
    a, b = P["ocn"].to_list(), P["cnk"].to_list()
    P = P.with_columns(ts=pl.Series(process.cpdist(a, b, scorer=fuzz.token_set_ratio, workers=-1), dtype=pl.Float32),
                       r=pl.Series(process.cpdist(a, b, scorer=fuzz.ratio, workers=-1), dtype=pl.Float32))
    P = P.with_columns(sc=(pl.col("ts") + pl.col("r")) / 2).sort(["oid", "sc"], descending=[False, True])
    g = P.group_by("oid", maintain_order=True).agg(rid=pl.col("rid").first(), s1=pl.col("sc").first(), s2=pl.col("sc").slice(1, 1).first(),
                                                   r1=pl.col("r").first(), nc=pl.len())
    return g.with_columns(pl.col("s2").fill_null(0.0), margin=pl.col("s1") - pl.col("s2").fill_null(0.0))


def canon(split, c):
    s1 = pl.read_parquet(f"{D}p1_canon_{split}_{c}_s1.parquet", columns=["idx", "cn", "cnk", "entity_id"]).select(pl.col("idx").cast(pl.UInt32).alias("rid"), "cnk", tok="cn", s="entity_id")
    o = pl.read_parquet(f"{D}p1_canon_{split}_{c}_o.parquet", columns=["idx", "cn", "cnk", "aempty", "entity_id"]).select(pl.col("idx").cast(pl.UInt32).alias("oid"), ocn="cnk", otok="cn", ae="aempty", m="entity_id")
    return s1, o


def val(c):
    keep = pl.read_parquet(f"{D}keep_v11.parquet").select(pl.col("rid").cast(pl.UInt32))
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid"]).with_columns(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32))
    orph = gt.join(keep, on="rid", how="anti").select("oid")
    s1, o = canon("train", c)
    s1 = s1.join(keep, on="rid", how="semi")
    U = pl.read_parquet(f"{DN}train_joint9.parquet", columns=["rid", "oid", "pj", "country"]).filter(pl.col("country") == c).select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32), "pj").join(s1.select("rid"), on="rid", how="semi")
    pred = decide.ef_predict(U.select("rid", "oid", p2="pj"), "p2", 0.0, 1.0, 0.05).select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32))
    R = o.filter(pl.col("ae")).join(orph, on="oid", how="anti").join(pred.select("oid").unique(), on="oid", how="anti")
    # only records of this country's universe: owned by kept S1 of this country, or unowned
    own = gt.join(s1.select("rid"), on="rid", how="semi")
    allown = gt.select("oid")
    R = R.filter(pl.col("oid").is_in(own["oid"].implode()) | ~pl.col("oid").is_in(allown["oid"].implode()))
    g = top2(R, s1.select("rid", "cnk", "tok"))
    g = g.join(own.with_columns(y=pl.lit(1)), on=["rid", "oid"], how="left").with_columns(pl.col("y").fill_null(0))
    g = g.join(U.select("rid", "oid", "pj"), on=["rid", "oid"], how="left").with_columns(incand=pl.col("pj").is_not_null())
    n = s1.height
    print(f"== {c} VAL: unassigned name-only records (test-like) {R.height:,}; with a rare-token S1 match {g.height:,}; "
          f"top1 true {g['y'].mean():.3f}; owner exists for {R.join(own, on='oid', how='semi').height / R.height:.3f}")
    t = (g.with_columns(sb=pl.col("s1").cut([80, 90, 95, 99.9]), mb=pl.col("margin").cut([5, 10, 20, 40]))
         .group_by("incand", "sb", "mb").agg(n=pl.len(), prec=pl.col("y").mean()).filter(pl.col("n") >= 50).sort("incand", "sb", "mb"))
    with pl.Config(tbl_rows=80, float_precision=3, tbl_hide_column_data_types=True):
        print(t)
    g.write_parquet(f"{N}nofz_val_{c}.parquet")
    print(f"   per-S1 scale: {n:,} kept S1")


def test(c, base="probeA14"):
    s1, o = canon("test", c)
    fin = pl.read_csv(f"{OUT}matching_results_{base}.tsv", **rd).filter(pl.col("matched_entity_ids") != "").with_columns(m=pl.col("matched_entity_ids").str.split(",")).explode("m")
    R = o.filter(pl.col("ae")).join(fin.select("m"), on="m", how="anti")
    g = top2(R, s1.select("rid", "cnk", "tok"))
    TP = pl.read_parquet(f"{DN}test_union_p.parquet").select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32), pl.col("p").alias("pj"))
    g = g.join(TP, on=["rid", "oid"], how="left").with_columns(incand=pl.col("pj").is_not_null())
    g = g.join(s1.select("rid", "s"), on="rid").join(o.select("oid", "m"), on="oid")
    print(f"== {c} TEST: unassigned name-only records {R.height:,}; with rare-token match {g.height:,}")
    g.write_parquet(f"{N}nofz_test_{c}.parquet")


if __name__ == "__main__":
    (val if sys.argv[1] == "val" else test)(sys.argv[2])
