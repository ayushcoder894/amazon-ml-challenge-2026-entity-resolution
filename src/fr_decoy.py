"""Does France test carry the US-style test-only decoys (same name, same street, house number changed)?
Per country: pattern candidates per S1 (direct name+street join, blocking-free) and accepted per S1 by
sibling agreement x |diff|, for test (v21 = before any decoy removal, and A8) and train truth (US, India)."""
import polars as pl
from nearby_number import pattern, sib_agree

D = "C:/ML/Challenges/amazon/work/data/"
OUT = "C:/ML/Challenges/amazon/work/output/"
rd = dict(separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)


def accepted(v, c):
    s1t = pl.read_parquet(f"{D}p1_canon_test_{c}_s1.parquet", columns=["idx", "entity_id"])
    ot = pl.read_parquet(f"{D}p1_canon_test_{c}_o.parquet", columns=["idx", "entity_id"])
    fin = pl.read_csv(f"{OUT}matching_results_{v}.tsv", **rd).filter(pl.col("matched_entity_ids") != "").with_columns(m=pl.col("matched_entity_ids").str.split(",")).explode("m")
    return (fin.join(s1t.rename({"entity_id": "source1_entity_id", "idx": "rid"}), on="source1_entity_id").join(ot.rename({"entity_id": "m", "idx": "oid"}), on="m")
            .select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32))), s1t.height


rows = []
for c in ("US", "India"):
    P, o = pattern("train", c)
    P = P.with_columns(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32))
    s1c = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == c).select(pl.col("rid").cast(pl.UInt32))
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid"]).select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32)).join(s1c, on="rid", how="semi")
    X = sib_agree(P, gt, o.with_columns(pl.col("idx").cast(pl.UInt32)), "sib").join(gt.with_columns(y=pl.lit(1)), on=["rid", "oid"], how="left").with_columns(pl.col("y").fill_null(0))
    n = s1c.height
    rows.append(X.group_by("sib", "dbin").agg(v=pl.len() / n).with_columns(what=pl.lit(f"{c} train cand")))
    rows.append(X.group_by("sib", "dbin").agg(v=pl.col("y").sum() / n).with_columns(what=pl.lit(f"{c} train TRUE")))
for c in ("US", "India", "France"):
    P, o = pattern("test", c)
    P = P.with_columns(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32))
    o = o.with_columns(pl.col("idx").cast(pl.UInt32))
    for v in ("v21", "probeA8"):
        A, n = accepted(v, c)
        X = sib_agree(P, A, o, "sib").join(A.with_columns(acc=pl.lit(1)), on=["rid", "oid"], how="left").with_columns(pl.col("acc").fill_null(0))
        if v == "v21":
            rows.append(X.group_by("sib", "dbin").agg(v=pl.len() / n).with_columns(what=pl.lit(f"{c} test cand")))
        rows.append(X.group_by("sib", "dbin").agg(v=pl.col("acc").sum() / n).with_columns(what=pl.lit(f"{c} test acc {v}")))
        if c == "France" and v == "probeA8":
            X.filter(pl.col("acc") == 1).write_parquet("D:/amazon_er/france_night/data/fr_decoy_acc.parquet")
T = pl.concat([r.select("what", "sib", "dbin", "v") for r in rows]).pivot(on="what", index=["sib", "dbin"], values="v").sort("sib", "dbin")
with pl.Config(tbl_rows=40, tbl_width_chars=400, float_precision=4, tbl_hide_column_data_types=True, fmt_str_lengths=20):
    cols = ["sib", "dbin", "US train TRUE", "India train TRUE", "US train cand", "US test cand", "India test cand", "France test cand",
            "US test acc v21", "US test acc probeA8", "India test acc v21", "France test acc v21", "France test acc probeA8"]
    T.write_parquet("D:/amazon_er/france_night/data/fr_decoy_table.parquet")
    for grp in (cols[:6], cols[:2] + cols[6:]):
        print(T.select([c for c in grp if c in T.columns]))
