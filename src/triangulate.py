"""Triangulation test (train, harder split, honest out-of-fold final probabilities).

Rule: if record R is an exact copy (same sorted word set of name + address, or same canonical name key AND canonical
address key) of a record M that is confidently matched to entity E (predicted, pf >= conf), then R also belongs to E.
Reports the precision of the rule, how many missed true pairs it recovers (candidates / never-candidates), and the
honest macro F0.5 when the triangle pairs are added (one owner per record respected).
"""
import sys

import polars as pl

import decide
import evaluate as ev

D = "C:/ML/Challenges/amazon/work/data/"


def word_key(col):
    return (pl.col(col).str.to_lowercase().str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars().str.split(" ")
            .list.unique().list.sort().list.join(" "))


def main(conf=0.9):
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    s1c = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).join(keep, on="rid", how="semi")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").join(keep, on="rid", how="semi").select("rid", "oid")
    own = pl.read_parquet(f"{D}train_gt_rid.parquet").select(owner="rid", oid="oid")
    K = pl.read_parquet(f"{D}train_oof_final_v12_twin.parquet", columns=["rid", "oid", "pf", "country"])
    pred = decide.ef_predict(K.select("rid", "oid", pl.col("pf").alias("p2")), "p2", 0.0, 1.0, 0.05).select("rid", "oid")
    members = pred.join(K.select("rid", "oid", "pf"), on=["rid", "oid"]).filter(pl.col("pf") >= conf).select("rid", m="oid")
    raw = pl.concat([pl.read_parquet(f"{D}train_source{k}.parquet", columns=["business_name", "business_address", "country"])
                     for k in (2, 3)]).with_row_index("oid").select(
        pl.col("oid").cast(pl.UInt32), "country", k_words=word_key("business_name") + pl.lit(" || ") + word_key("business_address"),
        aempty=pl.col("business_address").str.strip_chars() == "")
    canon = pl.concat([pl.read_parquet(f"{D}p1_canon_train_{c}_o.parquet", columns=["idx", "cnk", "cak"]) for c in ("India", "US")]
                      ).select(oid=pl.col("idx").cast(pl.UInt32), k_canon=pl.col("cnk") + pl.lit(" || ") + pl.col("cak"),
                               ok=(pl.col("cnk") != "") & (pl.col("cak") != ""))
    raw = raw.join(canon, on="oid", how="left")
    taken = pred.select(oid="oid", taken=pl.lit(True))
    base = ev.macro_f05(pred, gt, s1c["rid"])
    for kname in ("k_words", "k_canon"):
        rk = raw.filter(pl.col("ok") if kname == "k_canon" else pl.lit(True)).select("oid", "country", key=pl.col(kname), aempty="aempty")
        tri = (members.join(rk.rename({"oid": "m"}).select("m", "key"), on="m")
               .join(rk, on="key").filter(pl.col("oid") != pl.col("m"))
               .select("rid", "oid", "aempty").unique(["rid", "oid"]))
        tri = tri.join(pred, on=["rid", "oid"], how="anti").join(taken, on="oid", how="left")   # new pairs only
        tri = tri.join(own, on="oid", how="left").with_columns(
            y=(pl.col("owner") == pl.col("rid")).fill_null(False),
            cand=pl.struct("rid", "oid").is_in(K.select(pl.struct("rid", "oid")).to_series().implode()))
        free = tri.filter(pl.col("taken").is_null())                 # record not yet given to any entity
        amb = free.group_by("oid").agg(n=pl.len())                   # a record linked to 2+ entities stays out
        add = free.join(amb.filter(pl.col("n") == 1), on="oid", how="semi")
        new = pl.concat([pred, add.select("rid", "oid")])
        f = ev.macro_f05(new, gt, s1c["rid"])
        print(f"\n=== {kname}: triangle pairs not predicted {tri.height:,} (true {tri['y'].mean():.3f}); "
              f"record already given to another entity {tri['taken'].is_not_null().sum():,}")
        print(f"  addable (record free, one entity): {add.height:,}, precision {add['y'].mean():.3f}; "
              f"recovers {add.filter(pl.col('y') & pl.col('cand')).height:,} missed candidates and "
              f"{add.filter(pl.col('y') & ~pl.col('cand')).height:,} never-candidates")
        print(f"  by record address: " + ", ".join(f"{'empty' if e else 'present'}: {n:,} pairs, precision {p:.3f}"
                                                  for e, n, p in add.group_by("aempty").agg(pl.len(), pl.col("y").mean()).sort("aempty").iter_rows()))
        print(f"  macro F0.5 {base:.5f} -> {f:.5f} ({f - base:+.5f})")


if __name__ == "__main__":
    main(*[float(a) for a in sys.argv[1:2]])
