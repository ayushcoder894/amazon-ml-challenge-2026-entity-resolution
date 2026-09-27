"""Deflation check: per pattern key x band, train candidates/true/accepted vs test candidates/accepted (per entity).
If test true matches slid from high bands into rejected bands, test would show MORE candidates in 0.3-0.7 than train,
while its accepted total over all bands stays below train truth."""
import polars as pl
import decide
from shift_all import keyed, D, DN, OUT, rd
PB5 = (pl.when(pl.col("p") >= 0.97).then(pl.lit("5 >=.97")).when(pl.col("p") >= 0.7).then(pl.lit("4 .70-.97"))
       .when(pl.col("p") >= 0.5).then(pl.lit("3 .50-.70")).when(pl.col("p") >= 0.3).then(pl.lit("2 .30-.50")).otherwise(pl.lit("1 <.30")))
keep = pl.read_parquet(f"{D}keep_v11.parquet")
fin = pl.read_csv(f"{OUT}matching_results_probeA2.tsv", **rd).filter(pl.col("matched_entity_ids") != "").with_columns(m=pl.col("matched_entity_ids").str.split(",")).explode("m")
for c in ("US", "India"):
    s1c = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == c).select("rid").join(keep, on="rid", how="semi")
    U = pl.read_parquet(f"{DN}train_joint9.parquet", columns=["rid", "oid", "pj", "y", "country"]).filter(pl.col("country") == c).join(s1c, on="rid", how="semi").select("rid", "oid", "y", p="pj")
    pred = decide.ef_predict(U.select("rid", "oid", p2="p"), "p2", 0.0, 1.0, 0.05)
    tr = keyed(U.join(pred.with_columns(acc=pl.lit(1, pl.Int8)), on=["rid", "oid"], how="left").with_columns(pl.col("acc").fill_null(0)), "train", c).with_columns(pb=PB5)
    s1t = pl.read_parquet(f"{D}p1_canon_test_{c}_s1.parquet", columns=["idx", "entity_id"])
    ot = pl.read_parquet(f"{D}p1_canon_test_{c}_o.parquet", columns=["idx", "entity_id"])
    pt = (fin.join(s1t.rename({"entity_id": "source1_entity_id", "idx": "rid"}), on="source1_entity_id").join(ot.rename({"entity_id": "m", "idx": "oid"}), on="m")
          .select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32)))
    T = pl.read_parquet(f"{DN}test_union_p.parquet").select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32), "p").join(s1t.select(pl.col("idx").cast(pl.UInt32).alias("rid")), on="rid", how="semi")
    te = keyed(T.join(pt.with_columns(acc=pl.lit(1, pl.Int8)), on=["rid", "oid"], how="left").with_columns(pl.col("acc").fill_null(0)), "test", c).with_columns(pb=PB5)
    ntr, nte = s1c.height, s1t.height
    for key in ("name eq | name-only", "name diff | name-only", "name eq | num changed", "name diff | addr equal"):
        a = tr.filter(pl.col("key") == key).group_by("pb").agg(cand_tr=pl.len() / ntr, true_tr=pl.col("y").sum() / ntr, acc_tr=pl.col("acc").sum() / ntr)
        b = te.filter(pl.col("key") == key).group_by("pb").agg(cand_te=pl.len() / nte, acc_te=pl.col("acc").sum() / nte)
        t = a.join(b, on="pb", how="full", coalesce=True).fill_null(0).sort("pb")
        print(f"\n{c} | {key}   (per entity; totals: train true {t['true_tr'].sum():.4f}, test accepted {t['acc_te'].sum():.4f}, gap {(t['true_tr'].sum() - t['acc_te'].sum()) * nte:,.0f} pairs)")
        with pl.Config(float_precision=4, tbl_hide_column_data_types=True):
            print(t.with_columns(prec_tr=pl.col("true_tr") / pl.col("cand_tr")))
    te.filter(pl.col("key").str.contains("name-only") & (pl.col("acc") == 0) & (pl.col("p") >= 0.3)).select("rid", "oid", "p", "key").write_parquet(f"D:/amazon_er/france_night/data/deflate_cands_{c}.parquet")
