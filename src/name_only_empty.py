"""User's rule on TRAIN (harder split): if all of a Source-1 entity's matches are name-only records (none with an
address), leave the entity empty. Variants:
  E1  entity-level: S1 whose predicted records are ALL name-only -> predict nothing for it
  E2  same, only when every such record's name is shared by 2+ S1 entities
  R1  record-level: never match a name-only record (drop every name-only pair)
  R2  same, only for names shared by 2+ S1 entities
Compared with the current model (joint9 + v13 rule). Also counts what each rule would change in v19 on test."""
import polars as pl

import decide
import evaluate as ev

D = "C:/ML/Challenges/amazon/work/data/"
DN = "D:/amazon_er/dense/"
OUT = "C:/ML/Challenges/amazon/work/output/"
keep = pl.read_parquet(f"{D}keep_v11.parquet")


def flags(split, c, s1_rids):
    s1 = pl.read_parquet(f"{D}p1_canon_{split}_{c}_s1.parquet", columns=["idx", "cns"]).rename({"idx": "rid"}).join(s1_rids, on="rid", how="semi")
    o = pl.read_parquet(f"{D}p1_canon_{split}_{c}_o.parquet", columns=["idx", "cns", "aempty"]).rename({"idx": "oid"})
    sh = s1.filter(pl.col("cns") != "").group_by("cns").agg(nsh=pl.len())
    return o.join(sh, on="cns", how="left").select("oid", no=pl.col("aempty"), amb=pl.col("nsh").fill_null(0) >= 2)


def rules(pred, F):
    x = pred.join(F, on="oid", how="left").with_columns(pl.col("no", "amb").fill_null(False))
    g = x.group_by("rid").agg(all_no=pl.col("no").all(), all_no_amb=(pl.col("no") & pl.col("amb")).all())
    return {
        "E1  S1 has only name-only matches -> empty": x.join(g.filter(pl.col("all_no")).select("rid"), on="rid", how="anti"),
        "E2  ... and their names are shared -> empty": x.join(g.filter(pl.col("all_no_amb")).select("rid"), on="rid", how="anti"),
        "R1  never match a name-only record": x.filter(~pl.col("no")),
        "R2  never match a shared-name name-only record": x.filter(~(pl.col("no") & pl.col("amb"))),
    }


for c in ("India", "US"):
    s1c = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == c).select("rid").join(keep, on="rid", how="semi")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid"]).join(s1c, on="rid", how="semi")
    U = pl.read_parquet(f"{DN}train_joint9.parquet", columns=["rid", "oid", "pj", "country"]).filter(pl.col("country") == c).drop("country")
    pred = decide.ef_predict(U.select("rid", "oid", pl.col("pj").alias("p2")), "p2", 0.0, 1.0, 0.05)
    del U
    F = flags("train", c, s1c)
    base = ev.macro_f05(pred, gt, s1c["rid"])
    print(f"\n{c} TRAIN harder split ({s1c.height:,} S1), current model {base:.5f}")
    for lab, p in rules(pred, F).items():
        rem = pred.join(p.select("rid", "oid"), on=["rid", "oid"], how="anti")
        true = rem.join(gt, on=["rid", "oid"], how="semi").height
        f = ev.macro_f05(p.select("rid", "oid"), gt, s1c["rid"])
        print(f"  {lab:<48s} S1 changed {rem['rid'].n_unique():>6,}  pairs removed {rem.height:>6,} (of which truly matches {true / max(rem.height, 1):.1%}) -> {f:.5f} ({f - base:+.5f})")

print("\nTEST v19: what each rule would change")
rd = dict(separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
fin = (pl.read_csv(f"{OUT}matching_results_v19.tsv", **rd).filter(pl.col("matched_entity_ids") != "")
       .with_columns(m=pl.col("matched_entity_ids").str.split(",")).explode("m"))
for c in ("India", "US", "France"):
    s1 = pl.read_parquet(f"{D}p1_canon_test_{c}_s1.parquet", columns=["idx", "entity_id"])
    o = pl.read_parquet(f"{D}p1_canon_test_{c}_o.parquet", columns=["idx", "entity_id"])
    pred = (fin.join(s1.rename({"entity_id": "source1_entity_id", "idx": "rid"}), on="source1_entity_id")
            .join(o.rename({"entity_id": "m", "idx": "oid"}), on="m").select("rid", "oid"))
    F = flags("test", c, s1.select(rid="idx"))
    for lab, p in rules(pred, F).items():
        rem = pred.join(p.select("rid", "oid"), on=["rid", "oid"], how="anti")
        print(f"  {c:<7s} {lab:<48s} S1 changed {rem['rid'].n_unique():>6,}  pairs removed {rem.height:>6,}")
