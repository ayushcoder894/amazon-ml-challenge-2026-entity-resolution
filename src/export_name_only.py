"""Export every TEST name-only record (empty address) with its candidate Source-1 entities, raw text as it appears,
for manual review. One CSV per country (UTF-8 with BOM, opens correctly in Excel).

Columns: record_id, record_name, n_candidates, name_shared_by (Source-1 entities with the same normalised name),
         cand_rank, s1_entity_id, s1_name, s1_address, probability (our final score), assigned_in_v19 (Y/-),
         s1_matches_in_v19 (records already matched to that entity in v19), your_choice (empty: put Y on the ONE row per
         record you want to assign; leave all empty to assign nothing).
Order: ambiguous records (name shared by 2+ entities) first, then by record, then by candidate rank.
"""
import os

import polars as pl

from france_cats import canon

D = "C:/ML/Challenges/amazon/work/data/"
DN = "D:/amazon_er/dense/"
NIGHT = "D:/amazon_er/france_night/data/"
OUT = "C:/ML/Challenges/amazon/work/name_only_review/"


def main():
    os.makedirs(OUT, exist_ok=True)
    rd = dict(separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
    s1raw = pl.read_parquet(f"{D}test_source1.parquet", columns=["entity_id", "business_name", "business_address", "country"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
    oraw = pl.concat([pl.read_parquet(f"{D}test_source{k}.parquet", columns=["entity_id", "business_name", "business_address"]) for k in (2, 3)]).with_row_index("oid").with_columns(pl.col("oid").cast(pl.UInt32))
    v19 = (pl.read_csv("C:/ML/Challenges/amazon/work/output/matching_results_v19.tsv", **rd)
           .filter(pl.col("matched_entity_ids") != "").with_columns(r=pl.col("matched_entity_ids").str.split(",")).explode("r")
           .select(s1_entity_id="source1_entity_id", record_id="r"))
    nmatch = v19.group_by("s1_entity_id").agg(s1_matches_in_v19=pl.len())
    iu = pl.read_parquet(f"{DN}test_union_p.parquet").select("rid", "oid", probability="p")
    fr = pl.read_parquet(f"{NIGHT}france_union.parquet").select("rid", "oid", probability="pb")
    for c in ("India", "US", "France"):
        s1, o, _ = canon("test", c)
        shared = s1.filter(pl.col("cns") != "").group_by("cns").agg(name_shared_by=pl.len())
        no = o.filter(pl.col("aempty")).select(oid=pl.col("idx").cast(pl.UInt32), cns="cns").join(shared, on="cns", how="left").with_columns(pl.col("name_shared_by").fill_null(0))
        P = (fr if c == "France" else iu).join(no.select("oid"), on="oid", how="semi")
        P = P.join(s1raw.filter(pl.col("country") == c).select("rid"), on="rid", how="semi")
        P = P.with_columns(cand_rank=pl.col("probability").rank("ordinal", descending=True).over("oid").cast(pl.Int32)).filter(pl.col("cand_rank") <= 10)
        P = P.with_columns(n_candidates=pl.len().over("oid").cast(pl.Int32))
        x = (P.join(no.select("oid", "name_shared_by"), on="oid")
             .join(oraw.select("oid", record_id="entity_id", record_name="business_name"), on="oid")
             .join(s1raw.select("rid", s1_entity_id="entity_id", s1_name="business_name", s1_address="business_address"), on="rid")
             .join(v19.with_columns(assigned_in_v19=pl.lit("Y")), on=["s1_entity_id", "record_id"], how="left")
             .join(nmatch, on="s1_entity_id", how="left")
             .with_columns(pl.col("assigned_in_v19").fill_null("-"), pl.col("s1_matches_in_v19").fill_null(0),
                           probability=pl.col("probability").round(3), your_choice=pl.lit(""),
                           ambiguous=(pl.col("name_shared_by") >= 2)))
        x = x.sort(["ambiguous", "record_id", "cand_rank"], descending=[True, False, False])
        cols = ["record_id", "record_name", "n_candidates", "name_shared_by", "cand_rank", "s1_entity_id", "s1_name",
                "s1_address", "probability", "assigned_in_v19", "s1_matches_in_v19", "your_choice"]
        path = f"{OUT}name_only_{c}.csv"
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            f.write(x.select(cols).write_csv())
        nrec = x["record_id"].n_unique()
        namb = x.filter(pl.col("ambiguous"))["record_id"].n_unique()
        nass = x.filter(pl.col("assigned_in_v19") == "Y")["record_id"].n_unique()
        print(f"{c:<7s} {nrec:>7,} name-only records with candidates ({namb:,} ambiguous), {x.height:>9,} rows; "
              f"assigned in v19: {nass:,}  -> {path} ({os.path.getsize(path) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
