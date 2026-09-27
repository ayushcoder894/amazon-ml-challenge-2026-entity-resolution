"""Learn a native-script -> Latin word dictionary from training matches (no external data).

Native-script names in S2/S3 are token-by-token transliterations of the S1 name
("स्काई हॉस्पिटैलिटी प्राइवेट लिमिटेड" <-> "Sky Hospitality Private Limited"). For every training match whose
S2/S3 name is in an Indic script and has the same token count as the S1 name, align tokens by position and
count (native token, latin token). Keep the majority mapping when it is frequent and consistent.
"""
import json

import polars as pl

import norm

D = "C:/ML/Challenges/amazon/work/data/"
INDIC = "[\u0900-\u0d7f]"


def split_native(e):
    return e.str.replace_all(r"[^\u0900-\u0d7f\w]+", " ").str.strip_chars().str.split(" ")


def build(min_count=2, min_share=0.6):
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid"])
    s1 = pl.read_parquet(f"{D}train_source1.parquet", columns=["business_name"]).with_row_index("rid")
    o = pl.concat([pl.read_parquet(f"{D}train_source{k}.parquet", columns=["business_name"]) for k in (2, 3)]
                  ).with_row_index("oid").filter(pl.col("business_name").str.contains(INDIC))
    x = gt.join(o.rename({"business_name": "nat"}), on="oid").join(s1.rename({"business_name": "lat"}), on="rid")
    x = x.with_columns(lat=norm.fold_series(x["lat"])).with_columns(
        lt=pl.col("lat").str.replace_all(r"[^a-z0-9&]+", " ").str.strip_chars().str.split(" "),
        nt=split_native(pl.col("nat")))
    al = x.filter(pl.col("lt").list.len() == pl.col("nt").list.len()).select("nt", "lt").explode("nt", "lt")
    al = al.filter(pl.col("nt").str.contains(INDIC))
    cnt = al.group_by("nt", "lt").agg(n=pl.len())
    tot = cnt.group_by("nt").agg(tot=pl.col("n").sum())
    best = (cnt.join(tot, on="nt").sort("n", descending=True).group_by("nt").first()
            .with_columns(share=pl.col("n") / pl.col("tot"))
            .filter((pl.col("tot") >= min_count) & (pl.col("share") >= min_share)))
    print(f"aligned pairs {x.height:,} (same length {x.filter(pl.col('lt').list.len() == pl.col('nt').list.len()).height:,});"
          f" native tokens {tot.height:,} -> dictionary {best.height:,}")
    d = dict(zip(best["nt"].to_list(), best["lt"].to_list()))
    json.dump(d, open(f"{D}translit_dict.json", "w", encoding="utf8"), ensure_ascii=False)
    return d


def coverage(d):
    """Share of native name tokens (token occurrences) covered by the dictionary, train vs test."""
    for split in ("train", "test"):
        o = pl.concat([pl.read_parquet(f"{D}{split}_source{k}.parquet", columns=["business_name"]) for k in (2, 3)]
                      ).filter(pl.col("business_name").str.contains(INDIC))
        t = o.select(t=split_native(pl.col("business_name"))).explode("t").filter(pl.col("t").str.contains(INDIC))
        cov = t.select(pl.col("t").is_in(list(d.keys())).mean()).item()
        print(f"{split}: {o.height:,} native names, token coverage {cov:.3f}")


if __name__ == "__main__":
    d = build()
    coverage(d)
    for k in list(d)[:12]:
        print(k, "->", d[k])


def build_addr(min_count=5, min_share=0.5):
    """Native tokens in addresses are mostly state names: map each to the S1 address's last component."""
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid"])
    s1 = pl.read_parquet(f"{D}train_source1.parquet", columns=["business_address"]).with_row_index("rid")
    o = pl.concat([pl.read_parquet(f"{D}train_source{k}.parquet", columns=["business_address"]) for k in (2, 3)]
                  ).with_row_index("oid").filter(pl.col("business_address").str.contains(INDIC))
    x = gt.join(o.rename({"business_address": "nat"}), on="oid").join(s1.rename({"business_address": "lat"}), on="rid")
    x = x.with_columns(state=pl.col("lat").str.split(",").list.last().str.strip_chars().str.to_lowercase(),
                       nt=pl.col("nat").str.extract_all(INDIC + "+"))
    al = x.select("nt", "state").explode("nt").drop_nulls("nt")
    cnt = al.group_by("nt", "state").agg(n=pl.len())
    tot = cnt.group_by("nt").agg(tot=pl.col("n").sum())
    best = (cnt.join(tot, on="nt").sort("n", descending=True).group_by("nt").first()
            .with_columns(share=pl.col("n") / pl.col("tot"))
            .filter((pl.col("tot") >= min_count) & (pl.col("share") >= min_share)))
    print(f"address native tokens {tot.height:,} -> dictionary {best.height:,}")
    print(best.sort("tot", descending=True).head(8).select("nt", "state", "share"))
    d = dict(zip(best["nt"].to_list(), best["state"].to_list()))
    json.dump(d, open(f"{D}translit_addr_dict.json", "w", encoding="utf8"), ensure_ascii=False)
    return d
