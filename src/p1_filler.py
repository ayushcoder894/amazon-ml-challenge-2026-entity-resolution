"""Phase 1.4: learn generator filler tokens from data.

A token is a filler candidate when its share of S2/S3 names (or addresses) is far higher than its share of S1 names:
the generator adds it to records, S1 never (or rarely) carries it. Ratio = (df_o / N_o) / ((df_s1 + 1) / N_s1).
Computed per country (train + test; no labels needed) and printed with counts.
"""
import sys

import polars as pl

D = "C:/ML/Challenges/amazon/work/data/"
FOLD = [("[àáâãäåā]", "a"), ("[çćč]", "c"), ("[èéêëēė]", "e"), ("[ìíîïī]", "i"), ("[ñń]", "n"), ("[òóôõöøō]", "o"),
        ("[ùúûüū]", "u"), ("[ýÿ]", "y"), ("œ", "oe"), ("æ", "ae"), ("ß", "ss")]


def fold(e):
    e = e.str.to_lowercase()
    for a, b in FOLD:
        e = e.str.replace_all(a, b)
    return e


def tokens(df, col):
    return (df.select(pl.int_range(pl.len()).alias("i"), t=fold(pl.col(col)).str.replace_all(r"[^a-z0-9]+", " ")
                      .str.strip_chars().str.split(" ")).explode("t").filter(pl.col("t") != "").unique())


def ratios(split, country, col, min_df=200):
    s1 = pl.read_parquet(f"{D}{split}_source1.parquet", columns=[col, "country"]).filter(pl.col("country") == country)
    o = pl.concat([pl.read_parquet(f"{D}{split}_source{k}.parquet", columns=[col, "country"]) for k in (2, 3)]
                  ).filter(pl.col("country") == country)
    a = tokens(s1, col).group_by("t").agg(df1=pl.len())
    b = tokens(o, col).group_by("t").agg(df2=pl.len())
    r = b.join(a, on="t", how="left").with_columns(pl.col("df1").fill_null(0)).filter(pl.col("df2") >= min_df)
    r = r.with_columns(ratio=((pl.col("df2") / o.height) / ((pl.col("df1") + 1) / s1.height)).round(1),
                       pct_o=(100 * pl.col("df2") / o.height).round(3), pct_s1=(100 * pl.col("df1") / s1.height).round(3))
    return r.sort("ratio", descending=True)


def main(col="business_name", top=45):
    pl.Config.set_tbl_rows(top + 5)
    pl.Config.set_tbl_width_chars(160)
    for split, country in (("train", "US"), ("train", "India"), ("test", "France")):
        r = ratios(split, country, col)
        print(f"\n=== {split} {country}: {col} tokens over-represented in S2/S3 vs S1 (df_o >= 200), top {top}")
        print(r.head(top).select("t", "ratio", "df2", "df1", "pct_o", "pct_s1"))
        r.write_parquet(f"{D}p1_filler_{col}_{country}.parquet")


if __name__ == "__main__":
    main(*sys.argv[1:2])
