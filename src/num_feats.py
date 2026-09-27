"""House-number features that keep compound / alphanumeric numbers intact ("N87W17833", "4-144/A-SF1", "1/1").

The pipeline's number features parse plain integers, so a decoy that shifts the last digit run of a compound number
("1/1" -> "1/8", "N87W17833" -> "N87W17836") looks like a typo. Generator evidence (p1): a small arithmetic shift of
the house number is mostly a decoy, a digit substitution mostly noise.
Per candidate pair, S1 primary number token = first token of the S1 address that contains a digit:
  hn_has1 / hn_has2   S1 / record have a number token
  hn_exact            the primary token appears exactly in the record
  hn_skel             a record token has the same shape (digit runs -> #) as the primary token
  hn_runs_eq          ... and all its digit runs except the last are equal
  hn_lastdiff         |last digit run difference| for that token (-1 if none), capped at 10^6
  hn_n_tok            number tokens in the record
Writes data/hn_{split}{tag}.parquet.
"""
import sys

import polars as pl

D = "C:/ML/Challenges/amazon/work/data/"
HN_COLS = ["hn_has1", "hn_has2", "hn_exact", "hn_skel", "hn_runs_eq", "hn_lastdiff", "hn_n_tok"]
TOK = r"[a-z0-9]*\d[a-z0-9/\-]*"


def norm_tokens(col):
    s = (pl.col(col).str.to_lowercase().str.replace_all(r"n°|nº|#", " ").str.replace_all(r"\b[hd]\.?\s*no\b[\.:]?", " ")
         .str.replace_all(r"\bno[\.:]?\s*(\d)", "$1").str.replace_all(r"\s*([/\-])\s*", "$1"))
    return s.str.extract_all(TOK).list.eval(pl.element().str.strip_chars("/-").str.replace_all(r"\b0+(\d)", "$1"))


def main(split, src, tag):
    K = pl.read_parquet(f"{D}{src}.parquet", columns=["rid", "oid"])
    s1 = pl.read_parquet(f"{D}{split}_source1.parquet", columns=["business_address"]).with_row_index("rid").select(
        pl.col("rid").cast(pl.UInt32), t1=norm_tokens("business_address"))
    o = pl.concat([pl.read_parquet(f"{D}{split}_source{k}.parquet", columns=["business_address"]) for k in (2, 3)]).with_row_index(
        "oid").select(pl.col("oid").cast(pl.UInt32), t2=norm_tokens("business_address"))
    x = (K.with_columns(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32)).with_row_index("i")
         .join(s1, on="rid", how="left").join(o, on="oid", how="left"))
    x = x.with_columns(prim=pl.col("t1").list.first(), hn_has1=(pl.col("t1").list.len() > 0).cast(pl.Int8),
                       hn_has2=(pl.col("t2").list.len() > 0).cast(pl.Int8), hn_n_tok=pl.col("t2").list.len().cast(pl.Int16))
    x = x.with_columns(hn_exact=pl.col("t2").list.contains(pl.col("prim")).fill_null(False).cast(pl.Int8))
    skel = lambda e: e.str.replace_all(r"\d+", "#")
    last = lambda e: e.str.extract(r"(\d+)[^\d]*$", 1).str.slice(0, 7).cast(pl.Int64, strict=False)
    head = lambda e: e.str.replace(r"\d+([^\d]*)$", "#$1")
    e = (x.select("i", "prim", t="t2").explode("t").filter(pl.col("t").is_not_null() & pl.col("prim").is_not_null())
         .with_columns(sk=skel(pl.col("t")) == skel(pl.col("prim")),
                       req=head(pl.col("t")) == head(pl.col("prim")),
                       d=(last(pl.col("t")) - last(pl.col("prim"))).abs())
         .filter(pl.col("sk")))
    agg = e.group_by("i").agg(hn_skel=pl.lit(1, pl.Int8), hn_runs_eq=pl.col("req").max().cast(pl.Int8),
                              hn_lastdiff=pl.col("d").filter(pl.col("req")).min().fill_null(-1).clip(-1, 1_000_000))
    out = (x.select("i", "rid", "oid", *[c for c in HN_COLS if c in ("hn_has1", "hn_has2", "hn_exact", "hn_n_tok")])
           .join(agg, on="i", how="left").with_columns(pl.col("hn_skel", "hn_runs_eq").fill_null(0), pl.col("hn_lastdiff").fill_null(-1))
           .sort("i").select("rid", "oid", *HN_COLS))
    out.write_parquet(f"{D}hn_{split}{tag}.parquet")
    print(f"{split}: {out.height:,} pairs; exact {out['hn_exact'].mean():.3f}, same shape {out['hn_skel'].mean():.3f}, "
          f"shift 1..20 {out.filter((pl.col('hn_lastdiff') >= 1) & (pl.col('hn_lastdiff') <= 20)).height / out.height:.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
