"""Key-based blocking with IDF-weighted shared-key scoring (memory-lean, per country).

Keys per record (all scoped to country):
  n|tok        core name token
  a|num|word   address number x address word   (street identity; robust to name noise / Indic names)
  x|num|tok    address number x core name token (robust to street-name typos)
  b|tok|tok    consecutive core-name bigram      (sharper than single tokens for common words)
  p|skel       phonetic skeleton of a core token (typo / transliteration tolerant)
  w|word|word  consecutive address-word bigram   (numberless addresses, noisy house numbers)
  m|num|num    pair of address numbers
  z|skel|word  name-token skeleton x address word (names drawn from a shared vocabulary are only identifying
               in combination with their street/city; tolerant to transliterated names)
  f|core       full sorted core name             (name-only records with unique names)
Keys with df1 * df2 > CAP are dropped (uninformative + quadratic). Pair score = sum of shared-key IDF.
Output: data/<split>_cand.parquet with (rid, oid, score, sn, sa, sx, r1, r2) keeping top-K per S1.
"""
import gc
import math
import sys
import time

import polars as pl

import norm

D = "C:/ML/Challenges/amazon/work/data/"
KINDS = {"n": 0, "a": 1, "x": 2, "b": 3, "p": 4, "w": 5, "m": 6, "z": 7, "f": 8}


def load(split, cols=("rid", "country", "nt", "core", "nums", "words")):
    s1 = pl.read_parquet(f"{D}{split}_s1_norm.parquet", columns=list(cols))
    o = pl.concat([pl.read_parquet(f"{D}{split}_s{k}_norm.parquet", columns=list(cols)) for k in (2, 3)])
    o = o.drop("rid").with_row_index("rid")
    return s1, o


def _pairs(col, name):
    """Unordered consecutive pairs of a list column -> (rid, a, b)."""
    return (pl.col(col).list.slice(0, pl.col(col).list.len() - 1).alias("a"), pl.col(col).list.slice(1).alias("b"))


def _kpair(df, col, tag, kind):
    x = (df.filter(pl.col(col).list.len() >= 2).select("rid", *_pairs(col, tag)).explode("a", "b")
         .drop_nulls("a"))
    return x.select("rid", k=pl.when(pl.col("a") < pl.col("b")).then(tag + pl.col("a") + "|" + pl.col("b"))
                    .otherwise(tag + pl.col("b") + "|" + pl.col("a")), kind=pl.lit(kind, pl.UInt8))


# only true legal forms / honorifics / glue are dropped for the "soft" full name (generic business words stay:
# "om enterprises" and "om traders" are different businesses)
HARD = ["pvt", "ltd", "llc", "inc", "corp", "co", "llp", "plc", "lp", "pc", "pllc", "sa", "sas", "sasu", "sarl", "sci",
        "eurl", "snc", "the", "&", "of", "com", "www", "sri", "smt", "mr", "mrs", "ms", "dr", "dba", "aka", "fka"] + list(
    "abcdefghijklmnopqrstuvwxyz")


def _keys_chunk(df, max_nums=4, max_words=8):
    base = df.select("rid",
                     soft=pl.col("nt").list.eval(pl.element().filter(~pl.element().is_in(HARD))).list.unique().list.sort()
                     .list.join(" "),
                     core=pl.col("core").list.eval(pl.element().filter(pl.element().str.len_chars() >= 2)),
                     nums=pl.col("nums").list.head(max_nums),
                     words=pl.col("words").list.unique(maintain_order=True).list.head(max_words))
    core_u = base.select("rid", c=pl.col("core").list.unique())
    kn = core_u.explode("c").drop_nulls("c").select("rid", k="n|" + pl.col("c"), kind=pl.lit(0, pl.UInt8))
    kp = (core_u.explode("c").drop_nulls("c").filter(pl.col("c").str.len_chars() >= 4)
          .with_columns(c=norm.skeleton_expr(pl.col("c"))).filter(pl.col("c").str.len_chars() >= 3)
          .select("rid", k="p|" + pl.col("c"), kind=pl.lit(4, pl.UInt8)))
    kb = _kpair(base, "core", "b|", 3)
    # all (not only adjacent) pairs of the first 5 core tokens: a typo in the middle token keeps "first|last" intact
    ap = base.select("rid", c=pl.col("core").list.unique(maintain_order=True).list.head(5)).filter(pl.col("c").list.len() >= 3)
    ap = (ap.with_row_index("_i").explode("c").with_columns(j=pl.int_range(pl.len()).over("_i"))
          .join(ap.with_row_index("_i").explode("c").with_columns(j=pl.int_range(pl.len()).over("_i"))
                .select("_i", c2="c", j2="j"), on="_i")
          .filter(pl.col("j2") > pl.col("j") + 1))
    kq = ap.select("rid", k=pl.when(pl.col("c") < pl.col("c2")).then("b|" + pl.col("c") + "|" + pl.col("c2"))
                   .otherwise("b|" + pl.col("c2") + "|" + pl.col("c")), kind=pl.lit(3, pl.UInt8))
    kw = _kpair(base, "words", "w|", 5)
    km = _kpair(base.with_columns(nums=pl.col("nums").list.head(3)), "nums", "m|", 6)
    nw = base.select("rid", "nums", "words").explode("nums").drop_nulls("nums").explode("words").drop_nulls("words")
    ka = nw.select("rid", k="a|" + pl.col("nums") + "|" + pl.col("words"), kind=pl.lit(1, pl.UInt8))
    nx = base.select("rid", "nums", c="core").explode("nums").drop_nulls("nums").with_columns(
        c=pl.col("c").list.unique()).explode("c").drop_nulls("c")
    kx = nx.select("rid", k="x|" + pl.col("nums") + "|" + pl.col("c"), kind=pl.lit(2, pl.UInt8))
    kz = (base.select("rid", c=pl.col("core").list.unique(maintain_order=True).list.head(3),
                      w=pl.col("words").list.head(6))
          .explode("c").drop_nulls("c").filter(pl.col("c").str.len_chars() >= 3)
          .with_columns(c=norm.skeleton_expr(pl.col("c"))).filter(pl.col("c").str.len_chars() >= 2)
          .explode("w").drop_nulls("w")
          .select("rid", k="z|" + pl.col("c") + "|" + pl.col("w"), kind=pl.lit(7, pl.UInt8)))
    kf = (base.filter(pl.col("core").list.len() >= 1)
          .select("rid", k="f|" + pl.col("core").list.sort().list.join(" "), kind=pl.lit(8, pl.UInt8)))
    # typo-tolerant full-name key: sorted phonetic skeletons of the core tokens
    kg = (base.filter(pl.col("core").list.len() >= 1)
          .select("rid", k="g|" + pl.col("core").list.eval(norm.skeleton_expr(pl.element())).list.sort().list.join(" "),
                  kind=pl.lit(8, pl.UInt8)))
    # full soft name x address number / address word: specific even when every single name token is common
    sb = base.filter(pl.col("soft").str.len_chars() >= 2)
    ky = (sb.select("rid", "soft", n=pl.col("nums").list.head(3)).explode("n").drop_nulls("n")
          .select("rid", k="y|" + pl.col("soft") + "|" + pl.col("n"), kind=pl.lit(9, pl.UInt8)))
    kv = (sb.select("rid", "soft", w=pl.col("words").list.head(6)).explode("w").drop_nulls("w")
          .select("rid", k="v|" + pl.col("soft") + "|" + pl.col("w"), kind=pl.lit(9, pl.UInt8)))
    out = pl.concat([kn, kp, kb, kq, kw, km, ka, kx, kz, kf, kg, ky, kv])
    return out.select(pl.col("rid").cast(pl.UInt32), h=pl.col("k").hash(seed=7), kind="kind").unique(["rid", "h"])


def keys(df, step=1_000_000):
    return pl.concat([_keys_chunk(df.slice(i, step)) for i in range(0, df.height, step)])


def block_country(s1, o, cap, topk, rtopk, chunk, t):
    k1, k2 = keys(s1), keys(o)
    df1 = k1.group_by("h").agg(d1=pl.len())
    df2 = k2.group_by("h").agg(d2=pl.len())
    n = s1.height + o.height
    dd = df1.join(df2, on="h").filter(pl.col("d1") * pl.col("d2") <= cap).with_columns(
        w=(math.log(n) - (pl.col("d1") + pl.col("d2")).cast(pl.Float64).log()).cast(pl.Float32))
    raw = int((dd["d1"] * dd["d2"]).sum())
    k1 = k1.join(dd.select("h", "w"), on="h")
    k2 = k2.join(dd.select("h"), on="h").select(pl.col("rid").alias("oid"), "h").sort("h")
    del df1, df2, dd
    gc.collect()
    print(f"   keys s1 {k1.height:,} other {k2.height:,}; raw pairs {raw:,} ({time.time() - t:.0f}s)", flush=True)
    def agg(j, key, k):
        p = (j.group_by("rid", "oid")
             .agg(score=pl.col("w").sum(),
                  sn=(pl.col("w") * pl.col("kind").is_in([0, 3, 4, 8])).sum(),
                  sa=(pl.col("w") * pl.col("kind").is_in([1, 5, 6])).sum(),
                  sx=(pl.col("w") * (pl.col("kind") == 2)).sum(),
                  sz=(pl.col("w") * pl.col("kind").is_in([7, 9])).sum(),
                  nk=pl.len().cast(pl.UInt16)))
        return p.filter(pl.col("score").rank("ordinal", descending=True).over(key) <= k)

    outs = []
    rids = s1["rid"]
    for i in range(0, s1.height, chunk):
        lo, hi = rids[i], rids[min(i + chunk, s1.height) - 1]
        outs.append(agg(k1.filter(pl.col("rid").is_between(lo, hi)).join(k2, on="h"), "rid", topk))
    # reverse pass: each S2/S3 record keeps its best rtopk S1s (rescues name-only / crowded records)
    k1r = k1.select("rid", "h", "w", "kind")
    oids = o["rid"]
    for i in range(0, o.height, chunk * 4):
        lo, hi = oids[i], oids[min(i + chunk * 4, o.height) - 1]
        outs.append(agg(k2.filter(pl.col("oid").is_between(lo, hi)).join(k1r, on="h"), "oid", rtopk))
    p = pl.concat(outs, how="diagonal").unique(["rid", "oid"])
    return p.with_columns(r1=pl.col("score").rank("ordinal", descending=True).over("rid").cast(pl.UInt16),
                          r2=pl.col("score").rank("ordinal", descending=True).over("oid").cast(pl.UInt16))


def block(split, cap=5000, topk=40, rtopk=3, chunk=60_000):
    t = time.time()
    s1, o = load(split)
    outs = []
    for c in s1["country"].unique().sort().to_list():
        a, b = s1.filter(pl.col("country") == c), o.filter(pl.col("country") == c)
        print(f"[{c}] s1 {a.height:,} other {b.height:,}", flush=True)
        p = block_country(a, b, cap, topk, rtopk, chunk, t)
        print(f"[{c}] {p.height:,} pairs ({time.time() - t:.0f}s)", flush=True)
        outs.append(p)
        gc.collect()
    p = pl.concat(outs)
    p.write_parquet(f"{D}{split}_cand.parquet")
    print(f"total {p.height:,} pairs, {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    block(sys.argv[1] if len(sys.argv) > 1 else "train")
