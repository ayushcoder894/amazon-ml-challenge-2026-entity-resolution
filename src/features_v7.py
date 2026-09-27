"""Pairwise features for candidate pairs (v2), computed per country in chunks (CPU, rapidfuzz multi-threaded).

v2 adds: uncapped IDF-weighted overlap of name tokens / address numbers / address words (+ rarest missing token
on each side), phonetic-skeleton name similarity, exact core-name flags, house-number prefix flag, and carries
all blocking score components (sn, sa, sx, sz, nk).
"""
import math
import sys
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

import norm

D = "C:/ML/Challenges/amazon/work/data/"
REC_COLS = ["rid", "nf", "core", "nums", "words", "at"]
INDIC_RE = "[\u0900-\u0d7f]"
TOKCOLS = ("core", "nums", "words")


def load_records(split, country):
    s1 = (pl.scan_parquet(f"{D}{split}_s1_norm.parquet").filter(pl.col("country") == country)
          .select(REC_COLS).collect())
    o = (pl.concat([pl.scan_parquet(f"{D}{split}_s{k}_norm.parquet") for k in (2, 3)]).drop("rid").with_row_index("rid")
         .filter(pl.col("country") == country).select(REC_COLS + ["entity_id"]).collect())
    o = o.with_columns(src=pl.col("entity_id").str.slice(1, 1).cast(pl.UInt8)).drop("entity_id")
    # Indic-script flag from the raw (pre-transliteration) names; row order matches the norm files
    ind = (pl.concat([pl.scan_parquet(f"{D}{split}_source{k}.parquet") for k in (2, 3)])
           .filter(pl.col("country") == country)
           .select(indic=pl.col("business_name").str.contains(INDIC_RE).cast(pl.UInt8)).collect())
    o = o.with_columns(ind["indic"])
    s1 = s1.with_columns(indic=pl.lit(0, pl.UInt8), src=pl.lit(1, pl.UInt8))

    def enrich(df):
        return df.with_columns(
            cs=pl.col("core").list.join(" "),
            css=pl.col("core").list.sort().list.join(" "),
            skel=pl.col("core").list.eval(norm.skeleton_expr(pl.element())).list.join(" "),
            nfc=pl.col("nf").str.replace_all(r"[^a-z0-9 ]", " ").str.replace_all(r"\s+", " ").str.strip_chars(),
            ac=pl.col("at").list.join(" "),
            ws=pl.col("words").list.sort().list.join(" "),
        ).drop("nf", "at")
    return enrich(s1), enrich(o)


def token_tables(s1, o):
    """Per-record (rid, h, idf) tables for core tokens / numbers / words; IDF over both sides of the country."""
    n = s1.height + o.height
    out = {}
    for col in TOKCOLS:
        t1 = s1.select("rid", h=pl.col(col).list.unique()).explode("h").drop_nulls("h").with_columns(pl.col("h").hash(seed=3))
        t2 = o.select("rid", h=pl.col(col).list.unique()).explode("h").drop_nulls("h").with_columns(pl.col("h").hash(seed=3))
        df = pl.concat([t1.select("h"), t2.select("h")]).group_by("h").agg(df=pl.len())
        idf = df.select("h", idf=(math.log(n) - pl.col("df").cast(pl.Float64).log()).cast(pl.Float32))
        out[col] = (t1.join(idf, on="h"), t2.join(idf, on="h").rename({"rid": "oid"}))
    return out


def idf_overlap(p, tabs):
    """IDF-weighted overlap features for pairs p (pid, rid, oid)."""
    feats = p.select("pid")
    for col, (t1, t2) in tabs.items():
        a = p.select("pid", "rid").join(t1, on="rid").drop("rid")
        b = p.select("pid", "oid").join(t2, on="oid").drop("oid")
        inter = a.join(b.select("pid", "h"), on=["pid", "h"], how="semi")
        m1 = a.join(b.select("pid", "h"), on=["pid", "h"], how="anti")
        m2 = b.join(a.select("pid", "h"), on=["pid", "h"], how="anti")
        f = (a.group_by("pid").agg(t1=pl.col("idf").sum())
             .join(b.group_by("pid").agg(t2=pl.col("idf").sum()), on="pid", how="full", coalesce=True)
             .join(inter.group_by("pid").agg(i=pl.col("idf").sum(), imax=pl.col("idf").max()), on="pid", how="left")
             .join(m1.group_by("pid").agg(m1=pl.col("idf").max()), on="pid", how="left")
             .join(m2.group_by("pid").agg(m2=pl.col("idf").max()), on="pid", how="left")
             .with_columns(pl.exclude("pid").fill_null(0.0)))
        f = f.select("pid",
                     pl.col("i").alias(f"{col}_isum"), pl.col("imax").alias(f"{col}_imax"),
                     (pl.col("i") / pl.col("t1").clip(1e-6)).alias(f"{col}_cov1"),
                     (pl.col("i") / pl.col("t2").clip(1e-6)).alias(f"{col}_cov2"),
                     pl.col("m1").alias(f"{col}_miss1"), pl.col("m2").alias(f"{col}_miss2"))
        feats = feats.join(f, on="pid", how="left")
    return feats.with_columns(pl.exclude("pid").fill_null(0.0))


def _cp(a, b, scorer, **kw):
    return cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw)


def pair_features(p: pl.DataFrame, s1: pl.DataFrame, o: pl.DataFrame, tabs) -> pl.DataFrame:
    """p: candidate pairs (rid, oid, blocking columns...) -> features."""
    p = p.with_row_index("pid")
    a = s1.select(pl.all().name.suffix("_1")).rename({"rid_1": "rid"})
    b = o.select(pl.all().name.suffix("_2")).rename({"rid_2": "oid"})
    x = p.join(a, on="rid", how="left").join(b, on="oid", how="left").sort("pid")

    f = {}
    n1, n2 = x["nfc_1"].to_list(), x["nfc_2"].to_list()
    c1, c2 = x["cs_1"].to_list(), x["cs_2"].to_list()
    cs1, cs2 = x["css_1"].to_list(), x["css_2"].to_list()
    k1, k2 = x["skel_1"].to_list(), x["skel_2"].to_list()
    a1, a2 = x["ac_1"].to_list(), x["ac_2"].to_list()
    w1, w2 = x["ws_1"].to_list(), x["ws_2"].to_list()
    f["n_ratio"] = _cp(n1, n2, fuzz.ratio)
    f["n_tset"] = _cp(n1, n2, fuzz.token_set_ratio)
    f["n_tsort"] = _cp(n1, n2, fuzz.token_sort_ratio)
    f["n_partial"] = _cp(n1, n2, fuzz.partial_ratio)
    f["c_ratio"] = _cp(c1, c2, fuzz.ratio)
    f["c_jw"] = _cp(c1, c2, JaroWinkler.normalized_similarity)
    f["cs_ratio"] = _cp(cs1, cs2, fuzz.ratio)
    f["c_tset"] = _cp(c1, c2, fuzz.token_set_ratio)
    f["c_partial"] = _cp(c1, c2, fuzz.partial_ratio)
    f["c_lev"] = _cp(c1, c2, Levenshtein.distance)
    f["k_ratio"] = _cp(k1, k2, fuzz.ratio)
    f["k_tset"] = _cp(k1, k2, fuzz.token_set_ratio)
    f["a_ratio"] = _cp(a1, a2, fuzz.ratio)
    f["a_tset"] = _cp(a1, a2, fuzz.token_set_ratio)
    f["a_tsort"] = _cp(a1, a2, fuzz.token_sort_ratio)
    f["a_partial"] = _cp(a1, a2, fuzz.partial_ratio)
    f["w_tset"] = _cp(w1, w2, fuzz.token_set_ratio)
    f["w_ratio"] = _cp(w1, w2, fuzz.ratio)
    feats = pl.DataFrame(f)

    keep = [c for c in p.columns if c not in ("pid",)]
    x = x.select(*keep, "pid", "src_2", "indic_2", "core_1", "core_2", "nums_1", "nums_2", "words_1", "words_2",
                 "cs_1", "cs_2", "css_1", "css_2", "ac_2")
    x = x.with_columns(
        c_inter=pl.col("core_1").list.set_intersection("core_2").list.len(),
        c_len1=pl.col("core_1").list.len(), c_len2=pl.col("core_2").list.len(),
        n_inter=pl.col("nums_1").list.set_intersection("nums_2").list.len(),
        n_len1=pl.col("nums_1").list.len(), n_len2=pl.col("nums_2").list.len(),
        w_inter=pl.col("words_1").list.set_intersection("words_2").list.len(),
        w_len1=pl.col("words_1").list.len(), w_len2=pl.col("words_2").list.len(),
        first_num_eq=(pl.col("nums_1").list.first() == pl.col("nums_2").list.first()).fill_null(False).cast(pl.UInt8),
        first_core_eq=(pl.col("core_1").list.first() == pl.col("core_2").list.first()).fill_null(False).cast(pl.UInt8),
        num1_in2=pl.col("nums_2").list.contains(pl.col("nums_1").list.first()).fill_null(False).cast(pl.UInt8),
        num2_in1=pl.col("nums_1").list.contains(pl.col("nums_2").list.first()).fill_null(False).cast(pl.UInt8),
        # house number truncated / extended by noise (7235 vs 723)
        num_prefix=(pl.col("nums_1").list.first().str.starts_with(pl.col("nums_2").list.first())
                    | pl.col("nums_2").list.first().str.starts_with(pl.col("nums_1").list.first())).fill_null(False).cast(pl.UInt8),
        core_eq=(pl.col("cs_1") == pl.col("cs_2")).cast(pl.UInt8),
        css_eq=(pl.col("css_1") == pl.col("css_2")).cast(pl.UInt8),
        cl1=pl.col("cs_1").str.len_chars(), cl2=pl.col("cs_2").str.len_chars(),
        addr_empty2=(pl.col("ac_2").str.len_chars() == 0).cast(pl.UInt8),
        num_maxlen=pl.col("nums_1").list.set_intersection("nums_2").list.eval(
            pl.element().str.len_chars()).list.max().fill_null(0),
    ).with_columns(
        c_jac=pl.col("c_inter") / (pl.col("c_len1") + pl.col("c_len2") - pl.col("c_inter")).clip(1),
        c_cont=pl.col("c_inter") / pl.min_horizontal("c_len1", "c_len2").clip(1),
        n_jac=pl.col("n_inter") / (pl.col("n_len1") + pl.col("n_len2") - pl.col("n_inter")).clip(1),
        n_cont1=pl.col("n_inter") / pl.col("n_len1").clip(1),
        w_jac=pl.col("w_inter") / (pl.col("w_len1") + pl.col("w_len2") - pl.col("w_inter")).clip(1),
        w_cont=pl.col("w_inter") / pl.min_horizontal("w_len1", "w_len2").clip(1),
    ).drop("core_1", "core_2", "nums_1", "nums_2", "words_1", "words_2", "cs_1", "cs_2", "css_1", "css_2", "ac_2")
    x = pl.concat([x, feats], how="horizontal").rename({"src_2": "src", "indic_2": "indic"})
    x = x.join(idf_overlap(x.select("pid", "rid", "oid"), tabs), on="pid", how="left").drop("pid")
    return x


def dense_context(df: pl.DataFrame) -> pl.DataFrame:
    """Competition features on the dense cosine (only when the dense columns are present)."""
    if "cos" not in df.columns:
        return df
    c = pl.col("cos")
    return df.with_columns(
        cos_rank1=c.rank("ordinal", descending=True).over("rid").cast(pl.UInt16),
        cos_gap1=c.max().over("rid") - c,
        cos_rank2=c.rank("ordinal", descending=True).over("oid").cast(pl.UInt16),
        cos_gap2=c.max().over("oid") - c,
        cos_second2=c.sort(descending=True).slice(1, 1).first().over("oid"),
    ).with_columns(cos_margin2=pl.when(pl.col("cos_rank2") == 1).then(c - pl.col("cos_second2"))
                   .otherwise(c - c.max().over("oid")))


def context_features(df: pl.DataFrame) -> pl.DataFrame:
    """Rank/relative features within each S1's and each S2/S3 record's candidate list (needs full candidate set)."""
    df = dense_context(df)
    return df.with_columns(
        rel1=pl.col("score") / pl.col("score").max().over("rid"),
        gap1=pl.col("score").max().over("rid") - pl.col("score"),
        rel2=pl.col("score") / pl.col("score").max().over("oid"),
        gap2=pl.col("score").max().over("oid") - pl.col("score"),
        ncand1=pl.len().over("rid").cast(pl.UInt16),
        ncand2=pl.len().over("oid").cast(pl.UInt16),
        r1src=pl.col("score").rank("ordinal", descending=True).over("rid", "src").cast(pl.UInt16),
        # the same competition view on the uncapped name / address evidence
        cis_rank2=pl.col("core_isum").rank("ordinal", descending=True).over("oid").cast(pl.UInt16),
        cis_gap2=pl.col("core_isum").max().over("oid") - pl.col("core_isum"),
        wis_rank2=pl.col("words_isum").rank("ordinal", descending=True).over("oid").cast(pl.UInt16),
    )


def build(split, topk, rtopk=3, chunk=1_000_000, tag="", cand_name="cand"):
    t = time.time()
    countries = pl.scan_parquet(f"{D}{split}_s1_norm.parquet").select(pl.col("country").unique()).collect()
    cand_all = pl.scan_parquet(f"{D}{split}_{cand_name}.parquet")
    if "insel" in cand_all.collect_schema().names():
        cand_all = cand_all.filter(pl.col("insel") == 1).drop("insel")
    else:
        cand_all = cand_all.filter((pl.col("r1") <= topk) | (pl.col("r2") <= rtopk))
    for c in sorted(countries["country"].to_list()):
        s1, o = load_records(split, c)
        tabs = token_tables(s1, o)
        cand = cand_all.filter(pl.col("rid").is_in(s1["rid"].implode())).collect()
        print(f"[{c}] {cand.height:,} pairs (top{topk}/rev{rtopk}) ({time.time() - t:.0f}s)", flush=True)
        parts = [pair_features(cand.slice(i, chunk), s1, o, tabs) for i in range(0, cand.height, chunk)]
        del s1, o, tabs
        df = context_features(pl.concat(parts)).with_columns(pl.col(pl.Float64).cast(pl.Float32))
        df.write_parquet(f"{D}{split}_feat{tag}_{c}.parquet")
        print(f"[{c}] done {df.shape} ({time.time() - t:.0f}s)", flush=True)
        del df, parts


if __name__ == "__main__":
    build(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10, int(sys.argv[3]) if len(sys.argv) > 3 else 3,
          tag=sys.argv[4] if len(sys.argv) > 4 else "", cand_name=sys.argv[5] if len(sys.argv) > 5 else "cand")
