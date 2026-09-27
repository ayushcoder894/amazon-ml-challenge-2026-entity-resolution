"""Pairwise features for candidate pairs, computed per country in chunks (CPU, rapidfuzz multi-threaded)."""
import sys
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

D = "C:/ML/Challenges/amazon/work/data/"
REC_COLS = ["rid", "nf", "core", "nums", "words", "at"]
INDIC_RE = "[\u0900-\u0d7f]"


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
            nfc=pl.col("nf").str.replace_all(r"[^a-z0-9 ]", " ").str.replace_all(r"\s+", " ").str.strip_chars(),
            ac=pl.col("at").list.join(" "),
            ws=pl.col("words").list.sort().list.join(" "),
        ).drop("nf", "at")
    return enrich(s1), enrich(o)


def _cp(a, b, scorer, **kw):
    return cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw)


def pair_features(p: pl.DataFrame, s1: pl.DataFrame, o: pl.DataFrame) -> pl.DataFrame:
    """p: (rid, oid, score, sn, sa, sx, r1, r2) -> features."""
    a = s1.select(pl.all().name.suffix("_1")).rename({"rid_1": "rid"})
    b = o.select(pl.all().name.suffix("_2")).rename({"rid_2": "oid"})
    x = p.join(a, on="rid", how="left").join(b, on="oid", how="left")

    f = {}
    n1, n2 = x["nfc_1"].to_list(), x["nfc_2"].to_list()
    c1, c2 = x["cs_1"].to_list(), x["cs_2"].to_list()
    cs1, cs2 = x["css_1"].to_list(), x["css_2"].to_list()
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
    f["a_ratio"] = _cp(a1, a2, fuzz.ratio)
    f["a_tset"] = _cp(a1, a2, fuzz.token_set_ratio)
    f["a_tsort"] = _cp(a1, a2, fuzz.token_sort_ratio)
    f["a_partial"] = _cp(a1, a2, fuzz.partial_ratio)
    f["w_tset"] = _cp(w1, w2, fuzz.token_set_ratio)
    f["w_ratio"] = _cp(w1, w2, fuzz.ratio)
    feats = pl.DataFrame(f)

    x = x.select("rid", "oid", "score", "sn", "sa", "sx", "r1", "r2", "src_2", "indic_2",
                 "core_1", "core_2", "nums_1", "nums_2", "words_1", "words_2", "cs_1", "cs_2", "ac_2")
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
        cl1=pl.col("cs_1").str.len_chars(), cl2=pl.col("cs_2").str.len_chars(),
        addr_empty2=(pl.col("ac_2").str.len_chars() == 0).cast(pl.UInt8),
        # longest shared number (long numbers are far more identifying than "1")
        num_maxlen=pl.col("nums_1").list.set_intersection("nums_2").list.eval(
            pl.element().str.len_chars()).list.max().fill_null(0),
    ).with_columns(
        c_jac=pl.col("c_inter") / (pl.col("c_len1") + pl.col("c_len2") - pl.col("c_inter")).clip(1),
        c_cont=pl.col("c_inter") / pl.min_horizontal("c_len1", "c_len2").clip(1),
        n_jac=pl.col("n_inter") / (pl.col("n_len1") + pl.col("n_len2") - pl.col("n_inter")).clip(1),
        n_cont1=pl.col("n_inter") / pl.col("n_len1").clip(1),
        w_jac=pl.col("w_inter") / (pl.col("w_len1") + pl.col("w_len2") - pl.col("w_inter")).clip(1),
        w_cont=pl.col("w_inter") / pl.min_horizontal("w_len1", "w_len2").clip(1),
    ).drop("core_1", "core_2", "nums_1", "nums_2", "words_1", "words_2", "cs_1", "cs_2", "ac_2")
    return pl.concat([x, feats], how="horizontal").rename({"src_2": "src", "indic_2": "indic"})


def context_features(df: pl.DataFrame) -> pl.DataFrame:
    """Rank/relative features within each S1's and each S2/S3 record's candidate list (needs full candidate set)."""
    return df.with_columns(
        rel1=pl.col("score") / pl.col("score").max().over("rid"),
        gap1=pl.col("score").max().over("rid") - pl.col("score"),
        rel2=pl.col("score") / pl.col("score").max().over("oid"),
        gap2=pl.col("score").max().over("oid") - pl.col("score"),
        ncand1=pl.len().over("rid").cast(pl.UInt16),
        ncand2=pl.len().over("oid").cast(pl.UInt16),
        # rank of this pair among the S1's candidates from the same source
        r1src=pl.col("score").rank("ordinal", descending=True).over("rid", "src").cast(pl.UInt16),
    )


def build(split, topk, rtopk=3, chunk=1_000_000):
    t = time.time()
    countries = pl.scan_parquet(f"{D}{split}_s1_norm.parquet").select(pl.col("country").unique()).collect()
    cand_all = pl.scan_parquet(f"{D}{split}_cand.parquet").filter((pl.col("r1") <= topk) | (pl.col("r2") <= rtopk))
    for c in sorted(countries["country"].to_list()):
        s1, o = load_records(split, c)
        cand = cand_all.filter(pl.col("rid").is_in(s1["rid"].implode())).collect()
        print(f"[{c}] {cand.height:,} pairs (top{topk}/rev{rtopk}) ({time.time() - t:.0f}s)", flush=True)
        parts = [pair_features(cand.slice(i, chunk), s1, o) for i in range(0, cand.height, chunk)]
        del s1, o
        df = context_features(pl.concat(parts)).with_columns(pl.col(pl.Float64).cast(pl.Float32))
        df.write_parquet(f"{D}{split}_feat_{c}.parquet")
        print(f"[{c}] done {df.shape} ({time.time() - t:.0f}s)", flush=True)
        del df, parts


if __name__ == "__main__":
    build(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10)
