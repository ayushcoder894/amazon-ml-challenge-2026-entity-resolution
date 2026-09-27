"""Normalise every source file once: folded text + name/address token lists. Output: data/<split>_s<k>_norm.parquet"""
import sys
import time

import polars as pl

import norm
import segment

D = "C:/ML/Challenges/amazon/work/data/"
ADDR_STOP = {"no", "flr", "ste", "apt", "unit", "bldg", "po", "box", "null", "nr", "opp", "bh", "the", "of", "de",
             "du", "des", "la", "le", "les", "d", "near", "dist", "district", "tq", "taluk", "post", "via", "plot",
             "h", "house", "room", "rrom", "door", "shop", "office", "off", "main", "bis", "ter",
             "pmb", "cdp", "city", "ity", "ciy", "ciyt", "ciity", "citty"}


def prep(split, k):
    t = time.time()
    df = pl.read_parquet(f"{D}{split}_source{k}.parquet")
    df = df.with_row_index("rid").with_columns(
        nf=norm.fold_series(df["business_name"]), af=norm.fold_series(df["business_address"]))
    df = df.with_columns(nt=norm.name_tokens_expr("nf"), at=norm.addr_tokens_expr("af"))
    # split glued handles / hashtags / domains into known words ("sevenconstructions" -> "seven constructions")
    vocab = list(segment._vocab()[0].keys())
    if vocab:
        # only names that look glued: handle / hashtag / domain markers, or one long word standing for the name
        long_oov = pl.col("nt").list.eval((~pl.element().is_in(vocab)) & (pl.element().str.len_chars() >= 7)
                                          & pl.element().str.contains("^[a-z]+$")).list.any()
        glued = (pl.col("nf").str.contains(r"^\s*[#@]|\.com|www")
                 | (pl.col("nt").list.eval(~pl.element().is_in(["com", "www", "the"])).list.sum() == 1))
        need = df.select(long_oov & glued).to_series()
        idx = need.arg_true()
        if len(idx):
            new = [segment.resegment_tokens(t) for t in df["nt"].gather(idx).to_list()]
            rep = pl.DataFrame({"rid": df["rid"].gather(idx), "nt_new": pl.Series(new, dtype=pl.List(pl.Utf8))})
            df = (df.join(rep, on="rid", how="left", maintain_order="left")
                  .with_columns(nt=pl.coalesce("nt_new", "nt")).drop("nt_new"))
            print(f"  resegmented {len(idx):,} names", flush=True)
    stop = list(norm.NAME_STOP)
    astop = list(ADDR_STOP)
    df = df.with_columns(
        core=pl.col("nt").list.eval(pl.element().filter(~pl.element().is_in(stop))),
        nums=pl.col("at").list.eval(pl.element().filter(pl.element().str.contains(r"^\d+$"))).list.unique(maintain_order=True),
        words=pl.col("at").list.eval(pl.element().filter(
            pl.element().str.contains(r"^[a-z]") & (pl.element().str.len_chars() >= 3) & ~pl.element().is_in(astop))),
    )
    df.select("rid", "entity_id", "country", "nf", "af", "nt", "at", "core", "nums", "words").write_parquet(
        f"{D}{split}_s{k}_norm.parquet")
    print(split, k, df.height, f"{time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    for split in sys.argv[1:] or ["train", "test"]:
        for k in (1, 2, 3):
            prep(split, k)
