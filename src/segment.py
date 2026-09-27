"""Word segmentation for glued names (handles / hashtags / domains): 'sevenconstructions' -> 'seven constructions'.

Vocabulary = name tokens of all Source-1 records (train + test; no labels involved). A token is re-segmented only
if it is not itself a known word, is long enough, and splits completely into >= 2 known words (unigram-cost DP).
"""
import json
import math
import os

import polars as pl

D = "C:/ML/Challenges/amazon/work/data/"
VOCAB = f"{D}seg_vocab.json"
_V = None


def build_vocab(min_count=3):
    toks = pl.concat([pl.read_parquet(f"{D}{s}_s1_norm.parquet", columns=["nt"]) for s in ("train", "test")])
    c = (toks.select(t=pl.col("nt")).explode("t").drop_nulls().filter(pl.col("t").str.contains(r"^[a-z]{2,}$"))
         .group_by("t").agg(n=pl.len()).filter(pl.col("n") >= min_count))
    v = dict(zip(c["t"].to_list(), c["n"].to_list()))
    json.dump(v, open(VOCAB, "w"))
    print(f"vocabulary: {len(v):,} words")
    return v


def _vocab():
    global _V
    if _V is None:
        v = json.load(open(VOCAB)) if os.path.exists(VOCAB) else {}
        tot = sum(v.values()) or 1
        # only reasonably common words of >= 3 letters may be used as pieces (typos must not shatter)
        cost = {w: math.log(tot / n) for w, n in v.items() if len(w) >= 3 and n >= 10}
        # vocabulary tokens are post-abbreviation ("pvt", "solution"): also accept the long forms
        import norm
        for long, short in norm.ABBR.items():
            if long.isalpha() and len(long) >= 3 and short in cost:
                cost.setdefault(long, cost[short])
        _V = (cost, max((len(w) for w in cost), default=1))
    return _V


def segment(tok: str):
    """Best split of tok into >= 2 known words (min total unigram cost), else None."""
    cost, maxlen = _vocab()
    if tok in cost or len(tok) < 6 or not tok.isalpha():
        return None
    n = len(tok)
    best = [0.0] + [math.inf] * n
    back = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(max(0, i - maxlen), i - 2):  # pieces of length >= 3
            w = tok[j:i]
            c = cost.get(w)
            if c is not None and best[j] + c < best[i]:
                best[i], back[i] = best[j] + c, j
    if best[n] == math.inf:
        return None
    out, i = [], n
    while i > 0:
        out.append(tok[back[i]:i])
        i = back[i]
    out = out[::-1]
    if not (2 <= len(out) <= 4) or sum(map(len, out)) / len(out) < 4:
        return None
    import norm
    return [norm.ABBR.get(w, w) for w in out]


def resegment_tokens(tokens):
    out = []
    for t in tokens:
        s = segment(t)
        out.extend(s if s else [t])
    return out


if __name__ == "__main__":
    build_vocab()
    for t in ["sevenconstructions", "fundfrontier", "andrewsanimal", "womenshealth", "pricesun", "trendzindia",
              "kowalkowskicontracting", "millerfive", "physicaltherapyharbor", "consolidatedvidyalaya", "rumblenextera",
              "constructions", "hospitality"]:
        print(t, "->", segment(t))
