"""Cross-encoder scores for the uncertain dense new pairs (add-on probability pn in (0.02, 0.98)).
Reuses v12's pair cross-encoders (e5-small, models ce_v10_fold{f}): train pairs of fold-f entities are scored by the
model of the other fold (out-of-fold, same fold split as v12); test pairs by the average of both.
Output: DN/ce_<split>_new.parquet (rid, oid, ce_new).
"""
import sys

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import ce
from dense_addon import DN, log

import os

from dense_addon import D

CTX = os.environ.get("CE_CTX") == "1"
CM = "D:/amazon_er/models_ce/" + ("cectx" if CTX else "ce") + "_v10_fold{f}"
NAME = "cectx_new" if CTX else "ce_new"
LO, HI = 0.02, 0.98


def context(b, split, o, n=3):
    """Listwise context (as v12's cectx): the S1's n most likely OTHER candidates over the union (v12 pf + new pn),
    appended to the record text."""
    f = "train_oof_final_v12_twin" if split == "train" else "test_final_v12_ce_cectx_twin"
    U = pl.concat([pl.read_parquet(f"{D}{f}.parquet", columns=["rid", "oid", "pf"]).rename({"pf": "p"}),
                   pl.read_parquet(f"{DN}{split}_pn.parquet", columns=["rid", "oid", "pn"]).rename({"pn": "p"})])
    U = U.join(b.select("rid").unique(), on="rid", how="semi").join(o, on="oid")
    c = (U.sort(["rid", "p"], descending=[False, True]).group_by("rid", maintain_order=True)
         .agg(c_oid=pl.col("oid").head(n + 1), c_t=pl.col("t2").head(n + 1)))
    b = b.join(c, on="rid", how="left").with_row_index("_i")
    x = (b.select("_i", "oid", "c_oid", "c_t").explode("c_oid", "c_t").filter(pl.col("c_oid") != pl.col("oid"))
         .group_by("_i").agg(ctx=pl.col("c_t").head(n).str.join(" ; ")))
    return (b.join(x, on="_i", how="left").sort("_i")
            .with_columns(t2=pl.col("t2") + pl.lit(" || ") + pl.col("ctx").fill_null("")).drop("_i", "c_oid", "c_t", "ctx"))


def main(split):
    src = f"{DN}{split}_pn.parquet"
    b = pl.read_parquet(src, columns=["rid", "oid", "pn"]).filter((pl.col("pn") > LO) & (pl.col("pn") < HI))
    b = b.with_columns(fold=(pl.col("rid").hash(seed=11) % 2).cast(pl.Int8))
    s1, o = ce.texts(split)
    b = b.join(s1, on="rid").join(o, on="oid")
    if CTX:
        b = context(b, split, o)
    log(f"{split}: {b.height:,} uncertain new pairs to score")
    out = np.zeros(b.height, dtype=np.float32)
    for f in (0, 1):
        tok = AutoTokenizer.from_pretrained(CM.format(f=f))
        model = AutoModelForSequenceClassification.from_pretrained(CM.format(f=f)).to(ce.DEV)
        if split == "train":  # model f was trained on fold-f entities -> scores the other fold
            m = (b["fold"] != f).to_numpy()
            out[m] = ce.predict(model, tok, b.filter(pl.Series(m)))
        else:
            out += ce.predict(model, tok, b) / 2
        log(f"  model {f} done")
        del model
        torch.cuda.empty_cache()
    b.select("rid", "oid").with_columns(pl.Series(NAME, out)).write_parquet(f"{DN}{NAME.replace('_new', '')}_{split}_new.parquet")
    log(f"wrote {DN}{NAME.replace('_new', '')}_{split}_new.parquet")


if __name__ == "__main__":
    main(sys.argv[1])
