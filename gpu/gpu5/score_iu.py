"""Score India/US uncertain-band pairs with the ALREADY TRAINED Qwen2.5-7B LoRA models from the v5 run (no training).
India pairs <- model trained WITHOUT India (on US); US pairs <- model trained WITHOUT US (on India). The scores are honest
on train (the model never saw that country's labels) and consistent on test -> a new feature for the India/US joint model.
Writes one file per (split, country) as soon as it is done, so a partial run is still usable:
  out/ce6_q7_iu_{train,test}_{India,US}.parquet  (rid, oid, s)
"""
import argparse
import os
import time

import numpy as np
import polars as pl
import torch

from ce5 import Pairs, load_model, predict, texts
from common import log

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", default=f"{HERE}/out")
    ap.add_argument("--data", default=f"{HERE}/data")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--pred_bs", type=int, default=128)
    ap.add_argument("--max_len", type=int, default=128)
    a = ap.parse_args()
    a.lora, a.no_ckpt = True, True
    t_tr, t_te = texts(a.bundle, "train"), texts(a.bundle, "test")
    P = {"train": pl.read_parquet(f"{a.data}/iu_train.parquet"), "test": pl.read_parquet(f"{a.data}/iu_test.parquet")}
    for c in ("India", "US"):
        todo = [s for s in ("train", "test") if not os.path.exists(f"{a.out}/ce6_q7_iu_{s}_{c}.parquet")]
        if not todo:
            log(f"[{c}] already done")
            continue
        mdir = f"{a.out}/models/q7_loco_no{c}"
        if not os.path.exists(f"{mdir}/.done"):
            raise SystemExit(f"model {mdir} not found - was the v5 loco run completed?")
        tok, model = load_model(a, mdir)
        # merge LoRA for faster inference; keep the unmerged model if the merged one disagrees
        probe = P["test"].filter(pl.col("country") == c).head(64)
        pp = Pairs(a, tok, t_te[0], t_te[1], probe)
        before = predict(a, model, pp)
        try:
            merged = model.merge_and_unload()
            after = predict(a, merged, pp)
            if np.abs(before - after).max() < 0.02:
                model = merged
                log(f"[{c}] LoRA merged (max diff {np.abs(before - after).max():.4f})")
            else:
                log(f"[{c}] merged model disagrees (max diff {np.abs(before - after).max():.3f}) - using unmerged")
        except Exception as e:
            log(f"[{c}] merge failed ({e}) - using unmerged")
        for s in todo:
            df = P[s].filter(pl.col("country") == c)
            tx = t_tr if s == "train" else t_te
            t0 = time.time()
            log(f"[{c} {s}] scoring {df.height:,} pairs")
            out = predict(a, model, Pairs(a, tok, tx[0], tx[1], df))
            df.select("rid", "oid").with_columns(s=pl.Series(out)).write_parquet(f"{a.out}/ce6_q7_iu_{s}_{c}.parquet")
            log(f"[{c} {s}] DONE {df.height:,} pairs in {(time.time() - t0) / 60:.0f} min ({df.height / (time.time() - t0):.0f}/s)")
        del model
        torch.cuda.empty_cache()
    log("ALL DONE")


if __name__ == "__main__":
    main()
