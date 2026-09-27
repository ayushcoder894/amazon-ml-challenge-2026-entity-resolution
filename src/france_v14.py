"""France v14: validated France recipe = 50/50 blend of (stage-1 | dense add-on) probability and a joint pass trained on
the unseen-country (LOCO) unions of India + US, rule ef(0, 4)  [proxy: India +0.0085, US +0.0016 over the v13 recipe],
+ optional label-free rules (pseudo-word / single-prefix truncation / extra word at the entity's own unique address,
record taken by nobody)  [proxy precision 0.95-0.98].

  python france_v14.py joint     -> models/france_joint.json + data/france_union.parquet (rid, oid, p, pj, pb)
  python france_v14.py decide    -> data/france_v14_pairs.parquet (+ rule-added pairs flagged)
"""
import gc
import os
import sys

import numpy as np
import polars as pl
import xgboost as xgb

import decide
from dense_addon import D, DN, JCOLS, JP, log, union_frame

NIGHT = "D:/amazon_er/france_night/"


def loco_union():
    Kl = (pl.read_parquet(f"{D}train_oof_locoh.parquet", columns=["rid", "oid", "country", "p1"])
          .join(pl.read_parquet(f"{DN}dense_train_cos.parquet"), on=["rid", "oid"], how="left").with_columns(pf=pl.col("p1")))
    N = (pl.read_parquet(f"{DN}train_pn_loco.parquet", columns=["rid", "oid", "pn"])
         .join(pl.scan_parquet(f"{DN}dense_train_new_small.parquet").select("rid", "oid", "cos").join(
             pl.read_parquet(f"{DN}train_pn_loco.parquet", columns=["rid", "oid"]).lazy(), on=["rid", "oid"], how="semi").collect(),
               on=["rid", "oid"], how="left"))
    U = union_frame(Kl, N)
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").select("rid", "oid").with_columns(y=pl.lit(1, pl.Int8))
    return U.join(gt, on=["rid", "oid"], how="left").with_columns(pl.col("y").fill_null(0))


def france_union():
    K = (pl.read_parquet(f"{D}test_final_v12_ce_cectx_twin.parquet", columns=["rid", "oid", "country", "p1"])
         .filter(pl.col("country") == "France")
         .join(pl.read_parquet(f"{DN}dense_test_cos.parquet"), on=["rid", "oid"], how="left").with_columns(pf=pl.col("p1")))
    N = (pl.read_parquet(f"{DN}test_pn_france.parquet").select("rid", "oid", "pn")
         .join(pl.read_parquet(f"{DN}test_nfeat_France.parquet", columns=["rid", "oid", "cos"]), on=["rid", "oid"], how="left"))
    return union_frame(K, N)


def joint():
    cols = [c for c in JCOLS if c != "is_india"]
    U = loco_union()
    X = U.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
    y = U["y"].to_numpy()
    del U
    gc.collect()
    bst = xgb.train(JP, xgb.QuantileDMatrix(X, label=y, max_bin=JP["max_bin"]), 400)
    del X
    gc.collect()
    os.makedirs(f"{NIGHT}models", exist_ok=True)
    bst.save_model(f"{NIGHT}models/france_joint.json")
    log(f"France joint trained on the India + US unseen-country unions ({len(y):,} pairs)")
    F = france_union()
    XF = F.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
    pj = bst.predict(xgb.DMatrix(XF))
    F = F.select("rid", "oid", "is_new", "p").with_columns(pj=pl.Series(pj)).with_columns(pb=0.5 * pl.col("p") + 0.5 * pl.col("pj"))
    F.write_parquet(f"{NIGHT}data/france_union.parquet")
    log(f"France union {F.height:,} pairs scored; mean p {F['p'].mean():.4f} pj {F['pj'].mean():.4f}")


def decide_fr(rules=True):
    F = pl.read_parquet(f"{NIGHT}data/france_union.parquet")
    base = decide.ef_predict(F.select("rid", "oid", pl.col("pb").alias("p2")), "p2", 0.0, 4.0, 0.05).select("rid", "oid")
    old = decide.ef_predict(F.select("rid", "oid", pl.col("p").alias("p2")), "p2", 0.0, 4.0, 0.05).select("rid", "oid")
    log(f"France: v13 recipe {old.height:,} pairs; blend recipe {base.height:,} pairs "
        f"(added {base.join(old, on=['rid', 'oid'], how='anti').height:,}, removed {old.join(base, on=['rid', 'oid'], how='anti').height:,})")
    out = base.with_columns(rule=pl.lit(0, pl.Int8))
    if rules:  # R2, validated on top of the blend (proxy India +0.00066 prec 0.947, US +0.00085 prec 0.960)
        pool = pl.read_parquet(f"{NIGHT}data/pool_France_blend.parquet")
        R2 = [("invented_pseudo_word", "canon_equal", False), ("truncation_single_prefix", "canon_equal", False),
              ("extra_word", "canon_equal", False), ("acronym", "canon_equal", None), ("truncation", "canon_equal", False),
              ("truncation_single_prefix", "canon_equal", True)]
        m = pl.lit(False)
        for nm, ad, sh in R2:
            cond = (pl.col("name") == nm) & (pl.col("addr") == ad)
            m = m | (cond if sh is None else cond & (pl.col("sh") == sh))
        cand = (pool.filter(m).join(base.select("oid"), on="oid", how="anti").join(base, on=["rid", "oid"], how="anti")
                .sort("p", descending=True).unique("oid", keep="first").select("rid", "oid", "name"))
        out = pl.concat([out, cand.select("rid", "oid").with_columns(rule=pl.lit(1, pl.Int8))])
        log(f"France rules add {cand.height:,} pairs: " + ", ".join(f"{k} {v:,}" for k, v in cand.group_by("name").len().sort("len", descending=True).iter_rows()))
    out.write_parquet(f"{NIGHT}data/france_v14_pairs.parquet")
    return out


if __name__ == "__main__":
    {"joint": joint, "decide": decide_fr}[sys.argv[1]]()
