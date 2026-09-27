"""Separate name-only model (India/US), trained under the TEST condition: name-only candidate pairs of the harder split
with orphan records (owned by dropped S1s) removed - test has full-train name-only density (no orphans).
Features: joint-9 inputs + main p, second-best p / margin / rank / #candidates of the record, namesake count, exact name,
entity's current accepted count. 2 entity folds (cross-fitted). Validation: macro F0.5 of main decisions + adds of the
best candidate per unassigned name-only record (q >= t) vs the margin rule (p>=0.5 & margin>0.5).
  python nameonly_model.py US [apply]"""
import os
import sys
os.environ.setdefault("JOINT_V", "9")
import numpy as np
import polars as pl
import xgboost as xgb
import dense_addon as da
import decide
import evaluate as ev

D = "C:/ML/Challenges/amazon/work/data/"; DN = "D:/amazon_er/dense/"; OUT = "C:/ML/Challenges/amazon/work/output/"
rd = dict(separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
XP = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda", max_depth=6, learning_rate=0.05,
          subsample=0.8, colsample_bytree=0.8, min_child_weight=20, reg_lambda=2.0, max_bin=128)
EXTRA = ["p2nd", "margin", "rrank", "rn", "ns", "name_eq", "k_acc"]


def build(meta, X, s1, o, acc):
    """meta: rid, oid (+y) aligned with X; restrict to name-only records; add record/entity features."""
    idx = meta.with_row_index("_i").join(o.filter(pl.col("ae")).select("oid", "ocn"), on="oid")
    Z = idx.with_columns(p=pl.Series(X[idx["_i"].to_numpy(), 0]))
    Z = Z.with_columns(p2nd=pl.col("p").sort(descending=True).slice(1, 1).first().over("oid").fill_null(0.0),
                       rrank=pl.col("p").rank("ordinal", descending=True).over("oid").cast(pl.Float32),
                       rn=pl.len().over("oid").cast(pl.Float32)).with_columns(margin=pl.col("p") - pl.col("p2nd"))
    ns = s1.group_by("cnk").len("ns")
    Z = (Z.join(ns, left_on="ocn", right_on="cnk", how="left").with_columns(pl.col("ns").fill_null(0).cast(pl.Float32))
         .join(s1.select("rid", "cnk"), on="rid", how="left").with_columns(name_eq=(pl.col("cnk") == pl.col("ocn")).cast(pl.Float32))
         .join(acc.group_by("rid").len("k_acc"), on="rid", how="left").with_columns(pl.col("k_acc").fill_null(0).cast(pl.Float32)))
    F = np.concatenate([X[Z["_i"].to_numpy()], Z.select(pl.col(EXTRA).cast(pl.Float32)).to_numpy()], 1)
    return Z, F


def main(c, apply=False):
    keep = pl.read_parquet(f"{D}keep_v11.parquet").select(pl.col("rid").cast(pl.UInt32))
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid"]).with_columns(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32))
    orph = gt.join(keep, on="rid", how="anti").select("oid")
    s1 = pl.read_parquet(f"{D}p1_canon_train_{c}_s1.parquet", columns=["idx", "cnk"]).select(pl.col("idx").cast(pl.UInt32).alias("rid"), "cnk").join(keep, on="rid", how="semi")
    o = pl.read_parquet(f"{D}p1_canon_train_{c}_o.parquet", columns=["idx", "cnk", "aempty"]).select(pl.col("idx").cast(pl.UInt32).alias("oid"), ocn="cnk", ae="aempty")
    meta = pl.read_parquet(f"{DN}jalt_meta9.parquet").select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32), "y", "fold", "country")
    X = np.load(f"{DN}jalt_X9.npy", mmap_mode="r")
    m = (meta["country"] == c).to_numpy()
    meta_c = meta.filter(pl.Series(m))
    Xc = np.asarray(X[np.flatnonzero(m)])
    pj = meta_c.select("rid", "oid").join(pl.read_parquet(f"{DN}train_joint9.parquet", columns=["rid", "oid", "pj"]).select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32), "pj"), on=["rid", "oid"], how="left")["pj"].to_numpy()
    Xc[:, 0] = pj                      # column 0 = main OOF probability (as used for decisions)
    ok = meta_c.with_row_index("_r").join(s1.select("rid"), on="rid", how="semi").join(orph, on="oid", how="anti")
    meta_c, Xc = ok.drop("_r"), Xc[ok["_r"].to_numpy()]
    pred = decide.ef_predict(meta_c.select("rid", "oid").with_columns(p2=pl.Series(Xc[:, 0])), "p2", 0.0, 1.0, 0.05).select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32))
    Z, F = build(meta_c, Xc, s1, o, pred)
    y, fv = Z["y"].to_numpy(), Z["fold"].to_numpy()
    q = np.zeros(len(y), np.float32)
    for f in (0, 1):
        tr, te = np.flatnonzero(fv == f), np.flatnonzero(fv != f)
        bst = xgb.train(XP, xgb.QuantileDMatrix(F[tr], label=y[tr], max_bin=128), 300)
        q[te] = bst.predict(xgb.DMatrix(F[te]))
    Z = Z.with_columns(q=pl.Series(q))
    print(f"== {c}: name-only candidate pairs {Z.height:,} (orphans removed); logloss main p {-np.mean(y * np.log(np.clip(Z['p'].to_numpy(), 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - Z['p'].to_numpy(), 1e-6, 1))):.4f} vs name-only model {-np.mean(y * np.log(np.clip(q, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - q, 1e-6, 1))):.4f}")
    g = gt.join(s1.select("rid"), on="rid", how="semi")
    rids = s1["rid"]
    base = ev.macro_f05(pred, g, rids)
    taken = pred.select("oid").unique()
    U = Z.join(pred, on=["rid", "oid"], how="anti").join(taken, on="oid", how="anti")
    print(f"   base F0.5 {base:.5f}")
    bm = U.sort("p", descending=True).unique("oid", keep="first").filter((pl.col("p") >= 0.5) & (pl.col("margin") > 0.5)).select("rid", "oid")
    fm = ev.macro_f05(pl.concat([pred, bm]), g, rids)
    print(f"   margin rule (p>=.5 & margin>.5): +{bm.height:,} pairs -> {fm:.5f} ({fm - base:+.5f})")
    bq = U.sort("q", descending=True).unique("oid", keep="first")
    for t in (0.6, 0.7, 0.75, 0.8, 0.85, 0.9):
        a = bq.filter(pl.col("q") >= t).select("rid", "oid")
        fq = ev.macro_f05(pl.concat([pred, a]), g, rids)
        print(f"   name-only model q>={t}: +{a.height:,} pairs (precision {bq.filter(pl.col('q') >= t)['y'].mean():.3f}) -> {fq:.5f} ({fq - base:+.5f})")
    # removal of accepted name-only pairs with low q
    Acc = Z.join(pred, on=["rid", "oid"], how="semi")
    for t in (0.2, 0.3, 0.4):
        r = Acc.filter(pl.col("q") < t).select("rid", "oid")
        fr = ev.macro_f05(pred.join(r, on=["rid", "oid"], how="anti"), g, rids)
        print(f"   remove accepted name-only q<{t}: -{r.height:,} (precision {Acc.filter(pl.col('q') < t)['y'].mean() if r.height else 0:.3f}) -> {fr:.5f} ({fr - base:+.5f})")
    # integrated: replace name-only pair probabilities with q, rerun one-owner + expected-F0.5 on the whole set
    P = meta_c.select("rid", "oid").with_columns(p2=pl.Series(Xc[:, 0]))
    P = P.join(Z.select("rid", "oid", "q"), on=["rid", "oid"], how="left")
    for w in (1.0, 0.5):
        Pi = P.with_columns(p2=pl.when(pl.col("q").is_not_null()).then(w * pl.col("q") + (1 - w) * pl.col("p2")).otherwise(pl.col("p2")))
        pi = decide.ef_predict(Pi.select("rid", "oid", "p2"), "p2", 0.0, 1.0, 0.05).select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32))
        fi = ev.macro_f05(pi, g, rids)
        print(f"   INTEGRATED (name-only p := {w}*q + {1 - w:.1f}*p, rerun ef): pairs {pi.height - pred.height:+,} -> {fi:.5f} ({fi - base:+.5f})")
    if apply:
        full = xgb.train(XP, xgb.QuantileDMatrix(F, label=y, max_bin=128), 300)
        full.save_model(f"{DN}models/nameonly_{c}.json")
        print("   saved full model")


if __name__ == "__main__":
    main(sys.argv[1], len(sys.argv) > 2)
