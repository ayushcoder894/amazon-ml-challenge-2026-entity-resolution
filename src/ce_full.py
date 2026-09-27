"""Richer combiner on the uncertain band: all stage-1/2 features + sibling + stage-2 context + cross-encoder features.

Honest: combiner f trains on fold-f band pairs (CE scores are out-of-fold) and predicts fold 1-f.
  python ce_full.py train <tag>           -> honest score, models, meta
  python ce_full.py test  <tag> <name>    -> output/matching_results_<name>.tsv
"""
import json
import subprocess
import sys

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import xgboost as xgb

import decide
from ce_combine import FEATS as CFEATS
from ce_combine import ce_features
from train import stage2_features
from train3 import EXTRA, NON_FEAT

D = "C:/ML/Challenges/amazon/work/data/"
M = "C:/ML/Challenges/amazon/work/models_ce/"
OUT = "C:/ML/Challenges/amazon/work/output/"
RES = "C:/ML/Challenges/amazon/6ab10eb3b23ba_student_resource/student_resource/"
LO, HI = 0.02, 0.98
P = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cpu", nthread=18,
         max_depth=8, learning_rate=0.04, subsample=0.8, colsample_bytree=0.6, min_child_weight=20)


def band_table(split, tag, K, feat_paths):
    """K: all cascade pairs (rid, oid, src, p1, p2, [y, fold]) -> band rows with every feature."""
    from siblings import SIB_COLS
    K = stage2_features(K)
    sib = pl.read_parquet(f"{D}sib_{split}{tag}.parquet")
    K = K.join(sib, on=["rid", "oid"], how="left").with_columns(pl.col(SIB_COLS).fill_null(-1))
    ce = pl.read_parquet(f"{D}ce_{split}{tag}.parquet")
    K = K.join(ce, on=["rid", "oid"], how="left")
    b = ce_features(K)                                   # band rows (ce not null) + CE competition features
    keys = b.select("rid", "oid")
    fcols = [c for c in pq.read_schema(feat_paths[0]).names if c not in NON_FEAT and c not in ("src",)]
    parts = []
    for p in feat_paths:
        parts.append(pl.scan_parquet(p).select(["rid", "oid"] + fcols).join(keys.lazy(), on=["rid", "oid"], how="semi")
                     .collect())
    F = pl.concat(parts)
    b = b.join(F, on=["rid", "oid"], how="left")
    names = fcols + [c for c in EXTRA if c != "p1"] + SIB_COLS + [c for c in CFEATS if c not in ("p1",)] + ["p1"]
    names = list(dict.fromkeys(n for n in names if n in b.columns))
    return b, names


def train(tag):
    from train2 import per_country, report
    K = pl.read_parquet(f"{D}train_oof{tag}.parquet")
    src = pl.concat([pl.scan_parquet(f"{D}train_feat{tag}_fold{f}.parquet").select("rid", "oid", "src").collect()
                     for f in (0, 1)])
    K = K.join(src, on=["rid", "oid"], how="left")
    b, names = band_table("train", tag, K, [f"{D}train_feat{tag}_fold{f}.parquet" for f in (0, 1)])
    print(f"band rows {b.height:,}, {len(names)} features", flush=True)
    preds = []
    for f in (0, 1):
        tr, te = b.filter(pl.col("fold") == f), b.filter(pl.col("fold") != f)
        m = xgb.train(P, xgb.DMatrix(tr.select(names).to_numpy().astype(np.float32), tr["y"].to_numpy()), 700)
        m.save_model(f"{M}full{tag}_fold{f}.json")
        preds.append(te.select("rid", "oid").with_columns(
            pf=pl.Series(m.predict(xgb.DMatrix(te.select(names).to_numpy().astype(np.float32))))))
    K = K.join(pl.concat(preds), on=["rid", "oid"], how="left").with_columns(p3=pl.coalesce("pf", "p2"))
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").filter(pl.col("country").is_in(["India", "US"])).select("rid", "oid")
    s1_all = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(
        pl.col("country").is_in(["India", "US"]))
    s, rule = report(K, gt, s1_all, "p3")
    per_country(K, gt, s1_all, "p3", rule)
    json.dump(dict(score=s, rule=list(rule), feats=names), open(f"{M}full_meta{tag}.json", "w"), indent=1)


def test(tag, name):
    meta = json.load(open(f"{M}full_meta{tag}.json"))
    K = pl.read_parquet(f"{D}test_pred{tag}.parquet")
    K = K.join(pl.concat([pl.scan_parquet(f"{D}test_feat{tag}_{c}.parquet").select("rid", "oid", "src").collect()
                          for c in K["country"].unique().sort().to_list()]), on=["rid", "oid"], how="left")
    paths = [f"{D}test_feat{tag}_{c}.parquet" for c in K["country"].unique().sort().to_list()]
    b, names = band_table("test", tag, K, paths)
    names = meta["feats"]
    X = xgb.DMatrix(b.select(names).to_numpy().astype(np.float32))
    pf = np.mean([xgb.Booster(model_file=f"{M}full{tag}_fold{f}.json").predict(X) for f in (0, 1)], axis=0)
    K = K.join(b.select("rid", "oid").with_columns(pf=pl.Series(pf)), on=["rid", "oid"], how="left").with_columns(
        p3=pl.coalesce("pf", "p2"))
    rule = meta["rule"]
    oo = K.select("rid", "oid", pl.col("p3").alias("p2"))
    best = (oo.filter((pl.col("p2").rank("ordinal", descending=True).over("oid") == 1) & (pl.col("p2") >= rule[1]))
            if rule[0] == "t" else decide.ef_predict(oo, "p2", rule[1], rule[2], 0.05))
    s1 = pl.read_parquet(f"{D}test_s1_norm.parquet", columns=["rid", "entity_id"])
    o = pl.concat([pl.read_parquet(f"{D}test_s{k}_norm.parquet", columns=["entity_id"]) for k in (2, 3)]
                  ).with_row_index("oid")
    ids = lambda x: x.join(s1.rename({"entity_id": "s1"}), on="rid").join(o.rename({"entity_id": "m"}), on="oid")
    match = ids(best.select("rid", "oid")).group_by("s1").agg(pl.col("m").sort().str.join(","))
    cand = ids(K.select("rid", "oid")).group_by("s1").agg(pl.col("m").unique().sort().str.join(","))
    base = s1.select(source1_entity_id="entity_id")
    mp, cpth = f"{OUT}matching_results_{name}.tsv", f"{OUT}candidate_pairs_{name}.tsv"
    (base.join(match.rename({"s1": "source1_entity_id", "m": "matched_entity_ids"}), on="source1_entity_id", how="left")
     .fill_null("").write_csv(mp, separator="\t", quote_style="never"))
    (base.join(cand.rename({"s1": "source1_entity_id", "m": "candidate_entity_ids"}), on="source1_entity_id", how="left")
     .fill_null("").write_csv(cpth, separator="\t", quote_style="never"))
    print(f"wrote {mp}: {best.height:,} matched pairs")
    subprocess.run([sys.executable, f"{RES}utils/validate_submission.py", "--matching", mp, "--candidate", cpth,
                    "--test-dir", f"{RES}dataset/test"])


if __name__ == "__main__":
    train(sys.argv[2]) if sys.argv[1] == "train" else test(sys.argv[2], sys.argv[3])
