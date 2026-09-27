"""Final combiner over all cross-encoder scores (honest 2-fold), train and test.

Band = pairs with stage-2 probability p2 in (LO, HI). Features: p1, p2 and, for every available CE score column
(ce = e5-small round 1, cebase = e5-base round 1, ce2 = e5-base round 2 on the hard band; missing -> NaN), the
score, its logit and its competition features (rank / max / margin among the S1s competing for the same record,
rank / max within the S1). Combiner f trains on fold-f band pairs and predicts fold 1-f; test = mean of both.

  python ce_multi.py train <tag> <ce cols...>          -> honest score, models, OOF (train_oof_multi<tag>)
  python ce_multi.py test  <tag> <mname> <name> <ce cols...>  -> output/matching_results_<name>.tsv
"""
import json
import os
import subprocess
import sys

import numpy as np
import polars as pl
import xgboost as xgb

import decide
from ce_combine import CP, logit

D = "C:/ML/Challenges/amazon/work/data/"
M = "C:/ML/Challenges/amazon/work/models_ce/"
OUT = "C:/ML/Challenges/amazon/work/output/"
RES = "C:/ML/Challenges/amazon/6ab10eb3b23ba_student_resource/student_resource/"
LO, HI = 0.02, float(os.environ.get("MULTI_HI", 0.98))


def KEYSFX():
    return (("_twin" if os.environ.get("USE_TWIN") == "1" else "") + ("_ns" if os.environ.get("USE_NS") == "1" else "")
            + ("_hn" if os.environ.get("USE_HN") == "1" else "") + (f"_hi{HI}" if HI != 0.98 else ""))


def feats(K, cols):
    b = K.filter((pl.col("p2") > LO) & (pl.col("p2") < HI))
    ex, names = [], ["p1", "p2", "lg2"]
    b = b.with_columns(lg2=logit("p2"),
                       p2_rank2=pl.col("p2").rank("ordinal", descending=True).over("oid").cast(pl.UInt16),
                       nb1=pl.len().over("rid").cast(pl.UInt16), nb2=pl.len().over("oid").cast(pl.UInt16))
    names += ["p2_rank2", "nb1", "nb2"]
    for c in cols:
        v = pl.col(c)
        b = b.with_columns(**{
            f"lg_{c}": logit(c),
            f"{c}_rank2": v.rank("ordinal", descending=True).over("oid").cast(pl.Float32),
            f"{c}_max2": v.max().over("oid"),
            f"{c}_rank1": v.rank("ordinal", descending=True).over("rid").cast(pl.Float32),
            f"{c}_max1": v.max().over("rid"),
            f"{c}_second2": v.sort(descending=True, nulls_last=True).slice(1, 1).first().over("oid"),
        }).with_columns(**{f"{c}_margin2": pl.when(pl.col(f"{c}_rank2") == 1).then(v - pl.col(f"{c}_second2"))
                           .otherwise(v - pl.col(f"{c}_max2"))})
        names += [c, f"lg_{c}", f"{c}_rank2", f"{c}_max2", f"{c}_rank1", f"{c}_max1", f"{c}_margin2"]
    if os.environ.get("USE_TWIN") == "1":
        names += TWIN
    if os.environ.get("USE_NS") == "1":
        from namesake_feats import NS_COLS
        names += NS_COLS
    if os.environ.get("USE_HN") == "1":
        from num_feats import HN_COLS
        names += HN_COLS
    return b, names


TWIN = ["tw_best_sim", "tw_best_p", "tw_n90", "tw_p90_mean", "tw_sim_hi", "tw_sim_lo", "tw_nlo_sim"]


def load(split, tag, cols, mname=None):
    src = f"{D}train_oof{tag}.parquet" if split == "train" else f"{D}test_pred{mname}.parquet"
    K = pl.read_parquet(src)
    for c in cols:
        K = K.join(pl.read_parquet(f"{D}{c}_{split}{tag}.parquet"), on=["rid", "oid"], how="left")
    if os.environ.get("USE_TWIN") == "1":
        K = K.join(pl.read_parquet(f"{D}twin_{split}{tag}.parquet"), on=["rid", "oid"], how="left")
    if os.environ.get("USE_NS") == "1":
        K = K.join(pl.read_parquet(f"{D}ns_{split}{tag}.parquet"), on=["rid", "oid"], how="left")
    if os.environ.get("USE_HN") == "1":
        K = K.join(pl.read_parquet(f"{D}hn_{split}{tag}.parquet"), on=["rid", "oid"], how="left")
    return K


def train(tag, cols):
    from train2 import per_country, report
    K = load("train", tag, cols)
    b, names = feats(K, cols)
    preds = []
    for f in (0, 1):
        tr, te = b.filter(pl.col("fold") == f), b.filter(pl.col("fold") != f)
        m = xgb.train(dict(CP, max_depth=7), xgb.DMatrix(tr.select(names).to_numpy().astype(np.float32),
                                                         tr["y"].to_numpy()), 500)
        m.save_model(f"{M}multi{tag}_{'_'.join(cols)}{KEYSFX()}_fold{f}.json")
        preds.append(te.select("rid", "oid").with_columns(
            pm=pl.Series(m.predict(xgb.DMatrix(te.select(names).to_numpy().astype(np.float32))))))
    K = K.join(pl.concat(preds), on=["rid", "oid"], how="left").with_columns(pf=pl.coalesce("pm", "p2"))
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").filter(pl.col("country").is_in(["India", "US"])).select("rid", "oid")
    s1_all = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(
        pl.col("country").is_in(["India", "US"]))
    if os.environ.get("KEEP"):  # test-like condition (make_drop_folds.py): only these S1 entities exist
        kr = pl.read_parquet(f"{D}keep{os.environ['KEEP']}.parquet")
        gt, s1_all = gt.join(kr, on="rid", how="semi"), s1_all.join(kr, on="rid", how="semi")
    score, rule = report(K, gt, s1_all, "pf")
    K.select("rid", "oid", "y", "fold", "country", "p1", "p2", "pf").write_parquet(f"{D}train_oof_final{tag}{KEYSFX()}.parquet")
    per_country(K, gt, s1_all, "pf", rule)
    json.dump(dict(score=score, rule=list(rule), cols=cols, feats=names),
              open(f"{M}multi_meta{tag}_{'_'.join(cols)}{KEYSFX()}.json", "w"), indent=1)


def test(tag, mname, name, cols):
    key = f"{tag}_{'_'.join(cols)}{KEYSFX()}"
    meta = json.load(open(f"{M}multi_meta{key}.json"))
    K = load("test", tag, cols, mname)
    b, names = feats(K, cols)
    X = xgb.DMatrix(b.select(names).to_numpy().astype(np.float32))
    pm = np.mean([xgb.Booster(model_file=f"{M}multi{key}_fold{f}.json").predict(X) for f in (0, 1)], axis=0)
    K = K.join(b.select("rid", "oid").with_columns(pm=pl.Series(pm)), on=["rid", "oid"], how="left").with_columns(
        pf=pl.coalesce("pm", "p2"))
    rule = meta["rule"]
    K.select("rid", "oid", "country", "p1", "p2", "pf").write_parquet(f"{D}test_final{key}.parquet")
    oo = K.select("rid", "oid", pl.col("pf").alias("p2"))
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
    print(f"wrote {mp}: {best.height:,} matched pairs, {K.height / s1.height:.2f} candidates per S1")
    subprocess.run([sys.executable, f"{RES}utils/validate_submission.py", "--matching", mp, "--candidate", cpth,
                    "--test-dir", f"{RES}dataset/test"])


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(sys.argv[2], sys.argv[3:])
    else:
        test(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
