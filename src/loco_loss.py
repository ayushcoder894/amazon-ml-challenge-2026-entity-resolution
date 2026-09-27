"""Loss breakdown of the France recipe on the unseen-country proxy (India from a US-trained model and the reverse,
true labels): decisions = 50/50 blend of stage-1/dense probability and the joint pass, rule ef(0, 4).
Error pairs (FP / FN) get their generator category and, for misses, whether the record went to another entity
(rival), is free (rejected everywhere) or was never a candidate. Loss of an entity = (1 - F0.5) / #entities,
shared equally over its error pairs.
"""
import polars as pl

import decide
from france_cats import canon
from p1_catalog import addr_rel, name_rel
from p1_decoy2 import refine
from p2_france_diag import addr_cls

D = "C:/ML/Challenges/amazon/work/data/"


def main():
    U = pl.read_parquet("D:/amazon_er/france_night/data/loco_joint_U.parquet").with_columns(pb=0.5 * pl.col("p") + 0.5 * pl.col("pj"))
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    s1c = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).join(keep, on="rid", how="semi")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").join(keep, on="rid", how="semi").select("rid", "oid")
    pred = decide.ef_predict(U.select("rid", "oid", pl.col("pb").alias("p2")), "p2", 0.0, 4.0, 0.05).select("rid", "oid")
    taken = pred.select("oid", owner="rid")
    tp = pred.join(gt, on=["rid", "oid"]).group_by("rid").agg(tp=pl.len())
    m = (s1c.join(gt.group_by("rid").agg(nt=pl.len()), on="rid", how="left").join(pred.group_by("rid").agg(np_=pl.len()), on="rid", how="left")
         .join(tp, on="rid", how="left").fill_null(0))
    p = pl.when(pl.col("np_") > 0).then(pl.col("tp") / pl.col("np_")).otherwise(0.0)
    r = pl.when(pl.col("nt") > 0).then(pl.col("tp") / pl.col("nt")).otherwise(0.0)
    m = m.with_columns(f=pl.when((pl.col("nt") == 0) & (pl.col("np_") == 0)).then(1.0).when(pl.col("tp") == 0).then(0.0)
                       .otherwise(1.25 * p * r / (0.25 * p + r)))
    m = m.with_columns(loss=(1 - pl.col("f")) / pl.len().over("country"))
    print(m.group_by("country").agg(F=pl.col("f").mean()).sort("country"))
    fp = pred.join(gt, on=["rid", "oid"], how="anti").with_columns(t=pl.lit("FP"))
    fn = (gt.join(pred, on=["rid", "oid"], how="anti").join(U.select("rid", "oid", "pb"), on=["rid", "oid"], how="left")
          .join(taken, on="oid", how="left")
          .with_columns(t=pl.when(pl.col("pb").is_null()).then(pl.lit("FN not cand"))
                        .when(pl.col("owner").is_not_null()).then(pl.lit("FN rival took it")).otherwise(pl.lit("FN free"))))
    E = pl.concat([fp, fn.select("rid", "oid", "t")]).join(m.select("rid", "country", "loss"), on="rid")
    E = E.with_columns(share=pl.col("loss") / pl.len().over("rid"))
    out = []
    for c in ("India", "US"):
        s1, o, vocab = canon("train", c)
        x = (E.filter(pl.col("country") == c)
             .join(s1.select(rid="idx", cn1="cn", cns1="cns", cak1="cak", na="na"), on="rid")
             .join(o.select(oid="idx", cn2="cn", cns2="cns", cak2="cak", ae2="aempty"), on="oid"))
        names, addrs = [], []
        for q in x.select("cn1", "cn2", "cns1", "cns2", "cak1", "cak2", "ae2").iter_rows():
            A, B = list(q[0]), list(q[1])
            nr = refine(name_rel(A, B, q[2], q[3]), A, B, vocab)
            names.append(nr)
            addrs.append(addr_cls(addr_rel(q[4].split(" ") if q[4] else [], q[5].split(" ") if q[5] else [], q[6])))
        out.append(x.select("rid", "oid", "country", "t", "share").with_columns(name=pl.Series(names), addr=pl.Series(addrs)))
    E = pl.concat(out)
    E.write_parquet("D:/amazon_er/france_night/data/loco_errors.parquet")
    for c in ("India", "US"):
        x = E.filter(pl.col("country") == c)
        print(f"\n=== {c}: loss {x['share'].sum():.5f}")
        print(x.group_by("t").agg(pairs=pl.len(), loss=pl.col("share").sum().round(5)).sort("loss", descending=True))
        g = x.group_by("t", "name", "addr").agg(pairs=pl.len(), loss=pl.col("share").sum()).sort("loss", descending=True).head(18)
        for q in g.iter_rows(named=True):
            print(f"  {q['t']:<17s} {q['name'][:26]:<27s} {q['addr'][:30]:<31s} {q['pairs']:>7,} {q['loss']:.5f}")


if __name__ == "__main__":
    main()
