"""How many name-only records (empty address, at least one candidate) truly belong to NO Source-1 entity?
TRAIN harder split (18.6% of S1 dropped like test; records of dropped S1 stay as distractors), truth known,
split by ambiguity (name shared by 2+ S1). Compared with what the model does (joint9 + v13 rule) on train and with
what v19 does in the test review CSVs."""
import polars as pl

import decide

D = "C:/ML/Challenges/amazon/work/data/"
DN = "D:/amazon_er/dense/"
CSV = "C:/ML/Challenges/amazon/work/name_only_review/"
keep = pl.read_parquet(f"{D}keep_v11.parquet")
gt = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid"]).join(keep, on="rid", how="semi")
rates = {}
print("TRAIN harder split, name-only records with >=1 candidate")
print(f"  {'':<22s} {'records':>9s} {'owner nobody':>13s} {'owner in cands':>15s} {'owner not in cands':>19s} | {'model: nobody':>13s} {'model right':>11s}")
for c in ("India", "US"):
    s1 = pl.read_parquet(f"{D}p1_canon_train_{c}_s1.parquet", columns=["idx", "cns"]).rename({"idx": "rid"}).join(keep, on="rid", how="semi")
    o = pl.read_parquet(f"{D}p1_canon_train_{c}_o.parquet", columns=["idx", "cns", "aempty"]).filter(pl.col("aempty")).rename({"idx": "oid"})
    sh = s1.filter(pl.col("cns") != "").group_by("cns").agg(nsh=pl.len())
    o = o.join(sh, on="cns", how="left").with_columns(amb=pl.col("nsh").fill_null(0) >= 2).select("oid", "amb")
    U = pl.read_parquet(f"{DN}train_joint9.parquet", columns=["rid", "oid", "pj", "country"]).filter(pl.col("country") == c).drop("country")
    pred = decide.ef_predict(U.select("rid", "oid", pl.col("pj").alias("p2")), "p2", 0.0, 1.0, 0.05)
    Uo = U.join(o, on="oid", how="semi").select("rid", "oid")
    del U
    r = Uo.select("oid").unique().join(o, on="oid")
    own = gt.join(r.select("oid"), on="oid", how="semi")
    r = (r.join(own.rename({"rid": "true_rid"}), on="oid", how="left")
         .join(Uo.join(own, on=["rid", "oid"], how="semi").select("oid", inc=pl.lit(True)), on="oid", how="left")
         .join(pred.join(r.select("oid"), on="oid", how="semi").rename({"rid": "pred_rid"}), on="oid", how="left")
         .with_columns(pl.col("inc").fill_null(False)))
    for lab, x in (("ambiguous name", r.filter(pl.col("amb"))), ("unique name", r.filter(~pl.col("amb"))), ("all", r)):
        nob = x["true_rid"].is_null()
        mn = x["pred_rid"].is_null()
        right = (x["pred_rid"] == x["true_rid"]).fill_null(False) | (mn & nob)
        rates[(c, lab)] = nob.mean()
        print(f"  {c + ' ' + lab:<22s} {x.height:>9,} {nob.mean():>13.1%} {x['inc'].mean():>15.1%} {(~nob & ~x['inc']).mean():>19.1%} | {mn.mean():>13.1%} {right.mean():>11.1%}")

print("\nTEST review CSVs (v19 decisions) and expected 'belongs to nobody' at the train rate")
print(f"  {'':<22s} {'records':>9s} {'v19 nobody':>11s} {'expected nobody':>16s}")
for c in ("India", "US", "France"):
    x = pl.read_csv(f"{CSV}name_only_{c}.csv", columns=["record_id", "name_shared_by", "assigned_in_v19"], infer_schema=False)
    g = x.group_by("record_id").agg(amb=pl.col("name_shared_by").first().cast(pl.Int64) >= 2, ass=(pl.col("assigned_in_v19") == "Y").any())
    for lab, y in (("ambiguous name", g.filter(pl.col("amb"))), ("unique name", g.filter(~pl.col("amb"))), ("all", g)):
        if c == "France":
            rt = (rates[("India", lab)] + rates[("US", lab)]) / 2
            tag = " (India/US avg rate)"
        else:
            rt, tag = rates[(c, lab)], ""
        print(f"  {c + ' ' + lab:<22s} {y.height:>9,} {(~y['ass']).sum():>11,} {rt * y.height:>16,.0f}{tag}")
