"""User's rule on TRAIN (harder split): a name-only record (empty address) whose canonical name equals a Source-1
entity's canonical name is matched to it.
  A  name equals exactly ONE entity             -> assign to it (overrides the model for that record)
  B  name equals SEVERAL entities               -> assign to ALL of them
  C  name equals SEVERAL entities               -> assign to the one with the highest model probability
Macro F0.5 vs the current model (joint9 + v13 rule) on the kept entities.  python name_only_rule.py India"""
import sys

import polars as pl

import decide
import evaluate as ev

D = "C:/ML/Challenges/amazon/work/data/"
DN = "D:/amazon_er/dense/"
c = sys.argv[1] if len(sys.argv) > 1 else "India"
keep = pl.read_parquet(f"{D}keep_v11.parquet")
s1c = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == c).select("rid").join(keep, on="rid", how="semi")
gt = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid"]).join(s1c, on="rid", how="semi")
s1 = pl.read_parquet(f"{D}p1_canon_train_{c}_s1.parquet", columns=["idx", "cns"]).rename({"idx": "rid"}).join(s1c, on="rid", how="semi").filter(pl.col("cns") != "")
o = pl.read_parquet(f"{D}p1_canon_train_{c}_o.parquet", columns=["idx", "cns", "aempty"]).filter(pl.col("aempty") & (pl.col("cns") != "")).rename({"idx": "oid"}).drop("aempty")
U = pl.read_parquet(f"{DN}train_joint9.parquet", columns=["rid", "oid", "pj", "country"]).filter(pl.col("country") == c).drop("country")
pred = decide.ef_predict(U.select("rid", "oid", pl.col("pj").alias("p2")), "p2", 0.0, 1.0, 0.05)
X = (o.join(s1, on="cns").with_columns(k=pl.len().over("oid"))
     .join(U, on=["rid", "oid"], how="left").with_columns(pl.col("pj").fill_null(0.0))
     .join(gt.with_columns(y=pl.lit(1, pl.Int8)), on=["rid", "oid"], how="left").with_columns(pl.col("y").fill_null(0)))
del U
owner = gt.join(X.select("oid").unique(), on="oid", how="semi").rename({"rid": "owner"})
mp = pred.join(X.select("oid").unique(), on="oid", how="semi").rename({"rid": "mrid"})
R = (X.group_by("oid").agg(k=pl.len(), namesake_true=pl.col("y").max() == 1, top=pl.col("rid").sort_by("pj", descending=True).first())
     .join(owner, on="oid", how="left").join(mp, on="oid", how="left"))
bk = pl.when(pl.col("k") == 1).then(pl.lit("1")).when(pl.col("k") == 2).then(pl.lit("2")).when(pl.col("k") <= 5).then(pl.lit("3-5")) \
    .when(pl.col("k") <= 20).then(pl.lit("6-20")).otherwise(pl.lit("21+"))
R = R.with_columns(kb=bk)
print(f"{c} TRAIN harder split: name-only records whose canonical name equals >= 1 Source-1 entity: {R.height:,}")
print(f"  {'same-name entities':<19s} {'records':>8s} {'owner is one of them':>21s} {'owner nobody':>13s} {'owner other entity':>19s} | "
      f"{'model right':>11s} {'model nobody':>12s} | {'top-prob pick right':>19s} {'B precision':>11s}")
for b in ["1", "2", "3-5", "6-20", "21+"]:
    x = R.filter(pl.col("kb") == b)
    xb = X.join(x.select("oid"), on="oid", how="semi")
    right = ((x["mrid"] == x["owner"]).fill_null(False) | (x["mrid"].is_null() & x["owner"].is_null())).mean()
    print(f"  {b:<19s} {x.height:>8,} {x['namesake_true'].mean():>21.1%} {x['owner'].is_null().mean():>13.1%} "
          f"{(x['owner'].is_not_null() & ~x['namesake_true']).mean():>19.1%} | {right:>11.1%} {x['mrid'].is_null().mean():>12.1%} | "
          f"{(x['top'] == x['owner']).fill_null(False).mean():>19.1%} {xb['y'].mean():>11.1%}")

base = ev.macro_f05(pred, gt, s1c["rid"])


def apply(recs, pairs):
    """Replace the model's decisions for these records by the rule's pairs."""
    return pl.concat([pred.join(recs.select("oid"), on="oid", how="anti"), pairs.select("rid", "oid")]).unique()


one = R.filter(pl.col("k") == 1)
sev = R.filter(pl.col("k") >= 2)
A = X.join(one.select("oid"), on="oid", how="semi")
B = X.join(sev.select("oid"), on="oid", how="semi")
C = sev.select(pl.col("top").alias("rid"), "oid")
print(f"\n{c} macro F0.5 on the harder split (current model {base:.5f}):")
for lab, recs, pairs in (("A  unique name -> that entity", one, A), ("B  shared name -> ALL namesakes", sev, B),
                         ("C  shared name -> most likely namesake", sev, C), ("A + C", R, pl.concat([A.select("rid", "oid"), C]))):
    p = apply(recs, pairs)
    prec = pairs.join(gt, on=["rid", "oid"], how="semi").height / pairs.height
    print(f"  {lab:<40s} records {recs.height:>7,}  rule pairs {pairs.height:>9,} (precision {prec:.1%}) -> {ev.macro_f05(p, gt, s1c['rid']):.5f} "
          f"({ev.macro_f05(p, gt, s1c['rid']) - base:+.5f})")

# examples like "Indo Builders Private": name-only records with 20+ same-name entities
raw1 = pl.read_parquet(f"{D}train_source1.parquet", columns=["business_name", "business_address"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
raw2 = pl.concat([pl.read_parquet(f"{D}train_source{k}.parquet", columns=["business_name"]) for k in (2, 3)]).with_row_index("oid").with_columns(pl.col("oid").cast(pl.UInt32))
ex = R.filter(pl.col("k") >= 20).sample(8, seed=4).join(raw2, on="oid").join(raw1.rename({"rid": "owner", "business_name": "owner_name", "business_address": "owner_address"}), on="owner", how="left")
print(f"\nExamples ({c} train, name-only records with 20+ same-name entities):")
for r in ex.iter_rows(named=True):
    pown = X.filter((pl.col("oid") == r["oid"]) & (pl.col("rid") == r["owner"]))["pj"] if r["owner"] is not None else None
    pmax = X.filter(pl.col("oid") == r["oid"])["pj"].max()
    who = (f"owner = {r['owner_name']} | {r['owner_address']} (prob {pown[0] if pown is not None and pown.len() else 0:.3f}, best namesake prob {pmax:.3f})"
           if r["owner"] is not None else "owner = nobody")
    print(f"  '{r['business_name']}' - {r['k']} same-name entities; {who}; model assigns {'nobody' if r['mrid'] is None else ('the owner' if r['mrid'] == r['owner'] else 'a wrong entity')}")
