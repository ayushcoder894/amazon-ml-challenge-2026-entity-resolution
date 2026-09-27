"""US/India name-only records with ONE same-name S1 (k=1) and top probability 0.5-0.9: is there an unassigned addressed record with the
same canonical name (evidence of a missing same-name entity)? Train (truth) vs test (v19). And: gain on train of assigning the
records without such evidence."""
import sys
import polars as pl
import decide
import evaluate as ev
from no_shift import groups, D, DN, OUT, rd


def twins(split, c, pred, R):
    oc = pl.read_parquet(f"{D}p1_canon_{split}_{c}_o.parquet", columns=["idx", "cns", "aempty"]).rename({"idx": "oid"})
    un = oc.filter(~pl.col("aempty")).join(pred.select("oid"), on="oid", how="anti")      # addressed and unassigned
    cnt = un.group_by("cns").agg(twins=pl.len())
    return R.join(oc.select("oid", "cns"), on="oid").join(cnt, on="cns", how="left").with_columns(pl.col("twins").fill_null(0))


c = sys.argv[1]
keep = pl.read_parquet(f"{D}keep_v11.parquet")
s1c = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == c).select("rid").join(keep, on="rid", how="semi")
gtall = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid"])
gt = gtall.join(s1c, on="rid", how="semi")
U = pl.read_parquet(f"{DN}train_joint9.parquet", columns=["rid", "oid", "pj", "country"]).filter(pl.col("country") == c).select("rid", "oid", p="pj")
pred = decide.ef_predict(U.select("rid", "oid", p2="p"), "p2", 0.0, 1.0, 0.05)
tr = groups("train", c, s1c, U, pred).filter(pl.col("kb") == "k=1")
tr = twins("train", c, pred, tr).join(gtall.rename({"rid": "owner"}), on="oid", how="left").with_columns(right=(pl.col("owner") == pl.col("top_rid")).fill_null(False))
s1t = pl.read_parquet(f"{D}p1_canon_test_{c}_s1.parquet", columns=["idx", "entity_id"])
ot = pl.read_parquet(f"{D}p1_canon_test_{c}_o.parquet", columns=["idx", "entity_id"])
fin = pl.read_csv(f"{OUT}matching_results_v19.tsv", **rd).filter(pl.col("matched_entity_ids") != "").with_columns(m=pl.col("matched_entity_ids").str.split(",")).explode("m")
v19 = fin.join(s1t.rename({"entity_id": "source1_entity_id", "idx": "rid"}), on="source1_entity_id").join(ot.rename({"entity_id": "m", "idx": "oid"}), on="m").select("rid", "oid")
T = pl.read_parquet(f"{DN}test_union_p.parquet").join(s1t.select(rid="idx"), on="rid", how="semi").select("rid", "oid", "p")
te = twins("test", c, v19, groups("test", c, s1t.select(rid="idx"), T, v19).filter(pl.col("kb") == "k=1"))
ntr, nte = s1c.height, s1t.height
print(f"{c}: k=1 name-only records; 'twin' = unassigned addressed record with the same canonical name")
print(f"  {'top p':<8s} {'twin':<6s} | {'train /S1':>9s} {'owner=top':>9s} {'assigned':>8s} {'mean p':>6s} | {'test /S1':>8s} {'assigned':>8s} {'mean p':>6s}")
for pb in (">=0.9", "0.5-0.9", "0.2-0.5"):
    for tw in (False, True):
        a = tr.filter((pl.col("pb") == pb) & ((pl.col("twins") > 0) == tw))
        b = te.filter((pl.col("pb") == pb) & ((pl.col("twins") > 0) == tw))
        print(f"  {pb:<8s} {str(tw):<6s} | {a.height / ntr:9.4f} {a['right'].mean():9.1%} {a['ass_rid'].is_not_null().mean():8.1%} {a['top_p'].mean():6.3f} | "
              f"{b.height / nte:8.4f} {b['ass_rid'].is_not_null().mean():8.1%} {b['top_p'].mean():6.3f}")
# train gain: assign every unassigned k=1 record with top p >= 0.5 and NO twin to its top S1
f0 = ev.macro_f05(pred, gt, s1c["rid"])
for lo in (0.5, 0.6, 0.7):
    add = tr.filter((pl.col("top_p") >= lo) & (pl.col("top_p") < 0.9) & (pl.col("twins") == 0) & pl.col("ass_rid").is_null()).select(rid="top_rid", oid="oid")
    f1 = ev.macro_f05(pl.concat([pred, add]), gt, s1c["rid"])
    prec = add.join(gt, on=["rid", "oid"], how="semi").height / max(add.height, 1)
    tadd = te.filter((pl.col("top_p") >= lo) & (pl.col("top_p") < 0.9) & (pl.col("twins") == 0) & pl.col("ass_rid").is_null()).height
    print(f"  assign no-twin records with p in [{lo}, 0.9): train adds {add.height:,} (precision {prec:.1%}) -> {f1:.5f} ({f1 - f0:+.5f}); test would add {tadd:,}")
