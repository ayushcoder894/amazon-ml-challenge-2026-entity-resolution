"""Name-only records: train (harder split, model = joint9 + v13 rule, truth known) vs test (v19), same group definitions.
Groups by k = number of S1 sharing the record's canonical name (0 / 1 / 2-5 / 6+) and by the top candidate's probability."""
import polars as pl
import decide

D = "C:/ML/Challenges/amazon/work/data/"
DN = "D:/amazon_er/dense/"
OUT = "C:/ML/Challenges/amazon/work/output/"
rd = dict(separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)


def groups(split, c, s1_rids, U, pred):
    s1n = pl.read_parquet(f"{D}p1_canon_{split}_{c}_s1.parquet", columns=["idx", "cns"]).rename({"idx": "rid"}).join(s1_rids, on="rid", how="semi")
    sh = s1n.filter(pl.col("cns") != "").group_by("cns").agg(k=pl.len())
    on = (pl.read_parquet(f"{D}p1_canon_{split}_{c}_o.parquet", columns=["idx", "cns", "aempty"]).rename({"idx": "oid"})
          .filter(pl.col("aempty")).join(sh, on="cns", how="left").with_columns(pl.col("k").fill_null(0)).select("oid", "k"))
    Uc = U.join(on, on="oid", how="semi")
    top = Uc.sort("p", descending=True).unique("oid", keep="first").rename({"rid": "top_rid", "p": "top_p"})
    ass = pred.join(on, on="oid", how="semi").rename({"rid": "ass_rid"})
    r = top.join(on, on="oid").join(ass, on="oid", how="left")
    kb = pl.when(pl.col("k") == 0).then(pl.lit("k=0")).when(pl.col("k") == 1).then(pl.lit("k=1")).when(pl.col("k") <= 5).then(pl.lit("k=2-5")).otherwise(pl.lit("k=6+"))
    pb = pl.when(pl.col("top_p") >= 0.9).then(pl.lit(">=0.9")).when(pl.col("top_p") >= 0.5).then(pl.lit("0.5-0.9")).when(pl.col("top_p") >= 0.2).then(pl.lit("0.2-0.5")).otherwise(pl.lit("<0.2"))
    return r.with_columns(kb=kb, pb=pb)


def main():
  for c in ("India", "US"):
      keep = pl.read_parquet(f"{D}keep_v11.parquet")
      s1c = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == c).select("rid").join(keep, on="rid", how="semi")
      gt = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid"]).join(s1c, on="rid", how="semi").rename({"rid": "owner"})
      U = pl.read_parquet(f"{DN}train_joint9.parquet", columns=["rid", "oid", "pj", "country"]).filter(pl.col("country") == c).select("rid", "oid", p="pj")
      pred = decide.ef_predict(U.select("rid", "oid", p2="p"), "p2", 0.0, 1.0, 0.05)
      tr = groups("train", c, s1c, U, pred).join(gt, on="oid", how="left")
      ntr = s1c.height
      del U
      s1t = pl.read_parquet(f"{D}p1_canon_test_{c}_s1.parquet", columns=["idx", "entity_id"])
      ot = pl.read_parquet(f"{D}p1_canon_test_{c}_o.parquet", columns=["idx", "entity_id"])
      fin = (pl.read_csv(f"{OUT}matching_results_v19.tsv", **rd).filter(pl.col("matched_entity_ids") != "")
             .with_columns(m=pl.col("matched_entity_ids").str.split(",")).explode("m"))
      v19 = (fin.join(s1t.rename({"entity_id": "source1_entity_id", "idx": "rid"}), on="source1_entity_id")
             .join(ot.rename({"entity_id": "m", "idx": "oid"}), on="m").select("rid", "oid"))
      T = pl.read_parquet(f"{DN}test_union_p.parquet").join(s1t.select(rid="idx"), on="rid", how="semi").select("rid", "oid", "p")
      te = groups("test", c, s1t.select(rid="idx"), T, v19)
      nte = s1t.height
      print(f"\n==== {c}: name-only records with >=1 candidate, per S1: train {tr.height / ntr:.4f}  test {te.height / nte:.4f}")
      print(f"  {'group':<8s} {'top p':<8s} | {'train rec/S1':>12s} {'assigned':>8s} {'right':>6s} {'owner=top':>9s} {'nobody':>7s} | {'test rec/S1':>11s} {'assigned':>8s}")
      for kb in ("k=0", "k=1", "k=2-5", "k=6+"):
          for pb in (">=0.9", "0.5-0.9", "0.2-0.5", "<0.2"):
              a = tr.filter((pl.col("kb") == kb) & (pl.col("pb") == pb))
              b = te.filter((pl.col("kb") == kb) & (pl.col("pb") == pb))
              if a.height + b.height == 0:
                  continue
              asg = a.filter(pl.col("ass_rid").is_not_null())
              right = (asg["ass_rid"] == asg["owner"]).fill_null(False).mean() if asg.height else float("nan")
              print(f"  {kb:<8s} {pb:<8s} | {a.height / ntr:12.4f} {a['ass_rid'].is_not_null().mean():8.1%} {right:6.1%} "
                    f"{(a['top_rid'] == a['owner']).fill_null(False).mean():9.1%} {a['owner'].is_null().mean():7.1%} | {b.height / nte:11.4f} {b['ass_rid'].is_not_null().mean():8.1%}")


if __name__ == "__main__":
    main()
