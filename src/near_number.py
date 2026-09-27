"""Near-number category (same canonical name, same non-numeric address words, different numbers): candidates,
true and predicted pairs per S1 - train (harder split, truth known) vs test. Vectorised on the canonical keys."""
import polars as pl

import decide

D = "C:/ML/Challenges/amazon/work/data/"
OUT = "C:/ML/Challenges/amazon/work/output/"


def tag(P, split, c):
    s1 = pl.read_parquet(f"{D}p1_canon_{split}_{c}_s1.parquet", columns=["idx", "cnk", "cak"])
    o = pl.read_parquet(f"{D}p1_canon_{split}_{c}_o.parquet", columns=["idx", "cnk", "cak", "aempty"])
    w = lambda e: e.str.split(" ").list.eval(pl.element().filter(~pl.element().str.contains(r"\d"))).list.join(" ")
    n = lambda e: e.str.split(" ").list.eval(pl.element().filter(pl.element().str.contains(r"\d"))).list.join(" ")
    x = (P.join(s1.select(rid="idx", n1="cnk", a1="cak"), on="rid").join(o.select(oid="idx", n2="cnk", a2="cak", ae="aempty"), on="oid"))
    return x.with_columns(near=(pl.col("n1") == pl.col("n2")) & (pl.col("n1") != "") & ~pl.col("ae")
                          & (w(pl.col("a1")) == w(pl.col("a2"))) & (w(pl.col("a1")) != "")
                          & (n(pl.col("a1")) != n(pl.col("a2"))) & (n(pl.col("a1")) != "") & (n(pl.col("a2")) != ""))


def main():
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    K = pl.read_parquet(f"{D}train_oof_ce_v12.parquet", columns=["rid", "oid", "y", "country", "p3"])
    pr = decide.ef_predict(K.select("rid", "oid", pl.col("p3").alias("p2")), "p2", 0.0, 1.5, 0.05).with_columns(pred=pl.lit(1))
    K = K.join(pr, on=["rid", "oid"], how="left").with_columns(pl.col("pred").fill_null(0))
    s1t = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).join(keep, on="rid", how="semi")
    res = pl.read_csv(f"{OUT}matching_results_v12final.tsv", separator="\t", quote_char=None, infer_schema=False,
                      missing_utf8_is_empty_string=True)
    fin = res.filter(pl.col("matched_entity_ids") != "").with_columns(m=pl.col("matched_entity_ids").str.split(",")).explode("m")
    T = pl.read_parquet(f"{D}test_pred_v12.parquet", columns=["rid", "oid", "country"])
    print(f"{'':16s} {'cand/S1':>8s} {'true/S1':>8s} {'pred/S1':>8s} {'pred share':>10s}")
    for c in ("US", "India"):
        n = s1t.filter(pl.col("country") == c).height
        x = tag(K.filter(pl.col("country") == c).select("rid", "oid", "y", "pred"), "train", c).filter(pl.col("near"))
        print(f"train {c:<10s} {x.height / n:8.4f} {x['y'].sum() / n:8.4f} {x['pred'].sum() / n:8.4f} {x['pred'].mean():10.3f}")
    for c in ("US", "India", "France"):
        s1 = pl.read_parquet(f"{D}p1_canon_test_{c}_s1.parquet", columns=["idx", "entity_id"])
        o = pl.read_parquet(f"{D}p1_canon_test_{c}_o.parquet", columns=["idx", "entity_id"])
        p = (fin.join(s1.rename({"entity_id": "source1_entity_id", "idx": "rid"}), on="source1_entity_id")
             .join(o.rename({"entity_id": "m", "idx": "oid"}), on="m").select("rid", "oid", pred=pl.lit(1)))
        P = T.filter(pl.col("country") == c).select("rid", "oid").join(p, on=["rid", "oid"], how="left").with_columns(pl.col("pred").fill_null(0))
        x = tag(P, "test", c).filter(pl.col("near"))
        print(f"test  {c:<10s} {x.height / s1.height:8.4f} {'':>8s} {x['pred'].sum() / s1.height:8.4f} {x['pred'].mean():10.3f}")


if __name__ == "__main__":
    main()
