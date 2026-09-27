"""LLM group-judgment experiment (US nearby-number groups). Builds prompts: reference entity + up to 8 candidate records
(the model's candidates with p >= 0.05, incl. the near-number ones). Train: 1,500 groups (harder split, labels kept locally);
test: all groups with an odd/mixed/none near-number candidate within 50. -> gpu5/data/grp_{train,test}.parquet"""
import json
import polars as pl
import decide
from nearby_number import pattern, sib_agree

D = "C:/ML/Challenges/amazon/work/data/"
DN = "D:/amazon_er/dense/"
OUT = "C:/ML/Challenges/amazon/work/output/"
rd = dict(separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
K = 8


def build(split, U, pred, s1raw, oraw, n_groups=None, seed=0):
    P, o = pattern(split, "US")
    P = P.join(U.filter(pl.col("p") >= 0.05).select("rid", "oid"), on=["rid", "oid"], how="semi")
    P = sib_agree(P, pred, o, "sib").filter(pl.col("sib").is_in(["odd one out", "mixed", "none"]) & (pl.col("diff") <= 50))
    ents = P.select("rid").unique()
    if n_groups:
        ents = ents.sample(min(n_groups, ents.height), seed=seed)
    C = (U.join(ents, on="rid", how="semi").filter(pl.col("p") >= 0.05).sort(["rid", "p"], descending=[False, True])
         .with_columns(k=pl.int_range(pl.len()).over("rid")).filter(pl.col("k") < K))
    near = P.select("rid", "oid", near=pl.lit(True))
    C = C.join(near, on=["rid", "oid"], how="left").with_columns(pl.col("near").fill_null(False))
    C = C.join(pred.with_columns(acc=pl.lit(True)), on=["rid", "oid"], how="left").with_columns(pl.col("acc").fill_null(False))
    C = C.join(s1raw, on="rid").join(oraw, on="oid")
    return C


def main():
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    # train
    s1c = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == "US").select("rid").join(keep, on="rid", how="semi")
    U = pl.read_parquet(f"{DN}train_joint9.parquet", columns=["rid", "oid", "pj", "country"]).filter(pl.col("country") == "US").select("rid", "oid", p="pj")
    pred = decide.ef_predict(U.select("rid", "oid", p2="p"), "p2", 0.0, 1.0, 0.05)
    r1 = pl.read_parquet(f"{D}train_source1.parquet", columns=["business_name", "business_address"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32)).rename({"business_name": "en", "business_address": "ea"})
    r2 = pl.concat([pl.read_parquet(f"{D}train_source{k}.parquet", columns=["business_name", "business_address"]) for k in (2, 3)]).with_row_index("oid").with_columns(pl.col("oid").cast(pl.UInt32)).rename({"business_name": "rn", "business_address": "ra"})
    Ctr = build("train", U.join(s1c, on="rid", how="semi"), pred, r1, r2, 1500)
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid"]).with_columns(y=pl.lit(1, pl.Int8))
    Ctr = Ctr.join(gt, on=["rid", "oid"], how="left").with_columns(pl.col("y").fill_null(0))
    Ctr.write_parquet("D:/amazon_er/france_night/data/grp_train_labels.parquet")
    # test
    s1t = pl.read_parquet(f"{D}p1_canon_test_US_s1.parquet", columns=["idx", "entity_id"])
    ot = pl.read_parquet(f"{D}p1_canon_test_US_o.parquet", columns=["idx", "entity_id"])
    fin = pl.read_csv(f"{OUT}matching_results_v21.tsv", **rd).filter(pl.col("matched_entity_ids") != "").with_columns(m=pl.col("matched_entity_ids").str.split(",")).explode("m")
    predt = (fin.join(s1t.rename({"entity_id": "source1_entity_id", "idx": "rid"}), on="source1_entity_id")
             .join(ot.rename({"entity_id": "m", "idx": "oid"}), on="m").select(pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32)))
    T = pl.read_parquet(f"{DN}test_union_p.parquet").join(s1t.select(pl.col("idx").cast(pl.UInt32).alias("rid")), on="rid", how="semi")
    t1 = pl.read_parquet(f"{D}test_source1.parquet", columns=["business_name", "business_address"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32)).rename({"business_name": "en", "business_address": "ea"})
    t2 = pl.concat([pl.read_parquet(f"{D}test_source{k}.parquet", columns=["business_name", "business_address"]) for k in (2, 3)]).with_row_index("oid").with_columns(pl.col("oid").cast(pl.UInt32)).rename({"business_name": "rn", "business_address": "ra"})
    Cte = build("test", T, predt, t1, t2)
    for nm, C in (("train", Ctr), ("test", Cte)):
        C.select("rid", "oid", "k", "en", "ea", "rn", "ra", "near", "acc", "p").write_parquet(f"C:/ML/Challenges/amazon/work/gpu5/data/grp_{nm}.parquet")
        print(nm, "groups", C["rid"].n_unique(), "pairs", C.height, "near-number pairs", C["near"].sum(), "accepted", C["acc"].sum())
    print("train truth: near-number pairs true", Ctr.filter(pl.col("near"))["y"].mean(), "accepted near true", Ctr.filter(pl.col("near") & pl.col("acc"))["y"].mean())


if __name__ == "__main__":
    main()
