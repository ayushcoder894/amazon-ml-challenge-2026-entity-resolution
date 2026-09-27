"""What does the final model add (beyond stage-1 decisions) on US test, compared with train where truth is known?"""
from collections import Counter, defaultdict

import polars as pl

import decide
from p2_france_diag import addr_cls, canon
from p1_catalog import addr_rel, name_rel
from p1_decoy2 import refine

D = "C:/ML/Challenges/amazon/work/data/"


def cat(r, vocab):
    A, B = list(r["cn1"]), list(r["cn2"])
    nr = refine(name_rel(A, B, r["cns1"], r["cns2"]), A, B, vocab)
    ar = addr_cls(addr_rel(r["cak1"].split(" ") if r["cak1"] else [], r["cak2"].split(" ") if r["cak2"] else [], r["ae2"]))
    return nr, ar


def enrich(P, s1, o):
    return (P.join(s1.select(i1="idx", cn1="cn", cns1="cns", cak1="cak"), on="i1")
            .join(o.select(i2="idx", cn2="cn", cns2="cns", cak2="cak", ae2="aempty"), on="i2"))


def main(c="US"):
    s1, o, vocab = canon("train", c)
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    n_tr = s1.join(keep.rename({"rid": "idx"}), on="idx", how="semi").height
    K = pl.read_parquet(f"{D}train_oof_ce_v12.parquet", columns=["rid", "oid", "y", "country", "p1", "p3"]).filter(pl.col("country") == c)
    a = decide.ef_predict(K.select("rid", "oid", pl.col("p1").alias("p2")), "p2", 0.0, 1.5, 0.05).with_columns(a=pl.lit(1))
    b = decide.ef_predict(K.select("rid", "oid", pl.col("p3").alias("p2")), "p2", 0.0, 1.5, 0.05)
    tr = (b.join(a, on=["rid", "oid"], how="left").filter(pl.col("a").is_null()).join(K.select("rid", "oid", "y"), on=["rid", "oid"])
          .select(i1=pl.col("rid").cast(pl.UInt32), i2=pl.col("oid").cast(pl.UInt32), y="y"))
    st = defaultdict(lambda: [0, 0])
    for r in enrich(tr, s1, o).iter_rows(named=True):
        k = cat(r, vocab)
        st[k][0] += 1
        st[k][1] += r["y"]
    t1, t2, tv = canon("test", c)
    T = pl.read_parquet(f"{D}test_pred_v12.parquet", columns=["rid", "oid", "country", "p1"]).filter(pl.col("country") == c)
    a = decide.ef_predict(T.select("rid", "oid", pl.col("p1").alias("p2")), "p2", 0.0, 1.5, 0.05).with_columns(a=pl.lit(1))
    res = pl.read_csv("C:/ML/Challenges/amazon/work/output/matching_results_v12final.tsv", separator="\t", quote_char=None,
                      infer_schema=False, missing_utf8_is_empty_string=True)
    fin = (res.filter(pl.col("matched_entity_ids") != "").with_columns(m=pl.col("matched_entity_ids").str.split(",")).explode("m")
           .join(t1.select(source1_entity_id="entity_id", rid="idx"), on="source1_entity_id").join(t2.select(m="entity_id", oid="idx"), on="m")
           .select("rid", "oid"))
    te = fin.join(a, on=["rid", "oid"], how="left").filter(pl.col("a").is_null()).select(i1="rid", i2="oid")
    ts, ex = Counter(), defaultdict(list)
    raw1 = pl.read_parquet(f"{D}test_source1.parquet").with_row_index("i1").select("i1", "business_name", "business_address")
    raw2 = pl.concat([pl.read_parquet(f"{D}test_source{k}.parquet") for k in (2, 3)]).with_row_index("i2").select("i2", n2="business_name", a2="business_address")
    x = enrich(te, t1, t2).join(raw1, on="i1").join(raw2, on="i2")
    for r in x.iter_rows(named=True):
        k = cat(r, tv)
        ts[k] += 1
        if len(ex[k]) < 3:
            ex[k].append(f"{r['business_name']} | {r['business_address']}  ->  {r['n2']} | {r['a2']}")
    n_te = t1.height
    print(f"{c}: pairs added by the final model beyond stage-1 decisions, per S1 (train: with true rate)")
    print(f"{'name relation':<30s} {'address':<30s} {'train/S1':>8s} {'true':>6s} {'test/S1':>8s} {'excess':>8s}")
    keys = sorted(set(st) | set(ts), key=lambda k: -(ts.get(k, 0) / n_te - st[k][0] / n_tr))
    for k in keys[:18]:
        trn = st[k][0] / n_tr
        print(f"{k[0]:<30s} {k[1]:<30s} {trn:8.4f} {st[k][1] / max(1, st[k][0]):6.3f} {ts.get(k, 0) / n_te:8.4f} "
              f"{ts.get(k, 0) / n_te - trn:+8.4f}")
        for e in ex[k][:2]:
            print(f"        {e}")


if __name__ == "__main__":
    main()
