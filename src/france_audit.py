"""Hand audit of France (test, no labels): readable dump of sampled France entities with their candidates.
For each sampled S1: raw name | address, then every candidate with p >= 0.01 (v12 stage-1 p1 for v12 candidates, France
dense add-on pn for new pairs), whether v13 accepts it ([X]), and - if another entity is the record's top candidate -
that rival. Output: D:/amazon_er/france_night/audit/audit_<seed>.txt (+ a parquet with the same rows for labelling).
"""
import sys

import polars as pl

import decide

D = "C:/ML/Challenges/amazon/work/data/"
DN = "D:/amazon_er/dense/"
OUT = "D:/amazon_er/france_night/audit/"


def union():
    K = pl.read_parquet(f"{D}test_final_v12_ce_cectx_twin.parquet", columns=["rid", "oid", "country", "p1"]).filter(
        pl.col("country") == "France").select("rid", "oid", p="p1", src=pl.lit("K"))
    N = pl.read_parquet(f"{DN}test_pn_france.parquet").select("rid", "oid", p="pn", src=pl.lit("N"))
    return pl.concat([K, N])


def main(n=40, seed=0):
    U = union()
    pred = decide.ef_predict(U.select("rid", "oid", pl.col("p").alias("p2")), "p2", 0.0, 4.0, 0.05).with_columns(acc=pl.lit(1))
    U = U.join(pred, on=["rid", "oid"], how="left").with_columns(pl.col("acc").fill_null(0))
    U = U.with_columns(top=pl.col("p").rank("ordinal", descending=True).over("oid") == 1)
    s1 = pl.read_parquet(f"{D}test_source1.parquet", columns=["entity_id", "business_name", "business_address"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
    o = pl.concat([pl.read_parquet(f"{D}test_source{k}.parquet", columns=["entity_id", "business_name", "business_address"]) for k in (2, 3)]).with_row_index("oid").with_columns(pl.col("oid").cast(pl.UInt32))
    fr = pl.read_parquet(f"{D}test_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == "France")
    ents = fr.sample(n, seed=seed)["rid"].to_list()
    rows, lines = [], []
    for i, r in enumerate(ents):
        e = s1.filter(pl.col("rid") == r).row(0, named=True)
        lines.append(f"\n#{i:03d} {e['entity_id']}  {e['business_name']} | {e['business_address']}")
        c = U.filter((pl.col("rid") == r) & (pl.col("p") >= 0.01)).sort("p", descending=True).head(12).join(o, on="oid")
        for q in c.iter_rows(named=True):
            rival = ""
            if not q["top"]:
                t = U.filter((pl.col("oid") == q["oid"]) & pl.col("top")).join(s1, on="rid")
                if t.height:
                    tr = t.row(0, named=True)
                    rival = f"   <- top is {tr['entity_id']} p={tr['p']:.2f} ({tr['business_name']} | {tr['business_address'][:45]})"
            lines.append(f"   {'[X]' if q['acc'] else '[ ]'} p={q['p']:.2f} {q['src']} {q['entity_id']}  {q['business_name']} | {q['business_address']}{rival}")
            rows.append(dict(i=i, s1=e["entity_id"], rid=r, oid=q["oid"], rec=q["entity_id"], p=q["p"], acc=q["acc"], src=q["src"]))
    open(f"{OUT}audit_{seed}.txt", "w", encoding="utf-8").write("\n".join(lines))
    pl.DataFrame(rows).write_parquet(f"{OUT}audit_{seed}.parquet")
    print(f"wrote {OUT}audit_{seed}.txt: {len(ents)} entities, {len(rows)} candidate lines")


if __name__ == "__main__":
    main(*[int(x) for x in sys.argv[1:3]])
