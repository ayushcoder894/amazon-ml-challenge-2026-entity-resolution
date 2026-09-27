"""Score the LLM group answers on train (labels): near-number pairs decided by the LLM vs by the current model."""
import re
import sys
import polars as pl
import evaluate as ev

L = pl.read_parquet("D:/amazon_er/france_night/data/grp_train_labels.parquet")
A = pl.read_parquet(sys.argv[1] if len(sys.argv) > 1 else "C:/ML/Challenges/amazon/work/grp_train_ans.parquet")
print("answers", A.height, "examples:", A.head(5).to_dicts())


def parse(a, n):
    a = (a or "").lower()
    if "none" in a:
        return set()
    return {int(x) for x in re.findall(r"\d+", a) if 1 <= int(x) <= n}


L = L.sort(["rid", "k"]).with_columns(pos=pl.int_range(pl.len()).over("rid") + 1, n=pl.len().over("rid"))
g = L.join(A.with_columns(pl.col("rid").cast(L["rid"].dtype)), on="rid", how="inner")
sel = [p in parse(a, n) for p, a, n in zip(g["pos"].to_list(), g["answer"].to_list(), g["n"].to_list())]
g = g.with_columns(llm=pl.Series(sel))
nr = g.filter(pl.col("near"))
for lab, col in (("model", "acc"), ("LLM", "llm")):
    x = nr.filter(pl.col(col))
    print(f"near-number pairs ({nr.height}, true {nr['y'].mean():.3f}): {lab} accepts {x.height} precision {x['y'].mean():.3f} recall {nr.filter(pl.col(col))['y'].sum() / nr['y'].sum():.3f}")
allp = g
for lab, col in (("model", "acc"), ("LLM", "llm")):
    x = allp.filter(pl.col(col))
    print(f"all pairs ({allp.height}, true {allp['y'].mean():.3f}): {lab} accepts {x.height} precision {x['y'].mean():.3f} recall {x['y'].sum() / allp['y'].sum():.3f}")
# F0.5 on these entities: truth = all true pairs of the entities (incl. outside the 8 shown)
gt = pl.read_parquet("C:/ML/Challenges/amazon/work/data/train_gt_rid.parquet", columns=["rid", "oid"]).join(g.select("rid").unique(), on="rid", how="semi")
ents = g["rid"].unique()
base = g.filter(pl.col("acc")).select("rid", "oid")
mix = g.filter((pl.col("acc") & ~pl.col("near")) | (pl.col("llm") & pl.col("near"))).select("rid", "oid")
llm_all = g.filter(pl.col("llm")).select("rid", "oid")
f0, f1, f2 = (ev.macro_f05(p, gt, ents) for p in (base, mix, llm_all))
print(f"F0.5 on these {len(ents)} entities: model {f0:.5f} | LLM on near-number pairs only {f1:.5f} ({f1 - f0:+.5f}) | LLM on whole group {f2:.5f} ({f2 - f0:+.5f})")
