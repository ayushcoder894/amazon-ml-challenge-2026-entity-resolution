"""True pairs missing from the final candidate set (harder split): generator categories, whether they are in the
blocking pool at all, and whether they are intrinsically ambiguous (name shared by 2+ S1 and record without address)."""
import sys
from collections import Counter, defaultdict

import polars as pl

from p1_catalog import addr_rel, name_rel
from p1_decoy2 import refine
from p2_france_diag import addr_cls, canon

D = "C:/ML/Challenges/amazon/work/data/"


def main(c, n_sample=60000):
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").filter(pl.col("country") == c).join(keep, on="rid", how="semi").select("rid", "oid")
    K = pl.read_parquet(f"{D}train_oof_final_v12_twin.parquet", columns=["rid", "oid"])
    pool = pl.concat([pl.scan_parquet(f"{D}train_feat_v11_fold{f}.parquet").filter(pl.col("country") == c).select("rid", "oid")
                      for f in (0, 1)]).collect()
    miss = gt.join(K, on=["rid", "oid"], how="anti").with_columns(
        in_pool=pl.struct("rid", "oid").is_in(pool.select(pl.struct("rid", "oid")).to_series().implode()))
    s1, o, vocab = canon("train", c)
    nn = s1.filter(pl.col("cnk") != "").group_by("cnk").agg(nn=pl.len())
    x = (miss.sample(min(n_sample, miss.height), seed=0)
         .join(s1.select(rid="idx", cn1="cn", cns1="cns", cak1="cak", cnk1="cnk"), on="rid")
         .join(o.select(oid="idx", cn2="cn", cns2="cns", cak2="cak", ae2="aempty"), on="oid")
         .join(nn.rename({"cnk": "cnk1"}), on="cnk1", how="left"))
    raw1 = pl.read_parquet(f"{D}train_source1.parquet").with_row_index("rid").select("rid", "business_name", "business_address")
    raw2 = pl.concat([pl.read_parquet(f"{D}train_source{k}.parquet") for k in (2, 3)]).with_row_index("oid").select(
        "oid", n2="business_name", a2="business_address")
    x = x.join(raw1, on="rid").join(raw2, on="oid")
    st, ex = Counter(), defaultdict(list)
    amb = 0
    for r in x.iter_rows(named=True):
        A, B = list(r["cn1"]), list(r["cn2"])
        nr = refine(name_rel(A, B, r["cns1"], r["cns2"]), A, B, vocab)
        ar = addr_cls(addr_rel(r["cak1"].split(" ") if r["cak1"] else [], r["cak2"].split(" ") if r["cak2"] else [], r["ae2"]))
        a = r["ae2"] and (r["nn"] or 0) > 1
        amb += a
        k = (nr, ar, "ambiguous" if a else "")
        st[k] += 1
        if len(ex[k]) < 2:
            ex[k].append(f"{r['business_name']} | {r['business_address']}  ->  {r['n2']} | {r['a2']}")
    f = miss.height / x.height
    print(f"\n=== {c}: {miss.height:,} true pairs missing from candidates ({miss.height / gt.height:.4f}); in blocking pool "
          f"{miss['in_pool'].mean():.3f}; intrinsically ambiguous {amb / x.height:.3f}")
    for k, n in st.most_common(16):
        print(f"  {k[0]:<28s} {k[1]:<30s} {k[2]:<10s} {n * f / gt.height:.4f} of true pairs")
        for e in ex[k][:2]:
            print(f"        {e}")


if __name__ == "__main__":
    for c in sys.argv[1:] or ["India", "US"]:
        main(c)
