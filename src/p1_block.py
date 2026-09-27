"""Phase 1.5: blocking ceiling on train (per country, fixed random sample of S1 entities).

Tiers (all within country):
  T1  exact canonical address key
  T2a exact canonical name key, record address empty (name-only fallback)
  T2b exact canonical name key, any address
  T3  char 3-gram TF-IDF (canonical name + canonical address) top-k records per S1, k in {5, 10, 20, 50}
Reported: recall of true pairs of the sampled S1 and candidates per S1 (mean / median / p95), per tier and for unions.
Reference: the current pipeline's final candidate set (cascade, train_oof_v10) on the same S1.
"""
import sys
import time

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer

D = "C:/ML/Challenges/amazon/work/data/"
KS = (5, 10, 20, 50)


def text(df):
    return df.select(t=pl.col("cn").list.join(" ") + pl.lit(" | ") + pl.col("cak"))["t"].to_list()


def topk(Q, R, k, chunk=400):
    """Exact top-k by cosine (TF-IDF rows are L2-normalised) for each query row, sparse products in chunks."""
    RT = R.T.tocsr()
    out_i, out_s = [], []
    for s in range(0, Q.shape[0], chunk):
        M = (Q[s:s + chunk] @ RT).tocsr()
        for r in range(M.shape[0]):
            a, b = M.indptr[r], M.indptr[r + 1]
            d, j = M.data[a:b], M.indices[a:b]
            if len(d) > k:
                sel = np.argpartition(-d, k)[:k]
                d, j = d[sel], j[sel]
            o = np.argsort(-d)
            out_i.append(j[o])
            out_s.append(d[o])
    return out_i, out_s


def stats(cand, gt, rids, name):
    c = cand.unique(["rid", "oid"])
    rec = gt.join(c, on=["rid", "oid"], how="semi").height / gt.height
    n = pl.DataFrame({"rid": rids}).join(c.group_by("rid").agg(n=pl.len()), on="rid", how="left").fill_null(0)["n"]
    return dict(tier=name, recall=round(rec, 4), mean=round(n.mean(), 2), median=n.median(), p95=n.quantile(0.95))


def main(c, n_s1=20000, seed=0):
    t = time.time()
    s1 = pl.read_parquet(f"{D}p1_canon_train_{c}_s1.parquet")
    o = pl.read_parquet(f"{D}p1_canon_train_{c}_o.parquet")
    samp = s1.sample(n_s1, seed=seed)
    rids = samp["idx"]
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").filter(pl.col("country") == c).select("rid", "oid").join(
        samp.select(rid="idx"), on="rid", how="semi")
    rows = []
    t1 = (samp.filter(pl.col("cak") != "").select(rid="idx", k="cak")
          .join(o.filter(pl.col("cak") != "").select(oid="idx", k="cak"), on="k").select("rid", "oid"))
    t2a = (samp.filter(pl.col("cnk") != "").select(rid="idx", k="cnk")
           .join(o.filter((pl.col("cnk") != "") & pl.col("aempty")).select(oid="idx", k="cnk"), on="k").select("rid", "oid"))
    t2b = (samp.filter(pl.col("cnk") != "").select(rid="idx", k="cnk")
           .join(o.filter(pl.col("cnk") != "").select(oid="idx", k="cnk"), on="k").select("rid", "oid"))
    for name, cand in (("T1 address key", t1), ("T2a name key, empty-address records", t2a), ("T2b name key, any address", t2b),
                       ("T1 + T2a", pl.concat([t1, t2a])), ("T1 + T2b", pl.concat([t1, t2b]))):
        rows.append(stats(cand, gt, rids, name))
    print(f"[{c}] exact tiers done ({time.time() - t:.0f}s)", flush=True)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, max_df=0.02, sublinear_tf=True,
                          dtype=np.float32)
    R = vec.fit_transform(text(o))
    Q = vec.transform(text(samp))
    print(f"[{c}] tf-idf {R.shape} nnz {R.nnz:,} ({time.time() - t:.0f}s)", flush=True)
    idx, _ = topk(Q, R, max(KS))
    oids = o["idx"].to_numpy()
    for k in KS:
        t3 = pl.DataFrame({"rid": np.repeat(rids.to_numpy(), [min(k, len(i)) for i in idx]),
                           "oid": np.concatenate([oids[i[:k]] for i in idx])}).with_columns(
            pl.col("rid").cast(pl.UInt32), pl.col("oid").cast(pl.UInt32))
        rows.append(stats(t3, gt, rids, f"T3 tf-idf top-{k}"))
        rows.append(stats(pl.concat([t1, t2a, t3]), gt, rids, f"T1 + T2a + T3 top-{k}"))
        rows.append(stats(pl.concat([t1, t2b, t3]), gt, rids, f"T1 + T2b + T3 top-{k}"))
    ref = pl.read_parquet(f"{D}train_oof_v10.parquet", columns=["rid", "oid"]).join(samp.select(rid="idx"), on="rid",
                                                                                     how="semi")
    rows.append(stats(ref, gt, rids, "reference: current pipeline (cascade)"))
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_width_chars(160)
    print(f"\n=== {c}: blocking ceiling on {n_s1:,} random S1 ({gt.height:,} true pairs) ({time.time() - t:.0f}s)")
    print(pl.DataFrame(rows))


if __name__ == "__main__":
    for c in sys.argv[1:] or ["US", "India"]:
        main(c)
