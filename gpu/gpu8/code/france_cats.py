"""Label-free France check of the v13 decisions against the generator's behaviour on India / US.

Per generator category = name relation (p1_catalog, swap position) x address relation (canonical) x whether the S1's
address is shared by several S1 x whether the record is a SINGLE record or part of a CLUSTER (other records carry the
same canonical name and the same house numbers -> a separate entity, e.g. a sister company),
report candidates per S1 and
  trueIN / trueUS : share that is a true match on train (harder split, union candidates = v12 + dense)
  locoIN / locoUS : share accepted by the unseen-country recipe (LOCO stage-1 + dense add-on, ef(0,4)) on train
  accFR           : share accepted by v13 on France (stage-1 + France dense add-on, ef(0,4))
dev/S1 = France candidates/S1 x (mean train true share - France accepted share): > 0 = France accepts too few.
Output: D:/amazon_er/france_night/data/france_cats.parquet + printed table.
"""
import sys

import polars as pl

import decide
from p1_catalog import addr_rel, name_rel
from p1_decoy2 import refine
from p2_france_diag import addr_cls

D = "C:/ML/Challenges/amazon/work/data/"
DN = "D:/amazon_er/dense/"
OUT = "D:/amazon_er/france_night/data/"


def canon(split, c):
    s1 = pl.read_parquet(f"{D}p1_canon_{split}_{c}_s1.parquet")
    o = pl.read_parquet(f"{D}p1_canon_{split}_{c}_o.parquet")
    na = s1.filter(pl.col("cak") != "").group_by("cak").agg(na=pl.len())
    s1 = s1.join(na, on="cak", how="left")
    vocab = set(s1.select(pl.col("cn").explode()).to_series().drop_nulls().to_list())
    nums = pl.col("cak").str.extract_all(r"\d+").list.join(" ")
    o = o.with_columns(ckey=pl.col("cns") + "|" + nums)
    o = o.join(o.group_by("ckey").agg(csize=pl.len()), on="ckey", how="left")
    return s1, o, vocab


def category(r, vocab):
    A, B = list(r["cn1"]), list(r["cn2"])
    nr = refine(name_rel(A, B, r["cns1"], r["cns2"]), A, B, vocab)
    if nr == "word_swap_real_word":
        olds = [t for t in A if t not in B]
        if olds:
            i = A.index(olds[0])
            nr += " (first)" if i == 0 else " (last)" if i == len(A) - 1 else " (middle)"
    ar = addr_cls(addr_rel(r["cak1"].split(" ") if r["cak1"] else [], r["cak2"].split(" ") if r["cak2"] else [], r["ae2"]))
    if ar not in ("empty",):
        ar += " [S1 addr shared]" if (r["na"] or 0) > 1 else ""
    cl = "name-only" if r["ae2"] else ("cluster" if (r["csize"] or 1) >= 2 else "single")
    return nr, ar, cl


def stats(C, split, c, n_sample, cols):
    """C: (i1, i2, flags...) candidate pairs of one country. Returns per-category counts per S1 and flag means."""
    s1, o, vocab = canon(split, c)
    x = C.sample(min(n_sample, C.height), seed=0)
    x = (x.join(s1.select(i1="idx", cn1="cn", cns1="cns", cak1="cak", na="na"), on="i1")
         .join(o.select(i2="idx", cn2="cn", cns2="cns", cak2="cak", ae2="aempty", csize="csize"), on="i2"))
    f = C.height / x.height / s1.height if split == "test" else C.height / x.height / C["i1"].n_unique()
    acc = {}
    for r in x.iter_rows(named=True):
        k = category(r, vocab)
        a = acc.setdefault(k, [0] + [0.0] * len(cols))
        a[0] += 1
        for j, col in enumerate(cols):
            a[j + 1] += r[col] or 0
    rows = [dict(name=k[0], addr=k[1], clus=k[2], cand=v[0] * f, **{col: v[j + 1] / v[0] for j, col in enumerate(cols)})
            for k, v in acc.items()]
    return pl.DataFrame(rows)


def main(n_sample=200000):
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    ids = lambda df: df.select(i1=pl.col("rid").cast(pl.UInt32), i2=pl.col("oid").cast(pl.UInt32), *[c for c in df.columns if c not in ("rid", "oid")])
    J = pl.read_parquet(f"{DN}train_joint7.parquet", columns=["rid", "oid", "y", "country"]).join(keep, on="rid", how="semi")
    Kl = pl.read_parquet(f"{D}train_oof_locoh.parquet", columns=["rid", "oid", "p1", "country"]).rename({"p1": "p"})
    Nl = pl.read_parquet(f"{DN}train_pn_loco.parquet", columns=["rid", "oid", "pn", "country"]).rename({"pn": "p"})
    L = pl.concat([Kl, Nl.with_columns(pl.col("p").cast(pl.Float32))], how="vertical_relaxed")
    lp = decide.ef_predict(L.select("rid", "oid", pl.col("p").alias("p2")), "p2", 0.0, 4.0, 0.05).with_columns(loco=pl.lit(1.0))
    J = J.join(lp, on=["rid", "oid"], how="left").with_columns(pl.col("loco").fill_null(0.0), y=pl.col("y").cast(pl.Float64))
    tabs = {}
    for c in ("India", "US"):
        tabs[c] = stats(ids(J.filter(pl.col("country") == c).drop("country")), "train", c, n_sample, ["y", "loco"])
        print(f"train {c} done", flush=True)
    Kf = pl.read_parquet(f"{D}test_final_v12_ce_cectx_twin.parquet", columns=["rid", "oid", "country", "p1"]).filter(
        pl.col("country") == "France").select("rid", "oid", p="p1")
    Nf = pl.read_parquet(f"{DN}test_pn_france.parquet").select("rid", "oid", p=pl.col("pn").cast(pl.Float32))
    F = pl.concat([Kf, Nf])
    fp = decide.ef_predict(F.select("rid", "oid", pl.col("p").alias("p2")), "p2", 0.0, 4.0, 0.05).with_columns(acc=pl.lit(1.0))
    F = F.join(fp, on=["rid", "oid"], how="left").with_columns(pl.col("acc").fill_null(0.0), pmean=pl.col("p").cast(pl.Float64))
    tf = stats(ids(F.select("rid", "oid", "acc", "pmean")), "test", "France", n_sample * 2, ["acc", "pmean"])
    key = ["name", "addr", "clus"]
    t = (tf.rename({"cand": "candFR", "acc": "accFR", "pmean": "pFR"})
         .join(tabs["India"].rename({"cand": "candIN", "y": "trueIN", "loco": "locoIN"}), on=key, how="full", coalesce=True)
         .join(tabs["US"].rename({"cand": "candUS", "y": "trueUS", "loco": "locoUS"}), on=key, how="full", coalesce=True)
         .fill_null(0.0))
    t = t.with_columns(dev=pl.col("candFR") * ((pl.col("trueIN") + pl.col("trueUS")) / 2 - pl.col("accFR")))
    t.write_parquet(f"{OUT}france_cats.parquet")
    with pl.Config(tbl_rows=45, tbl_width_chars=260, float_precision=3, tbl_hide_column_data_types=True, fmt_str_lengths=36):
        print(t.sort(pl.col("dev").abs(), descending=True).select(
            "name", "addr", "clus", "candFR", "accFR", "pFR", "trueIN", "trueUS", "locoIN", "locoUS", "candIN", "candUS", "dev").head(40))
    print(f"France accepts {t.filter(pl.col('dev') > 0)['dev'].sum():.3f} fewer / {-t.filter(pl.col('dev') < 0)['dev'].sum():.3f} "
          f"more matches per S1 than the train truth rates of the same categories would give")


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:2]])
