"""Astra #6: ownership allocation. Same probabilities (final out-of-fold pf, harder split), same expected-F0.5 rule;
only the ownership step changes.
  A current : a record may only be taken by its argmax entity (then per-entity expected-F0.5 top-k)
  B all     : every entity runs expected-F0.5 top-k over ALL its candidates; a record claimed by several entities goes
              to the claimant with the highest p
  C second  : A, then records nobody took are offered to their 2nd-best entity; that entity keeps a record if doing so
              raises its expected F0.5 (records added greedily by p)
"""
import polars as pl

import decide
import evaluate as ev

D = "C:/ML/Challenges/amazon/work/data/"


def ef_all(K, mu, c, floor=0.05):
    """expected-F0.5 top-k per entity over all candidates (no argmax restriction)."""
    tot = K.group_by("rid").agg(psum=pl.col("p").sum())
    w = (K.filter(pl.col("p") >= floor).join(tot, on="rid").sort(["rid", "p"], descending=[False, True])
         .with_columns(k=pl.int_range(1, pl.len() + 1).over("rid"), cs=pl.col("p").cum_sum().over("rid"),
                       lq=(1 - pl.col("p")).clip(1e-6).log().sum().over("rid")))
    w = w.with_columns(ef=1.25 * pl.col("cs") / (0.25 * (pl.col("psum") + mu) + pl.col("k")))
    w = w.with_columns(best=pl.col("ef").max().over("rid"), e0=c * pl.col("lq").exp())
    ks = w.filter(pl.col("ef") == pl.col("best")).group_by("rid").agg(kstar=pl.col("k").min())
    return w.join(ks, on="rid").filter((pl.col("k") <= pl.col("kstar")) & (pl.col("best") > pl.col("e0"))).select("rid", "oid", "p")


def second_pass(K, first, mu, floor=0.05):
    taken = first.select("oid")
    free = K.join(taken, on="oid", how="anti").filter(pl.col("p") >= floor)
    free = free.with_columns(r=pl.col("p").rank("ordinal", descending=True).over("oid")).filter(pl.col("r") == 2)
    tot = K.group_by("rid").agg(psum=pl.col("p").sum())
    cur = first.join(K, on=["rid", "oid"]).group_by("rid").agg(cs=pl.col("p").sum(), k=pl.len())
    x = (free.join(tot, on="rid").join(cur, on="rid", how="left").with_columns(pl.col("cs").fill_null(0.0), pl.col("k").fill_null(0))
         .sort(["rid", "p"], descending=[False, True])
         .with_columns(j=pl.int_range(1, pl.len() + 1).over("rid"), add=pl.col("p").cum_sum().over("rid")))
    x = x.with_columns(
        ef_new=1.25 * (pl.col("cs") + pl.col("add")) / (0.25 * (pl.col("psum") + mu) + pl.col("k") + pl.col("j")),
        ef_old=pl.when(pl.col("k") > 0).then(1.25 * pl.col("cs") / (0.25 * (pl.col("psum") + mu) + pl.col("k"))).otherwise(0.0))
    x = x.with_columns(best=pl.col("ef_new").max().over("rid"))
    js = x.filter(pl.col("ef_new") == pl.col("best")).group_by("rid").agg(jstar=pl.col("j").min())
    add = x.join(js, on="rid").filter((pl.col("j") <= pl.col("jstar")) & (pl.col("best") > pl.col("ef_old")) & (pl.col("k") > 0))
    return pl.concat([first.select("rid", "oid"), add.select("rid", "oid")])


def main():
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    s1 = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).join(keep, on="rid", how="semi")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").join(keep, on="rid", how="semi").select("rid", "oid", "country")
    K = pl.read_parquet(f"{D}train_oof_final_v12_twin.parquet", columns=["rid", "oid", "pf"]).rename({"pf": "p"})
    mu, c = 0.0, 1.0

    def score(pred, name):
        out = []
        for cc in ("India", "US"):
            r = s1.filter(pl.col("country") == cc)["rid"]
            out.append(f"{cc} {ev.macro_f05(pred.filter(pl.col('rid').is_in(r.implode())).select('rid', 'oid'), gt.filter(pl.col('country') == cc).select('rid', 'oid'), r):.5f}")
        print(f"  {name:<44s} {' | '.join(out)}   ({pred.height:,} pairs)")
    A = decide.ef_predict(K.select("rid", "oid", pl.col("p").alias("p2")), "p2", mu, c, 0.05).join(K, on=["rid", "oid"])
    score(A, "A current (argmax owner first)")
    B = ef_all(K, mu, c)
    B = B.filter(pl.col("p").rank("ordinal", descending=True).over("oid") == 1)
    score(B, "B every entity chooses, conflicts -> highest p")
    C = second_pass(K, A.select("rid", "oid"), mu)
    score(C, "C current + 2nd-best entity may take leftovers")


if __name__ == "__main__":
    main()
