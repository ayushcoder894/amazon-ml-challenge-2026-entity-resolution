"""Phase 1: exact canonical-key statistics per split / country (and true-pair hit rates on train)."""
import polars as pl

D = "C:/ML/Challenges/amazon/work/data/"


def load(split, c, side):
    return pl.read_parquet(f"{D}p1_canon_{split}_{c}_{side}.parquet")


def key_stats(split, c):
    s1, o = load(split, c, "s1"), load(split, c, "o")
    ka = o.filter(pl.col("cak") != "").group_by("cak").agg(n=pl.len())
    x = s1.join(ka, on="cak", how="left").with_columns(pl.col("n").fill_null(0))
    shared = s1.group_by("cak").agg(k=pl.len())
    x = x.join(shared, on="cak")
    return dict(split=split, country=c, n_s1=s1.height, s1_addr_hit=round((x["n"] > 0).mean(), 4),
                cand_mean=round(x["n"].mean(), 2), cand_median=x["n"].median(), cand_p95=x["n"].quantile(0.95),
                s1_share_addr_with_other_s1=round((x["k"] > 1).mean(), 4))


def pair_hits(c):
    s1, o = load("train", c, "s1"), load("train", c, "o")
    g = pl.read_parquet(f"{D}train_gt_rid.parquet").filter(pl.col("country") == c)
    x = (g.join(s1.select(rid="idx", k1a="cak", k1n="cnk", k1s="cns"), on="rid")
         .join(o.select(oid="idx", k2a="cak", k2n="cnk", k2s="cns", ae="aempty"), on="oid"))
    a, n = pl.col("k1a") == pl.col("k2a"), (pl.col("k1n") == pl.col("k2n")) & (pl.col("k1n") != "")
    return x.select(country=pl.lit(c), pairs=pl.len(), addr_key_eq=a.mean().round(4), name_key_eq=n.mean().round(4),
                    both=(a & n).mean().round(4), either=(a | n).mean().round(4),
                    nospace_eq=(pl.col("k1s") == pl.col("k2s")).mean().round(4),
                    rec_addr_empty=pl.col("ae").mean().round(4))


if __name__ == "__main__":
    pl.Config.set_tbl_width_chars(220)
    rows = [key_stats(sp, c) for sp, cs in (("train", ("India", "US")), ("test", ("France", "India", "US"))) for c in cs]
    print(pl.DataFrame(rows))
    print(pl.concat([pair_hits(c) for c in ("India", "US")]))
