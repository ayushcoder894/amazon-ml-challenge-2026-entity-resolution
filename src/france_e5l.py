"""Does the e5-large cross-encoder (A100, trained on the other country's uncertain pairs) help the France recipe?
Proxy = India / US scored by models that never saw them (TRUE labels). Base = blend pb (France recipe), rule ef(0, 4).
Variants on the uncertain pairs (where the CE score s exists): linear mixes of pb and s, and a logistic combination
(logit pb, logit s, bias) fitted on the OTHER country only.
"""
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

import decide
import evaluate as ev

D = "C:/ML/Challenges/amazon/work/data/"
NIGHT = "D:/amazon_er/france_night/"
R = "D:/amazon_er/gpu3_results/"


def lg(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def main():
    U = pl.read_parquet(f"{NIGHT}data/loco_joint_U.parquet").with_columns(pb=0.5 * pl.col("p") + 0.5 * pl.col("pj"))
    U = U.join(pl.read_parquet(f"{R}ce3_e5l_loco.parquet"), on=["rid", "oid"], how="left")
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    s1 = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).join(keep, on="rid", how="semi")
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").join(keep, on="rid", how="semi").select("rid", "oid")
    # logistic combination fitted on the other country's uncertain pairs
    coef = {}
    for c in ("India", "US"):
        tr = U.filter((pl.col("country") != c) & pl.col("s").is_not_null())
        X = np.column_stack([lg(tr["pb"].to_numpy()), lg(tr["s"].to_numpy())])
        m = LogisticRegression(C=1.0, max_iter=500).fit(X, tr["y"].to_numpy())
        coef[c] = (m.coef_[0], m.intercept_[0])
        print(f"logistic fitted on {'US' if c == 'India' else 'India'}: w_pb {m.coef_[0][0]:.3f} w_s {m.coef_[0][1]:.3f} b {m.intercept_[0]:.3f}")
    pbv, sv = U["pb"].to_numpy(), U["s"].to_numpy()
    has = ~np.isnan(sv.astype(float)) if sv.dtype != object else np.array([v is not None for v in sv])
    sv = np.where(has, sv, 0.5).astype(float)
    variants = {"pb (France recipe)": pbv}
    for w in (0.15, 0.3, 0.5):
        variants[f"mix w_s={w}"] = np.where(has, (1 - w) * pbv + w * sv, pbv)
    cvec = U["country"].to_numpy()
    lgt = np.zeros(len(pbv))
    for c, (w, b) in coef.items():
        mk = (cvec == c) & has
        lgt[mk] = 1 / (1 + np.exp(-(w[0] * lg(pbv[mk]) + w[1] * lg(sv[mk]) + b)))
    variants["logistic (other country)"] = np.where(has, lgt, pbv)
    for name, v in variants.items():
        pred = decide.ef_predict(U.select("rid", "oid").with_columns(p2=pl.Series(v)), "p2", 0.0, 4.0, 0.05)
        sc = {c: ev.macro_f05(pred.join(s1.filter(pl.col("country") == c), on="rid", how="semi"),
                              gt.join(s1.filter(pl.col("country") == c), on="rid", how="semi"),
                              s1.filter(pl.col("country") == c)["rid"]) for c in ("India", "US")}
        print(f"{name:<26s} India {sc['India']:.5f}  US {sc['US']:.5f}", flush=True)


if __name__ == "__main__":
    main()
