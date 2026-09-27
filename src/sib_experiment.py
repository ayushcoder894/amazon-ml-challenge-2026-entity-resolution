"""Does adding sibling features help on the pairs the model is unsure about? (India, v5 OOF, honest folds)"""
import time

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score

from siblings import SIB_COLS, sibling_features

D = "C:/ML/Challenges/amazon/work/data/"
t = time.time()
K = pl.read_parquet(f"{D}train_oof_v5.parquet").filter(pl.col("country") == "India")
S = sibling_features(K.select("rid", "oid", "p1"), "train", "India")
print(f"sibling features for {K.height:,} pairs ({time.time() - t:.0f}s); coverage {S['sib_name_max'].is_not_null().mean():.3f}")
X = K.join(S, on=["rid", "oid"]).filter((pl.col("p2") > 0.05) & (pl.col("p2") < 0.95))
print(f"uncertain pairs: {X.height:,} (positive rate {X['y'].mean():.3f})")
P = dict(objective="binary:logistic", tree_method="hist", device="cuda", max_depth=6, learning_rate=0.1)
for name, cols in [("p2 only", ["p2"]), ("p2 + siblings", ["p2"] + SIB_COLS)]:
    res = []
    for f in (0, 1):
        tr, te = X.filter(pl.col("fold") == f), X.filter(pl.col("fold") != f)
        b = xgb.train(P, xgb.DMatrix(tr.select(cols).to_numpy(), tr["y"].to_numpy()), 200)
        p = b.predict(xgb.DMatrix(te.select(cols).to_numpy()))
        res.append((roc_auc_score(te["y"], p), log_loss(te["y"], np.clip(p, 1e-6, 1 - 1e-6)),
                    ((p > 0.5) == te["y"].to_numpy()).mean()))
    r = np.mean(res, axis=0)
    print(f"{name:15s} AUC {r[0]:.4f}  logloss {r[1]:.4f}  acc@0.5 {r[2]:.4f}")
