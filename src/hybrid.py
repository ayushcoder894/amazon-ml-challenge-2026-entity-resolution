"""v4 submission: per-country model choice.

France (never seen in training) -> the transfer-robust "drop" model (no vocabulary-dependent features);
India / US -> the full v3 model (best in-country). Each country's matching is independent (candidates never
cross countries), so the decisions can be mixed cleanly. Each model applies its own tuned decision rule.
"""
import json
import os
import subprocess
import sys

import polars as pl

import decide
import predict3
from train3 import D, M

OUT = predict3.OUT
RES = predict3.RES
PLAN = {"France": "_v3drop", "India": "_v3", "US": "_v3"}


def decisions(pred, rule):
    oo = pred.select("rid", "oid", "p2")
    if rule[0] == "t":
        return oo.filter((pl.col("p2").rank("ordinal", descending=True).over("oid") == 1) & (pl.col("p2") >= rule[1]))
    return decide.ef_predict(oo, "p2", rule[1], rule[2], 0.05)


def main(plan=PLAN, name="v4"):
    picks, allpairs = [], []
    for mname in sorted(set(plan.values())):
        cs = [c for c, m in plan.items() if m == mname]
        path = f"{D}test_pred{mname}.parquet"
        pred = pl.read_parquet(path) if os.path.exists(path) else predict3.main("_v3", mname=mname, write=False)
        pred = pred.filter(pl.col("country").is_in(cs))
        rule = json.load(open(f"{M}meta{mname}.json"))["rule"]
        d = decisions(pred, rule)
        print(f"{mname} for {cs}: rule {rule}, {d.height:,} matched pairs")
        picks.append(d.select("rid", "oid"))
        allpairs.append(pred.select("rid", "oid"))
    best, K = pl.concat(picks), pl.concat(allpairs)
    s1 = pl.read_parquet(f"{D}test_s1_norm.parquet", columns=["rid", "entity_id"])
    o = pl.concat([pl.read_parquet(f"{D}test_s{k}_norm.parquet", columns=["entity_id"]) for k in (2, 3)]
                  ).with_row_index("oid")
    ids = lambda x: (x.join(s1.rename({"entity_id": "s1"}), on="rid").join(o.rename({"entity_id": "m"}), on="oid"))
    match = ids(best).group_by("s1").agg(pl.col("m").sort().str.join(","))
    cand = ids(K).group_by("s1").agg(pl.col("m").unique().sort().str.join(","))
    base = s1.select(source1_entity_id="entity_id")
    mpath, cpath = f"{OUT}matching_results_{name}.tsv", f"{OUT}candidate_pairs_{name}.tsv"
    (base.join(match.rename({"s1": "source1_entity_id", "m": "matched_entity_ids"}), on="source1_entity_id", how="left")
     .fill_null("").write_csv(mpath, separator="\t", quote_style="never"))
    (base.join(cand.rename({"s1": "source1_entity_id", "m": "candidate_entity_ids"}), on="source1_entity_id", how="left")
     .fill_null("").write_csv(cpath, separator="\t", quote_style="never"))
    print(f"wrote {mpath}: {best.height:,} matched pairs")
    subprocess.run([sys.executable, f"{RES}utils/validate_submission.py", "--matching", mpath, "--candidate", cpath,
                    "--test-dir", f"{RES}dataset/test"])


if __name__ == "__main__":
    main()
