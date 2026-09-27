"""Pick the best honest version that has a validated test file, promote it to output/matching_results.tsv, repackage."""
import json
import os
import re
import shutil
import subprocess
import sys

W = "C:/ML/Challenges/amazon/work/"
MC = W + "models_ce/"


def honest(log):
    try:
        s = open(W + "logs_v10/" + log, encoding="utf8", errors="ignore").read()
        v = re.findall(r"\[pf\] HONEST macro F0.5 = ([0-9.]+)", s)
        return float(v[-1]) if v else None
    except FileNotFoundError:
        return None


cands = {  # name -> (honest score, test file suffix)
    "v10ce": (0.98783, "v10ce"),
    "v10full": (0.98798, "v10full"),
    "v10ctx": (honest("ctx_multi_train.log"), "v10ctx"),
    "v10ctx_twin": (honest("final2_train.log"), "v10ctx_twin"),
}
ok = {k: v for k, v in cands.items() if v[0] is not None and os.path.exists(f"{W}output/matching_results_{v[1]}.tsv")}
best = max(ok, key=lambda k: ok[k][0])
print("candidates:", {k: v[0] for k, v in ok.items()}, "-> best", best)
suf = ok[best][1]
shutil.copy(f"{W}output/matching_results_{suf}.tsv", f"{W}output/matching_results.tsv")
shutil.copy(f"{W}output/candidate_pairs_{suf}.tsv", f"{W}output/candidate_pairs.tsv")
res = "C:/ML/Challenges/amazon/6ab10eb3b23ba_student_resource/student_resource/"
subprocess.run([sys.executable, res + "utils/validate_submission.py", "--matching", W + "output/matching_results.tsv",
                "--candidate", W + "output/candidate_pairs.tsv", "--test-dir", res + "dataset/test"])
with open(W + "NIGHT_LOG.md", "a", encoding="utf8") as fh:
    fh.write(f"\n**Final pick ({best}, honest {ok[best][0]:.5f}) promoted to output/matching_results.tsv.** "
             f"Candidates: " + ", ".join(f"{k} {v[0]:.5f}" for k, v in ok.items()) + "\n")
subprocess.run([sys.executable, W + "package.py", "TEAM"])
