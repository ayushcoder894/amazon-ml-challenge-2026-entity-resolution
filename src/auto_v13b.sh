#!/bin/bash
# After auto_v13.sh (v13_dense) finishes: cross-encoder on the uncertain test new pairs -> v13_ce (joint v3),
# then the US near-number probe variant on top of it. Each output is validated.
W=/c/ML/Challenges/amazon/work
DN=D:/amazon_er/dense
R=C:/ML/Challenges/amazon/6ab10eb3b23ba_student_resource/student_resource
LOG=$DN/auto_v13b.log
echo "$(date +%H:%M:%S) waiting for v13_dense" > $LOG
until grep -q "ALL DONE\|FAILED" $DN/auto_v13.log; do sleep 20; done
grep -q "FAILED" $DN/auto_v13.log && { echo "v13_dense failed, stopping" >> $LOG; exit 1; }
cd $W/src
echo "$(date +%H:%M:%S) cross-encoder on test uncertain new pairs" >> $LOG
PYTHONUTF8=1 TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=../.smoke_libs:. python -u ce_dense.py test >> $LOG 2>&1 || { echo "CE FAILED" >> $LOG; exit 1; }
echo "$(date +%H:%M:%S) v13_ce" >> $LOG
PYTHONUTF8=1 FWD=10 REV=1 JOINT=1 JOINT_V=3 python -u dense_addon.py test v13_ce >> $LOG 2>&1 || { echo "V13_CE FAILED" >> $LOG; exit 1; }
PYTHONUTF8=1 python $R/utils/validate_submission.py --matching $W/output/matching_results_v13_ce.tsv \
  --candidate $W/output/candidate_pairs_v13_ce.tsv --test-dir $R/dataset/test >> $LOG 2>&1
echo "$(date +%H:%M:%S) US near-number probe on v13_ce" >> $LOG
PYTHONUTF8=1 python -u us_near_v13.py US v13_ce v13_ce_USnear >> $LOG 2>&1 || { echo "USNEAR FAILED" >> $LOG; exit 1; }
echo "$(date +%H:%M:%S) ALL DONE" >> $LOG
