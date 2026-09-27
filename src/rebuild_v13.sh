#!/bin/bash
# Rebuild the v13 files with complete candidate lists (every predicted pair listed), validating each.
cd /c/ML/Challenges/amazon/work/src
R=C:/ML/Challenges/amazon/6ab10eb3b23ba_student_resource/student_resource
LOG=D:/amazon_er/dense/rebuild_v13.log
val() { PYTHONUTF8=1 python $R/utils/validate_submission.py --matching ../output/matching_results_$1.tsv \
          --candidate ../output/candidate_pairs_$1.tsv --test-dir $R/dataset/test 2>&1 | grep -E "S1 entity|PASS|FAIL|ERROR" | sed "s/^/  [$1] /" >> $LOG; }
echo "$(date +%H:%M:%S) start" > $LOG
for v in 3:v13_ce 5:v13_ceu; do
  jv=${v%%:*}; name=${v#*:}
  PYTHONUTF8=1 FWD=10 REV=1 JOINT=1 JOINT_V=$jv python -u dense_addon.py test $name 2>&1 | grep -E "wrote|France:|test:" >> $LOG || echo "FAILED $name" >> $LOG
  val $name
  PYTHONUTF8=1 python -u us_near_v13.py US $name ${name}_USnear 2>&1 | grep -E "^US:" >> $LOG || echo "FAILED ${name}_USnear" >> $LOG
  val ${name}_USnear
done
echo "$(date +%H:%M:%S) ALL DONE" >> $LOG
