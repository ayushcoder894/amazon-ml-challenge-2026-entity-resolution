#!/bin/bash
# Waits for the A100 test dense files, then builds submission v13_dense (v12 + dense add-on + joint re-scoring + France add-on).
W=/c/ML/Challenges/amazon/work
DN=D:/amazon_er/dense
R=C:/ML/Challenges/amazon/6ab10eb3b23ba_student_resource/student_resource
LOG=$DN/auto_v13.log
echo "$(date +%H:%M:%S) waiting for dense_test files" > $LOG
stable() { a=$(stat -c %s "$1" 2>/dev/null); sleep 20; b=$(stat -c %s "$1" 2>/dev/null); [ -n "$a" ] && [ "$a" = "$b" ]; }
while true; do
  for z in $W/dense_test.zip D:/amazon_er/dense_test.zip $DN/dense_test.zip; do
    if [ -f "$z" ] && stable "$z"; then
      echo "$(date +%H:%M:%S) extracting $z" >> $LOG
      (cd $DN && python -m zipfile -e "$z" .) && mv "$z" "$DN/dense_test.zip.done" 2>/dev/null
    fi
  done
  for f in dense_test_new.parquet dense_test_cos.parquet; do
    for d in $W D:/amazon_er; do
      if [ -f "$d/$f" ] && stable "$d/$f"; then mv "$d/$f" $DN/ && echo "$(date +%H:%M:%S) moved $d/$f" >> $LOG; fi
    done
  done
  if [ -f $DN/dense_test_new.parquet ] && [ -f $DN/dense_test_cos.parquet ] && \
     python -c "import polars as pl; [pl.scan_parquet('$DN/'+f).select(pl.len()).collect() for f in ('dense_test_new.parquet','dense_test_cos.parquet')]" 2>/dev/null; then
    break
  fi
  sleep 30
done
echo "$(date +%H:%M:%S) files ready; building test features" >> $LOG
cd $W/src
PYTHONUTF8=1 FWD=10 REV=1 python -u dense_addon.py feat test >> $LOG 2>&1 || { echo "FEAT FAILED" >> $LOG; exit 1; }
echo "$(date +%H:%M:%S) predicting" >> $LOG
PYTHONUTF8=1 FWD=10 REV=1 JOINT=1 python -u dense_addon.py test v13_dense >> $LOG 2>&1 || { echo "TEST FAILED" >> $LOG; exit 1; }
PYTHONUTF8=1 python $R/utils/validate_submission.py --matching $W/output/matching_results_v13_dense.tsv \
  --candidate $W/output/candidate_pairs_v13_dense.tsv --test-dir $R/dataset/test >> $LOG 2>&1
echo "$(date +%H:%M:%S) ALL DONE" >> $LOG
