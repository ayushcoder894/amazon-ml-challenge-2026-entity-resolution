#!/bin/bash
# Waits for the bge-m3 France test scores, then builds France v18 and submission v18 (+ US probe), validated.
W=/c/ML/Challenges/amazon/work
R=D:/amazon_er/gpu3_results
LOG=D:/amazon_er/france_night/logs/auto_v18.txt
echo "$(date +%H:%M:%S) waiting for ce3_bgem3_france_test.parquet" > $LOG
f=$W/ce3_bgem3_france_test.parquet
until [ -f "$f" ] && a=$(stat -c %s "$f") && sleep 20 && [ "$a" = "$(stat -c %s "$f")" ] && python -c "import polars as pl; pl.read_parquet('$f')" 2>/dev/null; do sleep 20; done
mv "$f" $R/ && echo "$(date +%H:%M:%S) file ready" >> $LOG
cd $W/src
LIMIT=58 PYTHONUTF8=1 bash guard.sh "france_v18.py france" $LOG python -u france_v18.py france || { echo "FRANCE FAILED" >> $LOG; exit 1; }
FR_PAIRS=france_v18_pairs PYTHONUTF8=1 python -u build_v15.py v13_e9 v18 >> $LOG 2>&1 || { echo "BUILD FAILED" >> $LOG; exit 1; }
PYTHONUTF8=1 python -u us_near_v13.py US v18 v18_USnear >> $LOG 2>&1 || { echo "USNEAR FAILED" >> $LOG; exit 1; }
echo "$(date +%H:%M:%S) ALL DONE" >> $LOG
