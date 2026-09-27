#!/bin/bash
# Memory guard for long jobs: guard.sh <pattern> <log> <command...>
#  - refuses to start if a python process whose command line contains <pattern> is already running
#  - kills that process if the committed memory of the machine exceeds LIMIT GB (default 56; limit is ~69.5)
LIMIT=${LIMIT:-56}
PAT=$1; LOG=$2; shift 2
running=$(powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*$PAT*' }).Count")
if [ "${running:-0}" -gt 0 ]; then echo "GUARD: '$PAT' already running ($running) - not starting" | tee -a "$LOG"; exit 3; fi
"$@" >> "$LOG" 2>&1 &
BG=$!
peak=0
while kill -0 $BG 2>/dev/null; do
  c=$(powershell -NoProfile -Command "[int]((Get-Counter '\Memory\Committed Bytes').CounterSamples[0].CookedValue/1GB)")
  [ "${c:-0}" -gt "$peak" ] && peak=$c
  if [ "${c:-0}" -gt "$LIMIT" ]; then
    echo "GUARD: committed ${c} GB > ${LIMIT} GB - killing '$PAT'" | tee -a "$LOG"
    powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*$PAT*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
    wait $BG; exit 2
  fi
  sleep 8
done
wait $BG; rc=$?
echo "GUARD: done rc=$rc, peak committed ${peak} GB" | tee -a "$LOG"
exit $rc
