#!/bin/bash
# A100 v5 - LLM pair classifier: Qwen2.5-7B-Instruct (Apache-2.0), LoRA, P(same business) on the uncertain pairs.
#   1. loco  : train on US uncertain pairs -> score 10% of India's entities; train on India -> score 10% of US
#              (the France proxy: does the LLM help a country it never saw?)                         ~1-1.5 h
#   2. france: train on India + US -> score all 389k uncertain France test pairs                       ~1.5 h
# Re-running skips finished steps. Results: out/ce5_*.parquet, out/ce5_*.json, out/logs/*.log
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
BUNDLE=${BUNDLE:-/workspace/1_Amazon_Challenge/gpu2/bundle}
OUT=${OUT:-$HERE/out}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
mkdir -p "$OUT/logs"
export TOKENIZERS_PARALLELISM=true PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f "$BUNDLE/records_train.parquet" ] || { echo "bundle not found at $BUNDLE (set BUNDLE=...)"; exit 1; }
COMMON="--model $MODEL --tag q7 --lora --no_ckpt --bs 8 --grad_accum 2 --pred_bs 64 --max_len 128 --lr 1e-4"
run() {
  local mode=$1; shift
  if [ -f "$OUT/ce5_q7_${mode}.json" ]; then echo "$(date +%H:%M:%S) skip $mode (done)"; return; fi
  echo "$(date +%H:%M:%S) start $mode"
  python -u "$HERE/ce5.py" --mode "$mode" --bundle "$BUNDLE" --out "$OUT" $COMMON "$@" > "$OUT/logs/q7_${mode}.log" 2>&1 \
    && echo "$(date +%H:%M:%S) done  $mode" || echo "$(date +%H:%M:%S) FAILED $mode (see $OUT/logs/q7_${mode}.log)"
}
run loco --train_limit 40000 --eval_frac 0.1
run france --train_limit 60000
grep -h "AUC\|FAILED\|Traceback\|Error\|NOT LEARNING" "$OUT"/logs/*.log | grep -v "^ *batch" > "$OUT/SUMMARY.txt"
echo "ALL DONE - send back: out/ce5_q7_loco.parquet out/ce5_q7_france_test.parquet (and out/SUMMARY.txt)"
