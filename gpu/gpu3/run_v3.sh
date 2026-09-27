#!/bin/bash
# A100 v3 - stronger cross-encoders on the uncertain band (India/US in-country, France proxy, France test).
#   chain A (encoders, full fine-tune):  e5-large  inc -> loco -> france,  then bge-m3 inc -> loco -> france
#   chain B (LLM, LoRA):                 Qwen2.5-1.5B inc -> loco -> france
#   A and B run in parallel (~25-30 GB GPU memory together); then, alone on the GPU:
#   chain C (diagnostic, RUN_7B=0 to skip): Qwen2.5-7B LoRA, subsampled - does a 7B model read names much better?
# Re-running skips finished steps. Results: out/*.json (AUCs), out/ce3_*.parquet (scores), out/logs/*.log
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
BUNDLE=${BUNDLE:-/workspace/1_Amazon_Challenge/gpu2/bundle}
OUT=${OUT:-$HERE/out}
RUN_7B=${RUN_7B:-1}
mkdir -p "$OUT/logs"
export TOKENIZERS_PARALLELISM=true PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f "$BUNDLE/records_train.parquet" ] || { echo "bundle not found at $BUNDLE (set BUNDLE=...)"; exit 1; }

run() {  # tag mode args...
  local tag=$1 mode=$2; shift 2
  if [ -f "$OUT/ce3_${tag}_${mode}.json" ]; then echo "$(date +%H:%M:%S) skip $tag $mode (done)"; return; fi
  echo "$(date +%H:%M:%S) start $tag $mode"
  python -u "$HERE/ce3.py" --mode "$mode" --bundle "$BUNDLE" --out "$OUT" --tag "$tag" "$@" > "$OUT/logs/${tag}_${mode}.log" 2>&1 \
    && echo "$(date +%H:%M:%S) done  $tag $mode" || echo "$(date +%H:%M:%S) FAILED $tag $mode (see $OUT/logs/${tag}_${mode}.log)"
}
chainA() {
  for m in inc loco france; do run e5l "$m" --model intfloat/multilingual-e5-large --lr 1e-5 --bs 64 --pred_bs 512 --train_limit 800000; done
  for m in inc loco france; do run bgem3 "$m" --model BAAI/bge-m3 --lr 1e-5 --bs 64 --pred_bs 512 --train_limit 800000; done
}
chainB() {
  for m in inc loco france; do run q15 "$m" --model Qwen/Qwen2.5-1.5B --lora --lr 1e-4 --bs 16 --pred_bs 128 --max_len 160 --train_limit 200000; done
}
chainA > "$OUT/logs/chainA.txt" 2>&1 &
chainB > "$OUT/logs/chainB.txt" 2>&1 &
wait
if [ "$RUN_7B" = "1" ]; then
  run q7 inc --model Qwen/Qwen2.5-7B --lora --lr 1e-4 --bs 8 --grad_accum 2 --pred_bs 64 --max_len 160 \
    --train_limit 60000 --score_limit 40000 --no_test > "$OUT/logs/chainC.txt" 2>&1
fi
{ echo "=== SUMMARY $(date)"; cat "$OUT"/logs/chain*.txt; grep -h "AUC\|FAILED\|Traceback\|Error\|NOT LEARNING" "$OUT"/logs/*.log | grep -v "^ *batch"; } > "$OUT/SUMMARY.txt"
echo "ALL DONE - send back the folder $OUT without models/ (python -m zipfile -c v3_results.zip out/*.parquet out/*.json out/logs out/SUMMARY.txt)"
