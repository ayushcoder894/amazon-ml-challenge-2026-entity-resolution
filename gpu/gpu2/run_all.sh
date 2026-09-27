#!/usr/bin/env bash
# A100 experiments toward 0.995 per country (about 3-3.5 h sequential on one A100 40GB; see README_A100.md).
#   DATA=/path/to/dataset bash run_all.sh               sequential (default)
#   PARALLEL=1 DATA=... bash run_all.sh                 E1 and E2a share the GPU (both fit in 40 GB), then E2b
#   RUN_7B=1 ...                                        also the optional Qwen2.5-7B measurement (+ ~1.5 h)
# Re-running skips finished steps. Results: $OUT/SUMMARY.txt, $OUT/reports/*.json, $OUT/logs/*.log
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
BUNDLE=${BUNDLE:-$HERE/bundle}
OUT=${OUT:-$HERE/gpu2_out}
mkdir -p "$OUT/logs" "$OUT/reports"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; print('GPU:', torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory // 2**30, 'GB')" || exit 1
if [ ! -f "$BUNDLE/rule.json" ]; then
  : "${DATA:?set DATA to the dataset folder that contains train/ and test/}"
  python build_bundle.py --data "$DATA" --out "$BUNDLE" 2>&1 | tee "$OUT/logs/build_bundle.log" || exit 1
fi

E1() {   # dense retrieval: out-of-fold recall + ceiling on the harder split (~1 h incl. test retrieval later)
  python exp_dense.py --bundle "$BUNDLE" --out "$OUT" --model intfloat/multilingual-e5-base \
    --max_pairs 1500000 --bs 256 --skip_test > "$OUT/logs/E1_dense.log" 2>&1
}
E2a() {  # bge-m3 (MIT, 568M) full fine-tune, rival context (~40 min)
  python exp_rerank.py --bundle "$BUNDLE" --out "$OUT" --model BAAI/bge-m3 --tag bgem3 \
    --max_len 320 --bs 16 --grad_accum 2 --pred_bs 256 --lr 2e-5 > "$OUT/logs/E2a_bgem3.log" 2>&1
}
E2b() {  # Qwen2.5-1.5B (Apache-2.0) + LoRA, rival context (~1.5 h)
  python exp_rerank.py --bundle "$BUNDLE" --out "$OUT" --model Qwen/Qwen2.5-1.5B --tag qwen15 --lora --grad_ckpt \
    --train_limit 120000 --max_len 320 --bs 16 --grad_accum 2 --pred_bs 128 --lr 1e-4 > "$OUT/logs/E2b_qwen15.log" 2>&1
}
E2c() {  # optional: Qwen2.5-7B (Apache-2.0) + LoRA, measurement only (no test scoring) (~1.5 h)
  python exp_rerank.py --bundle "$BUNDLE" --out "$OUT" --model Qwen/Qwen2.5-7B --tag qwen7 --lora --grad_ckpt \
    --train_limit 40000 --max_len 320 --bs 8 --grad_accum 4 --pred_bs 64 --lr 1e-4 --skip_test > "$OUT/logs/E2c_qwen7.log" 2>&1
}
E1test() {  # dense candidates for test (only needed if E1 shows a gain; ~25 min)
  python exp_dense.py --bundle "$BUNDLE" --out "$OUT" --model intfloat/multilingual-e5-base > "$OUT/logs/E1_dense_test.log" 2>&1
}
step() { echo "=== $(date +%H:%M:%S) $1"; }

# multi-GPU servers: ONLY=<E1|E2a|E2b|E2c|E1test|build> runs a single step, e.g.
#   ONLY=build bash run_all.sh ; ONLY=E1 CUDA_VISIBLE_DEVICES=0 bash run_all.sh & ONLY=E2a CUDA_VISIBLE_DEVICES=1 bash run_all.sh &
if [ -n "${ONLY:-}" ]; then
  if [ "$ONLY" != "build" ]; then step "$ONLY"; $ONLY; fi
  exit 0
fi

if [ "${PARALLEL:-0}" = "1" ]; then
  step "E1 dense retrieval + E2a bge-m3 in parallel"
  E1 & P1=$!
  E2a & P2=$!
  wait $P1; wait $P2
else
  step "E1 dense retrieval"; E1
  step "E2a bge-m3 reranker"; E2a
fi
step "E2b Qwen2.5-1.5B reranker"; E2b
if [ "${RUN_7B:-0}" = "1" ]; then step "E2c Qwen2.5-7B reranker (measurement)"; E2c; fi
if [ "${SKIP_DENSE_TEST:-0}" != "1" ]; then step "E1 dense retrieval for test"; E1test; fi

{ echo "=== SUMMARY $(date)"; grep -h -E "RESULT|recall .* ceiling|in-batch acc|Traceback|Error" "$OUT"/logs/*.log | grep -vE "step [0-9]+/" ; } > "$OUT/SUMMARY.txt"
grep -h -E "in-batch acc" "$OUT/logs/E1_dense.log" | tail -2 >> "$OUT/SUMMARY.txt"
cat "$OUT/SUMMARY.txt"
step "ALL DONE - send back: $OUT/SUMMARY.txt, $OUT/reports/, $OUT/logs/, $OUT/rerank_*_test.parquet, $OUT/rerank_*_oof.parquet"
