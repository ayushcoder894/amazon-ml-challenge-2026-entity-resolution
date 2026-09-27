#!/bin/bash
# A100 v6 - score India/US uncertain-band pairs with the already-trained Qwen LoRA models (no training). ~1-2 h.
# Re-running skips finished files. Results: out/ce6_q7_iu_{train,test}_{India,US}.parquet ; log: out/logs/v6_iu.log
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
BUNDLE=${BUNDLE:-/workspace/1_Amazon_Challenge/gpu2/bundle}
mkdir -p "$HERE/out/logs"
export TOKENIZERS_PARALLELISM=true PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ls "$HERE"/out/models/q7_loco_noIndia/.done "$HERE"/out/models/q7_loco_noUS/.done || { echo "trained models missing in out/models"; exit 1; }
python -u "$HERE/score_iu.py" --bundle "$BUNDLE" > "$HERE/out/logs/v6_iu.log" 2>&1 && echo "DONE - send back out/ce6_q7_iu_*.parquet (4 files)" || echo "FAILED - see out/logs/v6_iu.log"
