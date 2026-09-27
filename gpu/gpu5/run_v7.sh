#!/bin/bash
# A100 v7 - LLM group judgment (Qwen2.5-7B-Instruct, no training) on US nearby-number groups: train 1,500 groups first, then test.
HERE=$(cd "$(dirname "$0")" && pwd); mkdir -p "$HERE/out/logs"
python -u "$HERE/group_llm.py" "$@" > "$HERE/out/logs/v7_group.log" 2>&1 && echo DONE || echo "FAILED - see out/logs/v7_group.log"
