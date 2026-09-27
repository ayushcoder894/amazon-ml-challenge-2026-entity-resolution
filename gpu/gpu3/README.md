# A100 v3 - stronger cross-encoders (≈ 5-6 h)

Unzip next to `gpu2` so that the existing bundle `/workspace/1_Amazon_Challenge/gpu2/bundle` is reused (no rebuild).

```bash
cd /workspace/1_Amazon_Challenge
python -m zipfile -e er_a100_v3.zip .
cd gpu3
nohup bash run_v3.sh > run.log 2>&1 &
```

Progress: `cat out/logs/chainA.txt out/logs/chainB.txt; tail -2 out/logs/*.log`
Every job logs `AUC new ... vs current p ...` (the new model vs today's probability on the same pairs).
A line containing `NOT LEARNING` means that model is stuck at the base rate (tell me; the other jobs continue).

| Job | Model (licence) | What it measures |
|---|---|---|
| e5l / bgem3 / q15 `inc` | multilingual-e5-large (MIT), bge-m3 (MIT), Qwen2.5-1.5B LoRA (Apache-2.0) | India/US uncertain band, 2-fold out-of-fold + test scores |
| `loco` | same | France proxy: trained on one country, scored on the other |
| `france` | same | trained on both countries, scores the France test band |
| q7 `inc` (diagnostic) | Qwen2.5-7B LoRA (Apache-2.0) | subsample only: does a 7B model read names much better? |

When `run.log` says ALL DONE, send back (≈ 100-200 MB, no model weights):
```bash
cd /workspace/1_Amazon_Challenge/gpu3
python -m zipfile -c v3_results.zip out/*.parquet out/*.json out/logs out/SUMMARY.txt
```
