# A100 experiments: where can the last 0.005–0.01 per country come from?

Current pipeline (honest 2-fold, harder test-like split): **India 0.98601, US 0.98863**. Every experiment prints
these baseline numbers again, computed by the same code, next to its own result.

Two limits were measured locally:
- **The candidate set caps a perfect matcher** at India 0.9901 / US 0.9944, because 3.1% / 1.8% of true pairs are
  never candidates.
- **The largest matcher error** is records that several same-name entities compete for. The current models never
  see the rival entities together.

| Exp | What it tests | Model (licence) | Time on A100 40GB |
|---|---|---|---|
| **E1** | Dense retrieval: how much of the candidate ceiling a fine-tuned bi-encoder recovers (recall + perfect-matcher ceiling, per country) | `intfloat/multilingual-e5-base` (MIT, 278M) | ~40 min + ~25 min test retrieval |
| **E2a** | Stronger cross-encoder that also sees the entity's other records **and the rival entities** for the record | `BAAI/bge-m3` (MIT, 568M), full fine-tune | ~40 min |
| **E2b** | Same with an LLM reranker | `Qwen/Qwen2.5-1.5B` (Apache-2.0) + LoRA | ~1.5 h |
| E2c (optional) | Same with a 7B LLM, measurement only | `Qwen/Qwen2.5-7B` (Apache-2.0) + LoRA | ~1.5 h |

The default run (E1 + E2a + E2b + E1 test) takes **about 3–3.5 h**. With `RUN_7B=1` it's about 5 h. The times
are estimates and every step logs its own ETA.

All evaluation is honest:
- 2 folds by Source-1 entity; each model scores only the fold it didn't train on.
- The combiner is fitted on one fold and applied to the other.
- The decision rule and the macro-F0.5 metric are identical to the local pipeline's.

## Run
```bash
unzip er_a100_v2.zip && cd gpu2
pip install -r requirements.txt                     # keeps the box's own CUDA torch
DATA=/path/to/dataset nohup bash run_all.sh > run.log 2>&1 &   # DATA = folder containing train/ and test/
tail -f run.log gpu2_out/logs/*.log
```
- The first step checks that the dataset's ids match the local pipeline (md5), then builds `bundle/` (~5 min).
- Models download from Hugging Face on first use: e5-base 1.1 GB, bge-m3 2.3 GB, Qwen 1.5B 3 GB (7B 15 GB).
- Re-running the same command skips finished steps.

## Running experiments at the same time
- **One A100 40GB:** `PARALLEL=1 DATA=... bash run_all.sh` runs E1 and E2a together (~15 + ~15 GB), then E2b alone.
  It saves roughly 30–40 min. Don't run E2b or E2c next to anything else: they need most of the GPU memory and
  compute, and sharing makes both slower.
- **Several GPUs:** build the bundle once, then give each experiment its own GPU:
  ```bash
  ONLY=build DATA=/path/to/dataset bash run_all.sh
  ONLY=E1  CUDA_VISIBLE_DEVICES=0 bash run_all.sh > e1.log 2>&1 &
  ONLY=E2a CUDA_VISIBLE_DEVICES=1 bash run_all.sh > e2a.log 2>&1 &
  ONLY=E2b CUDA_VISIBLE_DEVICES=2 bash run_all.sh > e2b.log 2>&1 &
  ```
  Run `ONLY=E1test` after E1 has finished.

## What to watch
- `gpu2_out/logs/E1_dense.log`:
  - `in-batch acc` should climb above ~0.95 during training.
  - Then comes one line per candidate configuration, e.g.
    `current + fwd10 + rev3  India: recall … ceiling … +x/S1`. **The ceiling numbers are the headline.** If India
    rises well above 0.9901, candidate generation is the lever.
- `gpu2_out/logs/E2*_*.log`: the `RESULT` lines.
  - `f05_current` is the baseline.
  - `f05_combined` is the baseline plus the new model. **This is the headline.**
  - The band AUCs show how much better the new model ranks the hard pairs.

## Send back
1. `gpu2_out/SUMMARY.txt`, `gpu2_out/reports/`, `gpu2_out/logs/` (small)
2. `gpu2_out/rerank_*_oof.parquet` and `gpu2_out/rerank_*_test.parquet` (a few MB each)
3. Keep `gpu2_out/dense_*.parquet` on the box (hundreds of MB). I'll ask for them only if E1 shows a gain.
