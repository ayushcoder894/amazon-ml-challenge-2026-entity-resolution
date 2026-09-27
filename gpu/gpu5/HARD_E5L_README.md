# A100 hard-negative e5-large experiment

Upload `hard_e5l_upload.zip` to `/workspace/1_Amazon_Challenge/gpu5/`, then extract there. It contains the new script and this README; the existing `ce5.py` and `common.py` remain in place. The bundle must be at `/workspace/1_Amazon_Challenge/gpu2/bundle`, and `gpu5/data/band_loco.parquet` and `band_france.parquet` must already be present. No external entity data or constructed labels are used.

```bash
cd /workspace/1_Amazon_Challenge/gpu5
python -m zipfile -e hard_e5l_upload.zip .
python -u ce_hard.py --mode loco --selftest
mkdir -p out/logs
python -u ce_hard.py --mode loco --train_limit 60000 --bs 48 --pred_bs 256 > out/logs/hard_e5l_loco.log 2>&1
tail -n 30 out/logs/hard_e5l_loco.log
```

The LOCO output is `out/cehard_hard_e5l_n60000_loco.parquet`. Copy it to `D:/amazon_er/gpu5_results/` on Windows. Run `src/q7_check.py` on that file under `guard.sh`; it accepts any [0,1] auxiliary score despite the historical Qwen wording. Its `eligible_for_v20` flag requires complete coverage and positive final F0.5 deltas in both proxy countries. If false, stop.

From `C:/ML/Challenges/amazon/work/src` on Windows, after copying the result:

```powershell
$env:LIMIT='58'; $env:PYTHONUTF8='1'
& 'C:/Program Files/Git/bin/bash.exe' guard.sh q7_check.py D:/amazon_er/france_night/logs/hard_e5l_check.txt python -u q7_check.py D:/amazon_er/gpu5_results/cehard_hard_e5l_n60000_loco.parquet
```

If positive, run France inference:

```bash
cd /workspace/1_Amazon_Challenge/gpu5
python -u ce_hard.py --mode france --train_limit 120000 --bs 48 --pred_bs 256 > out/logs/hard_e5l_france.log 2>&1
tail -n 30 out/logs/hard_e5l_france.log
```

Copy `out/cehard_hard_e5l_n120000_france_test.parquet` to Windows. The France file covers only `p >= 0.2`, exactly like the prior Qwen experiment. The local `france_v20.py` can assemble it with `--name v22` using the LOCO checker report and then the official validator. This is a new experiment; do not overwrite v19 or v20 outputs.

Use the exact `q7_check_<hash>.json` path printed by the check:

```powershell
& 'C:/Program Files/Git/bin/bash.exe' guard.sh france_v20.py D:/amazon_er/france_night/logs/hard_e5l_build.txt python -u france_v20.py D:/amazon_er/gpu5_results/cehard_hard_e5l_n120000_france_test.parquet D:/amazon_er/gpu5_results/q7_check_<hash>.json --name v22 --assemble
```

Runtime on one A100 40 GB, assuming cached model weights: approximately 45–100 minutes for LOCO (two 60k-pair trainings plus 10% proxy scoring), then 30–70 minutes for France (120k-pair training plus 167k scoring). Allow 15–30 minutes for transfers and local validation. The first training log prints throughput and ETA after 200 batches; stop if that ETA exceeds the remaining deadline. Runtime is an estimate, not a benchmark of this exact script.
