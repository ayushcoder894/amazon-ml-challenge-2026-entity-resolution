#!/bin/bash
# A100 v8 - test-reweighted retraining of the India/US final model (256 cores for pattern labelling, GPU for XGBoost). ~15-25 min.
HERE=$(cd "$(dirname "$0")" && pwd); mkdir -p "$HERE/out"
python -c "import polars, xgboost, rapidfuzz; print('deps ok', polars.__version__, xgboost.__version__)" || pip install -q polars xgboost rapidfuzz
python -u "$HERE/reweight.py" --data "$HERE/data" --out "$HERE/out" > "$HERE/out/v8.log" 2>&1 && echo "DONE - send back gpu8/out/test_union_p_rw.parquet, test_union_p_unw.parquet, report.txt" || echo "FAILED - see gpu8/out/v8.log"
