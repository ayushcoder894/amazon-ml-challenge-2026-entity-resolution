#!/bin/bash
# A100 v9: bagged density-ratio decoy estimator. ~10-20 min. Needs: python, polars, xgboost>=2 (cuda), torch (cuda).
cd "$(dirname "$0")"
mkdir -p out
python -u dr_ens.py --data data --out out --bags 16 --mlp_seeds 5 2>&1 | tee out/log.txt
