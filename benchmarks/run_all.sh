#!/bin/bash
# Runs the full Supplementary benchmark suite (several hours on 2 cores).
# bench_03 and bench_04 save partial results and resume if interrupted.
set -e
cd "$(dirname "$0")"
python3 bench_01_tps_accuracy.py      > results/bench01.log 2>&1
python3 bench_02_fit_bias_precision.py > results/bench02.log 2>&1
python3 bench_03_uncertainty_coverage.py > results/bench03.log 2>&1
python3 bench_04_spectral_shift.py    > results/bench04.log 2>&1
python3 make_figures.py
