# Supplementary benchmark suite

All scripts are self-contained Python. They use `hgtitr` plus the two
established open-source tools available in Python:

| Tool | Version used | How it is called |
|---|---|---|
| Musketeer (Soloviev & Hunter, *Chem. Sci.* 2024, 15, 15299) | 1.10.3 (`pip install musketeer`) | `adapters/musketeer_adapter.py`: tkinter GUI modules stubbed; `SpeciationSolver.run` and the `Titration.optimise` protocol (Nelder–Mead on log K, `FitSignalsUnconstrained.leastSquares`) used unchanged |
| Equilibrist (Masson, *J. Chem. Inf. Model.* 2026, 66, 9976) | v2.0.1 (git tag) | `adapters/equilibrist_adapter.py`: script → `parse_script` → `build_network` → `fit_spectra` unchanged; set `EQUILIBRIST_DIR` |

| Script | Question | Output |
|---|---|---|
| `bench_01_tps_accuracy.py` | Error of the legacy thin-plate-spline (TPS) manifold emulator against exact speciation, across stoichiometry, log K, c = K·H_T, grid density and `max_train`; effect on fitted constants | `results/bench01_*.csv` |
| `bench_02_fit_bias_precision.py` | Bias, SD, RMSE, failure rate and wall time of TPS-legacy, exact-full, exact-VARPRO, Musketeer and Equilibrist on identical Monte Carlo datasets | `results/bench02_*.csv` |
| `bench_03_uncertainty_coverage.py` | Coverage of nominal 95 % intervals: Wald, profile likelihood, bootstrap, NUTS (marginal / explicit responses), with and without unknown 3 % stock-concentration error (EIV) | `results/bench03_*.csv` |
| `bench_04_spectral_shift.py` | Synthetic CD spectra with known band displacement vs fixed-spectrum 1:2: recovery of log K and Δλ; AIC/BIC model selection under AR(1) noise | `results/bench04_spectral.csv` |
| `make_figures.py` | Summary tables and figures from the CSVs | `results/*.png`, `results/summary.md` |

`common.py` defines the scenarios (S1, S3, S4 = manuscript scenarios 1, 3 and 4
with the legacy design; S5 = tight near-stoichiometric 1:1; S6 = 4-channel 1:2).
Every replicate uses a fixed seed. `--quick` runs a smoke test.

Hardware for the reported numbers: 2-core x86-64 Linux VM, Python 3.13,
NumPy 2.5, SciPy 1.18, JAX 0.12, NumPyro 0.22 (printed at the top of each log).
