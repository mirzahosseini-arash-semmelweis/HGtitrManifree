# HGtitrManifree (Python)

Simulation, global fitting, model comparison and uncertainty analysis for
two-component host–guest titrations, plus a full-spectrum model with
occupancy-dependent spectral shifts.

Version 1.0.0 replaces the numbered R scripts of the first submission
(kept unchanged in [`legacy_R/`](legacy_R/)). It is a pip-installable
Python package with unit tests, a worked example, a reproduction script for
the experimental analyses and the Supplementary benchmark suite.

## Install

```bash
pip install -e ".[bayes,dev]"      # core needs numpy/scipy/pandas; [bayes] adds jax + numpyro
pytest -q                          # 19 tests (MCMC test: HGTITR_SLOW=1)
python examples/quickstart.py      # simulate -> fit -> AICc -> profile CI -> NUTS
```

## What is in the package

| Module | Content |
|---|---|
| `hgtitr.model` | `Model(["H1G1", "H1G2", "H2G1", "H1G1a", ...])`: stoichiometry parsed from labels; suffixes give same-stoichiometry states |
| `hgtitr.speciation` | **Exact inverse map** (totals → free): Newton's method with Armijo backtracking on the strictly convex potential F(x) = Σ β_j exp(s_j·x) − T·x. It converges globally from any start, is vectorised over points and solves 30 points in about 1 ms |
| `hgtitr.jax_speciation` | The same solver in JAX, with derivatives from the implicit-function theorem (exact and cheap) for NUTS |
| `hgtitr.forward` | **Free-component manifold** (the original idea): explicit forward map (h, g) → (H_T, G_T, C, signal); faithful port of the legacy thin-plate-spline emulator (`tps_species`, `tps_signal`) so that its error can be measured; `sensitivity_map`, which gives exact ∂signal/∂log β over the whole (H_T, G_T) plane with no inverse solve, for experiment design |
| `hgtitr.observables` | Linear observation models: additive intensity (UV-vis, CD, fluorescence) and **fast-exchange NMR shifts** δ_obs = Σ a_j[S_j]δ_j / H_T (or / G_T) |
| `hgtitr.fit` | Global fitting of any number of titrations and channels with shared constants; `method="varpro"` (responses profiled by linear least squares), `"full"` (responses explicit, legacy) or `"varpro_nm"` (derivative-free); multistart; ML per-channel noise; optional stock-concentration nuisance factors (errors in variables); Gauss–Newton covariance from the projected Jacobian; AIC/AICc/BIC; profile-likelihood CIs; residual and parametric bootstrap |
| `hgtitr.bayes` | NUTS (NumPyro) from the optimum. Response treatments: `explicit` (legacy joint posterior), `marginal` (responses integrated out analytically), `eiv` (marginal plus priors on stock-concentration factors) |
| `hgtitr.spectral` | Full-spectrum model with B-spline species spectra and shifts Δλ_ij = q_i Δλ_max,j. **q is defined explicitly** (`bound_host`: fraction of host bound; `saturation`: legacy definition, identical for 1:1). iid or AR(1) noise (feasible GLS); synthetic band generator |

## Minimal use

```python
import numpy as np
from hgtitr import Model, Experiment, fit, titration, compare_models, profile_ci

HT, GT = titration(c_cell_H=1e-6, c_titrant_G=0.1, V0=3.0, V_add=np.linspace(0, 0.01, 30))
exp = Experiment(HT, GT, Y)                         # Y: (n_points, n_channels)
r12 = fit(Model(["H1G1", "H1G2"]), exp, start=[5, 8], n_starts=8)
print(r12.summary(), profile_ci(r12, 0)[:2])
```

NMR fast exchange: `Experiment(HT, GT, shifts, kind="shift_H")`.
Free-host emission (e.g. Trp quenching): `Experiment(..., free_H_active=True)`.
Several titrations: `fit(model, [exp1, exp2, ...])`.
Uncertain titrant stock: `Experiment(..., titrant_factor_sd=np.log10(1.03))`.

## Reproducing the manuscript

```bash
python examples/analysis_experimental.py   # meloxicam + biliverdin: hgtitr, Musketeer, Equilibrist, NUTS, spectra
cd benchmarks
python bench_01_tps_accuracy.py            # spline-emulator error vs exact speciation
python bench_02_fit_bias_precision.py      # bias / precision / cost vs Musketeer, Equilibrist
python bench_03_uncertainty_coverage.py    # coverage of Wald, profile, bootstrap, NUTS (incl. EIV)
python bench_04_spectral_shift.py          # synthetic band-displacement validation
```

Benchmark adapters call **Musketeer** (`pip install musketeer`; its GUI
modules are stubbed, while its speciation solver and Nelder–Mead/least-squares
protocol are used unchanged) and **Equilibrist**
(`git clone https://github.com/masson-group-ohiouniversity/Equilibrist`,
then set `EQUILIBRIST_DIR`; `fit_spectra` is called unchanged).

## Scope and limitations

* Two components (host, guest) only. There are no competitors, protons/pH or
  metal ions, so no displacement assays or pH-metric titrations. For those,
  use HYPERQUAD/HypSpec, Musketeer or Equilibrist. Upgrade coming soon.
* Observables must be linear in the response parameters (intensities and
  fast-exchange shifts). There is no ITC, no slow exchange and no kinetics.
* No graphical interface.
