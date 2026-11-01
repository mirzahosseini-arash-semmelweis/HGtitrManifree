# Legacy R scripts

These are the original numbered R scripts (`01_…` to `07b_…`) that
produced the results of the original work. They are archived
unmodified for provenance and are no longer maintained. The Python package
in `src/hgtitr` supersedes them:

| Legacy script | Python replacement |
|---|---|
| `01_HGtitr_simulator.R` (`HG_titr_matrix`, `interpolate_HG_Tps`, `calibrate_corner`) | `hgtitr.forward.free_manifold`, `tps_species`; exact inverse in `hgtitr.speciation` |
| `02_HGtitr_fitter.R`, `04_HGtitr_multichannel_fitter.R` | `hgtitr.fit.fit` (`method="full"` reproduces the explicit-response treatment) |
| `03_…`, `05_…_uncertainty_estimator.R` (BayesianTools DEzs) | `hgtitr.bayes.sample_posterior` (NUTS, `treatment="explicit"` or `"marginal"`) |
| `06_HGtitr_spectral_shift_fitter.R`, `07_…` | `hgtitr.spectral` (explicit occupancy coordinate, AR(1) GLS) |
| `03b_…`, `05b_…`, `07b_…` | `benchmarks/`, `examples/analysis_experimental.py` |

Known issues fixed in the Python version:
* The simulated "data" in `03b` were generated with the same thin-plate-spline
  emulator used for fitting, so emulator error could not appear in the recovery tests.
* `fields::Tps` smoothing (GCV) and thinning to `max_train` points add
  interpolation error of up to several % of the signal range (benchmark 1).
* MCMC prior boxes of ±1.5–2 log units around the optimum can truncate
  non-identified directions (e.g. the H1G1a/H1G1b split, which is exactly
  non-identifiable).
