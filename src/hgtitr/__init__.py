"""HGtitrManifree: host-guest titration simulation, fitting and uncertainty analysis.

Quick start
-----------
>>> import numpy as np
>>> from hgtitr import Model, Experiment, fit, titration, solve_speciation
>>> HT, GT = titration(1e-6, 1e-1, 3.0, np.linspace(0, 0.01, 30))
>>> m = Model(["H1G1", "H1G2"])
>>> Y = solve_speciation(HT, GT, m, [5, 8]).C[:, 2:] @ [2e5, 3e5]
>>> res = fit(m, Experiment(HT, GT, Y + 1e-3 * np.random.default_rng(0).normal(size=Y.size)), start=[5, 8])
"""
from .design import job_series, titration
from .fit import Experiment, FitResult, bootstrap, compare_models, fit, profile_ci
from .model import Model, parse_species
from .speciation import solve_speciation, species_from_free, totals_from_free

__version__ = "1.0.0"

__all__ = [
    "Model", "parse_species", "Experiment", "FitResult", "fit", "compare_models", "profile_ci", "bootstrap",
    "solve_speciation", "species_from_free", "totals_from_free", "titration", "job_series", "__version__",
]
