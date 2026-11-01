"""Shared scenario definitions for the Supplementary benchmarks."""
from __future__ import annotations

import os
import platform
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "adapters"))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

from hgtitr import Model, solve_speciation, titration  # noqa: E402

# Manuscript design: 1 uM host (3 mL), 0.1 M guest stock, 30 additions of
# linearly increasing volume 0..10 uL (legacy V_G = seq(0, 0.01, length.out = 30)).
MS_DESIGN = dict(c_cell_H=1e-6, c_titrant_G=1e-1, V0=3.0, V_add=np.linspace(0, 0.01, 30))
# Biliverdin-like design: ~30 uM host, 13 points to 1.3 equivalents (experimental design)
BVD_DESIGN = dict(c_cell_H=3.0e-5, c_titrant_G=3.0e-3, V0=2.0, V_add=np.full(12, 2.2e-3))
# Same concentrations, extended to ~3 equivalents with 24 additions (multichannel benchmark)
MC_DESIGN = dict(c_cell_H=3.0e-5, c_titrant_G=3.0e-3, V0=2.0, V_add=np.full(24, 2.6e-3))
# Tight-binding, near-stoichiometric single-channel design (c = K*H_T = 300)
TIGHT_DESIGN = dict(c_cell_H=3.0e-5, c_titrant_G=1.0e-3, V0=2.0, V_add=np.full(24, 5.0e-3))


def design(d):
    HT, GT = titration(d["c_cell_H"], d["c_titrant_G"], d["V0"], d["V_add"])
    Vc = np.concatenate([[0.0], np.cumsum(d["V_add"])])
    return HT, GT, Vc


SCENARIOS = {
    # name: (species, logbeta, responses over complexes, design, noise_sd, free_H_response)
    "S1_1to1": (["H1G1"], [4.0], [3e5], MS_DESIGN, 1e-3, 0.0),
    "S3_1to2_weak": (["H1G1", "H1G2"], [4.5, 8.0], [2e5, 3e5], MS_DESIGN, 1e-3, 0.0),
    "S4_1to2": (["H1G1", "H1G2"], [5.0, 8.0], [2e5, 3e5], MS_DESIGN, 1e-3, 0.0),
    "S5_tight_1to1": (["H1G1"], [7.0], [2e4], TIGHT_DESIGN, 2e-3, 0.0),
    "S6_1to2_4ch": (["H1G1", "H1G2"], [5.1, 9.5],
                        [[-4.0e5, -2.0e5, -3.2e5, -1.2e5], [-3.0e5, -9.0e5, -2.0e5, 1.5e5]], MC_DESIGN, 0.15, 0.0),
}


def simulate(name, rng, conc_error_sd=0.0):
    """Noisy data for a scenario; optional true titrant-concentration error (relative SD)."""
    sp, lb, resp, d, sd, rH = SCENARIOS[name]
    m = Model(sp)
    HT, GT, Vc = design(d)
    f = 1.0 + conc_error_sd * rng.normal() if conc_error_sd > 0 else 1.0
    C = solve_speciation(HT, GT * f, m, lb).C
    R = np.atleast_2d(np.asarray(resp, float))
    R = R.T if R.shape[0] == 1 and len(sp) > 1 else R
    R = R.reshape(len(sp), -1)
    Y = C[:, 2:] @ R + rH * C[:, [0]]
    Y = Y + rng.normal(0, sd, Y.shape)
    return m, HT, GT, Vc, Y, np.asarray(lb, float), f


def env_info():
    import scipy
    info = dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__,
                platform=platform.platform(), cpu=platform.processor() or "n/a", n_cpu=os.cpu_count())
    try:
        import jax, numpyro  # noqa: E401
        info.update(jax=jax.__version__, numpyro=numpyro.__version__)
    except ImportError:
        pass
    return info
