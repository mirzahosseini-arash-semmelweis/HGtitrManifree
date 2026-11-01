"""Reproduces the numbers reported in the revised manuscript (experimental data)."""
import os

import numpy as np
import pandas as pd
import pytest

from hgtitr import Experiment, Model, fit, profile_ci
from hgtitr.spectral import SpectralData, SpectralProblem

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def test_meloxicam_one_to_one():
    d = pd.read_csv(os.path.join(DATA, "meloxicam_rabsa.csv"))
    Y = d.iloc[:, 2:].values
    # the two channels (282.9 and 283.0 nm) are effectively one measurement
    assert np.corrcoef(Y.T)[0, 1] > 0.9999
    r = fit(Model(["H1G1"]), Experiment(d.total_H.values, d.total_G.values, Y.mean(1), free_H_active=True),
            start=[5.0])
    assert r.logbeta[0] == pytest.approx(4.906, abs=2e-3)
    assert r.se[0] == pytest.approx(0.030, abs=3e-3)
    lo, hi = profile_ci(r, 0)[:2]
    assert lo < 4.86 and hi > 4.95


def test_biliverdin_split_ridge_is_flat():
    df = pd.read_csv(os.path.join(DATA, "biliverdin_rabsa_fullCD.csv"))
    cols = [c for c in df.columns if c.startswith("CD")]
    wl = np.array([float(c[2:]) for c in cols])
    data = SpectralData(df.total_H.values, df.total_G.values, wl, df[cols].values).subset(305.0, 445.0, step=10)
    pr = SpectralProblem(Model(["H1G1a", "H1G1b"]), data, ["H1G1a", "H1G1b"], n_basis=10)
    tot = 10 ** 5.97
    rss = [pr.resid(np.array([a, np.log10(tot - 10 ** a), 49.0, -37.5])) for a in (2.0, 3.63, 5.0)]
    rss = [r @ r for r in rss]
    assert np.ptp(rss) < 1e-9 * np.mean(rss)
