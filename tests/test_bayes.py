import os

import numpy as np
import pytest

pytest.importorskip("numpyro")
SLOW = pytest.mark.skipif(not os.environ.get("HGTITR_SLOW"), reason="set HGTITR_SLOW=1 to run MCMC tests")

from hgtitr import Experiment, Model, fit, solve_speciation, titration  # noqa: E402


@SLOW
def test_marginal_matches_explicit_and_wald():
    from hgtitr.bayes import sample_posterior

    HT, GT = titration(1e-6, 1e-1, 3.0, np.linspace(0, 0.01, 30))
    m = Model(["H1G1"])
    y = solve_speciation(HT, GT, m, [4.0]).C[:, 2] * 3e5
    y = y + np.random.default_rng(3).normal(0, 1e-3, y.size)
    r = fit(m, Experiment(HT, GT, y), start=[4.0])
    pm = sample_posterior(r, "marginal", num_warmup=300, num_samples=600, num_chains=2).draws("logbeta")[:, 0]
    pe = sample_posterior(r, "explicit", num_warmup=300, num_samples=600, num_chains=2).draws("logbeta")[:, 0]
    assert abs(pm.mean() - pe.mean()) < 0.3 * r.se[0]
    assert abs(pm.std() / pe.std() - 1) < 0.2
    assert abs(pm.std() / r.se[0] - 1) < 0.3
