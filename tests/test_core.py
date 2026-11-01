import numpy as np
import pytest

from hgtitr import Model, Experiment, fit, solve_speciation, titration
from hgtitr.forward import exact_signal, sensitivity_map, tps_signal, totals_from_free
from hgtitr.observables import design_matrix
from hgtitr.speciation import free_guest_1to1
from hgtitr.spectral import SpectralData, SpectralProblem, synthetic_band_spectra


def test_parse_and_stoich():
    m = Model(["H1G1", "H1G2", "H2G1", "H1G1b"])
    assert m.stoich.shape == (6, 2)
    assert list(m.nH) == [1, 1, 2, 1]
    with pytest.raises(ValueError):
        Model(["HG"])
    with pytest.raises(ValueError):
        Model(["H0G1"])


@pytest.mark.parametrize("logK", [0.0, 3.0, 6.0, 9.0, 12.0])
def test_one_to_one_matches_quadratic(logK):
    HT = np.full(60, 1e-5)
    GT = np.logspace(-9, -2, 60)
    sp = solve_speciation(HT, GT, Model(["H1G1"]), [logK])
    assert np.allclose(sp.g, free_guest_1to1(HT, GT, logK), rtol=1e-10)


def test_mass_balance_complex_model():
    m = Model(["H1G1", "H1G2", "H2G1", "H2G2", "H1G3"])
    lb = [6, 10.5, 11, 16, 13]
    HT = np.logspace(-7, -3, 40)
    GT = np.logspace(-3, -8, 40)
    sp = solve_speciation(HT, GT, m, lb)
    T = sp.C @ m.stoich
    assert np.allclose(T[:, 0], HT, rtol=1e-10) and np.allclose(T[:, 1], GT, rtol=1e-10)


def test_forward_inverse_roundtrip():
    m = Model(["H1G1", "H1G2"])
    h = np.logspace(-9, -4, 15)
    g = np.logspace(-4, -9, 15)
    HT, GT, _ = totals_from_free(h, g, m, [5, 8])
    sp = solve_speciation(HT, GT, m, [5, 8])
    assert np.allclose(sp.h, h, rtol=1e-9) and np.allclose(sp.g, g, rtol=1e-9)


def test_zero_totals():
    m = Model(["H1G1"])
    sp = solve_speciation([1e-5, 0.0, 1e-5], [0.0, 1e-5, 1e-5], m, [5])
    assert sp.C[0, 2] == 0 and sp.C[1, 2] == 0 and sp.C[2, 2] > 0


def test_sensitivity_map_matches_finite_difference():
    m = Model(["H1G1", "H1G2"])
    HT, GT = titration(1e-6, 1e-1, 3.0, np.linspace(0, 0.01, 30))
    r = [0, 0, 2e5, 3e5]
    sp = solve_speciation(HT, GT, m, [5, 8])
    _, _, _, dS = sensitivity_map(m, [5, 8], r, sp.h, sp.g)
    e = 1e-6
    fd = (exact_signal(m, [5 + e, 8], r, HT, GT) - exact_signal(m, [5 - e, 8], r, HT, GT)) / (2 * e)
    assert np.allclose(dS[:, 0], fd, atol=1e-6 * np.abs(fd).max())


def test_nmr_fast_exchange_shift():
    m = Model(["H1G1"])
    HT = np.full(5, 1e-3); GT = np.linspace(0, 3e-3, 5)
    sp = solve_speciation(HT, GT, m, [3])
    A = design_matrix(sp.C, m.stoich, HT, GT, "shift_H", active=[True, False, True])
    d = A @ np.array([7.0, 8.0])
    frac = sp.C[:, 2] / HT
    assert np.allclose(d, 7.0 + frac * 1.0)


@pytest.mark.parametrize("method", ["varpro", "full"])
def test_fit_recovers_noiseless(method):
    m = Model(["H1G1", "H1G2"])
    HT, GT = titration(1e-6, 1e-1, 3.0, np.linspace(0, 0.01, 30))
    C = solve_speciation(HT, GT, m, [5, 8]).C
    Y = C[:, 2:] @ np.array([2e5, 3e5])
    Y = Y + 1e-9 * np.sin(np.arange(Y.size))  # tiny deterministic noise
    res = fit(m, Experiment(HT, GT, Y), start=[5.2, 7.7], method=method)
    assert np.allclose(res.logbeta, [5, 8], atol=1e-4)


def test_global_fit_two_experiments_and_nuisance():
    m = Model(["H1G1"])
    rng = np.random.default_rng(0)
    exps = []
    for cH in (2e-6, 2e-5):
        HT, GT = titration(cH, 1e-3, 2.0, np.full(20, 0.01))
        C = solve_speciation(HT, GT * 1.05, m, [6]).C  # titrant 5 % stronger than nominal
        Y = C[:, [0, 2]] @ np.array([1e5, 2e4])
        exps.append(Experiment(HT, GT, Y + rng.normal(0, 1e-4, Y.size), free_H_active=True,
                               titrant_factor_sd=0.05))
    res = fit(m, exps, start=[5.5])
    assert abs(res.logbeta[0] - 6) < 0.05
    assert np.all(np.abs(res.nuisance - np.log10(1.05)) < 0.02)


def test_tps_emulator_close_but_not_exact():
    m = Model(["H1G1", "H1G2"])
    HT, GT = titration(1e-6, 1e-1, 3.0, np.linspace(0, 0.01, 30))
    r = [0, 0, 2e5, 3e5]
    ex = exact_signal(m, [5, 8], r, HT, GT)
    em = tps_signal(m, [5, 8], r, HT, GT, max_train=1200)
    err = np.max(np.abs(em - ex)) / np.ptp(ex)
    assert 1e-8 < err < 0.05


def test_same_stoichiometry_split_not_identifiable():
    m = Model(["H1G1a", "H1G1b"])
    HT = np.full(10, 3e-5); GT = np.linspace(0, 4e-5, 10)
    wl = np.linspace(300, 450, 151)
    Y, _, _ = synthetic_band_spectra(HT, GT, Model(["H1G1"]), [6], wl, {"H1G1": [(1e5, 380, 15)]},
                                     {"H1G1": 10}, noise_sd=0.0)
    prob = SpectralProblem(m, SpectralData(HT, GT, wl, Y + 1e-3 * np.cos(np.arange(Y.size)).reshape(Y.shape)),
                           ["H1G1a", "H1G1b"])
    tot = 10 ** 6
    rss = []
    for la in (3.0, 4.5, 5.5):
        r = prob.resid(np.array([la, np.log10(tot - 10 ** la), 10.0, -5.0]))
        rss.append(r @ r)
    assert np.ptp(rss) < 1e-9 * np.mean(rss)
