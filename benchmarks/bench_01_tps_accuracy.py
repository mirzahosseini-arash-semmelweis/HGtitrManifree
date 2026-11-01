"""Benchmark 1 - accuracy and cost of the legacy thin-plate-spline (TPS) emulator.

Reviewer 1 / Reviewer 2 item: quantify the interpolation error of the
free-component-manifold TPS surface against exact equilibrium solutions
across binding strengths, stoichiometries, concentration regimes (c-values)
and grid / training-set settings, and its effect on fitted constants.

Outputs: results/bench01_tps_error.csv, results/bench01_tps_fitbias.csv
Run:     python bench_01_tps_accuracy.py  (about 5-10 min on 2 cores)
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import RESULTS, env_info
from hgtitr import Experiment, Model, fit, solve_speciation, titration
from hgtitr.forward import tps_species

MODELS = {
    "1:1": (["H1G1"], lambda k: [k]),
    "1:2": (["H1G1", "H1G2"], lambda k: [k, 2 * k - np.log10(4)]),     # statistical (non-cooperative) K2 = K1/4
    "2:1": (["H1G1", "H2G1"], lambda k: [k, 2 * k - np.log10(4)]),
}
LOGK = [3, 5, 7, 9]
CVAL = [0.1, 1, 10, 100, 1000]
NSIM = [(25, 125), (50, 250), (100, 500)]
MAXTRAIN = [300, 1200, 3000]


def make_design(HT0):
    # 30 equal additions to 4 equivalents of guest, stock 100 x host (modest dilution)
    return titration(HT0, 100 * HT0, 2.0, np.full(30, 2.0 * 4 / 100 / 30 * 1.04))


def one_case(mname, logK, c, nsim, mt):
    sp, lbf = MODELS[mname]
    m, lb = Model(sp), lbf(logK)
    HT0 = c / 10 ** logK
    if not (1e-7 <= HT0 <= 1e-2):
        return None
    HT, GT = make_design(HT0)
    t0 = time.perf_counter()
    C_ex = solve_speciation(HT, GT, m, lb).C
    t_ex = time.perf_counter() - t0
    try:
        C_tps, info = tps_species(m, lb, HT, GT, n=nsim, max_train=mt, return_info=True)
    except RuntimeError as err:
        return dict(model=mname, logK=logK, c=c, n_h=nsim[0], n_g=nsim[1], max_train=mt, error=str(err))
    # fraction-of-host error for each complex (host-weighted)
    dfrac = np.abs(C_tps[:, 2:] - C_ex[:, 2:]) * m.nH[None, :] / HT[:, None]
    # signal with unit molar responses for every complex
    S_ex, S_tps = C_ex[:, 2:].sum(1), C_tps[:, 2:].sum(1)
    rel_sig = np.abs(S_tps - S_ex) / np.ptp(S_ex)
    mid = (GT > 0.5 * HT) & (GT < 1.5 * HT)  # around the equivalence point
    return dict(model=mname, logK=logK, c=c, n_h=nsim[0], n_g=nsim[1], max_train=mt,
                n_manifold=info["n_manifold"], n_train=info["n_train"],
                max_frac_err=dfrac.max(), rms_frac_err=np.sqrt(np.mean(dfrac ** 2)),
                max_rel_signal_err=rel_sig.max(), max_rel_signal_err_equiv=rel_sig[mid].max() if mid.any() else np.nan,
                first_point_err=rel_sig[1], t_tps_ms=1e3 * info["seconds"], t_exact_ms=1e3 * t_ex)


def fit_bias_case(mname, logK, c, mt, noise_rel=0.005, seed=0):
    """Fit noiseless (and noisy) exact data with the TPS emulator vs the exact solver."""
    sp, lbf = MODELS[mname]
    m, lb = Model(sp), np.array(lbf(logK))
    HT0 = c / 10 ** logK
    if not (1e-7 <= HT0 <= 1e-2):
        return None
    HT, GT = make_design(HT0)
    C = solve_speciation(HT, GT, m, lb).C
    R = np.arange(1, m.n_complex + 1, dtype=float) / HT0      # signal ~ O(1)
    y0 = C[:, 2:] @ R
    rng = np.random.default_rng(seed)
    y = y0 + noise_rel * np.ptp(y0) * rng.normal(size=y0.size)
    out = []
    for label, Y, sd in (("noiseless", y0 + 1e-12 * np.ptp(y0) * np.sin(np.arange(y0.size)), 1e-12 * np.ptp(y0)),
                         ("noisy", y, noise_rel * np.ptp(y0))):
        e = Experiment(HT, GT, Y, sigma=sd)  # known noise: no reweighting loop
        spec = lambda H, G, mm, b: tps_species(mm, b, H, G, n=(100, 500), max_train=mt)  # noqa: E731
        t0 = time.perf_counter()
        try:
            rt = fit(m, e, start=lb + 0.2, speciator=spec, compute_cov=False, method="varpro_nm", max_nfev=600)
            bt, tt = rt.logbeta, time.perf_counter() - t0
        except Exception:
            bt, tt = np.full(m.n_complex, np.nan), np.nan
        t0 = time.perf_counter()
        rx = fit(m, e, start=lb + 0.2)
        rxn = fit(m, e, start=lb + 0.2, method="varpro_nm", compute_cov=False, max_nfev=600)  # same optimiser
        tx = time.perf_counter() - t0
        for j, s in enumerate(sp):
            out.append(dict(model=mname, logK=logK, c=c, max_train=mt, data=label, species=s, true=lb[j],
                            tps_est=bt[j], exact_est=rx.logbeta[j], exact_se=rx.se[j], exact_nm_est=rxn.logbeta[j],
                            tps_minus_exact=bt[j] - rx.logbeta[j], t_fit_tps_s=tt, t_fit_exact_s=tx))
    return out


def run_accuracy():
    cases = list(itertools.product(MODELS, LOGK, CVAL, NSIM, MAXTRAIN))
    rows = Parallel(n_jobs=2, verbose=0)(delayed(one_case)(*c) for c in cases)
    df = pd.DataFrame([r for r in rows if r is not None])
    df.to_csv(f"{RESULTS}/bench01_tps_error.csv", index=False)
    print(df.groupby(["n_h", "max_train"])[["max_rel_signal_err", "max_frac_err", "t_tps_ms", "t_exact_ms"]]
          .median())
    leg = df[(df.n_h == 100) & (df.max_train == 1200)]
    print(leg.pivot_table(index=["model", "logK"], columns="c", values="max_rel_signal_err").round(5))


def run_fitbias():
    fcases = list(itertools.product(MODELS, [5, 7], [0.1, 1, 10, 100, 1000], [300, 1200]))
    frows = Parallel(n_jobs=2)(delayed(fit_bias_case)(*c) for c in fcases)
    fdf = pd.DataFrame([x for r in frows if r for x in r])
    fdf.to_csv(f"{RESULTS}/bench01_tps_fitbias.csv", index=False)
    fdf["bias_over_se"] = (fdf.tps_est - fdf.exact_est) / fdf.exact_se
    print(fdf[fdf.data == "noisy"].groupby(["model", "c", "max_train"])[["tps_minus_exact", "bias_over_se"]]
          .agg(lambda x: np.nanmax(np.abs(x))).round(3))


if __name__ == "__main__":
    import sys
    print(env_info())
    if "--fit-only" not in sys.argv:
        run_accuracy()
    run_fitbias()
