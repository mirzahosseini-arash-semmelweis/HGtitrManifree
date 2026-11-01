"""Benchmark 2 - fit bias, precision, robustness and compute cost.

Same simulated datasets (Monte Carlo replicates) are fitted by

  tps_legacy     legacy HGtitrManifree emulator (TPS of the free-component
                 manifold, nsim = 100 x 500, max_train = 1200) + VARPRO,
                 derivative-free Nelder-Mead (the TPS surface is not smooth in
                 log beta, gradient-based optimisers stall on it)
  exact_full     exact speciation, responses as explicit parameters (legacy
                 multichannel treatment, trust-region LM-type least squares)
  exact_varpro   exact speciation + variable projection (hgtitr default)
  musketeer      Musketeer 1.10 speciation + Nelder-Mead bilevel protocol
  equilibrist    Equilibrist 2.0.1 fit_spectra (unchanged), fewer replicates

All methods start from the same perturbed values (truth + 0.4 / -0.5 log units).
Outputs: results/bench02_fits.csv, results/bench02_summary.csv
Run:     python bench_02_fit_bias_precision.py [--quick]
"""
from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import RESULTS, SCENARIOS, env_info, simulate
from hgtitr import Experiment, fit
from hgtitr.forward import tps_species

QUICK = "--quick" in sys.argv
N_REP = {"exact_varpro": 200, "exact_full": 200, "tps_legacy": 40, "musketeer": 50, "equilibrist": 8}
if QUICK:
    N_REP = {k: max(3, v // 20) for k, v in N_REP.items()}
OFFSET = np.array([0.4, -0.5])


def run_one(scen, method, rep):
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(10_000 + rep)
    m, HT, GT, Vc, Y, truth, _ = simulate(scen, rng)
    start = truth + OFFSET[: m.n_complex]
    active = np.r_[False, False, np.ones(m.n_complex, bool)]
    t0 = time.perf_counter()
    est, ok, nfev, se = np.full(m.n_complex, np.nan), False, np.nan, np.full(m.n_complex, np.nan)
    try:
        if method in ("exact_varpro", "exact_full", "tps_legacy"):
            spec = None
            if method == "tps_legacy":
                spec = lambda H, G, mm, b: tps_species(mm, b, H, G, n=(100, 500), max_train=1200)  # noqa: E731
            meth = {"exact_full": "full", "exact_varpro": "varpro", "tps_legacy": "varpro_nm"}[method]
            r = fit(m, Experiment(HT, GT, Y), start=start, method=meth,
                    speciator=spec, compute_cov=(method != "tps_legacy"))
            est, ok, nfev, se = r.logbeta, r.success, r.nfev, r.se[: m.n_complex]
        elif method == "musketeer":
            from musketeer_adapter import fit_musketeer
            r = fit_musketeer(m, HT, GT, Y, active, start)
            est, ok, nfev = r["logbeta"], r["success"], r["nfev"]
        elif method == "equilibrist":
            from equilibrist_adapter import fit_equilibrist
            if any(s not in ("H1G1", "H1G2", "H1G3") for s in m.species):
                return []
            stepwise = np.diff(np.r_[0.0, start])
            d = SCENARIOS[scen][3]
            r = fit_equilibrist(m.n_complex, d["c_cell_H"], d["V0"], d["c_titrant_G"], Vc, Y, stepwise,
                                timeout_s=120)
            est, ok = r["logbeta"], r["success"] and not r["timed_out"]
            nfev = r["n_iter"]
            se = r["se_logK"]  # stepwise SE (reported for K1 only comparisons)
    except Exception as err:  # noqa: BLE001
        ok = False
    secs = time.perf_counter() - t0
    return [dict(scenario=scen, method=method, rep=rep, species=s, true=truth[j], est=est[j], se=se[j],
                 success=bool(ok), nfev=nfev, seconds=secs) for j, s in enumerate(m.species)]


def summarise(df):
    df = df.copy()
    df["err"] = df.est - df.true
    df["fail"] = (~df.success) | (df.err.abs() > 0.5) | df.est.isna()
    g = df.groupby(["scenario", "species", "method"])
    good = df[~df.fail].groupby(["scenario", "species", "method"])
    out = pd.DataFrame({
        "n": g.size(),
        "fail_rate": g.fail.mean(),
        "bias": good.err.mean(),
        "sd": good.err.std(),
        "rmse": good.err.apply(lambda x: np.sqrt(np.mean(x ** 2))),
        "median_s": g.seconds.median(),
        "median_nfev": g.nfev.median(),
    })
    return out.reset_index()


if __name__ == "__main__":
    print(env_info())
    jobs = [(s, meth, r) for s in SCENARIOS for meth, n in N_REP.items() for r in range(n)]
    # equilibrist and musketeer are slow: run them last
    rows = Parallel(n_jobs=2, verbose=5)(delayed(run_one)(*j) for j in jobs)
    df = pd.DataFrame([x for r in rows for x in r])
    df.to_csv(f"{RESULTS}/bench02_fits.csv", index=False)
    summ = summarise(df)
    summ.to_csv(f"{RESULTS}/bench02_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(summ.round(4))
