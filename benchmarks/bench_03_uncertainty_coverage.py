"""Benchmark 3 - frequentist coverage of 95 % intervals for log beta.

Is the Bayesian (MCMC) uncertainty actually better calibrated than cheap
alternatives?  For each replicate dataset we compute 95 % intervals by

  wald           Gauss-Newton covariance from the projected (Kaufman) Jacobian
  profile        profile likelihood (chi2_1 threshold)
  bootstrap      residual bootstrap, percentile interval (B = 200)
  nuts_marginal  NUTS, responses integrated out analytically
  nuts_explicit  NUTS, responses sampled explicitly (legacy treatment; fewer reps)
  wald_eiv /     with a log10 titrant-concentration factor ~ N(0, 0.0128)
  nuts_eiv       (3 % stock error) - only in the "+conc3%" scenarios

"+conc3%" scenarios simulate a true titrant concentration error with 3 %
relative SD that the analyst does not know (pipetting / weighing / purity).

Outputs: results/bench03_intervals.csv, results/bench03_coverage.csv
"""
from __future__ import annotations

import os
import sys
import time
import warnings

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import RESULTS, env_info, simulate
from hgtitr import Experiment, bootstrap, fit, profile_ci

QUICK = "--quick" in sys.argv
SCEN = [("S1_1to1", 0.0), ("S4_1to2", 0.0), ("S5_tight_1to1", 0.0), ("S1_1to1", 0.03), ("S4_1to2", 0.03)]
N_CHEAP = 5 if QUICK else 200
N_BOOT_REP = 3 if QUICK else 100
N_NUTS = 3 if QUICK else 30
N_EXPL = 2 if QUICK else 6
F_SD = np.log10(1.03)


def one(scen, conc_sd, rep, do_boot, do_nuts, do_expl):
    warnings.filterwarnings("ignore")
    from hgtitr.bayes import diagnostics, sample_posterior

    rng = np.random.default_rng(50_000 + rep + int(conc_sd * 1e4))
    m, HT, GT, Vc, Y, truth, f = simulate(scen, rng, conc_error_sd=conc_sd)
    label = scen + ("+conc3%" if conc_sd > 0 else "")
    rows = []

    def add(method, lo, hi, secs, extra=None):
        for j, s in enumerate(m.species):
            d = dict(scenario=label, rep=rep, method=method, species=s, true=truth[j], lo=lo[j], hi=hi[j],
                     covered=bool(lo[j] <= truth[j] <= hi[j]), width=hi[j] - lo[j], seconds=secs)
            if extra:
                d.update({k: (v[j] if np.ndim(v) else v) for k, v in extra.items()})
            rows.append(d)

    e = Experiment(HT, GT, Y)
    t0 = time.perf_counter()
    r = fit(m, e, start=truth + 0.1)
    t_fit = time.perf_counter() - t0
    add("wald", r.logbeta - 1.96 * r.se[: m.n_complex], r.logbeta + 1.96 * r.se[: m.n_complex], t_fit)
    t0 = time.perf_counter()
    pr = [profile_ci(r, j)[:2] for j in range(m.n_complex)]
    add("profile", np.array([p[0] for p in pr]), np.array([p[1] for p in pr]), time.perf_counter() - t0)
    if do_boot:
        t0 = time.perf_counter()
        b = bootstrap(r, n_boot=200, seed=rep)[:, : m.n_complex]
        add("bootstrap", np.quantile(b, 0.025, 0), np.quantile(b, 0.975, 0), time.perf_counter() - t0)
    if conc_sd > 0:
        ee = Experiment(HT, GT, Y, titrant_factor_sd=F_SD)
        t0 = time.perf_counter()
        re = fit(m, ee, start=truth + 0.1)
        add("wald_eiv", re.logbeta - 1.96 * re.se[: m.n_complex], re.logbeta + 1.96 * re.se[: m.n_complex],
            time.perf_counter() - t0)
    if do_nuts:
        for treat, res in (("marginal", r),) + ((("eiv", re),) if conc_sd > 0 else ()):
            po = sample_posterior(res, treat, num_warmup=500, num_samples=500, num_chains=2, seed=rep)
            lo, hi = po.interval("logbeta")
            dg = diagnostics(po)
            add(f"nuts_{treat}", lo, hi, po.seconds, dict(r_hat=dg["r_hat"], ess=dg["n_eff"],
                                                          ess_per_s=dg["ess_per_s"]))
    if do_expl:
        po = sample_posterior(r, "explicit", num_warmup=500, num_samples=500, num_chains=2, seed=rep)
        lo, hi = po.interval("logbeta")
        dg = diagnostics(po)
        add("nuts_explicit", lo, hi, po.seconds, dict(r_hat=dg["r_hat"], ess=dg["n_eff"], ess_per_s=dg["ess_per_s"]))
    return rows


def aggregate(df):
    cov = (df.groupby(["scenario", "species", "method"])
             .agg(n=("covered", "size"), coverage=("covered", "mean"), median_width=("width", "median"),
                  median_seconds=("seconds", "median"))
             .reset_index())
    cov["mc_se"] = np.sqrt(cov.coverage * (1 - cov.coverage) / cov.n)
    if "ess_per_s" in df:
        ess = df.groupby(["scenario", "species", "method"]).ess_per_s.median().rename("median_ess_per_s")
        cov = cov.merge(ess.reset_index(), how="left")
    return cov


if __name__ == "__main__":
    # Resumable: jobs run in blocks of 20 and each block is saved to results/bench03_parts/;
    # re-running the script skips finished blocks.
    import glob
    print(env_info())
    parts = os.path.join(RESULTS, "bench03_parts")
    os.makedirs(parts, exist_ok=True)
    jobs = [(s, c, rep, rep < N_BOOT_REP, rep < N_NUTS, rep < N_EXPL and c == 0 and s != "S5_tight_1to1")
            for s, c in SCEN for rep in range(N_CHEAP)]
    block = 20
    for b in range(0, len(jobs), block):
        fn = os.path.join(parts, f"part_{b // block:04d}.csv")
        if os.path.exists(fn):
            continue
        out = Parallel(n_jobs=2)(delayed(one)(*j) for j in jobs[b:b + block])
        pd.DataFrame([x for r in out for x in r]).to_csv(fn, index=False)
        print(f"block {b // block} done", flush=True)
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(parts, "part_*.csv")))], ignore_index=True)
    df.to_csv(f"{RESULTS}/bench03_intervals.csv", index=False)
    cov = aggregate(df)
    cov.to_csv(f"{RESULTS}/bench03_coverage.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(cov.round(4))
