"""Benchmark 4 - validation of the full-spectrum spectral-shift model.

Reviewer 1 item: "the model is not validated on synthetic spectra with a known
band displacement, which is the one test that would show it recovers both the
constants and dlam, and that the fixed-spectrum model fails in the way claimed."

Two generating truths, CD-like Gaussian bands, 305-445 nm at 1 nm:

  A  'shift 1:1'  1:1, log K = 5.9, bound spectrum displaced by
                  q * (+12 nm), q = fraction of host bound
  B  'fixed 1:2'  sequential 1:2, log beta = (5.3, 9.5), fixed spectra
                  (HG2 band 12 nm red of HG)

Two designs: the experimental biliverdin design (13 points to 1.3 equiv) and
an extended design (25 points to 3.1 equiv).  Noise: AR(1) along wavelength
(rho = 0.9, marginal SD 0.1) - realistic for densely sampled CD.

Candidate fits: 1:1 fixed, 1:1 shifted, 1:2 fixed, 1:2 both shifted,
H1G1a + H1G1b both shifted (the manuscript's preferred model).
Outputs: results/bench04_spectral.csv
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import BVD_DESIGN, MC_DESIGN, RESULTS, design, env_info
from hgtitr import Model
from hgtitr.spectral import SpectralData, fit_spectral, synthetic_band_spectra

QUICK = "--quick" in sys.argv
N_REP = 2 if QUICK else 8
WL = np.arange(305.0, 446.0, 1.0)
BANDS_HG = [(-6.0e5, 380.0, 18.0), (3.0e5, 420.0, 15.0)]
BANDS_H = [(-2.0e4, 330.0, 25.0)]                          # weak free-host CD
TRUTHS = {
    "A_shift_1to1": dict(model=Model(["H1G1"]), lb=[5.9],
                         bands={"H": BANDS_H, "H1G1": BANDS_HG}, shift={"H1G1": 12.0}),
    "B_fixed_1to2": dict(model=Model(["H1G1", "H1G2"]), lb=[5.3, 9.5],
                         bands={"H": BANDS_H, "H1G1": BANDS_HG,
                                "H1G2": [(a * 1.3, c + 12.0, w) for a, c, w in BANDS_HG]}, shift={}),
}
CANDIDATES = {
    "1:1 fixed": (["H1G1"], [], [5.5]),
    "1:1 shifted": (["H1G1"], ["H1G1"], [5.5, 5.0]),
    "1:2 fixed": (["H1G1", "H1G2"], [], [5.5, 9.0]),
    "1:2 shifted(both)": (["H1G1", "H1G2"], ["H1G1", "H1G2"], [5.5, 9.0, 5.0, 5.0]),
    "HG_I+HG_II shifted(both)": (["H1G1a", "H1G1b"], ["H1G1a", "H1G1b"], [4.5, 5.8, 10.0, -5.0]),
}


def ar1_noise(rng, shape, rho, sd):
    e = np.empty(shape)
    e[:, 0] = rng.normal(0, sd, shape[0])
    innov = sd * np.sqrt(1 - rho ** 2)
    for k in range(1, shape[1]):
        e[:, k] = rho * e[:, k - 1] + rng.normal(0, innov, shape[0])
    return e


def one(truth, dname, rep):
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(70_000 + rep)
    T = TRUTHS[truth]
    HT, GT, _ = design(BVD_DESIGN if dname == "bvd13" else MC_DESIGN)
    Y, C, q = synthetic_band_spectra(HT, GT, T["model"], T["lb"], WL, T["bands"], T["shift"])
    Y = Y + ar1_noise(rng, Y.shape, 0.9, 0.1)
    data = SpectralData(HT, GT, WL, Y)
    rows = []
    for cname, (sp, sh, st) in CANDIDATES.items():
        for noise in ("iid", "ar1"):
            try:
                f = fit_spectral(Model(sp), data, sh, n_basis=14, noise=noise, start=st, n_starts=4, seed=rep,
                                 dlam_bounds=(-40, 40))
                lb_tot = np.log10(np.sum(10 ** f.logbeta)) if cname.startswith("HG_I") else np.nan
                rows.append(dict(truth=truth, design=dname, rep=rep, candidate=cname, noise=noise,
                                 loglik=f.loglik, k=f.k, aic=f.aic, bic=f.bic, rho=f.rho, sigma=f.sigma,
                                 logbeta=";".join(f"{x:.4f}" for x in f.logbeta),
                                 dlam=";".join(f"{x:.2f}" for x in f.dlam), logbeta_total_split=lb_tot,
                                 se=";".join(f"{x:.4f}" for x in (f.se if f.se is not None else []))))
            except Exception as err:  # noqa: BLE001
                rows.append(dict(truth=truth, design=dname, rep=rep, candidate=cname, noise=noise, error=str(err)))
    return rows


if __name__ == "__main__":
    import glob
    import os
    print(env_info())
    parts = os.path.join(RESULTS, "bench04_parts")
    os.makedirs(parts, exist_ok=True)
    jobs = [(t, d, r) for t in TRUTHS for d in ("bvd13", "ext25") for r in range(N_REP)]
    for k in range(0, len(jobs), 2):  # resumable: two jobs (one per core) per saved block
        fn = os.path.join(parts, f"part_{k // 2:03d}.csv")
        if os.path.exists(fn):
            continue
        out = Parallel(n_jobs=2)(delayed(one)(*j) for j in jobs[k:k + 2])
        pd.DataFrame([x for r in out for x in r]).to_csv(fn, index=False)
        print(f"block {k // 2} done", flush=True)
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(parts, "part_*.csv")))], ignore_index=True)
    df.to_csv(f"{RESULTS}/bench04_spectral.csv", index=False)
    for crit in ("aic", "bic"):
        best = df.loc[df.dropna(subset=[crit]).groupby(["truth", "design", "rep", "noise"])[crit].idxmin()]
        print(f"\nSelected by {crit.upper()}:")
        print(best.groupby(["truth", "design", "noise"]).candidate.value_counts().unstack(fill_value=0))
