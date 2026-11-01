"""Re-analysis of the experimental datasets (meloxicam-RSA and biliverdin-RSA).

Addresses Reviewer 1 "Validation" and "Spectral-shift model" items:
  * fits with hgtitr (exact speciation, VARPRO), Musketeer 1.10 and
    Equilibrist 2.0.1 on the same data;
  * Wald, profile-likelihood and NUTS intervals;
  * effect of near-duplicate channels and of stock-concentration uncertainty;
  * selected-channel and full-spectrum biliverdin models, including the
    identifiability of the H1G1a/H1G1b split.

Run from the repository root:  python examples/analysis_experimental.py
Writes examples/output/experimental_results.json and figures.
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "adapters"))
OUT = os.path.join(ROOT, "examples", "output")
os.makedirs(OUT, exist_ok=True)
warnings.filterwarnings("ignore")

from hgtitr import Experiment, Model, compare_models, fit, profile_ci  # noqa: E402
from hgtitr.bayes import diagnostics, sample_posterior  # noqa: E402
from hgtitr.spectral import SpectralData, SpectralProblem, fit_spectral  # noqa: E402

RES: dict = {}


def fmt(r):
    return {"species": list(r.model.species), "logbeta": r.logbeta.round(4).tolist(),
            "se": r.se[: r.model.n_complex].round(4).tolist(), "loglik": round(r.loglik, 3), "k": r.k, "n": r.n,
            "aicc": round(r.aicc, 3), "bic": round(r.bic, 3), "sigmas": [s.round(5).tolist() for s in r.sigmas]}


# --------------------------------------------------------------------------- #
# Meloxicam - rabbit serum albumin, Trp fluorescence quenching, 2 channels
# --------------------------------------------------------------------------- #
mel = pd.read_csv(os.path.join(ROOT, "data", "meloxicam_rabsa.csv"))
HT, GT, Ym = mel.total_H.values, mel.total_G.values, mel.iloc[:, 2:].values
RES["meloxicam"] = {}
e2 = Experiment(HT, GT, Ym, free_H_active=True, channels=list(mel.columns[2:]))
m11, m12 = Model(["H1G1"]), Model(["H1G1", "H1G2"])
r11 = fit(m11, e2, start=[5.0], n_starts=6)
r12 = fit(m12, e2, start=[5.0, 9.0], n_starts=12)
RES["meloxicam"]["two_channels"] = {"1:1": fmt(r11), "1:2": fmt(r12),
                                    "compare": compare_models([r11, r12])}
RES["meloxicam"]["two_channels"]["1:1"]["profile95"] = list(profile_ci(r11, 0)[:2])
res_corr = float(np.corrcoef(r11.residuals[0].T)[0, 1])
RES["meloxicam"]["residual_correlation_between_channels"] = res_corr
# the two channels (282.9 / 283.0 nm) are 0.1 nm apart: treat as one averaged channel
e1 = Experiment(HT, GT, Ym.mean(1), free_H_active=True)
r11a = fit(m11, e1, start=[5.0])
RES["meloxicam"]["averaged_channel"] = fmt(r11a)
RES["meloxicam"]["averaged_channel"]["profile95"] = list(profile_ci(r11a, 0)[:2])
# stock-concentration uncertainty: 3 % on titrant and 5 % on albumin (log10 SD)
e1e = Experiment(HT, GT, Ym.mean(1), free_H_active=True, titrant_factor_sd=np.log10(1.03),
                 host_factor_sd=np.log10(1.05))
r11e = fit(m11, e1e, start=[5.0])
RES["meloxicam"]["averaged_channel_eiv"] = fmt(r11e)
RES["meloxicam"]["averaged_channel_eiv"]["nuisance_log10"] = r11e.nuisance.round(4).tolist()
for tag, rr, tr in (("nuts_two_channels", r11, "marginal"), ("nuts_averaged", r11a, "marginal"),
                    ("nuts_averaged_eiv", r11e, "eiv")):
    po = sample_posterior(rr, tr, num_warmup=1000, num_samples=2000, num_chains=4, seed=1)
    lb = po.draws("logbeta")[:, 0]
    dg = diagnostics(po)
    RES["meloxicam"][tag] = dict(mean=float(lb.mean()), sd=float(lb.std()),
                                 ci95=np.quantile(lb, [0.025, 0.975]).round(4).tolist(),
                                 r_hat=float(dg["r_hat"][0]), ess=float(dg["n_eff"][0]), seconds=po.seconds)

# Musketeer and Equilibrist on the same (two-channel) data
try:
    from musketeer_adapter import VERSION as MV, fit_musketeer
    mk = fit_musketeer(m11, HT, GT, Ym, [True, False, True], [5.0])
    mk2 = fit_musketeer(m12, HT, GT, Ym, [True, False, True, True], [5.0, 9.0])
    RES["meloxicam"]["musketeer"] = {"version": MV, "1:1": mk["logbeta"].round(4).tolist(),
                                     "1:2": mk2["logbeta"].round(4).tolist(), "rmse_1:1": float(mk["rmse"])}
except Exception as err:  # noqa: BLE001
    RES["meloxicam"]["musketeer"] = f"failed: {err}"
try:
    from equilibrist_adapter import VERSION as EV, fit_equilibrist
    # back-calculate the dilution design: HT = H0 V0/(V0+v), GT = c_tit v/(V0+v)
    V0 = 2.0
    v = V0 * (HT[0] / HT - 1.0)
    c_tit = np.median(GT[1:] * (V0 + v[1:]) / v[1:])
    RES["meloxicam"]["derived_titrant_M"] = float(c_tit)
    # free host (albumin Trp) emits; free guest is transparent
    eq = fit_equilibrist(1, HT[0], V0, c_tit, v, Ym, [5.0], timeout_s=300, transparent=("G",))
    RES["meloxicam"]["equilibrist"] = {"version": EV, "1:1": eq["logbeta"].round(4).tolist(),
                                       "se_hessian": eq["se_logK"].round(4).tolist(), "timed_out": eq["timed_out"]}
except Exception as err:  # noqa: BLE001
    RES["meloxicam"]["equilibrist"] = f"failed: {err}"

# --------------------------------------------------------------------------- #
# Biliverdin - rabbit serum albumin, CD at 4 selected wavelengths
# --------------------------------------------------------------------------- #
bv = pd.read_csv(os.path.join(ROOT, "data", "biliverdin_rabsa.csv"))
bHT, bGT, bY = bv.total_H.values, bv.total_G.values, bv.iloc[:, 2:].values
RES["biliverdin_4ch"] = {}
eb = Experiment(bHT, bGT, bY, channels=list(bv.columns[2:]))
cands = {"1:1": (["H1G1"], [5.0]), "HG2 only": (["H1G2"], [7.7]), "1:2": (["H1G1", "H1G2"], [5.06, 9.48]),
         "1:3": (["H1G1", "H1G2", "H1G3"], [5.0, 9.5, 13.0]), "1:2+2:1": (["H1G1", "H1G2", "H2G1"], [5, 9.5, 10])}
fits = {}
for k, (sp, st) in cands.items():
    fits[k] = fit(Model(sp), eb, start=st, n_starts=16, bounds=(0.0, 18.0))
    RES["biliverdin_4ch"][k] = fmt(fits[k])
RES["biliverdin_4ch"]["compare"] = compare_models(list(fits.values()))
r = fits["1:2"]
RES["biliverdin_4ch"]["1:2"]["profile95"] = [list(profile_ci(r, j)[:2]) for j in range(2)]
# NOTE: the 1:2 likelihood is bimodal (an "HG2 only" basin at log beta11 -> -inf is only ~4.3
# deviance units worse), so NUTS chains started at the optimum report only part of the posterior
po = sample_posterior(r, "marginal", num_warmup=1000, num_samples=2000, num_chains=4, seed=2, bounds=(0.0, 18.0))
lb = po.draws("logbeta")
RES["biliverdin_4ch"]["1:2"]["nuts"] = dict(mean=lb.mean(0).round(3).tolist(), sd=lb.std(0).round(3).tolist(),
                                            ci95=np.quantile(lb, [0.025, 0.975], axis=0).round(3).tolist(),
                                            r_hat=diagnostics(po)["r_hat"].round(3).tolist())
try:
    mk = fit_musketeer(Model(["H1G1", "H1G2"]), bHT, bGT, bY, [False, False, True, True], [5.06, 9.48])
    RES["biliverdin_4ch"]["musketeer_1:2"] = mk["logbeta"].round(4).tolist()
except Exception as err:  # noqa: BLE001
    RES["biliverdin_4ch"]["musketeer_1:2"] = f"failed: {err}"

# --------------------------------------------------------------------------- #
# Biliverdin full CD spectra, 305-445 nm, 1 nm (every 10th point of 0.1 nm data)
# --------------------------------------------------------------------------- #
df = pd.read_csv(os.path.join(ROOT, "data", "biliverdin_rabsa_fullCD.csv"))
cols = [c for c in df.columns if c.startswith("CD")]
wl = np.array([float(c[2:]) for c in cols])
full = SpectralData(df.total_H.values, df.total_G.values, wl, df[cols].values)
d1 = full.subset(305.0, 445.0, step=10)
RES["biliverdin_full"] = {"n_wavelengths": int(d1.wl.size), "wl_range": [float(d1.wl.min()), float(d1.wl.max())]}
spec_c = {"1:1 fixed": (["H1G1"], [], [5.5]), "1:1 shifted": (["H1G1"], ["H1G1"], [5.5, -20.0]),
          "1:2 fixed": (["H1G1", "H1G2"], [], [5.3, 9.7]),
          "1:2 shifted(both)": (["H1G1", "H1G2"], ["H1G1", "H1G2"], [5.8, 8.8, -35.0, 45.0]),
          "HG_I+HG_II shifted(both)": (["H1G1a", "H1G1b"], ["H1G1a", "H1G1b"], [4.6, 5.9, 49.0, -38.0])}
for noise in ("iid", "ar1"):
    for k, (sp, sh, st) in spec_c.items():
        f = fit_spectral(Model(sp), d1, sh, n_basis=10, noise=noise, start=st, n_starts=8, dlam_bounds=(-80, 80))
        RES["biliverdin_full"][f"{k} [{noise}]"] = dict(logbeta=f.logbeta.round(4).tolist(),
                                                       dlam=f.dlam.round(2).tolist(), rho=round(f.rho, 4),
                                                       sigma=round(f.sigma, 4), loglik=round(f.loglik, 2), k=f.k,
                                                       aic=round(f.aic, 2), bic=round(f.bic, 2))
# Identifiability of the H1G1a / H1G1b split on the real data
best = RES["biliverdin_full"]["HG_I+HG_II shifted(both) [iid]"]
la, lbb = best["logbeta"]
tot = 10 ** la + 10 ** lbb
pr = SpectralProblem(Model(["H1G1a", "H1G1b"]), d1, ["H1G1a", "H1G1b"], n_basis=10)
ridge = []
for x in (2.0, 3.0, 3.63, 4.0, la, 5.0, 5.5):
    if 10 ** x < tot:
        y = np.log10(tot - 10 ** x)
        rr = pr.resid(np.array([x, y] + best["dlam"]))
        ridge.append(dict(logbeta_a=x, logbeta_b=float(y), rss=float(rr @ rr)))
RES["biliverdin_full"]["split_ridge"] = ridge
RES["biliverdin_full"]["logbeta_total"] = float(np.log10(tot))

with open(os.path.join(OUT, "experimental_results.json"), "w") as fh:
    json.dump(RES, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print(json.dumps(RES, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))[:6000])
