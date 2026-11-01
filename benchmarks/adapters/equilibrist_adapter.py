"""Headless use of Equilibrist's own spectral fitting routine.

Equilibrist (E. Masson, J. Chem. Inf. Model. 2026, 66, 9976; MIT licence;
https://github.com/masson-group-ohiouniversity/Equilibrist) is a Streamlit
app.  Its numerical modules are importable without Streamlit; here we build
the script/network/params objects exactly as the app does and call
``equilibrist_fit_spectra.fit_spectra`` unchanged.

Set the environment variable EQUILIBRIST_DIR to the cloned repository.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np

EQ_DIR = os.environ.get("EQUILIBRIST_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "..", "ext", "Equilibrist"))
EQ_DIR = os.path.abspath(EQ_DIR)
if EQ_DIR not in sys.path:
    sys.path.insert(0, EQ_DIR)

from equilibrist_curve import _x_per_equiv  # noqa: E402
from equilibrist_fit_spectra import fit_spectra  # noqa: E402
from equilibrist_network import build_network  # noqa: E402
from equilibrist_parser import parse_script  # noqa: E402

try:
    VERSION = subprocess.run(["git", "-C", EQ_DIR, "describe", "--tags", "--always"], capture_output=True,
                             text=True).stdout.strip() or "unknown"
except Exception:  # pragma: no cover
    VERSION = "unknown"


def _build_params(parsed, network):
    """Copy of Equilibrist batch_process.build_params (module names updated)."""
    titrant_key = network["titrant_key"]
    titrant_name = network["titrant_name"]
    titrant_free_names = network["titrant_free_names"]
    titrant_keys = network["titrant_keys"]
    conc_roots = {}
    for cname, cval in parsed["concentrations"].items():
        root = cname[:-1] if cname.endswith("0") else cname
        conc_roots[root] = cval
    primary = list(conc_roots.keys())[0] if conc_roots else network["free_species"][0]
    V0_mL = list(parsed["volumes"].values())[0] if parsed["volumes"] else 0.5
    titrant_mMs, titrant_ratios = {}, {}
    for tkey in titrant_keys:
        tfree = tkey[:-1] if (tkey.endswith("t") or tkey.endswith("0")) else tkey
        titrant_mMs[tfree] = parsed["titrant"].get(tkey, 10.0)
        titrant_ratios[tfree] = 1.0
    x_expr = parsed.get("plot_x_expr") or f"{titrant_key}/{list(parsed['concentrations'].keys())[0]}"
    xmax = float(parsed["plot_xmax"])
    xpe = _x_per_equiv(x_expr, parsed, conc_roots, float(V0_mL), titrant_free_names, titrant_keys, titrant_mMs,
                       titrant_ratios, parsed["titrant_is_solid"], primary)
    return {"conc0": conc_roots, "V0_mL": float(V0_mL), "titrant_name": titrant_name, "titrant_key": titrant_key,
            "titrant_free_names": titrant_free_names, "titrant_keys": titrant_keys, "titrant_mMs": titrant_mMs,
            "titrant_ratios": titrant_ratios, "titrant_is_solid": parsed["titrant_is_solid"],
            "maxEquiv": xmax / max(xpe, 1e-12), "nPts": 200, "primary_component": primary}, x_expr


def script_HGn(n_steps: int, H0_mM: float, V0_mL: float, Gt_mM: float, logK_start, xmax=3.0,
               transparent=("H", "G")):
    """Sequential H + G = HG, HG + G = HG2, ... script (stepwise constants)."""
    lines = ["$concentrations", f"H0 = {H0_mM} mM", "", "$volumes", f"V0 = {V0_mL} mL", "", "$titrant",
             f"Gt = {Gt_mM} mM", "", "$reactions"]
    prev = "H"
    for i in range(n_steps):
        prod = "HG" if i == 0 else f"HG{i+1}"
        lines.append(f"{prev} + G = {prod}; log K{i+1} = {logK_start[i]:.4f}")
        prev = prod
    lines += ["", "$plot", f"xmax = {xmax}", "x = G0/H0", "y = H, G", "", "$spectra"]
    if transparent:
        lines.append("transparent: " + ", ".join(transparent))  # Equilibrist syntax uses a colon
    return "\n".join(lines)


def fit_equilibrist(n_steps, H0_M, V0_mL, Gt_M, V_add_cum_mL, Y, logK_start, timeout_s=60.0, maxiter=2000,
                    tolerance=1e-10, allow_negative_eps=True, transparent=("H", "G")):
    """Fit stepwise log K with Equilibrist's fit_spectra.  Returns cumulative log beta."""
    text = script_HGn(n_steps, H0_M * 1e3, V0_mL, Gt_M * 1e3, logK_start, transparent=transparent)
    parsed = parse_script(text)
    network = build_network(parsed)
    params, x_expr = _build_params(parsed, network)
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    wl = np.arange(Y.shape[1], dtype=float) + 300.0  # channels as pseudo-wavelengths
    spectra = {"wavelengths": wl, "x_vals": np.asarray(V_add_cum_mL, float), "A": Y, "x_col_header": "V",
               "known_spectra_raw": {}}
    logK_vals = {eq["kname"]: eq["logK"] for eq in parsed["equilibria"]}
    fit_keys = list(logK_vals)
    t0 = time.perf_counter()
    ok, fitted, stats, msg = fit_spectra(parsed, network, spectra, params, logK_vals, fit_keys, x_expr, wl[0], wl[-1],
                                         tolerance, maxiter, timeout_s=timeout_s, allow_negative_eps=allow_negative_eps)
    secs = time.perf_counter() - t0
    logK = np.array([fitted[k] for k in fit_keys]) if ok else np.full(n_steps, np.nan)
    pe = stats.get("param_errors", {}) if ok else {}
    se = np.array([pe.get(k, np.nan) for k in fit_keys]) if ok else np.full(n_steps, np.nan)
    return dict(logbeta=np.cumsum(logK), logK=logK, se_logK=se, success=bool(ok), seconds=secs, message=msg,
                timed_out=bool(stats.get("timed_out", False)) if ok else None, n_iter=stats.get("n_iter") if ok else None,
                stats=stats)
