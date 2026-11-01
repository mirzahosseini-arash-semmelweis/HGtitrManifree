"""Summary tables (Markdown) and figures (PNG) from the benchmark CSVs.

python make_figures.py   ->  results/summary.md, results/fig_*.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import RESULTS

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

out = ["# Benchmark summary\n"]

# ---- 1. TPS accuracy -------------------------------------------------------
e = pd.read_csv(f"{RESULTS}/bench01_tps_error.csv")
t = e.groupby(["n_h", "n_g", "max_train"]).agg(median_err_pct=("max_rel_signal_err", lambda x: 100 * x.median()),
                                               max_err_pct=("max_rel_signal_err", lambda x: 100 * x.max()),
                                               t_tps_ms=("t_tps_ms", "median"), t_exact_ms=("t_exact_ms", "median"))
out += ["## 1. TPS emulator error (max over titration, % of signal range)\n", t.round(3).to_markdown(), "\n"]
f = pd.read_csv(f"{RESULTS}/bench01_tps_fitbias.csv")
f["abs_err_noiseless"] = (f.tps_est - f.true).abs()
f["dev_in_se"] = ((f.tps_est - f.exact_est) / f.exact_se).abs()
tf = pd.DataFrame({
    "median |err| noiseless (log units)": f[f.data == "noiseless"].groupby("max_train").abs_err_noiseless.median(),
    "median |TPS-exact|/SE noisy": f[f.data == "noisy"].groupby("max_train").dev_in_se.median(),
    "median fit time TPS (s)": f.groupby("max_train").t_fit_tps_s.median(),
    "median fit time exact (s)": f.groupby("max_train").t_fit_exact_s.median()})
out += ["### Effect on fitted constants\n", tf.round(3).to_markdown(), "\n"]
if plt:
    leg = e[(e.n_h == 100) & (e.logK == 5)]
    fig, ax = plt.subplots(figsize=(6, 4))
    for mt, g in leg.groupby("max_train"):
        w = g.groupby("c").max_rel_signal_err.max() * 100
        ax.plot(w.index, w.values, "o-", label=f"max_train {mt}")
    ax.axhline(0.4, ls="--", c="grey", label="noise in manuscript scenarios")
    ax.set(xscale="log", yscale="log", xlabel="c = K·H_T", ylabel="max error (% of signal range)")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{RESULTS}/fig_tps_error.png", dpi=200)

# ---- 2. bias / precision / cost --------------------------------------------
try:
    s = pd.read_csv(f"{RESULTS}/bench02_summary.csv")
    out += ["## 2. Bias, precision, robustness and cost\n", s.round(4).to_markdown(index=False), "\n"]
except FileNotFoundError:
    pass

# ---- 3. coverage ------------------------------------------------------------
try:
    c = pd.read_csv(f"{RESULTS}/bench03_coverage.csv")
    piv = c.pivot_table(index=["scenario", "species"], columns="method", values="coverage")
    out += ["## 3. Coverage of nominal 95 % intervals\n", piv.round(3).to_markdown(), "\n",
            "### Median interval width\n",
            c.pivot_table(index=["scenario", "species"], columns="method", values="median_width").round(4)
            .to_markdown(), "\n", "### Median seconds per dataset\n",
            c.pivot_table(index=["scenario"], columns="method", values="median_seconds").round(2).to_markdown(), "\n"]
except FileNotFoundError:
    pass

# ---- 4. spectral shift ------------------------------------------------------
try:
    sp = pd.read_csv(f"{RESULTS}/bench04_spectral.csv").dropna(subset=["aic"])
    for crit in ("aic", "bic"):
        best = sp.loc[sp.groupby(["truth", "design", "rep", "noise"])[crit].idxmin()]
        tab = best.groupby(["truth", "design", "noise"]).candidate.value_counts().unstack(fill_value=0)
        out += [f"## 4. Spectral-shift validation: model selected by {crit.upper()}\n", tab.to_markdown(), "\n"]
    a = sp[(sp.truth == "A_shift_1to1") & (sp.candidate == "1:1 shifted")].copy()
    a["logK"] = a.logbeta.astype(float)
    a["dlam_hat"] = a.dlam.astype(float)
    rec = a.groupby(["design", "noise"]).agg(logK_mean=("logK", "mean"), logK_sd=("logK", "std"),
                                              dlam_mean=("dlam_hat", "mean"), dlam_sd=("dlam_hat", "std"))
    out += ["### Recovery under truth A (log K = 5.9, dlam = +12 nm) with the 1:1 shifted model\n",
            rec.round(3).to_markdown(), "\n"]
except (FileNotFoundError, KeyError):
    pass

open(f"{RESULTS}/summary.md", "w").write("\n".join(out))
print("\n".join(out))
