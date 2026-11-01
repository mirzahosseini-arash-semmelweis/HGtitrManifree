"""Headless use of Musketeer's own speciation solver and fitting protocol.

Musketeer (Soloviev & Hunter, Chem. Sci. 2024, 15, 15299; pip install
musketeer) is a GUI application.  Its numerical core is pure SciPy, but the
modules import tkinter at load time.  We stub the GUI modules, use Musketeer's
`SpeciationSolver.run` unchanged for speciation, and reproduce
`Titration.optimise` (Nelder-Mead on log10 K, linear least squares for the
signals with Musketeer's `FitSignalsUnconstrained.leastSquares`).
"""
from __future__ import annotations

import sys
import time
import types
from unittest import mock

import numpy as np
from scipy.optimize import minimize


def _stub_gui():
    for name in ["tkinter", "tkinter.ttk", "tkinter.font", "tkinter.filedialog", "tkinter.messagebox",
                 "tksheet", "tkinter.scrolledtext", "ttkbootstrap", "darkdetect"]:
        if name not in sys.modules:
            sys.modules[name] = mock.MagicMock(name=name)


_stub_gui()
from musketeer import speciation as _msp  # noqa: E402
from musketeer.fitSignals import FitSignalsUnconstrained  # noqa: E402

VERSION = __import__("importlib.metadata").metadata.version("musketeer")


class _Tot:
    freeNames = np.array(["Host", "Guest"])


class _Tit:
    totalConcentrations = _Tot()


class MusketeerSpeciation(_msp.SpeciationSolver):
    """Musketeer's general solver with a programmatic stoichiometry table."""

    def __init__(self, stoich):
        super().__init__(_Tit())
        self.stoichiometries = np.asarray(stoich, dtype=int)

    # The GUI normally provides these via the polymer/complex tables
    @property
    def outputCount(self):
        return self.freeCount + self.boundCount


def musketeer_speciate(model, logbeta, HT, GT):
    sp = MusketeerSpeciation(np.column_stack([model.nH, model.nG]))
    totals = np.column_stack([HT, GT])
    return sp.run(10.0 ** np.asarray(logbeta, float), totals), sp


def fit_musketeer(model, HT, GT, Y, active, start, maxiter=None):
    """Musketeer fitting protocol: Nelder-Mead on log10 K, signals by lstsq.

    active : boolean mask over [H, G, complexes] of signal-contributing species.
    Returns dict(logbeta, rmse, nfev, seconds).
    """
    sp = MusketeerSpeciation(np.column_stack([model.nH, model.nG]))
    totals = np.column_stack([HT, GT])
    lsq = FitSignalsUnconstrained.leastSquares
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]

    def obj(logk):
        conc = sp.run(10.0 ** logk, totals)
        X = conc[:, np.asarray(active)]
        _, res = lsq(None, X, Y)
        return np.sqrt(np.sum(res))

    t0 = time.perf_counter()
    opts = {} if maxiter is None else {"maxiter": maxiter}
    r = minimize(obj, x0=np.asarray(start, float), method="nelder-mead", options=opts)
    return dict(logbeta=r.x, rmse=obj(r.x) / np.sqrt(Y.size), nfev=r.nfev, seconds=time.perf_counter() - t0,
                success=r.success)
