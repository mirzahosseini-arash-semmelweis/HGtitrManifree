"""The free-component manifold (forward map) and its uses.

For fixed constants, every point (h, g) of free-concentration space maps
*explicitly* to totals (H_T, G_T), species concentrations and signals.  The
legacy HGtitrManifree simulator sampled this manifold on an adaptive log grid
and interpolated the signal at the experimental totals with a thin-plate
spline (TPS).  This module

1. ports that emulator faithfully (``tps_signal``) so that its error can be
   quantified against the exact inverse (``hgtitr.speciation``);
2. provides the uses of the forward map that need *no* inverse solve and no
   interpolation:  speciation/signal surfaces over the (H_T, G_T) plane, and
   exact local sensitivities d(signal)/d(log beta) everywhere on the manifold
   (``sensitivity_map``) for experimental design.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RBFInterpolator

from .model import LN10, Model
from .speciation import solve_speciation, totals_from_free


# --------------------------------------------------------------------------- #
@dataclass
class Manifold:
    h: np.ndarray
    g: np.ndarray
    HT: np.ndarray
    GT: np.ndarray
    C: np.ndarray
    n_grid_total: int
    seconds: float


def _log_axis(lo, hi, n, corner_vals, zero_floor_decades=8.0):
    if hi <= 0:
        return np.array([0.0])
    if lo == hi:
        return np.array([lo])
    if lo > 0:
        return np.exp(np.linspace(np.log(lo), np.log(hi), n))
    pos = corner_vals[corner_vals > 0]
    plo = max(min(np.concatenate([pos, [hi * 10 ** (-zero_floor_decades)]])), np.finfo(float).tiny)
    return np.unique(np.concatenate([[0.0], np.exp(np.linspace(np.log(plo), np.log(hi), n - 1))]))


def free_manifold(model: Model, logbeta, HT_range, GT_range, n=(100, 500), pad_decades=0.05) -> Manifold:
    """Sample the forward map on the free-concentration box that spans the
    requested total-concentration domain (legacy ``HG_titr_matrix``)."""
    t0 = time.perf_counter()
    Hl, Hh = float(np.min(HT_range)), float(np.max(HT_range))
    Gl, Gh = float(np.min(GT_range)), float(np.max(GT_range))
    cH = np.array([Hl, Hl, Hh, Hh]); cG = np.array([Gl, Gh, Gl, Gh])
    sp = solve_speciation(cH, cG, model, logbeta)  # 4 corner solves, as in the legacy code
    h_ax = _log_axis(sp.h.min(), sp.h.max(), n[0], sp.h)
    g_ax = _log_axis(sp.g.min(), sp.g.max(), n[1], sp.g)
    hh, gg = np.meshgrid(h_ax, g_ax, indexing="ij")
    hh, gg = hh.ravel(), gg.ravel()
    HT, GT, C = totals_from_free(hh, gg, model, logbeta)
    f = 10 ** pad_decades
    keep = (HT >= Hl / f) & (HT <= Hh * f) & (GT >= (Gl / f if Gl > 0 else 0.0)) & (GT <= Gh * f)
    return Manifold(hh[keep], gg[keep], HT[keep], GT[keep], C[keep], hh.size, time.perf_counter() - t0)


def _thin(X, max_train):
    if X.shape[0] <= max_train:
        return np.arange(X.shape[0])
    nb = max(2, int(np.floor(np.sqrt(max_train))))
    lo, hi = X.min(0), X.max(0)
    b = np.minimum(((X - lo) / np.where(hi > lo, hi - lo, 1) * nb).astype(int), nb - 1)
    key = b[:, 0] * nb + b[:, 1]
    keep = []
    for k in np.unique(key):
        idx = np.where(key == k)[0]
        c = X[idx].mean(0)
        keep.append(idx[np.argmin(((X[idx] - c) ** 2).sum(1))])
    return np.sort(np.array(keep))


def tps_species(model: Model, logbeta, HT, GT, n=(100, 500), max_train=1200, padding_decades=0.25,
                smoothing=0.0, G_offset=1e-10, return_info=False):
    """Legacy emulator: species concentrations at (HT, GT) by thin-plate-spline
    interpolation of the forward manifold in (log10 H_T, log10(G_T + offset)).

    Because a TPS with fixed nodes and smoothing is linear in the data, the
    emulated signal for any response vector r equals ``tps_species(...) @ r``,
    i.e. exactly what the legacy code obtained by interpolating the signal.
    ``smoothing = 0`` interpolates the noiseless manifold exactly (the most
    favourable choice; the legacy R code used a GCV-selected smoothing).
    """
    t0 = time.perf_counter()
    HT = np.asarray(HT, float); GT = np.asarray(GT, float)
    man = free_manifold(model, logbeta, (HT.min(), HT.max()), (GT.min(), GT.max()), n=n)
    ok = man.HT > 0
    X = np.column_stack([np.log10(man.HT[ok]), np.log10(man.GT[ok] + G_offset)])
    Yc = man.C[ok]
    Xp = np.column_stack([np.log10(HT), np.log10(GT + G_offset)])
    lo = Xp.min(0) - padding_decades; hi = Xp.max(0) + padding_decades
    loc = np.all((X >= lo) & (X <= hi), axis=1)
    X, Yc = X[loc], Yc[loc]
    if X.shape[0] < 20:
        raise RuntimeError("Too few manifold points near the experimental domain.")
    X, inv = np.unique(X, axis=0, return_inverse=True)
    inv = inv.ravel()
    cnt = np.bincount(inv)
    Yc = np.column_stack([np.bincount(inv, weights=Yc[:, j]) / cnt for j in range(Yc.shape[1])])
    n_local = X.shape[0]
    idx = _thin(X, max_train)
    X, Yc = X[idx], Yc[idx]
    lo2, rng = X.min(0), np.ptp(X, 0)
    rng[rng == 0] = 1.0
    rbf = RBFInterpolator((X - lo2) / rng, Yc, kernel="thin_plate_spline", degree=1, smoothing=smoothing)
    pred = rbf((Xp - lo2) / rng)
    if return_info:
        return pred, dict(n_manifold=man.HT.size, n_local=n_local, n_train=X.shape[0],
                          seconds=time.perf_counter() - t0)
    return pred


def tps_signal(model: Model, logbeta, response, HT, GT, return_info=False, **kw):
    """Legacy TPS emulator of the signal for response vector over [H, G, complexes]."""
    out = tps_species(model, logbeta, HT, GT, return_info=return_info, **kw)
    if return_info:
        return out[0] @ np.asarray(response, float), out[1]
    return out @ np.asarray(response, float)


def exact_signal(model: Model, logbeta, response, HT, GT):
    return solve_speciation(HT, GT, model, logbeta).C @ np.asarray(response, float)


# --------------------------------------------------------------------------- #
def sensitivity_map(model: Model, logbeta, response, h, g):
    """Exact d(signal)/d(log10 beta_j) at fixed totals, evaluated at free (h, g).

    Uses the implicit-function theorem on the explicit forward map, so no
    inverse solve is required:  dx/dlnb = -H^{-1} S^T diag(c) E_j.
    Returns (HT, GT, signal, dS) with dS of shape (N, n_complex).
    """
    HT, GT, C = totals_from_free(h, g, model, logbeta)
    S = model.stoich
    r = np.asarray(response, float)
    a, b = S[:, 0], S[:, 1]
    H00, H01, H11 = C @ (a * a), C @ (a * b), C @ (b * b)
    det = H00 * H11 - H01 ** 2
    det = np.where(det > 0, det, np.inf)  # blank points (h or g = 0): no complexes, zero sensitivity
    dS = np.zeros((C.shape[0], model.n_complex))
    for j in range(model.n_complex):
        k = j + 2
        rhs0, rhs1 = -C[:, k] * a[k], -C[:, k] * b[k]
        dx0 = (H11 * rhs0 - H01 * rhs1) / det
        dx1 = (-H01 * rhs0 + H00 * rhs1) / det
        dC = C * (np.outer(dx0, a) + np.outer(dx1, b))
        dC[:, k] += C[:, k]
        dS[:, j] = LN10 * dC @ r
    return HT, GT, C @ r, dS
