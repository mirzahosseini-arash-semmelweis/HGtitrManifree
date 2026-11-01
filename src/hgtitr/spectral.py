"""Full-spectrum model with smooth species spectra that may shift in wavelength.

Observation model for titration point i and wavelength lambda:

    y_i(lambda) = sum_j C_ij * r_j(lambda - q_i * dlam_j) + e_i(lambda)

* r_j is a cubic B-spline with K uniform knots (linear coefficients,
  profiled by least squares = variable projection, or GLS under AR(1)).
* dlam_j is a species-specific maximal displacement (nm); unshifted species
  have dlam_j = 0.
* q_i in [0, 1] is an explicit, *defined* equilibrium coordinate:

  ``bound_host``  q = sum_complexes a_j [S_j] / H_T     (fraction of host bound)
  ``saturation``  q = sum_complexes b_j [S_j] / (H_T * max_j(b_j / a_j))
                  (legacy definition; identical to bound_host for 1:1 models)

* Noise: iid Gaussian, or AR(1) along wavelength within each spectrum
  (feasible GLS: Prais-Winsten whitening of data *and* design).

Important identifiability note (checked in tests): two same-stoichiometry
states (e.g. H1G1a, H1G1b) always keep the ratio [a]/[b] = beta_a/beta_b, so
only beta_a + beta_b is identifiable; the split is absorbed by the spectra.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc

from .model import Model
from .speciation import solve_speciation


# --------------------------------------------------------------------------- #
def cubic_bspline(u, xp=np):
    """Centred cardinal cubic B-spline, support [-2, 2]."""
    a = xp.abs(u)
    return xp.where(a < 1, 2.0 / 3.0 - a ** 2 + 0.5 * a ** 3, xp.where(a < 2, (2.0 - a) ** 3 / 6.0, 0.0))


@dataclass(frozen=True)
class Basis:
    lo: float
    hi: float
    K: int

    @property
    def h(self):
        return (self.hi - self.lo) / (self.K - 3)

    @property
    def centres(self):
        return self.lo + (np.arange(self.K) - 1) * self.h

    def __call__(self, lam, xp=np):
        return cubic_bspline((lam[..., None] - xp.asarray(self.centres)) / self.h, xp)


@dataclass
class SpectralData:
    HT: np.ndarray
    GT: np.ndarray
    wl: np.ndarray       # (W,)
    Y: np.ndarray        # (N, W)

    def subset(self, lo=None, hi=None, step: int = 1):
        m = np.ones(self.wl.size, bool)
        if lo is not None:
            m &= self.wl >= lo
        if hi is not None:
            m &= self.wl <= hi
        idx = np.where(m)[0][::step]
        return SpectralData(self.HT, self.GT, self.wl[idx], self.Y[:, idx])


def occupancy(C, model: Model, HT, mode: str = "bound_host"):
    Cx = C[:, 2:]
    if mode == "bound_host":
        q = Cx @ model.nH / HT
    elif mode == "saturation":
        q = Cx @ model.nG / (HT * model.max_guest_per_host)
    else:
        raise ValueError("q mode must be 'bound_host' or 'saturation'")
    return np.clip(q, 0.0, 1.0)


def ar1_whiten(E, rho):
    """Prais-Winsten whitening along the last axis (wavelength)."""
    W = np.empty_like(E)
    W[..., 0] = np.sqrt(1 - rho ** 2) * E[..., 0]
    W[..., 1:] = E[..., 1:] - rho * E[..., :-1]
    return W


@dataclass
class SpectralFit:
    model: Model
    shifted: tuple
    logbeta: np.ndarray
    dlam: np.ndarray
    rho: float
    sigma: float
    coef: np.ndarray
    fitted: np.ndarray
    residual: np.ndarray
    q: np.ndarray
    loglik: float
    k: int
    n: int
    aic: float
    aicc: float
    bic: float
    seconds: float
    se: Optional[np.ndarray] = None

    def summary(self):
        s = [f"{' + '.join(self.model.species)} | shifted: {list(self.shifted) or 'none'} | rho={self.rho:.4f}"]
        for sp, b in zip(self.model.species, self.logbeta):
            s.append(f"  log beta {sp:7s} = {b:.4f}")
        for sp, d in zip(self.shifted, self.dlam):
            s.append(f"  dlam_max {sp:7s} = {d:+.2f} nm")
        s.append(f"  sigma = {self.sigma:.4g}, -2logL = {-2*self.loglik:.2f}, k = {self.k}, AIC = {self.aic:.2f}, BIC = {self.bic:.2f}")
        return "\n".join(s)


class SpectralProblem:
    def __init__(self, model: Model, data: SpectralData, shifted: Sequence[str] = (), n_basis: int = 10,
                 q_mode: str = "bound_host", active: Optional[Sequence[bool]] = None, basis: str = "bspline",
                 margin: float = 0.0):
        self.model, self.data = model, data
        self.shifted = tuple(shifted)
        for s in self.shifted:
            if s not in model.species:
                raise ValueError(f"Shifted species {s} not in model.")
        self.sidx = [model.species.index(s) + 2 for s in self.shifted]
        self.q_mode = q_mode
        n_sp = model.n_complex + 2
        self.active = np.ones(n_sp, bool) if active is None else np.asarray(active, bool)
        self.kind = basis
        if basis == "pointwise" and self.shifted:
            raise ValueError("Pointwise (per-wavelength) spectra cannot be shifted.")
        self.basis = Basis(data.wl.min() - margin, data.wl.max() + margin, n_basis)
        self.p = model.n_complex
        self.m = len(self.shifted)

    def design(self, theta):
        lb, dl = theta[: self.p], theta[self.p :]
        d = self.data
        C = solve_speciation(d.HT, d.GT, self.model, lb).C
        q = occupancy(C, self.model, d.HT, self.q_mode)
        N, W = d.Y.shape
        cols = []
        for j in np.where(self.active)[0]:
            if self.kind == "pointwise":
                B = np.broadcast_to(np.eye(W), (N, W, W))
            else:
                shift = np.zeros(N)
                if j in self.sidx:
                    shift = q * dl[self.sidx.index(j)]
                B = self.basis(d.wl[None, :] - shift[:, None])  # (N, W, K)
            cols.append(C[:, j, None, None] * B)
        X = np.concatenate(cols, axis=2)  # (N, W, P)
        return X, C, q

    def solve(self, theta, rho=0.0):
        X, C, q = self.design(theta)
        Y = self.data.Y
        N, W, P = X.shape
        if rho != 0.0:
            Xw = ar1_whiten(np.moveaxis(X, 2, 1), rho)       # (N, P, W)
            Xw = np.moveaxis(Xw, 1, 2).reshape(N * W, P)
            yw = ar1_whiten(Y, rho).ravel()
        else:
            Xw, yw = X.reshape(N * W, P), Y.ravel()
        sc = np.sqrt((Xw ** 2).sum(0)); sc[sc == 0] = 1.0
        b = np.linalg.lstsq(Xw / sc, yw, rcond=1e-12)[0] / sc
        fitted = (X.reshape(N * W, P) @ b).reshape(N, W)
        rw = yw - Xw @ b
        rank = np.linalg.matrix_rank(Xw / sc, tol=1e-10 * np.sqrt(N * W))
        return rw, b, fitted, C, q, rank

    def resid(self, theta, rho=0.0):
        return self.solve(theta, rho)[0]


def _loglik(rw, n, rho, n_spec):
    rss = float(rw @ rw)
    return -0.5 * n * (np.log(2 * np.pi * rss / n) + 1.0) + 0.5 * n_spec * np.log(1 - rho ** 2), np.sqrt(rss / n)


def estimate_rho(residual):
    E = residual
    num = np.sum(E[:, 1:] * E[:, :-1])
    den = np.sum(E[:, :-1] ** 2)
    return float(np.clip(num / den, -0.999, 0.9995))


def fit_spectral(model: Model, data: SpectralData, shifted: Sequence[str] = (), n_basis: int = 10,
                 q_mode: str = "bound_host", noise: str = "iid", start=None, n_starts: int = 8,
                 logbeta_bounds=(1.0, 14.0), dlam_bounds=(-60.0, 60.0), active=None, basis="bspline",
                 seed: int = 0, margin: float = 0.0, compute_se: bool = True) -> SpectralFit:
    t0 = time.perf_counter()
    prob = SpectralProblem(model, data, shifted, n_basis, q_mode, active, basis, margin)
    p, m = prob.p, prob.m
    lo = np.concatenate([np.full(p, logbeta_bounds[0]), np.full(m, dlam_bounds[0])])
    hi = np.concatenate([np.full(p, logbeta_bounds[1]), np.full(m, dlam_bounds[1])])
    starts = [] if start is None else [np.asarray(s, float) for s in np.atleast_2d(start)]
    if len(starts) < n_starts:
        u = qmc.Sobol(p + m, seed=seed).random_base2(int(np.ceil(np.log2(max(2, n_starts)))))[: n_starts - len(starts)]
        starts += list(lo + u * (hi - lo))
    N, W = data.Y.shape
    n = N * W
    best = None
    for s in starts:
        s = np.clip(s, lo + 1e-6, hi - 1e-6)
        try:
            sol = least_squares(prob.resid, s, bounds=(lo, hi), x_scale="jac", method="trf", args=(0.0,))
            th, rho = sol.x, 0.0
            if noise == "ar1":
                for _ in range(3):
                    rw, b, fitted, C, q, rank = prob.solve(th, 0.0) if rho == 0.0 else prob.solve(th, rho)
                    E = data.Y - fitted
                    rho = estimate_rho(E)
                    sol = least_squares(prob.resid, th, bounds=(lo, hi), x_scale="jac", method="trf", args=(rho,))
                    th = sol.x
            rw = prob.resid(th, rho)
            ll, sig = _loglik(rw, n, rho, N)
            if best is None or ll > best[0]:
                best = (ll, th, rho, sol)
        except (RuntimeError, FloatingPointError, np.linalg.LinAlgError):
            continue
    if best is None:
        raise RuntimeError("All spectral starts failed.")
    ll, th, rho, sol = best
    rw, b, fitted, C, q, rank = prob.solve(th, rho)
    ll, sig = _loglik(rw, n, rho, N)
    k = p + m + rank + 1 + int(noise == "ar1")
    se = None
    if compute_se:
        try:
            J = sol.jac if sol.jac is not None else None
            dof_sig2 = float(rw @ rw) / max(n - k, 1)
            se = np.sqrt(np.diag(np.linalg.pinv(J.T @ J)) * dof_sig2)
        except Exception:
            se = None
    aic = -2 * ll + 2 * k
    return SpectralFit(model, prob.shifted, th[:p], th[p:], rho, sig, b, fitted, data.Y - fitted, q, ll, k, n,
                       aic, aic + 2 * k * (k + 1) / max(n - k - 1, 1), -2 * ll + k * np.log(n),
                       time.perf_counter() - t0, se)


# --------------------------------------------------------------------------- #
def synthetic_band_spectra(HT, GT, model: Model, logbeta, wl, bands: dict, shift_species: dict,
                           q_mode="bound_host", noise_sd=0.0, rng=None):
    """Simulate spectra from Gaussian bands.

    bands : {species_label_or_'H'_'G': [(amplitude, centre_nm, width_nm), ...]}
    shift_species : {label: dlam_max_nm} applied as centre + q * dlam.
    """
    C = solve_speciation(HT, GT, model, logbeta).C
    q = occupancy(C, model, np.asarray(HT, float), q_mode)
    names = list(model.all_species)
    Y = np.zeros((len(HT), len(wl)))
    for sp, bl in bands.items():
        j = names.index(sp)
        d = shift_species.get(sp, 0.0)
        for amp, cen, wid in bl:
            centre = cen + q[:, None] * d
            Y += C[:, j, None] * amp * np.exp(-0.5 * ((wl[None, :] - centre) / wid) ** 2)
    if noise_sd > 0:
        rng = np.random.default_rng() if rng is None else rng
        Y = Y + rng.normal(0, noise_sd, Y.shape)
    return Y, C, q
