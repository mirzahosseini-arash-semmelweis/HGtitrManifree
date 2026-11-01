"""Deterministic global fitting with shared formation constants.

* Any number of experiments (titrations) and channels share the log10
  formation constants; each experiment has its own linear responses.
* ``method="varpro"`` (default) profiles the linear responses by least
  squares at every trial constant vector (variable projection, as in
  SPECFIT / Musketeer / Equilibrist).  ``method="full"`` keeps responses as
  explicit nonlinear parameters (the legacy HGtitrManifree treatment).  Both
  converge to the same optimum; VARPRO is faster and better conditioned.
* Channel noise is either given (``sigma``), or estimated by maximum
  likelihood per channel (iteratively reweighted).
* Optional nuisance factors on stock concentrations (errors in variables).
* Gauss-Newton covariance uses the projected (Kaufman) Jacobian, which equals
  the theta-block of the full-parameter covariance, so response uncertainty
  is propagated into the constants.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.stats import chi2, qmc

from .model import Model
from .observables import default_active, design_matrix
from .speciation import solve_speciation


# --------------------------------------------------------------------------- #
@dataclass
class Experiment:
    """One titration: totals, signals and the observation model."""

    HT: np.ndarray
    GT: np.ndarray
    Y: np.ndarray                       # (N, Q)
    kind: str = "intensity"
    free_H_active: bool = False         # intensity only
    free_G_active: bool = False
    baseline: bool = False
    sigma: Optional[np.ndarray] = None  # (Q,) known noise SD per channel
    name: str = "exp"
    channels: Optional[Sequence[str]] = None
    titrant_factor_sd: Optional[float] = None   # log10 prior SD on G_T scale factor
    host_factor_sd: Optional[float] = None      # log10 prior SD on H_T scale factor

    def __post_init__(self):
        self.HT = np.asarray(self.HT, float)
        self.GT = np.asarray(self.GT, float)
        self.Y = np.asarray(self.Y, float)
        if self.Y.ndim == 1:
            self.Y = self.Y[:, None]
        if not (self.HT.shape == self.GT.shape == (self.Y.shape[0],)):
            raise ValueError("HT, GT and rows of Y must match.")
        if self.channels is None:
            self.channels = [f"ch{i+1}" for i in range(self.Y.shape[1])]
        if self.sigma is not None:
            self.sigma = np.broadcast_to(np.asarray(self.sigma, float), (self.Y.shape[1],)).copy()

    @property
    def n_nuisance(self) -> int:
        return int(self.titrant_factor_sd is not None) + int(self.host_factor_sd is not None)

    def active_mask(self, model: Model):
        return default_active(model.n_complex + 2, self.kind, self.free_H_active, self.free_G_active)


# --------------------------------------------------------------------------- #
class Problem:
    """Residual machinery for a model + list of experiments."""

    def __init__(self, model: Model, experiments: Sequence[Experiment], speciator=None):
        self.speciator = speciator  # optional callable(HT, GT, model, logbeta) -> C (e.g. TPS emulator)
        self.model = model
        self.exps = list(experiments)
        self.S = model.stoich
        self.p = model.n_complex
        self.n_nuis = sum(e.n_nuisance for e in self.exps)
        self.masks = [e.active_mask(model) for e in self.exps]
        self._x0 = [None] * len(self.exps)

    # unpack theta = [logbeta (p), nuisance (n_nuis)]
    def split(self, theta):
        lb = np.asarray(theta[: self.p], float)
        nu = np.asarray(theta[self.p :], float)
        return lb, nu

    def totals(self, nu):
        out, k = [], 0
        for e in self.exps:
            HT, GT = e.HT, e.GT
            if e.titrant_factor_sd is not None:
                GT = GT * 10.0 ** nu[k]; k += 1
            if e.host_factor_sd is not None:
                HT = HT * 10.0 ** nu[k]; k += 1
            out.append((HT, GT))
        return out

    def designs(self, theta):
        lb, nu = self.split(theta)
        As = []
        for j, (e, (HT, GT)) in enumerate(zip(self.exps, self.totals(nu))):
            if self.speciator is None:
                sp = solve_speciation(HT, GT, self.model, lb, x0=self._x0[j])
                self._x0[j] = np.log(np.maximum(sp.free, 1e-300))
                C = sp.C
            else:
                C = self.speciator(HT, GT, self.model, lb)
            As.append(design_matrix(C, self.S, HT, GT, e.kind, self.masks[j], e.baseline))
        return As

    def weights(self, sigmas):
        return [1.0 / s for s in sigmas]

    def nuisance_penalty(self, nu):
        sds = []
        for e in self.exps:
            if e.titrant_factor_sd is not None:
                sds.append(e.titrant_factor_sd)
            if e.host_factor_sd is not None:
                sds.append(e.host_factor_sd)
        return np.asarray(nu) / np.asarray(sds) if sds else np.zeros(0)

    # VARPRO residual
    def resid_varpro(self, theta, sigmas, return_R=False):
        As = self.designs(theta)
        out, Rs = [], []
        for A, e, s in zip(As, self.exps, sigmas):
            R = np.linalg.lstsq(A, e.Y, rcond=None)[0]
            Rs.append(R)
            out.append(((e.Y - A @ R) / s[None, :]).ravel())
        r = np.concatenate(out + [self.nuisance_penalty(self.split(theta)[1])])
        return (r, Rs, As) if return_R else r

    # full-parameter residual (responses explicit)
    def r_sizes(self):
        sz = []
        for e, m in zip(self.exps, self.masks):
            P = int(m.sum()) + int(e.baseline)
            sz.append((P, e.Y.shape[1]))
        return sz

    def unpack_R(self, vecR):
        Rs, k = [], 0
        for P, Q in self.r_sizes():
            Rs.append(np.asarray(vecR[k : k + P * Q]).reshape(P, Q)); k += P * Q
        return Rs

    def resid_full(self, phi, sigmas):
        nt = self.p + self.n_nuis
        theta, Rs = phi[:nt], self.unpack_R(phi[nt:])
        As = self.designs(theta)
        out = [((e.Y - A @ R) / s[None, :]).ravel() for A, R, e, s in zip(As, Rs, self.exps, sigmas)]
        return np.concatenate(out + [self.nuisance_penalty(self.split(theta)[1])])

    def jac_full(self, phi, sigmas, h=1e-6):
        nt = self.p + self.n_nuis
        theta, Rs = phi[:nt], self.unpack_R(phi[nt:])
        As0 = self.designs(theta)
        blocks_t = []
        for k in range(nt):
            tp = theta.copy(); tp[k] += h
            tm = theta.copy(); tm[k] -= h
            Ap, Am = self.designs(tp), self.designs(tm)
            col = [(-((ap - am) / (2 * h)) @ R / s[None, :]).ravel() for ap, am, R, s in zip(Ap, Am, Rs, sigmas)]
            pen = np.zeros(self.n_nuis)
            if k >= self.p:
                pen = (self.nuisance_penalty(np.eye(self.n_nuis)[k - self.p]))
            blocks_t.append(np.concatenate(col + [pen]))
        Jt = np.column_stack(blocks_t)
        nrows = Jt.shape[0]
        JR = np.zeros((nrows, len(phi) - nt))
        row, colk = 0, 0
        for A, e, s in zip(As0, self.exps, sigmas):
            N, Q = e.Y.shape
            P = A.shape[1]
            for q in range(Q):
                rows = row + np.arange(N) * Q + q
                cols = colk + np.arange(P) * Q + q
                JR[np.ix_(rows, cols)] = -A / s[q]
            row += N * Q; colk += P * Q
        return np.hstack([Jt, JR])

    def kaufman_jacobian(self, theta, sigmas, h=1e-6):
        """Projected Jacobian d r / d theta with responses profiled out."""
        _, Rs, As = self.resid_varpro(theta, sigmas, return_R=True)
        cols = []
        nt = len(theta)
        for k in range(nt):
            tp = theta.copy(); tp[k] += h
            tm = theta.copy(); tm[k] -= h
            Ap, Am = self.designs(tp), self.designs(tm)
            parts = []
            for A, ap, am, R, s in zip(As, Ap, Am, Rs, sigmas):
                D = -((ap - am) / (2 * h)) @ R
                D = D - A @ np.linalg.lstsq(A, D, rcond=None)[0]   # project out span(A)
                parts.append((D / s[None, :]).ravel())
            pen = np.zeros(self.n_nuis)
            if k >= self.p:
                pen = self.nuisance_penalty(np.eye(self.n_nuis)[k - self.p])
            cols.append(np.concatenate(parts + [pen]))
        self.designs(theta)  # restore warm starts
        return np.column_stack(cols)

    def n_obs(self):
        return sum(e.Y.size for e in self.exps)

    def n_linear(self):
        return sum(P * Q for P, Q in self.r_sizes())


# --------------------------------------------------------------------------- #
@dataclass
class FitResult:
    model: Model
    logbeta: np.ndarray
    nuisance: np.ndarray
    se: np.ndarray
    cov: np.ndarray
    responses: list
    sigmas: list
    residuals: list
    rss_weighted: float
    loglik: float
    k: int
    n: int
    aic: float
    aicc: float
    bic: float
    success: bool
    nfev: int
    seconds: float
    method: str
    starts: list = field(default_factory=list)
    problem: Optional[Problem] = field(default=None, repr=False)

    def summary(self) -> str:
        lines = [f"Model {' + '.join(self.model.species)}  ({self.method}, k={self.k}, n={self.n})"]
        for s, b, e in zip(self.model.species, self.logbeta, self.se[: len(self.logbeta)]):
            lines.append(f"  log beta {s:8s} = {b:8.4f} +/- {e:.4f}")
        for i, v in enumerate(self.nuisance):
            lines.append(f"  log10 conc. factor {i} = {v:+.4f} +/- {self.se[len(self.logbeta)+i]:.4f}")
        lines.append(f"  -2 logL = {-2*self.loglik:.3f}  AICc = {self.aicc:.3f}  BIC = {self.bic:.3f}")
        return "\n".join(lines)


def _gaussian_loglik(residuals, sigmas, sigma_known):
    """Log-likelihood; ML per-channel sigma when unknown."""
    ll = 0.0
    for r, s, known in zip(residuals, sigmas, sigma_known):
        N = r.shape[0]
        if known:
            ll += -0.5 * np.sum((r / s[None, :]) ** 2) - N * np.sum(np.log(s)) - 0.5 * r.size * np.log(2 * np.pi)
        else:
            rss = np.sum(r ** 2, axis=0)
            ll += -0.5 * N * np.sum(np.log(2 * np.pi * rss / N) + 1.0)
    return ll


def fit(
    model: Model,
    experiments: Sequence[Experiment] | Experiment,
    start=None,
    bounds=(-2.0, 18.0),
    method: str = "varpro",
    n_starts: int = 1,
    seed: int = 0,
    reweight_iter: int = 4,
    max_nfev: int = 2000,
    compute_cov: bool = True,
    speciator=None,
) -> FitResult:
    """Fit log10 formation constants shared across experiments.

    Parameters
    ----------
    start : initial log10 constants (length = n complexes) or list of them.
        Additional random starts (Sobol, inside ``bounds``) are added until
        ``n_starts`` starts are available.
    speciator : optional replacement for the exact solver, callable
        (HT, GT, model, logbeta) -> C; used to benchmark the legacy TPS emulator.
    bounds : (low, high) applied to every log beta, or a pair of arrays.
    """
    if isinstance(experiments, Experiment):
        experiments = [experiments]
    t0 = time.perf_counter()
    prob = Problem(model, experiments, speciator)
    p, nn = prob.p, prob.n_nuis
    lo, hi = np.broadcast_to(bounds[0], (p,)).astype(float), np.broadcast_to(bounds[1], (p,)).astype(float)
    starts = []
    if start is not None:
        start = np.atleast_2d(np.asarray(start, float))
        starts += [s for s in start]
    if len(starts) < n_starts:
        need = n_starts - len(starts)
        sob = qmc.Sobol(p, seed=seed).random_base2(int(np.ceil(np.log2(max(2, need)))))
        starts += list(lo + sob[: n_starts - len(starts)] * (hi - lo))
    lo_t = np.concatenate([lo, np.full(nn, -1.0)])
    hi_t = np.concatenate([hi, np.full(nn, 1.0)])

    sigma_known = [e.sigma is not None for e in experiments]
    sig0 = [e.sigma if e.sigma is not None else np.maximum(np.std(e.Y, axis=0), 1e-12) * 1e-2 for e in experiments]

    best, records, total_nfev = None, [], 0
    for s0 in starts:
        theta0 = np.concatenate([np.clip(s0, lo + 1e-9, hi - 1e-9), np.zeros(nn)])
        sigmas = [s.copy() for s in sig0]
        try:
            for _ in range(reweight_iter if not all(sigma_known) else 1):
                if method == "varpro":
                    sol = least_squares(prob.resid_varpro, theta0, args=(sigmas,), bounds=(lo_t, hi_t),
                                        x_scale="jac", max_nfev=max_nfev, method="trf")
                    theta = sol.x
                    _, Rs, As = prob.resid_varpro(theta, sigmas, return_R=True)
                elif method == "varpro_nm":
                    # derivative-free (Nelder-Mead) on the VARPRO objective; needed for
                    # non-smooth emulators such as the legacy TPS surface
                    f = lambda th: 0.5 * np.sum(prob.resid_varpro(np.clip(th, lo_t, hi_t), sigmas) ** 2)  # noqa: E731
                    nm = minimize(f, theta0, method="Nelder-Mead",
                                  options=dict(xatol=1e-5, fatol=1e-10, maxfev=max_nfev, adaptive=True))
                    theta = np.clip(nm.x, lo_t, hi_t)
                    sol = type("S", (), dict(nfev=nm.nfev, success=nm.success))()
                    _, Rs, As = prob.resid_varpro(theta, sigmas, return_R=True)
                elif method == "full":
                    As = prob.designs(theta0)
                    R0 = [np.linalg.lstsq(A, e.Y, rcond=None)[0] for A, e in zip(As, experiments)]
                    phi0 = np.concatenate([theta0] + [R.ravel() for R in R0])
                    big = np.full(prob.n_linear(), np.inf)
                    sol = least_squares(prob.resid_full, phi0, jac=prob.jac_full, args=(sigmas,),
                                        bounds=(np.concatenate([lo_t, -big]), np.concatenate([hi_t, big])),
                                        max_nfev=max_nfev, method="trf", x_scale="jac")
                    theta = sol.x[: p + nn]
                    Rs = prob.unpack_R(sol.x[p + nn :])
                    As = prob.designs(theta)
                else:
                    raise ValueError("method must be 'varpro', 'varpro_nm' or 'full'")
                total_nfev += sol.nfev
                res = [e.Y - A @ R for A, R, e in zip(As, Rs, experiments)]
                new = [e.sigma if known else np.sqrt(np.maximum(np.mean(r ** 2, axis=0), 1e-300))
                       for e, r, known in zip(experiments, res, sigma_known)]
                theta0 = theta
                if all(np.allclose(a, b, rtol=1e-4) for a, b in zip(new, sigmas)):
                    sigmas = new
                    break
                sigmas = new
            ll = _gaussian_loglik(res, sigmas, sigma_known)
            ll += -0.5 * np.sum(prob.nuisance_penalty(theta[p:]) ** 2)
            records.append(dict(start=s0, logbeta=theta[:p].copy(), loglik=ll, success=bool(sol.success)))
            if best is None or ll > best[0]:
                best = (ll, theta.copy(), Rs, sigmas, res, sol.success)
        except (RuntimeError, FloatingPointError, np.linalg.LinAlgError) as err:
            records.append(dict(start=s0, logbeta=None, loglik=-np.inf, success=False, error=str(err)))
    if best is None:
        raise RuntimeError("All starts failed.")
    ll, theta, Rs, sigmas, res, ok = best
    n = prob.n_obs()
    k = p + nn + prob.n_linear() + sum(e.Y.shape[1] for e, kn in zip(experiments, sigma_known) if not kn)
    cov = np.full((p + nn, p + nn), np.nan)
    if compute_cov:
        try:
            J = prob.kaufman_jacobian(theta, sigmas)
            cov = np.linalg.pinv(J.T @ J)
            if not all(sigma_known):  # small-sample correction of ML sigma
                cov *= n / max(n - k, 1)
        except Exception as err:  # pragma: no cover
            warnings.warn(f"Covariance failed: {err}")
    se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    rssw = float(sum(np.sum((r / s[None, :]) ** 2) for r, s in zip(res, sigmas)))
    aic = -2 * ll + 2 * k
    aicc = aic + 2 * k * (k + 1) / max(n - k - 1, 1)
    bic = -2 * ll + k * np.log(n)
    return FitResult(model, theta[:p], theta[p:], se, cov, Rs, sigmas, res, rssw, ll, k, n, aic, aicc, bic,
                     bool(ok), total_nfev, time.perf_counter() - t0, method, records, prob)


# --------------------------------------------------------------------------- #
def compare_models(results: Sequence[FitResult], criterion: str = "aicc"):
    """Table of information criteria with delta and Akaike weights."""
    vals = np.array([getattr(r, criterion) for r in results])
    d = vals - vals.min()
    w = np.exp(-0.5 * d); w /= w.sum()
    rows = []
    for r, v, dd, ww in zip(results, vals, d, w):
        rows.append(dict(model=" + ".join(r.model.species), k=r.k, loglik=r.loglik, **{criterion: v},
                         delta=dd, weight=ww, logbeta=np.round(r.logbeta, 4).tolist()))
    return rows


def profile_ci(result: FitResult, index: int, level: float = 0.95, span: float | None = None, n_grid: int = 0):
    """Profile-likelihood interval for one parameter (others re-optimised).

    The interval ends are located by root finding on the profile deviance
    (chi2_1 threshold), bracketing outwards from the optimum in steps of
    ~2 Wald SE.  Returns (lo, hi, grid, deviance); grid/deviance are only
    filled when ``n_grid > 0`` (for plotting).  NaN means the profile does
    not reach the threshold inside the parameter bounds (non-identifiable
    direction).
    """
    from scipy.optimize import brentq

    prob = result.problem
    exps = prob.exps
    target = chi2.ppf(level, 1)
    theta_hat = np.concatenate([result.logbeta, result.nuisance])
    sig_known = [e.sigma is not None for e in exps]
    free_idx = [i for i in range(len(theta_hat)) if i != index]
    warm = {"x": theta_hat[free_idx].copy()}

    def dev(v):
        th = theta_hat.copy(); th[index] = v
        if free_idx:
            def r(tf):
                t = th.copy(); t[free_idx] = tf
                return prob.resid_varpro(t, result.sigmas)
            sol = least_squares(r, warm["x"], method="trf", x_scale="jac")
            th[free_idx] = sol.x
        _, Rs, As = prob.resid_varpro(th, result.sigmas, return_R=True)
        res = [e.Y - A @ R for A, R, e in zip(As, Rs, exps)]
        ll = _gaussian_loglik(res, result.sigmas, sig_known) - 0.5 * np.sum(prob.nuisance_penalty(th[prob.p:]) ** 2)
        return 2 * (result.loglik - ll)

    se = result.se[index] if np.isfinite(result.se[index]) and result.se[index] > 0 else 0.1
    step = 2.0 * se if span is None else span / 4
    ends = []
    for sgn in (-1.0, 1.0):
        warm["x"] = theta_hat[free_idx].copy()
        a, fa = theta_hat[index], -target
        end = np.nan
        for _ in range(12):
            b = a + sgn * step
            try:
                fb = dev(b) - target
            except (RuntimeError, FloatingPointError, np.linalg.LinAlgError):
                break
            if fb > 0:
                end = brentq(lambda v: dev(v) - target, min(a, b), max(a, b), xtol=1e-4 * se)
                break
            a, fa = b, fb
            step *= 1.6
        ends.append(end)
    grid = dv = None
    if n_grid:
        w = 1.5 * np.nanmax(np.abs(np.array(ends) - theta_hat[index])) if np.any(np.isfinite(ends)) else 1.0
        grid = theta_hat[index] + np.linspace(-w, w, n_grid)
        dv = np.array([dev(v) for v in grid])
    return ends[0], ends[1], grid, dv


def bootstrap(result: FitResult, n_boot: int = 200, seed: int = 0, kind: str = "residual"):
    """Residual (or parametric) bootstrap of the log constants."""
    rng = np.random.default_rng(seed)
    prob = result.problem
    exps = prob.exps
    th = np.concatenate([result.logbeta, result.nuisance])
    As = prob.designs(th)
    fitted = [A @ R for A, R in zip(As, result.responses)]
    out = []
    for _ in range(n_boot):
        new = []
        for e, f, r, s in zip(exps, fitted, result.residuals, result.sigmas):
            if kind == "residual":
                idx = rng.integers(0, r.shape[0], r.shape[0])
                eps = r[idx] * np.sqrt(result.n / max(result.n - result.k, 1))
            else:
                eps = rng.normal(size=r.shape) * s[None, :]
            ne = Experiment(e.HT, e.GT, f + eps, e.kind, e.free_H_active, e.free_G_active, e.baseline,
                            e.sigma, e.name, e.channels, e.titrant_factor_sd, e.host_factor_sd)
            new.append(ne)
        try:
            b = fit(result.model, new, start=result.logbeta, n_starts=1, compute_cov=False)
            out.append(np.concatenate([b.logbeta, b.nuisance]))
        except RuntimeError:
            continue
    return np.array(out)
