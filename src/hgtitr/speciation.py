"""Equilibrium speciation for two-component host-guest systems.

Two directions of the same mass-action model are provided:

* **Forward map** (explicit, no solving):  free (h, g)  ->  totals (H_T, G_T),
  species concentrations and signals.  This is the "free-component manifold"
  idea of HGtitrManifree.
* **Inverse map** (exact):  totals (H_T, G_T)  ->  free (h, g).

The inverse is computed by Newton's method on the strictly convex potential

    F(x) = sum_j beta_j exp(s_j . x)  -  T . x,      x = (ln h, ln g),

whose gradient is the mass-balance residual (totals(x) - T) and whose Hessian
S^T diag(c) S is positive definite.  Newton with Armijo backtracking on a
strictly convex function converges globally from any start, so no starting
value, damping heuristic or optimiser fallback is needed.  The same
potential underlies COGS-type and Musketeer speciation.

A JAX implementation with implicit-function-theorem derivatives is provided
in :mod:`hgtitr.jax_speciation` for gradient-based Bayesian sampling.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import LN10, Model

_EXP_CLIP = 700.0


# --------------------------------------------------------------------------- #
# Forward map (explicit)
# --------------------------------------------------------------------------- #
def species_from_free(h, g, model: Model, logbeta) -> np.ndarray:
    """Concentrations of [H, G, complexes...] from free h, g (explicit)."""
    h = np.atleast_1d(np.asarray(h, float))
    g = np.atleast_1d(np.asarray(g, float))
    lb = np.asarray(logbeta, float)
    with np.errstate(divide="ignore"):
        lh, lg = np.log(h), np.log(g)
    expo = LN10 * lb[None, :] + model.nH[None, :] * lh[:, None] + model.nG[None, :] * lg[:, None]
    if np.any(expo > _EXP_CLIP):
        raise FloatingPointError("Species concentration overflow: constants/concentrations out of range.")
    cx = np.exp(expo)
    return np.column_stack([h, g, cx])


def totals_from_free(h, g, model: Model, logbeta):
    """Forward map (h, g) -> (H_T, G_T, C).  Explicit, never fails."""
    C = species_from_free(h, g, model, logbeta)
    S = model.stoich
    T = C @ S
    return T[:, 0], T[:, 1], C


# --------------------------------------------------------------------------- #
# Inverse map (exact)
# --------------------------------------------------------------------------- #
@dataclass
class SpeciationResult:
    free: np.ndarray        # (N, 2) free h, g
    C: np.ndarray           # (N, n_complex + 2) concentrations [H, G, complexes]
    iterations: int
    max_rel_residual: float
    species: tuple

    @property
    def h(self):
        return self.free[:, 0]

    @property
    def g(self):
        return self.free[:, 1]


def _potential(x, lnb, S, T):
    expo = np.clip(lnb[None, :] + x @ S.T, -np.inf, _EXP_CLIP)
    c = np.exp(expo)
    F = c.sum(1) - (T * x).sum(1)
    return F, c


def solve_speciation(
    HT,
    GT,
    model: Model,
    logbeta,
    x0=None,
    tol: float = 1e-12,
    maxiter: int = 200,
) -> SpeciationResult:
    """Exact free concentrations at given totals (vectorised over points).

    Parameters
    ----------
    HT, GT : array_like, total host / guest concentrations (M), >= 0.
    model : Model
    logbeta : array_like, log10 cumulative constants, one per complex.
    x0 : optional (N, 2) warm start for (ln h, ln g).
    tol : relative mass-balance tolerance, max |total_calc/total - 1|.
    """
    HT = np.atleast_1d(np.asarray(HT, float))
    GT = np.atleast_1d(np.asarray(GT, float))
    if HT.shape != GT.shape:
        raise ValueError("HT and GT must have the same shape.")
    if np.any(HT < 0) or np.any(GT < 0) or not np.all(np.isfinite(HT + GT)):
        raise ValueError("Total concentrations must be finite and non-negative.")
    lb = np.asarray(logbeta, float)
    if lb.shape != (model.n_complex,):
        raise ValueError(f"logbeta must have length {model.n_complex}.")
    N = HT.size
    S = model.stoich
    lnb = np.concatenate([[0.0, 0.0], LN10 * lb])
    free = np.column_stack([HT, GT]).copy()
    active = (HT > 0) & (GT > 0)  # complexes need both components
    it_total = 0
    max_res = 0.0
    if np.any(active):
        T = np.column_stack([HT[active], GT[active]])
        scale = T.sum(1)
        x = np.log(T) if x0 is None else np.asarray(x0, float)[active].copy()
        F, c = _potential(x, lnb, S, T)
        for it in range(maxiter):
            grad = c @ S - T
            rel = np.max(np.abs(grad) / T, axis=1)
            todo = rel > tol
            it_total = it
            if not np.any(todo):
                break
            # 2x2 Hessian S^T diag(c) S, solved explicitly
            a, b = S[:, 0], S[:, 1]
            H00 = c @ (a * a)
            H01 = c @ (a * b)
            H11 = c @ (b * b)
            det = H00 * H11 - H01 * H01
            dx0 = -(H11 * grad[:, 0] - H01 * grad[:, 1]) / det
            dx1 = -(-H01 * grad[:, 0] + H00 * grad[:, 1]) / det
            dx = np.column_stack([dx0, dx1])
            dx[~todo] = 0.0
            slope = (grad * dx).sum(1)
            alpha = np.ones(T.shape[0])
            near = rel < 1e-3  # quadratic-convergence region: take full steps
            accept = np.zeros(T.shape[0], bool) | ~todo
            x_new, F_new, c_new = x.copy(), F.copy(), c.copy()
            for _ in range(60):
                trial = x + alpha[:, None] * dx
                Ft, ct = _potential(trial, lnb, S, T)
                ok = (Ft <= F + 1e-4 * alpha * slope + 1e-15 * np.abs(F) + 1e-30 * scale) | near
                newly = ok & ~accept
                x_new[newly], F_new[newly], c_new[newly] = trial[newly], Ft[newly], ct[newly]
                accept |= ok
                if np.all(accept):
                    break
                alpha = np.where(accept, alpha, 0.5 * alpha)
            x, F, c = x_new, F_new, c_new
        grad = c @ S - T
        max_res = float(np.max(np.abs(grad) / T))
        free[active] = np.exp(x)
    C = np.column_stack([free, np.zeros((N, model.n_complex))])
    if np.any(active):
        C[active] = species_from_free(free[active, 0], free[active, 1], model, lb)
    if max_res > 1e3 * tol and max_res > 1e-8:
        raise RuntimeError(f"Speciation did not converge (max relative residual {max_res:.2e}).")
    return SpeciationResult(free=free, C=C, iterations=it_total, max_rel_residual=max_res, species=model.all_species)


# --------------------------------------------------------------------------- #
# Closed forms used for testing
# --------------------------------------------------------------------------- #
def free_guest_1to1(HT, GT, logK):
    """Exact free guest for 1:1 binding (quadratic, numerically stable root)."""
    HT, GT = np.asarray(HT, float), np.asarray(GT, float)
    K = 10.0 ** logK
    b = HT - GT + 1.0 / K
    disc = np.sqrt(b * b + 4.0 * GT / K)
    # g = (-b + disc)/2 written stably
    return np.where(b > 0, 2.0 * GT / K / (b + disc), 0.5 * (disc - b))
