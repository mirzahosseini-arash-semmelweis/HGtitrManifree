"""Differentiable speciation in JAX (for NUTS / HMC and autodiff Jacobians).

The primal solve is the same globally convergent Newton/Armijo iteration on
the convex potential used in :mod:`hgtitr.speciation`.  Derivatives are not
obtained by differentiating through the iterations; they follow from the
implicit function theorem at the solution x*(lnb, T):

    S^T c(x*, lnb) = T   =>   dx* = H^{-1} (dT - S^T (c * dlnb)),
    H = S^T diag(c) S.

This makes gradients exact, cheap and independent of the iteration count.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

LN10 = jnp.log(10.0)
_ALPHAS = 0.5 ** jnp.arange(0, 40)


def _pot(x, lnb, S, T):
    c = jnp.exp(jnp.clip(lnb[None, :] + x @ S.T, -745.0, 700.0))
    return c.sum(1) - (T * x).sum(1), c


def _hess_solve(c, S, r):
    a, b = S[:, 0], S[:, 1]
    H00, H01, H11 = c @ (a * a), c @ (a * b), c @ (b * b)
    det = H00 * H11 - H01 * H01
    return jnp.stack([(H11 * r[:, 0] - H01 * r[:, 1]) / det, (-H01 * r[:, 0] + H00 * r[:, 1]) / det], 1)


def _newton_solve(lnb, T, S, tol=1e-13, maxiter=100):
    x0 = jnp.log(T)

    def cond(state):
        x, it = state
        _, c = _pot(x, lnb, S, T)
        rel = jnp.max(jnp.abs(c @ S - T) / T)
        return (rel > tol) & (it < maxiter)

    def body(state):
        x, it = state
        F, c = _pot(x, lnb, S, T)
        grad = c @ S - T
        dx = -_hess_solve(c, S, grad)
        slope = (grad * dx).sum(1)
        rel = jnp.max(jnp.abs(grad) / T, axis=1)
        trial = x[None] + _ALPHAS[:, None, None] * dx[None]
        Ft = jax.vmap(lambda xt: _pot(xt, lnb, S, T)[0])(trial)
        ok = Ft <= F[None] + 1e-4 * _ALPHAS[:, None] * slope[None] + 1e-15 * jnp.abs(F)[None]
        ok = ok.at[0].set(ok[0] | (rel < 1e-3))
        idx = jnp.argmax(ok, axis=0)  # first acceptable step length
        alpha = jnp.where(ok.any(0), _ALPHAS[idx], _ALPHAS[-1])
        return x + alpha[:, None] * dx, it + 1

    x, _ = jax.lax.while_loop(cond, body, (x0, 0))
    return x


@jax.custom_jvp
def solve_logfree(lnb_full, T, S):
    """ln free (h, g) at totals T (N, 2).  lnb_full includes 0, 0 for free H, G."""
    return _newton_solve(lnb_full, T, S)


@solve_logfree.defjvp
def _solve_jvp(primals, tangents):
    lnb, T, S = primals
    dlnb, dT, _ = tangents
    x = solve_logfree(lnb, T, S)
    _, c = _pot(x, lnb, S, T)
    rhs = dT - (c * dlnb[None, :]) @ S
    return x, _hess_solve(c, S, rhs)


def species_conc(logbeta, HT, GT, S):
    """Species concentrations (N, n_sp) [H, G, complexes] for log10 constants.

    Zero totals are floored at 1e-25 M, which is exact to double precision.
    """
    lnb = jnp.concatenate([jnp.zeros(2), LN10 * jnp.asarray(logbeta)])
    T = jnp.maximum(jnp.stack([HT, GT], 1), 1e-25)  # blank points: G_T = 0 -> 1e-25 M
    x = solve_logfree(lnb, T, S)
    return jnp.exp(jnp.clip(lnb[None, :] + x @ S.T, -745.0, 700.0))
