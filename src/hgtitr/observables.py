"""Observation models: the linear map from species concentrations to signals.

Every observable used here is *linear in its response parameters* given the
species concentrations, Y = A(C) R, which is what makes variable projection
(profiling R by linear least squares) exact.

``intensity``  additive intensity / absorbance / CD / fluorescence:
               A = C[:, active]                         (Eq. 11 of the manuscript)
``shift_H``    fast-exchange NMR chemical shift observed on the host:
               delta_obs = sum_j a_j [S_j] delta_j / H_T,  A = C * a / H_T
``shift_G``    fast-exchange shift observed on the guest:  A = C * b / G_T

For shifts the free-species column carries the free chemical shift; use
``baseline=True`` for intensity data with an additive per-channel offset.
"""
from __future__ import annotations

import numpy as np

KINDS = ("intensity", "shift_H", "shift_G")


def design_matrix(C, stoich, HT, GT, kind: str = "intensity", active=None, baseline: bool = False):
    """Return A (N, p) so that the noiseless signal is A @ R.

    Parameters
    ----------
    C : (N, n_sp) concentrations [H, G, complexes...].
    stoich : (n_sp, 2) stoichiometry incl. free H and G.
    active : boolean mask or index list over the n_sp species that contribute.
    """
    xp = np
    try:  # allow jax arrays / tracers transparently
        import jax
        import jax.numpy as jnp

        if isinstance(C, jax.Array):
            xp = jnp
    except ImportError:  # pragma: no cover
        pass
    if xp is np:
        C = np.asarray(C)
    if kind == "intensity":
        A = C
    elif kind == "shift_H":
        A = C * stoich[:, 0][None, :] / xp.maximum(HT, 1e-300)[:, None]
    elif kind == "shift_G":
        A = C * stoich[:, 1][None, :] / xp.maximum(GT, 1e-300)[:, None]
    else:
        raise ValueError(f"Unknown observable kind {kind!r}; choose from {KINDS}.")
    if active is not None:
        A = A[:, np.asarray(active)]
    if baseline:
        A = xp.concatenate([A, xp.ones((A.shape[0], 1))], axis=1)
    return A


def default_active(n_sp: int, kind: str, free_H=None, free_G=None):
    """Default contributing species.

    intensity: free host and free guest are silent unless requested.
    shift_H:   all host-containing species (free H + complexes) contribute.
    shift_G:   all guest-containing species contribute.
    """
    act = np.ones(n_sp, bool)
    if kind == "intensity":
        act[0] = bool(free_H) if free_H is not None else False
        act[1] = bool(free_G) if free_G is not None else False
    elif kind == "shift_H":
        act[1] = False
    elif kind == "shift_G":
        act[0] = False
    return act
