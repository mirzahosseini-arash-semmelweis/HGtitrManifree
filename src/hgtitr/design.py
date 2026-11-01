"""Experimental concentration designs (titrations with dilution, Job series)."""
from __future__ import annotations

import numpy as np


def titration(
    c_cell_H: float,
    c_titrant_G: float,
    V0: float,
    V_add,
    cumulative: bool = False,
    include_initial: bool = True,
    c_cell_G: float = 0.0,
    c_titrant_H: float = 0.0,
):
    """Total concentrations for a titration of a cell solution with titrant.

    Parameters
    ----------
    c_cell_H, c_cell_G : concentrations in the initial cell solution (M).
    c_titrant_G, c_titrant_H : concentrations in the titrant (M).  Setting
        ``c_titrant_H = c_cell_H`` gives a constant-host titration.
    V0 : initial cell volume.
    V_add : addition volumes (same unit as V0); increments unless ``cumulative``.
    """
    V_add = np.asarray(V_add, float)
    if np.any(V_add < 0):
        raise ValueError("Addition volumes must be non-negative.")
    Vc = V_add if cumulative else np.cumsum(V_add)
    if include_initial:
        Vc = np.concatenate([[0.0], Vc])
    Vt = V0 + Vc
    HT = (c_cell_H * V0 + c_titrant_H * Vc) / Vt
    GT = (c_cell_G * V0 + c_titrant_G * Vc) / Vt
    return HT, GT


def job_series(C_total: float, x_guest):
    """Continuous-variation (Job) design at constant H_T + G_T = C_total."""
    x = np.asarray(x_guest, float)
    return C_total * (1.0 - x), C_total * x
