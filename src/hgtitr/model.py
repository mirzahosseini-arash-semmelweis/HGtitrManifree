"""Chemical model: species labels, stoichiometry and formation constants.

A model is a set of complexes H_aG_b (a >= 1, b >= 1), each with a cumulative
formation constant beta_ab = [H_aG_b] / (h^a g^b).  The free host H and free
guest G are always included implicitly.

Labels follow the legacy R convention ``H<a>G<b>[suffix]``; the optional
suffix distinguishes same-stoichiometry states, e.g. ``H1G1a`` and ``H1G1b``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

_LABEL = re.compile(r"^H(\d+)G(\d+)([A-Za-z][A-Za-z0-9_]*)?$")
LN10 = np.log(10.0)


def parse_species(label: str) -> tuple[int, int, str]:
    """Return (a, b, suffix) for a label such as 'H1G2' or 'H1G1a'."""
    m = _LABEL.match(label.strip())
    if m is None:
        raise ValueError(
            f"Invalid species label {label!r}; expected H<a>G<b>[suffix], e.g. H1G1, H1G2, H2G1, H1G1a."
        )
    a, b = int(m.group(1)), int(m.group(2))
    if a < 1 or b < 1:
        raise ValueError(f"Complex {label!r} must contain at least one host and one guest.")
    return a, b, m.group(3) or ""


@dataclass(frozen=True)
class Model:
    """A two-component host-guest binding model.

    Parameters
    ----------
    species : sequence of complex labels, e.g. ``["H1G1", "H1G2"]``.
    """

    species: tuple[str, ...]
    nH: np.ndarray = field(init=False, repr=False)
    nG: np.ndarray = field(init=False, repr=False)

    def __init__(self, species: Sequence[str]):
        species = tuple(str(s).strip() for s in species)
        if len(species) == 0:
            raise ValueError("A model needs at least one complex.")
        if len(set(species)) != len(species):
            raise ValueError("Duplicate species labels.")
        ab = [parse_species(s) for s in species]
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "nH", np.array([x[0] for x in ab], dtype=float))
        object.__setattr__(self, "nG", np.array([x[1] for x in ab], dtype=float))

    # ------------------------------------------------------------------ #
    @property
    def n_complex(self) -> int:
        return len(self.species)

    @property
    def all_species(self) -> tuple[str, ...]:
        """Free host, free guest, then complexes (column order of C)."""
        return ("H", "G") + self.species

    @property
    def stoich(self) -> np.ndarray:
        """(n_complex + 2, 2) stoichiometry matrix [a, b] including free H, G."""
        S = np.zeros((self.n_complex + 2, 2))
        S[0, 0] = 1.0
        S[1, 1] = 1.0
        S[2:, 0] = self.nH
        S[2:, 1] = self.nG
        return S

    @property
    def max_guest_per_host(self) -> float:
        return float(np.max(self.nG / self.nH))

    def stepwise_logK(self, logbeta: Sequence[float]) -> dict[str, float]:
        """Stepwise log K for sequential HG_n series present in the model."""
        lb = dict(zip(self.species, logbeta))
        out = {}
        for s in self.species:
            a, b, suf = parse_species(s)
            prev = f"H{a}G{b-1}{suf}" if b > 1 else None
            if a == 1 and b == 1:
                out[s] = lb[s]
            elif prev in lb:
                out[s] = lb[s] - lb[prev]
        return out

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return " + ".join(self.species)
