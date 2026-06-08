# -*- coding: utf-8 -*-
"""Py-fatigue bundles the main functionality for performing cyclic stress (fatigue) analysis and cycle-counting."""  # noqa: E501  # pylint: disable=C0301

from typing import TYPE_CHECKING

from .cycle_count.cycle_count import CycleCount
from .material.sn_curve import SNCurve
from .material.crack_growth_curve import ParisCurve, WalkerCurve
from .version import __version__
from . import cycle_count, geometry, material, damage, styling, testing

if TYPE_CHECKING:
    from .damage import crack_growth
    from .utils import warmup_numba

__all__ = [
    "CycleCount",
    "SNCurve",
    "ParisCurve",
    "WalkerCurve",
    "warmup_numba",
    "cycle_count",
    "crack_growth",
    "damage",
    "geometry",
    "material",
    "styling",
    "testing",
    "__version__",
]


def __getattr__(name: str):
    """Lazily expose optional package-level helpers."""

    if name == "warmup_numba":
        from .utils import warmup_numba  # pylint: disable=C0415

        return warmup_numba
    if name == "crack_growth":
        from .damage import crack_growth  # pylint: disable=C0415

        return crack_growth
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
