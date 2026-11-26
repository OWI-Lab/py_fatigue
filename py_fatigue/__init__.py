# -*- coding: utf-8 -*-
"""Py-fatigue bundles the main functionality for performing cyclic stress (fatigue) analysis and cycle-counting."""  # noqa: E501  # pylint: disable=C0301

from importlib import import_module
from typing import TYPE_CHECKING

from .cycle_count.cycle_count import CycleCount
from .material.sn_curve import SNCurve
from .material.crack_growth_curve import ParisCurve, WalkerCurve
from .version import __version__
from . import cycle_count, geometry, material, damage, styling, testing

if TYPE_CHECKING:
    from . import crack_growth

__all__ = [
    "CycleCount",
    "SNCurve",
    "ParisCurve",
    "WalkerCurve",
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

    if name == "crack_growth":
        return import_module(".crack_growth", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
