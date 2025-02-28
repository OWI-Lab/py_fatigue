"""The module :mod:`py_fatigue.mean_stress.mean_stress` contains
the main mean stress-related utilities.
"""

from __future__ import annotations
from typing import ClassVar

import numpy as np

from ..utils import FatigueStress


class MeanStress(FatigueStress):
    """Mean stress class."""

    category: ClassVar[str] = "mean"


def calculate_mean_stress(
    load_ratio: float | np.ndarray | list,
    stress_amplitude: float | np.ndarray | list,
) -> float | np.ndarray:
    """
    Calculate the mean stress based on the load ratio and stress amplitude.

    The relationship is:

        mean_stress = stress_amplitude * (1 + load_ratio) / 2

    Parameters
    ----------
    load_ratio : float | np.ndarray | list
        Load ratio (minimum stress / maximum stress).
    stress_amplitude : float | np.ndarray | list
        Stress amplitude (half the range of stress variation).

    Returns
    -------
    float | np.ndarray
        Calculated mean stress.
    """
    if isinstance(load_ratio, list):
        load_ratio = np.array(load_ratio)
    if isinstance(stress_amplitude, list):
        stress_amplitude = np.array(stress_amplitude)
    return stress_amplitude * (1 + load_ratio) / (1 - load_ratio)
