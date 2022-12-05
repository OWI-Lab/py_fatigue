"""The module :mod:`py_fatigue.mean_stress.mean_stress` contains
the main mean stress-related utilities.
"""

from typing import ClassVar

from py_fatigue.utils import FatigueStress


class MeanStress(FatigueStress):
    """Mean stress class."""

    category: ClassVar[str] = "mean"
