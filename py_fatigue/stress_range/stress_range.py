"""The module :mod:`py_fatigue.stress_range.stress_range` contains
the main stress_range-related utilities.
"""

from typing import ClassVar

import numpy as np

from py_fatigue.utils import FatigueStress

__all__ = ["StressRange"]


class StressRange(FatigueStress):
    """Stress range class."""

    category: ClassVar[str] = "range"

    def small_cycles_idx(self):
        """The index of the small cycles."""
        return np.where(self.full >= self.bin_lb)[0]
