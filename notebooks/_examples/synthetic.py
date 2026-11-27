"""Shared input and display helpers for the long-signal notebook cases."""

from datetime import datetime, timedelta, timezone

import marimo as mo
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray

from py_fatigue import CycleCount
from .charts import plot_views as plot_views
from py_fatigue.testing import get_random_data, get_sampled_time

MINIMUM_STRESS = 10.0


def synthetic_peaks() -> NDArray[np.float64]:
    """Generate the same 300,000 tensile peaks (MPa) used in lesson 34."""
    cycle_time = get_sampled_time(fs=1, duration=300_000)
    return get_random_data(cycle_time, seed=2, min_=40.0, range_=60.0)


def tensile_waveform(peaks: NDArray[np.float64]) -> NDArray[np.float64]:
    """Place one peak between successive visits to the fixed tensile minimum."""
    stress = np.full(2 * peaks.size + 1, MINIMUM_STRESS)
    stress[1::2] = peaks
    return stress


def count_windows(stress: NDArray[np.float64]) -> CycleCount:
    """Count two consecutive windows and add them in timestamp order."""
    split = stress.size // 2
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    first = CycleCount.from_timeseries(stress[:split], timestamp=start)
    second = CycleCount.from_timeseries(
        stress[split:], timestamp=start + timedelta(seconds=split / 2)
    )
    return first + second


def display_indices(size: int) -> NDArray[np.intp]:
    """Keep both endpoints while limiting the size of the chart payload."""
    return np.linspace(0, size - 1, min(size, 1001), dtype=np.intp)


def long_case_view(
    fig: Figure, description: str, *, height: str = "420px"
) -> mo.Html:
    """Show the long-signal example in both rendering modes."""
    return mo.vstack(
        [
            mo.md(f"## Long synthetic signal\n\n{description}"),
            plot_views(fig, height=height),
        ]
    )
