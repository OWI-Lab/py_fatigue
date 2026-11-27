"""Recover low-frequency fatigue cycles across ordered stress windows."""

from datetime import datetime, timedelta, timezone
from typing import Annotated

from mcp.server.mcpserver.exceptions import ToolError
import numpy as np
from pydantic import BaseModel, Field, FiniteFloat

from py_fatigue import CycleCount

from .models import Number, preview_indices
from .cycle_count import RainflowRow

MAX_LFFD_SAMPLES = 100_000
StressWindow = Annotated[
    list[Number], Field(min_length=2, max_length=MAX_LFFD_SAMPLES)
]
StressWindows = Annotated[
    list[StressWindow], Field(min_length=2, max_length=64)
]


class LffdSummary(BaseModel):
    """Compare independent window counts with the recovered spectrum."""

    window_count: int
    windowed_total_cycles: FiniteFloat
    windowed_max_stress_range: FiniteFloat
    total_cycles: FiniteFloat
    max_stress_range: FiniteFloat
    remaining_turning_points: int
    row_count: int
    preview: list[RainflowRow]


def lffd_recovery(windows: StressWindows) -> LffdSummary:
    """Recover rainflow cycles that span adjacent stress windows.

    Supply contiguous, non-overlapping windows of numeric samples in MPa
    in acquisition order. Each window must yield at least one ASTM
    rainflow row. The result compares the summed window counts with the
    spectrum after recounting their joined residual turning points.

    Parameters
    ----------
    windows : list of list of float
        Ordered stress windows with at most 100,000 samples in total.

    Returns
    -------
    LffdSummary
        Cycle weights and maximum ranges before and after recovery, the
        remaining turning-point count, and a preview of the recovered rows.
    """
    if sum(map(len, windows)) > MAX_LFFD_SAMPLES:
        raise ToolError(
            f"Supply at most {MAX_LFFD_SAMPLES:,} samples in total."
        )

    counts = []
    start = datetime(2000, 1, 1, tzinfo=timezone.utc)
    for index, samples in enumerate(windows):
        try:
            with np.errstate(over="raise", invalid="raise"):
                count = CycleCount.from_timeseries(
                    samples, timestamp=start + timedelta(seconds=index)
                )
        except ValueError as error:
            raise ToolError(
                f"Cannot count window {index + 1}: {error} "
                "Merge short or flat windows with an adjacent window."
            ) from error
        except (FloatingPointError, OverflowError) as error:
            raise ToolError(
                f"Stress range in window {index + 1} is too large."
            ) from error
        counts.append(count)

    combined = sum(counts[1:], counts[0])
    try:
        with np.errstate(over="raise", invalid="raise"):
            recovered = combined.solve_lffd()
    except (FloatingPointError, OverflowError, ValueError) as error:
        raise ToolError("Could not recount the joined residuals.") from error

    row_count = len(recovered.count_cycle)
    return LffdSummary(
        window_count=len(windows),
        windowed_total_cycles=float(combined.count_cycle.sum()),
        windowed_max_stress_range=float(combined.stress_range.max()),
        total_cycles=float(recovered.count_cycle.sum()),
        max_stress_range=float(recovered.stress_range.max()),
        remaining_turning_points=len(recovered.residuals_sequence),
        row_count=row_count,
        preview=[
            RainflowRow(
                stress_range=recovered.stress_range[index],
                mean_stress=recovered.mean_stress[index],
                count_cycle=recovered.count_cycle[index],
            )
            for index in preview_indices(row_count)
        ],
    )
