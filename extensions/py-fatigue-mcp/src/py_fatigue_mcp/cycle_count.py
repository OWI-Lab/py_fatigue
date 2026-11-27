"""Expose the native CycleCount object and its distinct rainflow methods."""

import math
from datetime import datetime, timedelta, timezone
from typing import Annotated, Generic, Literal, TypeVar

from mcp.server.mcpserver.exceptions import ToolError
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, FiniteFloat, StrictBool

from py_fatigue import CycleCount

from .csv_io import CsvExport, CsvFile, read_csv_rows, write_csv_rows
from .models import InputModel, Number, Positive, preview_indices
from .synthetic import SyntheticHistory, synthetic_history

MAX_CYCLE_ROWS = 1_000_000
MAX_INLINE_ROWS = 100_000
RAINFLOW_COLUMNS = ("stress_range", "mean_stress", "count_cycle")


StressSamples = Annotated[
    list[Number], Field(min_length=2, max_length=MAX_CYCLE_ROWS)
]
InlineStress = (
    Annotated[list[Number], Field(min_length=2, max_length=MAX_INLINE_ROWS)]
    | SyntheticHistory
)
StressInput = StressSamples | CsvFile | SyntheticHistory


RainflowMethod = Literal["astm", "fourpoint"]
WindowStress = TypeVar("WindowStress")


class CycleWindow(InputModel, Generic[WindowStress]):
    """One recording window and an optional stress concentration factor."""

    stress: WindowStress
    stress_concentration_factor: Positive = 1.0


class CycleCountWindows(InputModel, Generic[WindowStress]):
    """Ordered windows to count, optionally scale, add, and recover."""

    kind: Literal["cycle_count_windows"]
    windows: Annotated[
        list[CycleWindow[WindowStress]], Field(min_length=1, max_length=64)
    ]
    resolve_residuals: StrictBool = False
    stress_concentration_factor: Positive = 1.0


CycleInput = StressInput | CycleCountWindows[StressInput]
InlineCycleInput = InlineStress | CycleCountWindows[InlineStress]


class RainflowRow(BaseModel):
    """An unbinned rainflow cycle with a stress range and mean in MPa."""

    stress_range: FiniteFloat
    mean_stress: FiniteFloat
    count_cycle: FiniteFloat


class CycleCountSummary(BaseModel):
    """Native cycle counts, extrema, and preview of residual turning points."""

    row_count: int
    total_cycles: Number
    full_cycles: Number
    half_cycle_rows: int
    max_stress_range: Number
    residual_count: int
    lffd_solved: bool
    residual_preview: list[Number]
    preview: list[RainflowRow]
    export: CsvExport | None = None


def stress_samples(
    stress: StressInput, limit: int = MAX_CYCLE_ROWS
) -> NDArray[np.float64]:
    """Read stresses in acquisition order from inline, CSV, or synthetic input."""
    if isinstance(stress, CsvFile):
        stress = [
            sample[0] for sample in read_csv_rows(stress, ("stress",), limit)
        ]
    elif isinstance(stress, SyntheticHistory):
        stress = synthetic_history(stress)[1]
    if len(stress) > limit:
        raise ToolError(f"Supply at most {limit:,} stress samples.")
    return np.asarray(stress, dtype=float)


def count_window(
    samples: NDArray[np.float64], method: RainflowMethod, timestamp: datetime
) -> CycleCount:
    """Build an unbinned native count for one validated window."""
    if samples.size < 2 or not math.isfinite(
        float(samples.max()) - float(samples.min())
    ):
        raise ToolError(
            "Supply at least two samples with a finite stress span."
        )
    return CycleCount.from_timeseries(
        samples, rainflow_method=method, timestamp=timestamp
    )


def build_cycle_count(
    stress: CycleInput | InlineCycleInput, method: RainflowMethod
) -> CycleCount:
    """Apply native CycleCount addition, residual recovery, and positive scaling."""
    start = datetime(2000, 1, 1, tzinfo=timezone.utc)
    try:
        with np.errstate(invalid="raise", over="raise"):
            if not isinstance(stress, CycleCountWindows):
                return count_window(stress_samples(stress), method, start)

            counts = []
            remaining = MAX_INLINE_ROWS
            for index, window in enumerate(stress.windows):
                samples = stress_samples(window.stress, remaining)
                remaining -= len(samples)
                count = count_window(
                    samples, method, start + timedelta(seconds=index)
                )
                if window.stress_concentration_factor != 1.0:
                    count = count * window.stress_concentration_factor
                counts.append(count)
            combined = sum(counts[1:], counts[0])
            if stress.resolve_residuals:
                combined = combined.solve_lffd(rainflow_method=method)
            if stress.stress_concentration_factor != 1.0:
                combined = combined * stress.stress_concentration_factor
            return combined
    except (ValueError, FloatingPointError, OverflowError) as error:
        raise ToolError(f"Cannot build CycleCount: {error}") from error


def cycle_count(
    stress: CycleInput,
    method: RainflowMethod = "astm",
    export_csv: StrictBool = False,
) -> CycleCountSummary:
    """Build CycleCount with ASTM or four-point rainflow and inspect its residuals.

    Supply a stress history or `cycle_count_windows` in acquisition order.
    Window and overall stress concentration factors scale stress ranges,
    means and residuals. Use resolve_residuals to recover cycles spanning
    adjacent windows. Request a CSV to save every unbinned cycle row.
    Use plot_py_fatigue for the native histogram and residual plots.
    """
    count = build_cycle_count(stress, method)
    table = np.column_stack(
        (count.stress_range, count.mean_stress, count.count_cycle)
    )
    row_count = len(table)
    residuals = count.residuals_sequence
    return CycleCountSummary(
        row_count=row_count,
        total_cycles=float(count.count_cycle.sum()),
        full_cycles=float(count.count_cycle[count.count_cycle > 0.5].sum()),
        half_cycle_rows=int(np.count_nonzero(count.count_cycle == 0.5)),
        max_stress_range=float(count.stress_range.max()),
        residual_count=len(residuals),
        lffd_solved=bool(count.lffd_solved),
        residual_preview=[
            float(residuals[index]) for index in preview_indices(len(residuals))
        ],
        preview=[
            RainflowRow(**dict(zip(RAINFLOW_COLUMNS, row)))
            for row in table[preview_indices(row_count)]
        ],
        export=(
            write_csv_rows("cycle-count", RAINFLOW_COLUMNS, table, row_count)
            if export_csv
            else None
        ),
    )
