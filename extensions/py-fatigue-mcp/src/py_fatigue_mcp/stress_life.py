"""MCP tools for ASTM rainflow cycles and Palmgren–Miner damage."""

import math
from typing import Annotated, Literal

from mcp.server.mcpserver.exceptions import ToolError
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, FiniteFloat, StrictBool

from py_fatigue import SNCurve
from py_fatigue.cycle_count.rainflow import rainflow
from py_fatigue.damage.stress_life import calc_pm

from .csv_io import CsvExport, CsvFile, read_csv_rows, write_csv_rows
from .models import InputModel, Nonnegative, Number, Positive, preview_indices

MAX_CYCLE_ROWS = 1_000_000
CYCLE_COLUMNS = ("stress_range", "count_cycle")
RAINFLOW_COLUMNS = ("stress_range", "mean_stress", "count_cycle")


StressSamples = Annotated[
    list[Number], Field(min_length=2, max_length=MAX_CYCLE_ROWS)
]
StressInput = StressSamples | CsvFile


class CountedCycle(InputModel):
    """A stress range in MPa and its cycle weight."""

    stress_range: Nonnegative
    count_cycle: Nonnegative


class StressHistory(InputModel):
    """An ordered stress history in MPa."""

    kind: Literal["stress_history"]
    stress: StressInput


class CountedCycles(InputModel):
    """Previously counted cycles or the CSV that contains them."""

    kind: Literal["counted_cycles"]
    cycles: (
        Annotated[list[CountedCycle], Field(max_length=MAX_CYCLE_ROWS)]
        | CsvFile
    )


MinerSource = Annotated[
    StressHistory | CountedCycles, Field(discriminator="kind")
]


class SNParameters(InputModel):
    """S–N slopes, log10 intercepts, and optional endurance in cycles."""

    slope: Positive | Annotated[list[Positive], Field(min_length=1)]
    intercept: Number | Annotated[list[Number], Field(min_length=1)]
    endurance: Positive | None = None


class RainflowRow(BaseModel):
    """An unbinned rainflow cycle with a stress range and mean in MPa."""

    stress_range: FiniteFloat
    mean_stress: FiniteFloat
    count_cycle: FiniteFloat


class RainflowSummary(BaseModel):
    """Cycle totals and a bounded preview of unbinned results."""

    row_count: int
    total_cycles: FiniteFloat
    preview: list[RainflowRow]
    export: CsvExport | None = None


class MinerRow(BaseModel):
    """One stress range in MPa, its cycle count, and damage contribution."""

    stress_range: FiniteFloat
    count_cycle: FiniteFloat
    damage: FiniteFloat


class MinerSummary(BaseModel):
    """Total dimensionless damage and a bounded contribution preview."""

    row_count: int
    total_damage: FiniteFloat
    preview: list[MinerRow]
    export: CsvExport | None = None


def rainflow_table(stress: StressInput) -> NDArray[np.float64]:
    """Count unbinned ASTM cycles in MPa using the existing rainflow kernel."""
    if isinstance(stress, CsvFile):
        stress = [
            sample[0]
            for sample in read_csv_rows(stress, ("stress",), MAX_CYCLE_ROWS)
        ]
    if len(stress) < 2:
        raise ToolError(
            "A stress history requires at least two stress samples."
        )
    cycles = rainflow(stress, extended_output=False)
    if not len(cycles) and min(stress) != max(stress):
        raise ToolError(
            "Rainflow returned no cycles for a varying stress history. "
            "Supply a complete history or pass counted cycles to miner_damage."
        )
    table = cycles[:, :3].copy()
    table[:, 0] *= 2
    if not np.isfinite(table).all():
        raise ToolError("Rainflow counting produced non-finite cycle values.")
    return table


def rainflow_count(
    stress: StressInput, export_csv: StrictBool = False
) -> RainflowSummary:
    """Count ASTM rainflow cycles from ordered stress samples in MPa.

    Pass numeric samples or a relative CSV path with a stress column.
    Results include full and half-cycle weights. Set export_csv to save
    the complete cycle table.
    """
    table = rainflow_table(stress)
    row_count = len(table)
    result = RainflowSummary(
        row_count=row_count,
        total_cycles=float(table[:, 2].sum()),
        preview=[
            RainflowRow(**dict(zip(RAINFLOW_COLUMNS, row)))
            for row in table[preview_indices(row_count)]
        ],
    )
    if export_csv:
        result.export = write_csv_rows(
            "rainflow", RAINFLOW_COLUMNS, table, row_count
        )
    return result


def miner_damage(
    source: MinerSource, curve: SNParameters, export_csv: StrictBool = False
) -> MinerSummary:
    """Calculate Palmgren–Miner damage with caller-supplied S–N parameters.

    Use stresses in MPa. Pass an ordered history for ASTM counting or
    supply counted cycles directly. S–N intercepts are log10 intercepts.
    Damage is dimensionless. Set export_csv to save every contribution.
    """
    if isinstance(source, StressHistory):
        cycles = rainflow_table(source.stress)[:, [0, 2]]
    else:
        rows = source.cycles
        if isinstance(rows, CsvFile):
            values = list(read_csv_rows(rows, CYCLE_COLUMNS, MAX_CYCLE_ROWS))
        else:
            values = [(row.stress_range, row.count_cycle) for row in rows]
        cycles = np.asarray(values, dtype=float).reshape(-1, 2)
        if np.any(cycles < 0):
            raise ToolError("stress_range and count_cycle must be nonnegative.")

    if np.size(curve.slope) != np.size(curve.intercept):
        raise ToolError("Supply one S–N intercept per slope.")
    endurance = np.inf if curve.endurance is None else curve.endurance
    try:
        sn_curve = SNCurve(curve.slope, curve.intercept, endurance)
        damage = (
            calc_pm(cycles[:, 0], cycles[:, 1], sn_curve)
            if len(cycles)
            else np.empty(0)
        )
    except (ValueError, AssertionError) as error:
        raise ToolError(
            f"Invalid S–N curve or stress range: {error}"
        ) from error

    total_damage = float(damage.sum())
    if not np.isfinite(damage).all() or not math.isfinite(total_damage):
        raise ToolError("Miner damage is non-finite for the supplied inputs.")
    table = np.column_stack((cycles, damage))
    row_count = len(table)
    columns = (*CYCLE_COLUMNS, "damage")
    result = MinerSummary(
        row_count=row_count,
        total_damage=total_damage,
        preview=[
            MinerRow(**dict(zip(columns, row)))
            for row in table[preview_indices(row_count)]
        ],
    )
    if export_csv:
        result.export = write_csv_rows("miner", columns, table, row_count)
    return result
