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
from .cycle_count import (
    MAX_CYCLE_ROWS,
    MAX_INLINE_ROWS,
    CycleCountWindows,
    CycleInput,
    InlineCycleInput,
    InlineStress,
    RAINFLOW_COLUMNS,
    RainflowRow,
    RainflowMethod,
    StressInput,
    build_cycle_count,
    stress_samples,
)
from .models import InputModel, Nonnegative, Number, Positive, preview_indices

CYCLE_COLUMNS = ("stress_range", "count_cycle")
EvaluationPoints = Annotated[
    list[Positive], Field(min_length=1, max_length=1000)
]


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


class NativeCycles(InputModel):
    """Native CycleCount with optional window addition and stress scaling."""

    kind: Literal["cycle_count"]
    stress: CycleInput
    method: RainflowMethod = "astm"


MinerSource = Annotated[
    StressHistory | CountedCycles | NativeCycles, Field(discriminator="kind")
]


class InlineHistory(InputModel):
    """Inline stress samples or a synthetic history in MPa."""

    kind: Literal["stress_history"]
    stress: InlineStress


class InlineCycles(InputModel):
    """Inline stress ranges and cycle weights."""

    kind: Literal["counted_cycles"]
    cycles: Annotated[list[CountedCycle], Field(max_length=MAX_INLINE_ROWS)]


class InlineNativeCycles(InputModel):
    """Inline or synthetic CycleCount inputs with optional window operations."""

    kind: Literal["cycle_count"]
    stress: InlineCycleInput
    method: RainflowMethod = "astm"


InlineMinerSource = Annotated[
    InlineHistory | InlineCycles | InlineNativeCycles,
    Field(discriminator="kind"),
]


def local_miner_source(
    source: InlineHistory | InlineCycles | InlineNativeCycles,
) -> StressHistory | CountedCycles | NativeCycles:
    """Reuse the same calculation with the inline portion of the stdio schema."""
    if isinstance(source, InlineHistory):
        return StressHistory(kind=source.kind, stress=source.stress)
    if isinstance(source, InlineNativeCycles):
        payload = source.stress
        if isinstance(payload, CycleCountWindows):
            payload = CycleCountWindows[StressInput].model_validate(
                payload.model_dump()
            )
        return NativeCycles(
            kind=source.kind, stress=payload, method=source.method
        )
    return CountedCycles(kind=source.kind, cycles=source.cycles)


class SNParameters(InputModel):
    """S–N slopes, log10 intercepts, and optional endurance in cycles."""

    slope: Positive | Annotated[list[Positive], Field(min_length=1)]
    intercept: Number | Annotated[list[Number], Field(min_length=1)]
    endurance: Positive | None = None


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
    stress = stress_samples(stress)
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

    Pass numeric samples, a synthetic specification, or a relative CSV path.
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


class SNResult(BaseModel):
    """S–N queries in input order. Null life denotes infinite life."""

    cycles_to_failure: list[FiniteFloat | None]
    stress_ranges_mpa: list[FiniteFloat]


def build_sn_curve(curve: SNParameters) -> SNCurve:
    """Construct the same S–N curve used for evaluation and Miner damage."""
    if np.size(curve.slope) != np.size(curve.intercept):
        raise ToolError("Supply one S–N intercept per slope.")
    endurance = np.inf if curve.endurance is None else curve.endurance
    try:
        return SNCurve(curve.slope, curve.intercept, endurance)
    except (ValueError, AssertionError, OverflowError) as error:
        raise ToolError(f"Invalid S–N curve: {error}") from error


def sn_curve(
    curve: SNParameters,
    stress_ranges: EvaluationPoints | None = None,
    cycles: EvaluationPoints | None = None,
) -> SNResult:
    """Evaluate cycles to failure at ranges in MPa or ranges at cycle counts.

    Supply slopes and log10 intercepts. Pass either query array or both.
    An infinite life is returned as null.
    """
    if stress_ranges is None and cycles is None:
        raise ToolError("Supply stress ranges or cycle counts to evaluate.")
    model = build_sn_curve(curve)
    try:
        life = (
            np.asarray(model.get_cycles(stress_ranges)) if stress_ranges else []
        )
        ranges = np.asarray(model.get_stress(cycles)) if cycles else []
    except (ValueError, AssertionError, OverflowError) as error:
        raise ToolError(f"Cannot evaluate this S–N curve: {error}") from error
    if any(np.isnan(value) or value <= 0 for value in life):
        raise ToolError("S–N evaluation produced an invalid life.")
    if any(not np.isfinite(value) or value < 0 for value in ranges):
        raise ToolError("S–N evaluation produced an invalid stress range.")
    return SNResult(
        cycles_to_failure=[
            None if np.isinf(value) else float(value) for value in life
        ],
        stress_ranges_mpa=list(ranges),
    )


def miner_table(
    source: MinerSource, curve: SNParameters
) -> NDArray[np.float64]:
    """Return unbinned stress ranges, cycle weights and Miner contributions."""
    if isinstance(source, StressHistory):
        cycles = rainflow_table(source.stress)[:, [0, 2]]
    elif isinstance(source, NativeCycles):
        count = build_cycle_count(source.stress, source.method)
        cycles = np.column_stack((count.stress_range, count.count_cycle))
    else:
        rows = source.cycles
        if isinstance(rows, CsvFile):
            values = list(read_csv_rows(rows, CYCLE_COLUMNS, MAX_CYCLE_ROWS))
        else:
            values = [(row.stress_range, row.count_cycle) for row in rows]
        cycles = np.asarray(values, dtype=float).reshape(-1, 2)
        if np.any(cycles < 0):
            raise ToolError("stress_range and count_cycle must be nonnegative.")

    sn_model = build_sn_curve(curve)
    try:
        damage = (
            calc_pm(cycles[:, 0], cycles[:, 1], sn_model)
            if len(cycles)
            else np.empty(0)
        )
    except (ValueError, AssertionError) as error:
        raise ToolError(
            f"Invalid S–N curve or stress range: {error}"
        ) from error

    if not np.isfinite(damage).all() or not math.isfinite(float(damage.sum())):
        raise ToolError("Miner damage is non-finite for the supplied inputs.")
    return np.column_stack((cycles, damage))


def miner_damage(
    source: MinerSource, curve: SNParameters, export_csv: StrictBool = False
) -> MinerSummary:
    """Calculate Palmgren–Miner damage with caller-supplied S–N parameters.

    Use stresses in MPa. Pass an ordered history for ASTM counting or
    supply counted cycles directly. S–N intercepts are log10 intercepts.
    Damage is dimensionless. Set export_csv to save every contribution.
    """
    table = miner_table(source, curve)
    total_damage = float(table[:, 2].sum())
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
