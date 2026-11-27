"""Expose Py-Fatigue calculations to local MCP clients over stdio."""

import csv
import io
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Iterable, Iterator, Literal, Sequence
from uuid import uuid4

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictBool

from .crack_growth import (
    AxialCylinderThroughCrack,
    CenterThroughCrack,
    CircumferentialCylinderSurfaceCrack,
    CircumferentialCylinderThroughCrack,
    EdgeThroughCrack,
    GrowthPoint,
    HoleBoreSurfaceCrack,
    LoadBlock,
    NasgroCurve,
    RolledBoltThreadSurfaceCrack,
    RoundBarSurfaceCrack,
    SemiEllipticalSurfaceCrack,
    SphereThroughCrack,
    simulate_crack_growth,
)
from .cycle_count.rainflow import rainflow
from .damage.stress_life import calc_pm
from .material.sn_curve import SNCurve

MAX_CSV_BYTES = 64 * 1024 * 1024
MAX_CYCLE_ROWS = 1_000_000
MAX_LOAD_BLOCKS = 10_000
MAX_TRIALS = 100_000

INPUT_DIR = "PY_FATIGUE_MCP_INPUT_DIR"
OUTPUT_DIR = "PY_FATIGUE_MCP_OUTPUT_DIR"
CYCLE_COLUMNS = ("stress_range", "count_cycle")
RAINFLOW_COLUMNS = ("stress_range", "mean_stress", "count_cycle")
LOAD_COLUMNS = ("minimum", "maximum", "cycles")

Number = Annotated[FiniteFloat, Field(strict=True)]
Positive = Annotated[Number, Field(gt=0)]
Nonnegative = Annotated[Number, Field(ge=0)]

mcp = MCPServer(
    "py-fatigue",
    instructions=(
        "Use MPa for stress and mm for lengths. Supply the analysis inputs "
        "and calibrated material parameters. Results contain up to five "
        "preview rows. Offer CSV export for complete results when the preview "
        "omits rows, and export only at the user's request."
    ),
)


class InputModel(BaseModel):
    """Validate the JSON shape and reject unknown input fields."""

    model_config = ConfigDict(extra="forbid", strict=True)


class CsvFile(InputModel):
    """A CSV path relative to PY_FATIGUE_MCP_INPUT_DIR."""

    csv_path: str


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


class TC01(InputModel):
    """Center through crack with plate half-width and crack half-length."""

    case: Literal["TC01"]
    half_width: Positive
    half_length: Positive


class TC02(InputModel):
    """Single edge through crack with full plate width and crack length."""

    case: Literal["TC02"]
    width: Positive
    length: Positive


class ShellDimensions(InputModel):
    """Shell dimensions with mid-wall radius and crack half-length."""

    mean_radius: Positive
    thickness: Positive
    half_length: Positive


class TC06(ShellDimensions):
    """Sphere through crack opened by membrane tension."""

    case: Literal["TC06"]


class TC07(ShellDimensions):
    """Axial cylinder through crack opened by hoop membrane stress."""

    case: Literal["TC07"]


class TC08(ShellDimensions):
    """Circumferential cylinder through crack under axial tension."""

    case: Literal["TC08"]
    length: Positive
    poisson: Number


class SC01(InputModel):
    """Semi-elliptical surface crack under membrane tension."""

    case: Literal["SC01"]
    half_width: Positive
    thickness: Positive
    depth: Positive
    half_length: Positive


class SC05(InputModel):
    """External circumferential pipe surface crack under axial tension."""

    case: Literal["SC05"]
    inner_radius: Positive
    thickness: Positive
    depth: Positive
    half_length: Positive


class SC07(InputModel):
    """Round-bar surface crack under remote tension."""

    case: Literal["SC07"]
    diameter: Positive
    depth: Positive


class SC08(InputModel):
    """Surface crack in a rolled bolt thread under axial tension."""

    case: Literal["SC08"]
    major_diameter: Positive
    minor_diameter: Positive
    depth: Positive


class SC11(InputModel):
    """Centered plate-hole bore crack under remote tension."""

    case: Literal["SC11"]
    width: Positive
    thickness: Positive
    hole_diameter: Positive
    depth: Positive
    half_length: Positive
    crack_count: Literal[1, 2]


CrackGeometry = Annotated[
    TC01 | TC02 | TC06 | TC07 | TC08 | SC01 | SC05 | SC07 | SC08 | SC11,
    Field(discriminator="case"),
]

GEOMETRY_TYPES = {
    "TC01": CenterThroughCrack,
    "TC02": EdgeThroughCrack,
    "TC06": SphereThroughCrack,
    "TC07": AxialCylinderThroughCrack,
    "TC08": CircumferentialCylinderThroughCrack,
    "SC01": SemiEllipticalSurfaceCrack,
    "SC05": CircumferentialCylinderSurfaceCrack,
    "SC07": RoundBarSurfaceCrack,
    "SC08": RolledBoltThreadSurfaceCrack,
    "SC11": HoleBoreSurfaceCrack,
}


class NasgroParameters(InputModel):
    """Calibrated NASGRO parameters in the MPa and mm unit system."""

    coefficient: Positive
    exponent: Positive
    p: Nonnegative
    q: Nonnegative
    threshold_zero: Nonnegative
    threshold_fanning: Number
    toughness: Positive
    constraint: Annotated[Number, Field(ge=1, le=3)]
    peak_to_flow: Annotated[Number, Field(ge=0, lt=1)]


class TensileLoad(InputModel):
    """Ordered tensile stress extrema in MPa and a fractional cycle count."""

    minimum: Nonnegative
    maximum: Positive
    cycles: Positive


LoadInput = (
    Annotated[
        list[TensileLoad], Field(min_length=1, max_length=MAX_LOAD_BLOCKS)
    ]
    | CsvFile
)


class SolverControls(InputModel):
    """Step cap in cycles, length tolerance in mm, and trial controls."""

    max_cycle_step: Positive
    length_tolerance: Positive
    cycle_tolerance: Positive
    max_trials: Annotated[int, Field(gt=0, le=MAX_TRIALS)]


class CsvExport(BaseModel):
    """The location and data row count of a complete local CSV."""

    relative_path: str
    row_count: int


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


class CrackGrowthSummary(BaseModel):
    """Solver outcome and accepted states in cycles, mm, and MPa sqrt(mm)."""

    termination: str
    final_cycles: FiniteFloat
    final_state: GrowthPoint | None
    dimension_names: tuple[str, ...]
    row_count: int
    preview: list[GrowthPoint]
    export: CsvExport | None = None


def configured_directory(variable: str) -> Path:
    """Resolve the directory configured by an MCP client."""
    value = os.environ.get(variable)
    if not value:
        raise ToolError(f"Set {variable} to enable this CSV operation.")
    try:
        directory = Path(value).resolve(strict=True)
        if directory.is_dir():
            return directory
    except (OSError, RuntimeError, ValueError):
        pass
    raise ToolError(f"{variable} must point to an existing directory.")


def read_csv_rows(
    source: CsvFile, columns: tuple[str, ...], limit: int
) -> Iterator[tuple[float, ...]]:
    """Read finite numbers from required columns in their requested order."""
    directory = configured_directory(INPUT_DIR)
    try:
        relative_path = Path(source.csv_path)
        if relative_path.anchor or ".." in relative_path.parts:
            raise ToolError(
                "CSV paths must be relative to the input directory."
            )
        path = (directory / relative_path).resolve(strict=True)
        if not path.is_relative_to(directory) or not path.is_file():
            raise ToolError(
                "CSV input must be a file inside the input directory."
            )
        if path.stat().st_size > MAX_CSV_BYTES:
            raise ToolError("Input CSV exceeds the 64 MiB size limit.")
        with path.open("rb") as stream:
            payload = stream.read(MAX_CSV_BYTES + 1)
    except (OSError, RuntimeError, ValueError) as error:
        raise ToolError("Could not read the input CSV.") from error
    if len(payload) > MAX_CSV_BYTES:
        raise ToolError("Input CSV exceeds the 64 MiB size limit.")

    with io.TextIOWrapper(
        io.BytesIO(payload), encoding="utf-8-sig", newline=""
    ) as stream:
        reader = csv.reader(stream, strict=True)
        try:
            header = next(reader, [])
            if len(header) != len(set(header)) or not set(columns) <= set(
                header
            ):
                required = ", ".join(columns)
                raise ToolError(
                    f"CSV header must contain unique columns for {required}."
                )
            indices = [header.index(column) for column in columns]
            for count, row in enumerate(reader, start=1):
                if count > limit:
                    raise ToolError(
                        f"Input CSV exceeds the {limit:,}-row limit."
                    )
                if len(row) != len(header):
                    raise ToolError(
                        f"CSV row {reader.line_num} has the wrong field count."
                    )
                try:
                    values = tuple(float(row[index]) for index in indices)
                except ValueError as error:
                    raise ToolError(
                        f"CSV row {reader.line_num} has an invalid number."
                    ) from error
                if not all(map(math.isfinite, values)):
                    raise ToolError(
                        f"CSV row {reader.line_num} requires finite numbers."
                    )
                yield values
        except (csv.Error, UnicodeError) as error:
            raise ToolError(
                f"Malformed UTF-8 CSV near row {reader.line_num + 1}."
            ) from error


def write_csv_rows(
    prefix: str,
    columns: Iterable[str],
    rows: Iterable[Iterable[float]],
    row_count: int,
) -> CsvExport:
    """Publish a complete CSV under a generated name in the output directory."""
    directory = configured_directory(OUTPUT_DIR)
    filename = f"{prefix}-{uuid4().hex}.csv"
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=directory, delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(rows)
        temporary_path.replace(directory / filename)
    except (OSError, csv.Error) as error:
        raise ToolError("Could not write the result CSV.") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return CsvExport(relative_path=filename, row_count=row_count)


def preview_indices(row_count: int) -> list[int]:
    """Select at most five rows from the beginning and end of a result.

    Examples
    --------
    >>> preview_indices(8)
    [0, 1, 2, 6, 7]
    """
    if row_count <= 5:
        return list(range(row_count))
    return [0, 1, 2, row_count - 2, row_count - 1]


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
    table = cycles[:, :3].copy()
    table[:, 0] *= 2
    if not np.isfinite(table).all():
        raise ToolError("Rainflow counting produced non-finite cycle values.")
    return table


@mcp.tool()
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


@mcp.tool()
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


def growth_csv_rows(
    history: Sequence[GrowthPoint],
) -> Iterator[tuple[float, ...]]:
    """Flatten accepted crack states for a CSV with ordered tip columns."""
    for point in history:
        yield (
            point.cycles,
            point.block,
            *point.sizes,
            *point.k_min,
            *point.k_max,
        )


@mcp.tool()
def ordered_crack_growth(
    geometry: CrackGeometry,
    curve: NasgroParameters,
    loads: LoadInput,
    controls: SolverControls,
    export_csv: StrictBool = False,
) -> CrackGrowthSummary:
    """Simulate ordered tensile crack growth with a calibrated NASGRO law.

    Use MPa, mm, and MPa sqrt(mm). Supply all solver controls and preserve
    the applied load block order. The final state is the last accepted
    point on the admissible side of a toughness or geometry limit.
    """
    crack = GEOMETRY_TYPES[geometry.case](
        **geometry.model_dump(exclude={"case"})
    )
    if isinstance(loads, CsvFile):
        blocks = [
            LoadBlock(*row)
            for row in read_csv_rows(loads, LOAD_COLUMNS, MAX_LOAD_BLOCKS)
        ]
    else:
        blocks = [LoadBlock(**row.model_dump()) for row in loads]
    if not blocks:
        raise ToolError("Supply at least one ordered load block.")
    for index, block in enumerate(blocks):
        if (
            not 0 <= block.minimum <= block.maximum
            or block.maximum <= 0
            or block.cycles <= 0
        ):
            raise ToolError(
                f"Load block {index} requires 0 <= minimum <= maximum, "
                "with positive maximum and cycles."
            )
    try:
        result = simulate_crack_growth(
            crack,
            NasgroCurve(**curve.model_dump()),
            blocks,
            **controls.model_dump(),
        )
    except (ValueError, RuntimeError) as error:
        raise ToolError(str(error)) from error

    history = result.history
    summary = CrackGrowthSummary(
        termination=result.termination,
        final_cycles=result.final_cycles,
        final_state=history[-1] if history else None,
        dimension_names=result.dimension_names,
        row_count=len(history),
        preview=[history[index] for index in preview_indices(len(history))],
    )
    if export_csv:
        columns = (
            "cycles",
            "block",
            *(
                f"{label}_{index}"
                for label in (
                    "size_mm",
                    "k_min_mpa_sqrt_mm",
                    "k_max_mpa_sqrt_mm",
                )
                for index in range(1, len(result.dimension_names) + 1)
            ),
        )
        summary.export = write_csv_rows(
            "crack-growth", columns, growth_csv_rows(history), len(history)
        )
    return summary


if __name__ == "__main__":
    mcp.run()
