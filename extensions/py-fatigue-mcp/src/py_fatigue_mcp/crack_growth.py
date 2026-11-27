"""MCP tool for ordered NASGRO crack growth."""

from typing import Annotated, Iterator, Literal, Sequence

from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field, FiniteFloat, StrictBool

from py_fatigue.crack_growth import (
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

from .csv_io import CsvExport, CsvFile, read_csv_rows, write_csv_rows
from .models import InputModel, Nonnegative, Number, Positive, preview_indices

MAX_LOAD_BLOCKS = 10_000
MAX_TRIALS = 100_000
LOAD_COLUMNS = ("minimum", "maximum", "cycles")


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
    crack_count: Annotated[int, Field(ge=1, le=2)]


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


class CrackGrowthSummary(BaseModel):
    """Solver outcome and accepted states in cycles, mm, and MPa sqrt(mm)."""

    termination: str
    final_cycles: FiniteFloat
    final_state: GrowthPoint | None
    dimension_names: tuple[str, ...]
    row_count: int
    preview: list[GrowthPoint]
    export: CsvExport | None = None


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
