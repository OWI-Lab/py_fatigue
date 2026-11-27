"""Typed requests for native plots and transport-specific input bounds."""

from typing import Annotated, Generic, Literal, TypeVar

from pydantic import Field

from .crack_growth import (
    CrackGeometry,
    InlineLoads,
    LoadInput,
    NasgroParameters,
    SolverControls,
)
from .cycle_count import CycleInput, InlineCycleInput, RainflowMethod
from .geometry import PlotGeometry
from .models import InputModel, Nonnegative, Number, Positive
from .stress_life import (
    EvaluationPoints,
    InlineMinerSource,
    MinerSource,
    SNParameters,
)

PlotFormat = Literal["png", "echarts"]
Scale = Literal["loglog", "semilogx", "semilogy"]
BinCount = Annotated[int, Field(ge=2, le=200)]
BinEdges = Annotated[list[Number], Field(min_length=2, max_length=201)]
AxisRange = Annotated[list[Number], Field(min_length=2, max_length=2)]
StressSource = TypeVar("StressSource")
DamageSource = TypeVar("DamageSource")
OrderedLoads = TypeVar("OrderedLoads")


class CyclePlot(InputModel, Generic[StressSource]):
    """CycleCount histogram, residual sequence or half-cycle sequence."""

    kind: Literal["cycle_count"]
    stress: StressSource
    method: RainflowMethod = "astm"
    view: Literal["histogram", "residuals", "half_cycles"] = "histogram"
    axes: Literal[
        "min-max",
        "mean-range",
        "counts-range",
        "counts-range-cumsum",
        "counts-range-2D",
    ] = "min-max"
    bins: (
        BinCount
        | Annotated[
            list[BinCount | BinEdges], Field(min_length=2, max_length=2)
        ]
    ) = 50
    bin_range: (
        Annotated[list[AxisRange], Field(min_length=2, max_length=2)] | None
    ) = None
    mode: Literal["mountain", "valley", "clip"] | None = "mountain"
    normalize: bool = False


class SNPlot(InputModel):
    """S–N curve with an optional measured cycle-range overlay."""

    kind: Literal["sn_curve"]
    curve: SNParameters
    cycles: EvaluationPoints | None = None
    stress_ranges: EvaluationPoints | None = None
    scale: Scale = "loglog"


class MinerPlot(InputModel, Generic[DamageSource]):
    """Miner damage contributions or the native contribution histogram."""

    kind: Literal["miner_damage"]
    source: DamageSource
    curve: SNParameters
    view: Literal["contributions", "histogram"] = "contributions"


class GeometryPlot(InputModel):
    """Initial geometry for an ordered or legacy crack geometry."""

    kind: Literal["geometry"]
    geometry: PlotGeometry


class GrowthPlot(InputModel, Generic[OrderedLoads]):
    """Ordered crack growth history or initial and final crack profile."""

    kind: Literal["crack_growth"]
    geometry: CrackGeometry
    curve: NasgroParameters
    loads: OrderedLoads
    controls: SolverControls
    view: Literal["history", "profile"] = "history"


class NasgroPlot(InputModel):
    """NASGRO growth rate against peak stress intensity."""

    kind: Literal["nasgro_curve"]
    curve: NasgroParameters
    length: Positive | None = None
    k_min: Nonnegative = 0.0
    k_max: Positive | EvaluationPoints | None = None
    scale: Scale = "semilogy"


class ParisPlot(InputModel):
    """Paris or Walker growth curve with optional paired observations."""

    kind: Literal["paris_curve", "walker_curve"]
    slope: Positive | EvaluationPoints
    intercept: Positive | EvaluationPoints
    threshold: Nonnegative = 0.0
    critical: Positive | None = None
    load_ratio: Annotated[float, Field(ge=-1, lt=1)] = 0.0
    walker_exponent: Nonnegative = 0.0
    sif: EvaluationPoints | None = None
    growth_rate: EvaluationPoints | None = None
    scale: Scale = "loglog"


MaterialPlot = SNPlot | NasgroPlot | ParisPlot
PlotRequest = Annotated[
    CyclePlot[CycleInput]
    | MinerPlot[MinerSource]
    | GrowthPlot[LoadInput]
    | GeometryPlot
    | MaterialPlot,
    Field(discriminator="kind"),
]
InlinePlotRequest = Annotated[
    CyclePlot[InlineCycleInput]
    | MinerPlot[InlineMinerSource]
    | GrowthPlot[InlineLoads]
    | GeometryPlot
    | MaterialPlot,
    Field(discriminator="kind"),
]
