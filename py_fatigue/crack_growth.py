"""Crack growth under ordered tensile membrane load blocks."""

from collections.abc import Sequence
from dataclasses import astuple, dataclass

from ._rust import nasgro_rate, ordered_growth, plate_factors
from .damage.crack_growth import (
    CalcCrackGrowth,
    CrackGrowth,
    get_crack_growth,
    get_geometry_factor,
    get_sif,
)
from .geometry import HollowCylinder, InfiniteSurface
from .material.crack_growth_curve import ParisCurve, WalkerCurve

__all__ = [
    "CalcCrackGrowth",
    "CenterThroughCrack",
    "CrackGrowth",
    "GrowthPoint",
    "GrowthResult",
    "HollowCylinder",
    "InfiniteSurface",
    "LoadBlock",
    "NasgroCurve",
    "ParisCurve",
    "SemiEllipticalSurfaceCrack",
    "WalkerCurve",
    "get_crack_growth",
    "get_geometry_factor",
    "get_sif",
    "simulate_crack_growth",
]


@dataclass(frozen=True)
class LoadBlock:
    """One ordered block with tensile extrema in MPa and fractional cycles."""

    minimum: float
    maximum: float
    cycles: float


@dataclass(frozen=True)
class CenterThroughCrack:
    """Symmetric center crack with plate and crack half-widths in mm."""

    half_width: float
    half_length: float

    def stress_intensity(self, stress: float) -> float:
        """Evaluate opening-mode intensity in MPa sqrt(mm)."""
        factor, _ = plate_factors(
            self.half_width, None, (self.half_length, 0.0)
        )
        return stress * factor


@dataclass(frozen=True)
class SemiEllipticalSurfaceCrack:
    """Surface crack with plate and crack dimensions in mm."""

    half_width: float
    thickness: float
    depth: float
    half_length: float

    def stress_intensity(self, stress: float) -> tuple[float, float]:
        """Return deepest-point and surface intensities in MPa sqrt(mm)."""
        depth, surface = plate_factors(
            self.half_width, self.thickness, (self.depth, self.half_length)
        )
        return stress * depth, stress * surface


@dataclass(frozen=True)
class NasgroCurve:
    """NASGRO 3.0 rate law for nonnegative stress ratios.

    Supply the calibrated C and n, threshold and toughness exponents p and q,
    long-crack threshold at R=0, threshold fanning constant, fixed toughness,
    Newman constraint factor and fitted maximum-to-flow stress ratio. Use
    MPa sqrt(mm) for stress intensity parameters and mm/cycle for growth rates.
    """

    coefficient: float
    exponent: float
    p: float  # pylint: disable=invalid-name
    q: float  # pylint: disable=invalid-name
    threshold_zero: float
    threshold_fanning: float
    toughness: float
    constraint: float
    peak_to_flow: float

    def get_growth_rate(
        self, length: float, k_min: float, k_max: float
    ) -> float:
        """Evaluate growth at a crack dimension and two stress intensities."""
        return nasgro_rate(astuple(self), length, k_min, k_max)


@dataclass(frozen=True)
class GrowthPoint:
    """Accepted state at a cumulative cycle count in a zero-based load block."""

    cycles: float
    sizes: tuple[float, ...]
    k_min: tuple[float, ...]
    k_max: tuple[float, ...]
    block: int


@dataclass(frozen=True)
class GrowthResult:
    """Accepted points and the reason propagation ended."""

    history: tuple[GrowthPoint, ...]
    termination: str

    @property
    def final_cycles(self) -> float:
        """Return the cycle count of the last accepted state, or zero."""
        return self.history[-1].cycles if self.history else 0.0


def simulate_crack_growth(
    geometry: CenterThroughCrack | SemiEllipticalSurfaceCrack,
    curve: NasgroCurve,
    loads: Sequence[LoadBlock],
    *,
    max_cycle_step: float,
    length_tolerance: float,
    cycle_tolerance: float,
    max_trials: int,
) -> GrowthResult:
    """Propagate a plate crack through the load blocks in the supplied order.

    Use MPa, mm and MPa sqrt(mm) throughout. Controls set the maximum step
    in cycles, absolute step error in mm, terminal bracket width in cycles and
    total number of trial steps. History contains accepted states. At a
    geometry or toughness limit, its last state is on the admissible side
    of the bracket.
    """
    if isinstance(geometry, CenterThroughCrack):
        section = (geometry.half_width, None)
        initial = (geometry.half_length, 0.0)
        dimensions = 1
    elif isinstance(geometry, SemiEllipticalSurfaceCrack):
        section = (geometry.half_width, geometry.thickness)
        initial = (geometry.depth, geometry.half_length)
        dimensions = 2
    else:
        raise TypeError("expected a supported plate crack geometry")
    raw, termination = ordered_growth(
        section,
        initial,
        astuple(curve),
        [(load.minimum, load.maximum, load.cycles) for load in loads],
        (max_cycle_step, length_tolerance, cycle_tolerance, max_trials),
    )
    history = tuple(
        GrowthPoint(
            cycles,
            tuple(sizes[:dimensions]),
            tuple(k_min[:dimensions]),
            tuple(k_max[:dimensions]),
            block,
        )
        for cycles, sizes, k_min, k_max, block in raw
    )
    return GrowthResult(history, termination)
