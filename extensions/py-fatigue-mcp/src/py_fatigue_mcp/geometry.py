"""Resolve ordered and legacy Py-Fatigue geometry inputs."""

from typing import Annotated, Literal

from mcp.server.mcpserver.exceptions import ToolError
import numpy as np
from pydantic import BaseModel, Field

from py_fatigue.geometry import (
    Cylinder,
    HollowCylinder,
    InfiniteSurface,
    TabulatedGeometryFactor,
)

from .crack_growth import CrackGeometry, GEOMETRY_TYPES
from .models import InputModel, Nonnegative, Number, Positive

FactorTable = Annotated[list[Positive], Field(min_length=2, max_length=10_000)]


class InitialDepth(InputModel):
    """Initial crack depth in mm for a legacy geometry."""

    initial_depth: Positive


class InfiniteSurfaceInput(InitialDepth):
    """Depth-only initial crack at an infinite surface."""

    case: Literal["infinite_surface"]


class CylinderInput(InitialDepth):
    """Solid cylinder side-view dimensions."""

    case: Literal["cylinder"]
    diameter: Positive
    height: Positive


class HollowCylinderInput(InitialDepth):
    """Hollow cylinder with an internal or external initial flaw."""

    case: Literal["hollow_cylinder"]
    outer_diameter: Positive
    thickness: Positive
    height: Positive
    crack_position: Literal["internal", "external"]
    width_to_depth_ratio: Positive = 2.0


class TabulatedInput(InitialDepth):
    """Depths in mm and corresponding dimensionless factors."""

    case: Literal["tabulated"]
    depths: FactorTable
    factors: FactorTable


PlotGeometry = Annotated[
    CrackGeometry
    | InfiniteSurfaceInput
    | CylinderInput
    | HollowCylinderInput
    | TabulatedInput,
    Field(discriminator="case"),
]
PLOT_GEOMETRY_TYPES = {
    **GEOMETRY_TYPES,
    "infinite_surface": InfiniteSurface,
    "cylinder": Cylinder,
    "hollow_cylinder": HollowCylinder,
    "tabulated": TabulatedGeometryFactor,
}


class IntensityResult(BaseModel):
    """One row per supplied tensile stress, in native crack-tip order."""

    stress_mpa: list[Number]
    intensity_mpa_sqrt_mm: list[list[Number]]


def geometry_stress_intensity(
    geometry: CrackGeometry,
    stress_mpa: Annotated[
        list[Nonnegative], Field(min_length=1, max_length=1000)
    ],
) -> IntensityResult:
    """Evaluate opening-mode stress intensity for any ordered geometry.

    Use the opening stress specified by the geometry case. The intensities
    are returned in MPa sqrt(mm), in the native crack-tip order.
    """
    try:
        crack = GEOMETRY_TYPES[geometry.case](
            **geometry.model_dump(exclude={"case"})
        )
        values = [
            np.atleast_1d(crack.stress_intensity(stress))
            for stress in stress_mpa
        ]
    except (ValueError, OverflowError) as error:
        raise ToolError(f"Cannot evaluate the geometry: {error}") from error
    if any(not np.isfinite(row).all() for row in values):
        raise ToolError("Geometry evaluation produced a non-finite intensity.")
    return IntensityResult(
        stress_mpa=stress_mpa,
        intensity_mpa_sqrt_mm=[row.tolist() for row in values],
    )
