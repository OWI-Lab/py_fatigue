"""Crack growth under ordered tensile membrane load blocks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import astuple, dataclass, field
from functools import partial
from typing import ClassVar

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import ArrayLike

from ._rust import (
    nasgro_long_crack_rate,
    nasgro_rate,
    ordered_growth,
    plate_factors,
    plate_net_section_stress,
)
from .damage.crack_growth import (
    CalcCrackGrowth,
    CrackGrowth,
    get_crack_growth,
    get_geometry_factor,
    get_sif,
)
from .geometry import HollowCylinder, InfiniteSurface
from .material.crack_growth_curve import ParisCurve, WalkerCurve
from .plotting import PlotMixin, _CurveScale, _plot_curve
from .utils import make_axes

__all__ = [
    "CalcCrackGrowth",
    "AxialCylinderThroughCrack",
    "CenterThroughCrack",
    "CircumferentialCylinderSurfaceCrack",
    "CircumferentialCylinderThroughCrack",
    "CrackGrowth",
    "EdgeThroughCrack",
    "GrowthPoint",
    "GrowthResult",
    "HoleBoreSurfaceCrack",
    "HollowCylinder",
    "InfiniteSurface",
    "LoadBlock",
    "NasgroCurve",
    "ParisCurve",
    "RolledBoltThreadSurfaceCrack",
    "RoundBarSurfaceCrack",
    "SemiEllipticalSurfaceCrack",
    "SphereThroughCrack",
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


class _PlateCrack(PlotMixin):
    def net_section_stress(self, stress: float) -> float:
        """Return Appendix B membrane net-section stress in MPa."""
        section, sizes, _ = _plate_state(self)
        return plate_net_section_stress(section, sizes, stress)


@dataclass(frozen=True)
class CenterThroughCrack(_PlateCrack):
    """TC01 center crack with plate and crack half-widths in mm."""

    half_width: float
    half_length: float

    def stress_intensity(self, stress: float) -> float:
        """Evaluate opening-mode intensity in MPa sqrt(mm)."""
        section, sizes, _ = _plate_state(self)
        factor, _ = plate_factors(section, sizes)
        return stress * factor

    def plot(
        self,
        fig: Figure | None = None,
        ax: Axes | None = None,
        *,
        result: GrowthResult | None = None,
    ) -> tuple[Figure, Axes]:
        """Draw the crack and optionally overlay its last accepted profile."""
        final = _final_profile((self.half_length,), result)
        end = ([-final[0], final[0]], [0, 0]) if final is not None else None
        fig, ax = make_axes(fig, ax)
        ax.plot(
            [-self.half_width, self.half_width], [0, 0], label="Plate width"
        )
        _plot_profiles(
            ax, ([-self.half_length, self.half_length], [0, 0]), end, "Crack"
        )
        ax.set(
            xlabel="Position across plate (mm)", title="Center through crack"
        )
        ax.yaxis.set_visible(False)
        ax.legend()
        return fig, ax


@dataclass(frozen=True)
class EdgeThroughCrack(_PlateCrack):
    """TC02 single edge crack under remote tension with dimensions in mm.

    Supply the full plate width and the crack length measured from one
    edge. The tension solution is supported through a length/width of 0.8.
    """

    width: float
    length: float

    def stress_intensity(self, stress: float) -> float:
        """Evaluate opening-mode intensity in MPa sqrt(mm)."""
        section, sizes, _ = _plate_state(self)
        factor, _ = plate_factors(section, sizes)
        return stress * factor

    def plot(
        self,
        fig: Figure | None = None,
        ax: Axes | None = None,
        *,
        result: GrowthResult | None = None,
    ) -> tuple[Figure, Axes]:
        """Draw the crack and optionally overlay its last accepted profile."""
        final = _final_profile((self.length,), result)
        end = ([0, final[0]], [0, 0]) if final is not None else None
        fig, ax = make_axes(fig, ax)
        ax.plot([0, self.width], [0, 0], label="Plate width")
        _plot_profiles(ax, ([0, self.length], [0, 0]), end, "Crack")
        ax.set(xlabel="Position across plate (mm)", title="Edge through crack")
        ax.yaxis.set_visible(False)
        ax.legend()
        return fig, ax


@dataclass(frozen=True)
class SemiEllipticalSurfaceCrack(_PlateCrack):
    """SC01 surface crack with plate and crack dimensions in mm."""

    half_width: float
    thickness: float
    depth: float
    half_length: float

    def stress_intensity(self, stress: float) -> tuple[float, float]:
        """Return deepest-point and surface intensities in MPa sqrt(mm)."""
        section, sizes, _ = _plate_state(self)
        depth, surface = plate_factors(section, sizes)
        return stress * depth, stress * surface

    def plot(
        self,
        fig: Figure | None = None,
        ax: Axes | None = None,
        *,
        result: GrowthResult | None = None,
    ) -> tuple[Figure, Axes]:
        """Draw the crack front and optionally overlay its final profile."""
        final = _final_profile((self.depth, self.half_length), result)
        fig, ax = make_axes(fig, ax)
        b, t = self.half_width, self.thickness
        ax.plot([-b, b, b, -b, -b], [0, 0, t, t, 0], label="Plate section")
        angle = np.linspace(0, np.pi, 101)
        cosine, sine = np.cos(angle), np.sin(angle)
        sine[[0, -1]] = 0.0
        end = (
            (final[1] * cosine, final[0] * sine) if final is not None else None
        )
        _plot_profiles(
            ax,
            (self.half_length * cosine, self.depth * sine),
            end,
            "Crack front",
        )
        ax.set(
            xlabel="Position across plate (mm)",
            ylabel="Depth (mm)",
            title="Semi-elliptical surface crack",
            aspect="equal",
        )
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
        ax.legend()
        return fig, ax


@dataclass(frozen=True)
class RoundBarSurfaceCrack(PlotMixin):
    """SC07 circular-arc surface crack in a solid round bar in tension.

    Diameter and depth are in mm. The opening-mode solution applies for
    0 < depth < diameter / 2. The front meets the bar surface at right angles.
    """

    diameter: float
    depth: float

    def stress_intensity(self, stress: float) -> float:
        """Return deepest-point stress intensity in MPa sqrt(mm)."""
        section, sizes, _ = _plate_state(self)
        factor, _ = plate_factors(section, sizes)
        return stress * factor

    def plot(
        self,
        fig: Figure | None = None,
        ax: Axes | None = None,
        *,
        result: GrowthResult | None = None,
    ) -> tuple[Figure, Axes]:
        """Draw the bar cross section and the initial and final crack fronts."""
        self.stress_intensity(1.0)
        final = _final_profile((self.depth,), result)
        radius = self.diameter / 2
        end = None
        if final is not None:
            RoundBarSurfaceCrack(self.diameter, final[0]).stress_intensity(1.0)
            end = _round_bar_front(radius, final[0])
        fig, ax = make_axes(fig, ax)
        angle = np.linspace(0, 2 * np.pi, 201)
        ax.plot(radius * np.cos(angle), radius * np.sin(angle), label="Bar")
        _plot_profiles(
            ax, _round_bar_front(radius, self.depth), end, "Crack front"
        )
        ax.set(
            xlabel="Position across bar (mm)",
            ylabel="Position across bar (mm)",
            title="Circular surface crack in a solid round bar",
            aspect="equal",
        )
        ax.legend()
        return fig, ax


@dataclass(frozen=True)
class CircumferentialCylinderSurfaceCrack(PlotMixin):
    """External SC05 thumbnail crack subject to remote axial tension.

    Parameters
    ----------
    inner_radius, thickness : float
        Pipe inner radius and wall thickness in mm.
    depth, half_length : float
        Radial depth and outer-surface arc half-length in mm.

    Notes
    -----
    Interpolates NASA TM-87594 Table 1 within 1 <= R/t <= 10,
    0.2 <= a/t <= 0.8 and 0.6 <= a/c <= 1. The crack is external.
    """

    inner_radius: float
    thickness: float
    depth: float
    half_length: float

    def stress_intensity(self, stress: float) -> tuple[float, float]:
        """Return deepest and free-surface K in MPa sqrt(mm)."""
        section, sizes, _ = _plate_state(self)
        depth, surface = plate_factors(section, sizes)
        return stress * depth, stress * surface

    def plot(
        self,
        fig: Figure | None = None,
        ax: Axes | None = None,
        *,
        result: GrowthResult | None = None,
    ) -> tuple[Figure, Axes]:
        """Draw the pipe section and the initial and last accepted fronts."""
        self.stress_intensity(1.0)
        final = _final_profile((self.depth, self.half_length), result)
        fig, ax = make_axes(fig, ax)
        radius = self.inner_radius + self.thickness
        angle = np.linspace(0, 2 * np.pi, 201)
        for wall, label in (
            (radius, "Outer wall"),
            (self.inner_radius, "Inner wall"),
        ):
            ax.plot(wall * np.cos(angle), wall * np.sin(angle), label=label)
        _plot_profiles(
            ax,
            _curved_surface_front(radius, self.depth, self.half_length),
            _curved_surface_front(radius, *final) if final else None,
            "Crack front",
        )
        ax.set(
            xlabel="Section x (mm)",
            ylabel="Section y (mm)",
            title="SC05 · External circumferential pipe crack",
            aspect="equal",
        )
        ax.legend()
        return fig, ax


@dataclass(frozen=True)
class RolledBoltThreadSurfaceCrack(PlotMixin):
    """SC08 thumbnail at first contact in a rolled aerospace bolt thread.

    Parameters
    ----------
    major_diameter, minor_diameter : float
        Thread crest and root diameters in mm.
    depth : float
        Radial crack depth from the thread root in mm.

    Notes
    -----
    Use nominal axial tension on the minor-diameter section. The model
    interpolates the rolled-thread column of NLR-TP-99313 Table 2 for
    0 < a/d <= 0.5. The surface arc half-length is equal to the depth.
    """

    major_diameter: float
    minor_diameter: float
    depth: float

    def stress_intensity(self, stress: float) -> float:
        """Return deepest-point K in MPa sqrt(mm)."""
        section, sizes, _ = _plate_state(self)
        factor, _ = plate_factors(section, sizes)
        return stress * factor

    def plot(
        self,
        fig: Figure | None = None,
        ax: Axes | None = None,
        *,
        result: GrowthResult | None = None,
    ) -> tuple[Figure, Axes]:
        """Draw a schematic thread section with initial and final fronts."""
        self.stress_intensity(1.0)
        final = _final_profile((self.depth,), result)
        root = self.minor_diameter / 2
        fig, ax = make_axes(fig, ax)
        angle = np.linspace(0, 2 * np.pi, 201)
        for radius, label in (
            (self.major_diameter / 2, "Thread crest"),
            (root, "Thread root"),
        ):
            ax.plot(radius * np.cos(angle), radius * np.sin(angle), label=label)
        phi = np.linspace(0, np.pi, 101)

        def front(depth: float) -> tuple[ArrayLike, ArrayLike]:
            radius = root - depth * np.sin(phi)
            circumferential = depth / root * np.cos(phi)
            return radius * np.cos(circumferential), radius * np.sin(
                circumferential
            )

        _plot_profiles(
            ax,
            front(self.depth),
            front(final[0]) if final else None,
            "Crack front",
        )
        ax.set(
            xlabel="Section x (mm)",
            ylabel="Section y (mm)",
            title="SC08 · Rolled bolt thread",
            aspect="equal",
        )
        ax.legend()
        return fig, ax


@dataclass(frozen=True)
class HoleBoreSurfaceCrack(PlotMixin):
    """SC11 bore crack at a centered circular hole under remote tension.

    Parameters
    ----------
    width, thickness, hole_diameter : float
        Full plate width and thickness and the hole diameter in mm.
    depth, half_length : float
        Radial depth into the ligament and half-length along the bore in
        mm. These are c and a, respectively, in NASA TM-85793.
    crack_count : int
        One crack, or two identical cracks on opposite sides of the bore.
        Each starts halfway through the plate.

    Notes
    -----
    NASA TM-85793 requires 0.2 <= a/c <= 2, a < t/2, 0.5 <= D/t <= 2,
    and (D + 2c)/W < 0.5, where t and W below denote full plate dimensions.
    """

    width: float
    thickness: float
    hole_diameter: float
    depth: float
    half_length: float
    crack_count: int

    def stress_intensity(self, stress: float) -> tuple[float, float]:
        """Return radial-tip and bore-tip K in MPa sqrt(mm)."""
        section, sizes, _ = _plate_state(self)
        radial, bore = plate_factors(section, sizes)
        return stress * radial, stress * bore

    def plot(
        self,
        fig: Figure | None = None,
        ax: Axes | None = None,
        *,
        result: GrowthResult | None = None,
    ) -> tuple[Figure, Axes]:
        """Draw the plate section and the initial and last accepted fronts."""
        self.stress_intensity(1.0)
        final = _final_profile((self.depth, self.half_length), result)
        fig, ax = make_axes(fig, ax)
        radius, width, height = (
            self.hole_diameter / 2,
            self.width / 2,
            self.thickness / 2,
        )
        for side, label in ((-1, "Left ligament"), (1, "Right ligament")):
            ax.plot(
                [
                    side * radius,
                    side * width,
                    side * width,
                    side * radius,
                    side * radius,
                ],
                [-height, -height, height, height, -height],
                label=label,
            )
        angle = np.linspace(-np.pi / 2, np.pi / 2, 101)
        cosine, sine = np.cos(angle), np.sin(angle)
        cosine[[0, -1]] = 0.0

        def fronts(depth: float, span: float) -> tuple[ArrayLike, ArrayLike]:
            radial = radius + depth * cosine
            bore = span * sine
            if self.crack_count == 2:
                return np.r_[radial, np.nan, -radial], np.r_[bore, np.nan, bore]
            return radial, bore

        _plot_profiles(
            ax,
            fronts(self.depth, self.half_length),
            fronts(*final) if final else None,
            "Crack front",
        )
        ax.set(
            xlabel="Position across plate (mm)",
            ylabel="Position along bore (mm)",
            title=f"SC11 · Plate hole ({self.crack_count} crack(s))",
            aspect="equal",
        )
        ax.legend()
        return fig, ax


@dataclass(frozen=True)
class _ShellThroughCrack(PlotMixin):
    mean_radius: float
    thickness: float
    half_length: float
    _case: ClassVar[str]
    _title: ClassVar[str]

    def stress_intensity(self, stress: float) -> float:
        """Return Mode I intensity in MPa sqrt(mm) from membrane stress in MPa."""
        section, sizes, _ = _plate_state(self)
        factor, _ = plate_factors(section, sizes)
        return stress * factor

    def plot(
        self,
        fig: Figure | None = None,
        ax: Axes | None = None,
        *,
        result: GrowthResult | None = None,
    ) -> tuple[Figure, Axes]:
        """Draw the shell section and overlay the last accepted crack."""
        self.stress_intensity(1.0)
        final = _final_profile((self.half_length,), result)
        fig, ax = make_axes(fig, ax)
        if self._case == "TC07":
            half_span = max(self.half_length, final[0] if final else 0.0)
            span = [-half_span, half_span]
            wall = self.thickness / 2
            ax.plot(span, [wall, wall], label="Outer wall")
            ax.plot(span, [-wall, -wall], label="Inner wall")
            initial = ([-self.half_length, self.half_length], [0, 0])
            end = ([-final[0], final[0]], [0, 0]) if final else None
            xlabel = "Axial position (mm)"
            ylabel = "Position through wall (mm)"
        else:
            radius = self.mean_radius
            angle = np.linspace(0, 2 * np.pi, 201)
            for surface, name in (
                (radius + self.thickness / 2, "Outer wall"),
                (radius - self.thickness / 2, "Inner wall"),
            ):
                ax.plot(
                    surface * np.cos(angle), surface * np.sin(angle), label=name
                )
            initial = _shell_arc(radius, self.half_length)
            end = _shell_arc(radius, final[0]) if final else None
            xlabel, ylabel = "Section x (mm)", "Section y (mm)"
            ax.set_aspect("equal")
        _plot_profiles(ax, initial, end, "Crack")
        ax.set(xlabel=xlabel, ylabel=ylabel, title=self._title)
        ax.legend()
        return fig, ax


@dataclass(frozen=True)
class SphereThroughCrack(_ShellThroughCrack):
    """TC06 meridional arc in a sphere under membrane tension, in mm."""

    _case = "TC06"
    _title = "TC06 · Sphere through crack"


@dataclass(frozen=True)
class AxialCylinderThroughCrack(_ShellThroughCrack):
    """TC07 axial through crack in a cylinder opened by hoop stress, in mm."""

    _case = "TC07"
    _title = "TC07 · Axial cylinder through crack"


@dataclass(frozen=True)
class CircumferentialCylinderThroughCrack(_ShellThroughCrack):
    """TC08 circumferential arc in a thin cylinder under axial tension.

    Use the mid-wall radius and arc half-length. Length is the cylinder's
    full axial length. All lengths are in mm. Supply Poisson's ratio.
    """

    length: float
    poisson: float
    _case = "TC08"
    _title = "TC08 · Circumferential cylinder through crack"


def _shell_arc(
    radius: float, half_length: float
) -> tuple[ArrayLike, ArrayLike]:
    angle = np.linspace(-half_length / radius, half_length / radius, 101)
    return radius * np.cos(angle), radius * np.sin(angle)


def _round_bar_front(
    radius: float, depth: float
) -> tuple[ArrayLike, ArrayLike]:
    # Orthogonal circles satisfy center² = bar radius² + front radius².
    tip = radius - depth
    ratio = tip / radius
    theta_limit = 2 * np.arctan(ratio)
    theta = np.linspace(-theta_limit, theta_limit, 101)
    # Divide the sine by the ratio before multiplying by the front radius.
    scaled_front_radius = depth * (1 + ratio) / 2
    x = tip + scaled_front_radius * (np.sin(theta / 2) / ratio) * (
        2 * np.sin(theta / 2)
    )
    y = scaled_front_radius * (np.sin(theta) / ratio)
    return x, y


def _curved_surface_front(
    radius: float, depth: float, half_length: float
) -> tuple[ArrayLike, ArrayLike]:
    # NASA TM-87594 Appendix. The transformation preserves the outer arc.
    angle = np.linspace(0, np.pi, 101)
    sine = np.sin(angle)
    sine[[0, -1]] = 0.0
    radial = radius * np.exp(np.log1p(-depth / radius) * sine)
    circumferential = half_length / radius * np.cos(angle)
    return radial * np.cos(circumferential), radial * np.sin(circumferential)


def _final_profile(
    initial: tuple[float, ...], result: GrowthResult | None
) -> tuple[float, ...] | None:
    if result is None or not result.history:
        return None
    final = result.history[-1].sizes
    if result.history[0].sizes != initial or len(final) != len(initial):
        raise ValueError(
            "growth history does not match the initial crack dimensions"
        )
    return final


def _plot_profiles(
    ax: Axes,
    initial: tuple[ArrayLike, ArrayLike],
    final: tuple[ArrayLike, ArrayLike] | None,
    label: str,
) -> None:
    if final is None:
        ax.plot(*initial, color="firebrick", linewidth=3, label=label)
        return
    ax.plot(
        *initial, color="navy", linestyle="--", zorder=3, label="Initial crack"
    )
    ax.plot(
        *final, color="firebrick", linewidth=3, label="Final accepted crack"
    )


def _plate_state(
    geometry: (
        _PlateCrack
        | RoundBarSurfaceCrack
        | CircumferentialCylinderSurfaceCrack
        | RolledBoltThreadSurfaceCrack
        | HoleBoreSurfaceCrack
        | _ShellThroughCrack
    ),
) -> tuple[
    tuple[str, float, float | None, tuple[float, float] | None],
    tuple[float, float],
    tuple[str, ...],
]:
    if isinstance(geometry, CenterThroughCrack):
        return (
            ("TC01", geometry.half_width, None, None),
            (geometry.half_length, 0.0),
            ("Half-length",),
        )
    if isinstance(geometry, EdgeThroughCrack):
        return (
            ("TC02", geometry.width, None, None),
            (geometry.length, 0.0),
            ("Length",),
        )
    if isinstance(geometry, SemiEllipticalSurfaceCrack):
        return (
            ("SC01", geometry.half_width, geometry.thickness, None),
            (geometry.depth, geometry.half_length),
            ("Depth", "Half-length"),
        )
    if isinstance(geometry, RoundBarSurfaceCrack):
        return (
            ("SC07", geometry.diameter, None, None),
            (geometry.depth, 0.0),
            ("Depth",),
        )
    if isinstance(geometry, CircumferentialCylinderSurfaceCrack):
        return (
            ("SC05", geometry.inner_radius, geometry.thickness, None),
            (geometry.depth, geometry.half_length),
            ("Radial depth", "Surface arc half-length"),
        )
    if isinstance(geometry, RolledBoltThreadSurfaceCrack):
        return (
            ("SC08", geometry.minor_diameter, geometry.major_diameter, None),
            (geometry.depth, 0.0),
            ("Depth",),
        )
    if isinstance(geometry, HoleBoreSurfaceCrack):
        return (
            (
                "SC11",
                geometry.width,
                geometry.thickness,
                (geometry.hole_diameter, float(geometry.crack_count)),
            ),
            (geometry.depth, geometry.half_length),
            ("Radial depth", "Bore half-length"),
        )
    if isinstance(geometry, _ShellThroughCrack):
        cylinder = (
            (geometry.length, geometry.poisson)
            if isinstance(geometry, CircumferentialCylinderThroughCrack)
            else None
        )
        return (
            (
                geometry._case,
                geometry.mean_radius,
                geometry.thickness,
                cylinder,
            ),
            (geometry.half_length, 0.0),
            ("Half-length",),
        )
    raise TypeError("expected a supported crack geometry")


@dataclass(frozen=True)
class NasgroCurve(PlotMixin):
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

    def plot(
        self,
        length: float | None = None,
        k_min: ArrayLike = 0.0,
        k_max: ArrayLike | None = None,
        fig: Figure | None = None,
        ax: Axes | None = None,
        *,
        scale: _CurveScale = "semilogy",
    ) -> tuple[Figure, Axes]:
        """Plot NASGRO growth rate against maximum stress intensity.

        By default, show the long-crack reference curve at stress ratio R = 0.

        Parameters
        ----------
        length : float, optional
            Fixed crack dimension in mm for the short-crack threshold
            correction. Omit to use the long-crack limit.
        k_min : array_like, optional
            Fixed minimum intensity in MPa √mm, by default zero. With explicit
            sample arrays, supply one value or one per sample.
        k_max : array_like, optional
            Omit to sample up to the curve's toughness. A scalar sets the
            upper bound of the sweep. An array supplies exact sample points,
            each strictly below toughness. To plot one point, pass a
            one-element array.
        fig, ax : matplotlib.figure.Figure and matplotlib.axes.Axes, optional
            Existing figure and axes for overlays.
        scale : {"loglog", "semilogx", "semilogy"}, optional
            Axis scales, by default "semilogy". Use "semilogx" to display
            zero growth on a linear rate axis.

        Returns
        -------
        tuple of Figure and Axes
            Figure and axes containing the sampled curve.
        """
        minimum = np.asarray(k_min, dtype=float)
        maxima = np.asarray(
            self.toughness if k_max is None else k_max, dtype=float
        )
        sweep = maxima.ndim == 0
        if sweep:
            if minimum.ndim != 0:
                raise ValueError("A sweep requires a scalar k_min.")
            lower, upper = minimum.item(), maxima.item()
            if not (
                np.isfinite(upper) and 0 <= lower < upper <= self.toughness
            ):
                raise ValueError(
                    "Require 0 <= k_min < sweep upper bound <= toughness."
                )
            samples = np.linspace(lower, upper, 201)
            maxima = np.unique(
                samples[(samples > lower) & (samples < self.toughness)]
            )
        if maxima.ndim != 1 or not maxima.size:
            raise ValueError("k_max must contain at least one intensity")
        minima = np.broadcast_to(minimum, maxima.shape)
        if length is None:
            rate_at = partial(nasgro_long_crack_rate, astuple(self))
            label = "NASGRO long-crack growth rate"
        else:
            rate_at = partial(self.get_growth_rate, length)
            label = "NASGRO growth rate"
        rates = [rate_at(lower, upper) for lower, upper in zip(minima, maxima)]
        fig, ax = make_axes(fig, ax)
        _plot_curve(
            ax,
            maxima,
            rates,
            scale,
            marker="o" if maxima.size == 1 else None,
            label=label,
            nonpositive="mask",
        )
        ax.set(
            xlabel="Maximum stress intensity (MPa √mm)",
            ylabel="Growth rate (mm/cycle)",
        )
        if ax.get_yscale() == "linear":
            ax.set_ylim(bottom=0)
        if sweep:
            left = lower if lower > 0 or scale == "semilogy" else None
            ax.set_xlim(left=left, right=upper)
            if upper == self.toughness:
                ax.axvline(
                    upper, color="0.4", linestyle="--", label="Toughness"
                )
        return fig, ax


@dataclass(frozen=True)
class GrowthPoint:
    """Accepted state at a cumulative cycle count in a zero-based load block."""

    cycles: float
    sizes: tuple[float, ...]
    k_min: tuple[float, ...]
    k_max: tuple[float, ...]
    block: int


@dataclass(frozen=True)
class GrowthResult(PlotMixin):
    """Accepted points and the reason propagation ended."""

    history: tuple[GrowthPoint, ...]
    termination: str
    dimension_names: tuple[str, ...] = field(
        default=(), compare=False, repr=False
    )

    @property
    def final_cycles(self) -> float:
        """Return the cycle count of the last accepted state, or zero."""
        return self.history[-1].cycles if self.history else 0.0

    def plot(
        self,
        fig: Figure | None = None,
        ax: Axes | None = None,
        *,
        label: str | None = None,
    ) -> tuple[Figure, Axes]:
        """Plot the accepted crack dimensions against cumulative cycles.

        Use `label` as a legend prefix when overlaying several simulations.
        """
        fig, ax = make_axes(fig, ax)
        ax.set(xlabel="Cycles", ylabel="Crack dimension (mm)")
        if not self.history:
            if not ax.has_data():
                ax.set_title("No accepted crack states")
            return fig, ax

        dimensions = self.dimension_names or (
            ("Half-length",)
            if len(self.history[0].sizes) == 1
            else ("Depth", "Half-length")
        )
        cycles = [point.cycles for point in self.history]
        for index, name in enumerate(dimensions):
            ax.plot(
                cycles,
                [point.sizes[index] for point in self.history],
                marker="o" if len(cycles) == 1 else None,
                label=f"{label} · {name}" if label else name,
            )
        ax.legend()
        return fig, ax


def simulate_crack_growth(
    geometry: (
        CenterThroughCrack
        | EdgeThroughCrack
        | SemiEllipticalSurfaceCrack
        | RoundBarSurfaceCrack
        | CircumferentialCylinderSurfaceCrack
        | RolledBoltThreadSurfaceCrack
        | HoleBoreSurfaceCrack
        | SphereThroughCrack
        | AxialCylinderThroughCrack
        | CircumferentialCylinderThroughCrack
    ),
    curve: NasgroCurve,
    loads: Sequence[LoadBlock],
    *,
    max_cycle_step: float,
    length_tolerance: float,
    cycle_tolerance: float,
    max_trials: int,
) -> GrowthResult:
    """Propagate a crack through the load blocks in the supplied order.

    Use MPa, mm and MPa sqrt(mm) throughout. Controls set the maximum step
    in cycles, absolute step error in mm, terminal bracket width in cycles and
    total number of trial steps. History contains accepted states. At a
    geometry or toughness limit, its last state is on the admissible side
    of the bracket.
    """
    section, initial, names = _plate_state(geometry)
    dimensions = len(names)
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
    return GrowthResult(history, termination, names)
