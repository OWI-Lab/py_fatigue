# Packages from non-standard libraries
from typing import Dict, Optional

import matplotlib
import numpy as np
from pydantic import PositiveFloat, validator
from pydantic.dataclasses import dataclass

# Local imports
from .. import utils as pfu
from .._rust import hollow_cylinder_factor
from .generic import AbstractCrackGeometry


@dataclass(repr=False)
class HollowCylinder(AbstractCrackGeometry):
    """Class to initalise and store the crack geometry data on a
    hollow cylindrical section.

    The crack is assumed to maintain its width to depth ratio.

    >>> # Example drawing of a hollow cylinder with external crack
    >>> #        ┌────────────────────────────────────────┐
    >>> #        │              ▄▄▄▀▀▀▀▀▀▀▚▄▄▖            │
    >>> #        │          ▗▄▀▀             ▝▀▚▄         │
    >>> #        │        ▄▀▘                    ▀▖       │
    >>> #        │       ▞    crack front         ▝▀▖     │
    >>> #        │     ▗▀     │      ▄▄▄▖           ▝▖    │
    >>> #        │    ▗███▄▖<─┘  ▗▟▀▀   ▀▀▚▖         ▐    │
    >>> #        │    ▟█████▖   ▟▘         ▀▙         ▌   │
    >>> #        │2c─>███████   ▌       r_i ▐         ▐   │
    >>> #        │    ███████   ▌     ↑<───>▐         ▐   │
    >>> #        │    ▜█████▘   ▜▖    │    ▄▛         ▌   │
    >>> #        │   │▝███▀▘│    ▝▜▄▄ │ ▄▄▞▘         ▐    │
    >>> #        │   │ ▝▄   │        ▀│▀▘           ▗▘    │
    >>> #        │   │   ▚  │      r_o│           ▗▄▘     │
    >>> #        │   │    ▚▄│         │          ▗▘       │
    >>> #        │   │   a  ▀▚▄▖      ↓       ▄▞▀▘        │
    >>> #        │   │<────>│  ▝▀▀▚▄▄▄▄▄▄▄▄▀▀▀            │
    >>> #        └────────────────────────────────────────┘
    >>> #

    Attributes
    ----------
    initial_depth : float
        The initial depth of the crack.
    outer_diameter : float
        The outer diameter of the part.
    thickness : float
        The thickness of the part.
    height : float
        The height of the part.
    crack_position : float
        The position of the crack on the part.
    width_to_depth_ratio : float
        The ratio of the width to the depth of the crack.
    """

    outer_diameter: PositiveFloat
    thickness: PositiveFloat
    height: PositiveFloat
    crack_position: str
    width_to_depth_ratio: PositiveFloat = 2.0

    _id = property(
        lambda self: (
            "HOL_CYL_00" if self.crack_position == "internal" else "HOL_CYL_01"
        )
    )

    @validator("thickness")
    @classmethod
    def thickness_validator(
        cls, v: PositiveFloat, values: dict
    ) -> PositiveFloat:
        """Validate the thickness of the part."""
        if v >= values["outer_diameter"] / 2:
            e_msg = "thickness is greater than or equal to the outer radius"
            raise ValueError(e_msg)
        if v <= values["initial_depth"]:
            e_msg = "thickness is less than or equal to the initial depth"
            raise ValueError(e_msg)
        return v

    @validator("crack_position")
    @classmethod
    def crack_position_validator(cls, v: str) -> str:
        """Validate the crack position string."""
        if v not in ["internal", "external"]:
            raise ValueError(
                "crack position must be either 'internal' or 'external'"
            )
        return v

    @property
    def geometry_factor(self) -> np.ndarray:
        """Get the geometric factor. This should be a function of the crack
        size. The default value is 1.0."""
        return np.empty(0)

    def __str__(self) -> str:
        return (
            super().__str__()[:-3]
            + f",\n  outer_diameter={self.outer_diameter}"
            + f",\n  thickness={self.thickness}"
            + f",\n  height={self.height}"
            + f",\n  width_to_depth_ratio={self.width_to_depth_ratio}"
            + f",\n  crack_position={self.crack_position},\n)"
        )

    def plot(
        self,
        fig: Optional[matplotlib.figure.Figure] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        **kwargs: dict,
    ) -> tuple:  # pragma: no cover
        """Draw the hollow cylinder cross-section and crack.

        For an internal crack, the line indicates depth from the inner wall.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            The figure to plot on.
        ax : matplotlib.axes.Axes, optional
            The axis to plot on.
        **kwargs : dict
            Keyword arguments to pass to the matplotlib plot function.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure that was plotted on.
        ax : matplotlib.axes.Axes
        """

        if self.crack_position == "external":
            return self._plot_external_crack(fig=fig, ax=ax, **kwargs)

        fig, axes = pfu.make_axes(fig, ax)
        outer_radius = self.outer_diameter / 2
        inner_radius = outer_radius - self.thickness
        radii = (outer_radius, inner_radius)
        angles = np.linspace(0, 2 * np.pi, 101)
        for radius in radii:
            axes.plot(radius * np.cos(angles), radius * np.sin(angles), c="k")
        axes.plot(
            [inner_radius, inner_radius + self.initial_depth], [0, 0],
            c="firebrick", label="Internal crack depth", **kwargs,
        )
        axes.set(aspect="equal", title="Hollow cylinder cross-section")
        axes.legend()
        return fig, axes

    def _plot_external_crack(
        self,
        depth: Optional[PositiveFloat] = None,
        fig: Optional[matplotlib.figure.Figure] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        **kwargs: dict,
    ) -> tuple:  # pragma: no cover
        """Plot the crack front when the crack is located on the external
        surface of the hollow cylinder.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            The figure to plot on.
        ax : matplotlib.axes.Axes, optional
            The axis to plot on.
        **kwargs : dict
            Keyword arguments to pass to the matplotlib plot function.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure that was plotted on.
        ax : matplotlib.axes.Axes
        """
        if depth is None:
            depth = self.initial_depth
        angles = np.linspace(0, 2 * np.pi, 100)
        r_o = self.outer_diameter / 2
        r_i = r_o - self.thickness

        x_o = r_o * np.cos(angles)
        y_o = r_o * np.sin(angles)
        x_i = r_i * np.cos(angles)
        y_i = r_i * np.sin(angles)

        crack_width = self.width_to_depth_ratio * depth
        c_angle = crack_width / r_o  # crack angle
        c_v = r_o * np.sin(c_angle)
        a_h = depth - r_o * (1 - np.cos(c_angle))
        r_e = (c_v**2 + a_h**2) / (2 * a_h)  # radius of crack-front arc
        # angle of crack-front arc
        beta = np.arcsin(c_v / r_e)
        theta = np.linspace(-beta, beta, 100)  # θ (crack-front angle)
        phi = np.linspace(np.pi - c_angle, np.pi + c_angle, 100)

        # crack-front arc
        x_c = r_e * np.cos(theta) - (r_o + r_e - depth)
        y_c = r_e * np.sin(theta)
        x_c = np.append(x_c, r_o * np.cos(phi))
        y_c = np.append(y_c, r_o * np.sin(phi))

        # plot
        fig, axes = pfu.make_axes(fig=fig, ax=ax)
        axes.set_aspect("equal")
        axes.plot(x_o, y_o, c="k", lw=0.5, **kwargs)  # type: ignore
        axes.plot(x_i, y_i, c="k", lw=0.5, **kwargs)  # type: ignore
        axes.plot(x_c, y_c, c="r", lw=1.0, **kwargs)  # type: ignore
        return fig, axes


@dataclass(repr=False)
class Cylinder(AbstractCrackGeometry):
    """Class to initalise and store the crack geometry data on a
    cylindrical section.

    Parameters
    ----------
    diameter : float
        The outer diameter of the part.
    height : float
        The height of the part.
    """

    diameter: PositiveFloat
    height: PositiveFloat
    _id = property(lambda _: "FUL_CYL_00")

    def plot(
        self,
        fig: Optional[matplotlib.figure.Figure] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Draw the full cylinder's diameter and height in a side view."""
        fig, axes = pfu.make_axes(fig, ax)
        radius = self.diameter / 2
        axes.plot(
            [-radius, radius, radius, -radius, -radius],
            [0, 0, self.height, self.height, 0],
        )
        axes.set(
            xlabel="Position across diameter",
            ylabel="Height",
            title="Solid cylinder side view",
            aspect="equal",
        )
        return fig, axes

    def __post_init__(self):  # pragma: no cover
        if not all((self.diameter, self.height)) > 0.0:
            raise ValueError("Diameter and height must be positive.")

    @property
    def geometry_factor(self) -> np.ndarray:  # pragma: no cover
        """Get the geometric factor. This should be a function of the crack
        size. The default value is an empty."""
        return np.empty(0)

    def __str__(self) -> str:
        return (
            super().__str__()[:-3]
            + f",\n  diameter={self.diameter}"
            + f",\n  height={self.height},\n)"
        )


def cylinder_dimensions(
    geometry: Dict[str, float],
) -> tuple[float, float, float]:
    """Extract the dimensions used by the hollow-cylinder solution."""
    return (
        geometry["outer_diameter"],
        geometry["thickness"],
        geometry["width_to_depth_ratio"],
    )


def f_hol_cyl_01(crack_depth: float, crack_geometry: Dict[str, float]) -> float:
    """Calculate the factor for an external crack in a hollow cylinder."""
    if crack_depth >= crack_geometry["thickness"]:
        return 0.0
    return hollow_cylinder_factor(
        crack_depth, cylinder_dimensions(crack_geometry)
    )
