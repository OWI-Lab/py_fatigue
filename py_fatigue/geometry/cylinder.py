# Packages from non-standard libraries
from typing import Dict, Optional

import matplotlib
import numba as nb
import numpy as np
from pydantic import PositiveFloat, validator
from pydantic.dataclasses import dataclass

# Local imports
import py_fatigue.utils as pfu
from py_fatigue.geometry import AbstractCrackGeometry


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

    Parameters
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
        lambda self: "HOL_CYL_00"
        if self.crack_position == "internal"
        else "HOL_CYL_01"
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
        """Plot the crack front on a hollow cylinder.

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

        if self.crack_position == "internal":
            raise NotImplementedError(
                "Plotting of internal crack not implemented"
            )

        # case crack_position external
        return self._plot_external_crack(fig, ax, **kwargs)

    def _plot_external_crack(
        self,
        depth: Optional[PositiveFloat] = None,
        fig: Optional[matplotlib.figure.Figure] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        **kwargs: dict,
    ) -> tuple:
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
        axes.plot(x_o, y_o, c="k", lw=0.5, **kwargs)
        axes.plot(x_i, y_i, c="k", lw=0.5, **kwargs)
        axes.plot(x_c, y_c, c="r", lw=1.0, **kwargs)
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

    def __post_init__(self):
        if not all((self.diameter, self.height)) > 0.0:
            raise ValueError("Diameter and height must be positive.")

    @property
    def geometry_factor(self) -> np.ndarray:
        """Get the geometric factor. This should be a function of the crack
        size. The default value is an empty."""
        return np.empty(0)

    def __str__(self) -> str:
        return (
            super().__str__()[:-3]
            + f",\n  diameter={self.diameter}"
            + f",\n  height={self.height},\n)"
        )


@nb.njit(
    nb.float64(
        nb.float64, nb.types.DictType(nb.types.unicode_type, nb.float64)
    ),
    cache=True,
    fastmath=True,
)
def f_hol_cyl_01(
    crack_depth: float,
    crack_geometry: Dict[str, float],
):
    """Calculate the crack geometry factor for a hollow cylinder with
    an external crack.

    Parameters
    ----------
    crack_depth : float
        The crack depth.
    outer_diameter : float
        The outer diameter of the part.
    thickness : float
        The thickness of the part.
    height : float
        The height of the part.

    Returns
    -------
    float
        The crack size factor.
    """
    if crack_depth >= crack_geometry["thickness"]:
        return 0.0
    outer_diameter: float = crack_geometry["outer_diameter"]
    thickness: float = crack_geometry["thickness"]
    width_to_depth_ratio: float = crack_geometry["width_to_depth_ratio"]
    crack_width: float = crack_depth * width_to_depth_ratio
    outer_radius: float = outer_diameter / 2
    inner_radius: float = outer_radius - thickness
    # raise NotImplementedError
    # crack angle
    alpha: float = crack_width / outer_radius
    c_v: float = outer_radius * np.sin(alpha)
    a_h: float = crack_depth - outer_radius * (1 - np.cos(alpha))
    # radius of crack-front arc
    crack_radius: float = (c_v**2 + a_h**2) / (2 * a_h)
    # angle of crack-front arc
    beta: float = np.arcsin(c_v / crack_radius)
    # areas
    # * circular ring
    area_0: float = np.pi * (outer_radius**2 - inner_radius**2)
    # * external circular segment
    area_1: float = (alpha - np.sin(2 * alpha) / 2) * outer_radius**2
    # internal circular segment
    area_2: float = (beta - np.sin(2 * beta) / 2) * crack_radius**2
    # * net crack area
    area_net: float = area_0 - area_1 - area_2
    # centroids of the circular segments
    x_1: float = (
        -4
        * outer_radius
        * np.sin(alpha) ** 3
        / (3 * (2 * alpha - np.sin(2 * alpha)))
    )
    x_2: float = (
        4
        * crack_radius
        * np.sin(beta) ** 3
        / (3 * (2 * beta - np.sin(2 * beta)))
        - outer_radius * np.cos(alpha)
        - crack_radius * np.cos(beta)
    )
    # location of the net section centroid
    q_x: float = -(area_1 * x_1 + area_2 * x_2) / area_net
    # moments of inertia
    # * circular ring
    i_0: float = (
        np.pi / 4 * (outer_radius**4 - inner_radius**4) + area_0 * q_x**2
    )
    # * external circular segment
    i_1: float = (outer_radius**4) / 4 * (
        alpha
        - np.sin(4 * alpha) / 4
        - (16 * np.sin(alpha) ** 6)
        / (9 * (alpha - np.sin(alpha) * np.cos(alpha)))
    ) + area_1 * (q_x - x_1) ** 2
    # * internal circular segment
    i_2: float = (
        crack_radius**4
        / 4
        * (
            beta
            - np.sin(4 * beta) / 4
            - (16 * np.sin(beta) ** 6)
            / (9 * (beta - np.sin(beta) * np.cos(beta)))
        )
        + area_2 * (q_x - x_2) ** 2
    )
    # * net area moment of inertia
    i_net: float = i_0 - i_1 - i_2
    # net section stress center
    epsilon: float = 0.1
    # * loc a
    x_a: float = -(
        outer_radius * np.cos(alpha) + q_x
    ) + epsilon * outer_radius * (1 + np.cos(alpha))
    # # * loc b
    # x_b: float = (outer_radius - q_x) - epsilon * outer_radius * \
    # (1 + np.cos(alpha))
    # net section stress geometry factors
    # geometry factors
    # # * net section stress factor on sigma_0
    # g_0: float = area_0 / area_net - area_0 * q_x * x_a / i_net
    # * net section stress factor on sigma_1
    g_1: float = (
        -np.pi
        * (outer_radius**4 - inner_radius**4)
        * x_a
        / (4 * outer_radius * i_net)
    )
    return g_1
