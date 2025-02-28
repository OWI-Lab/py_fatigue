# -*- coding: utf-8 -*-
"""The module :mod:`py_fatigue.damage.stress_life` contains all the
damage models related to the stress-life approach.
"""

# Packages from standard library

from __future__ import annotations
from types import SimpleNamespace
from typing import Any, Callable, Optional, Tuple, Union
import logging
import warnings

# Packages from external libraries
import matplotlib
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd

# py-fatigue imports
from ..cycle_count.cycle_count import CycleCount
from ..material.sn_curve import (
    _check_param_couple_types,
    __jit_sn_curve_residuals,
    SNCurve,
)
from ..styling import py_fatigue_formatwarning
from ..utils import make_axes, numba_bisect, _plot_damage_accumulation

try:
    # delete the accessor to avoid warning
    del pd.DataFrame.miner  # type: ignore
except AttributeError:
    pass


warnings.formatwarning = py_fatigue_formatwarning


@pd.api.extensions.register_dataframe_accessor("miner")
class PalmgrenMiner:
    """Accessor for the Palmgren-Miner damage model."""

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj
        self.sn_curve = None
        self.des_data = {}
        self.dem_data = {}

    @staticmethod
    def _validate(obj):
        """Validate the input DataFrame. Raise an error if the input
        DataFrame does not contain the right columns.
        """
        if {
            "cycles_to_failure",
        }.issubset(obj.columns):
            e_msg = "'cycles_to_failure' already calculated"
            raise AttributeError(e_msg)

        if not {"count_cycle", "mean_stress", "stress_range"}.issubset(
            obj.columns
        ):
            e_msg = (
                "Must have 'count_cycle', 'mean_stress' and 'stress_range'."
            )
            raise AttributeError(e_msg)

    def damage(self, sn_curve: SNCurve):
        """Calculate the damage of the samples.

        Parameters
        ----------
        sn_curve : SNCurve
            The SNCurve object.

        Returns
        -------
        DataFrame
            The DataFrame with the damage.
        """
        if self._obj._metadata["unit"] != sn_curve.unit:
            e_msg = (
                f"Units of cycle_count ({self._obj._metadata['unit']}) and "
                f"sn_curve ({sn_curve.unit}) do not match."
            )
            raise ValueError(e_msg)
        self._obj.sn_curve = sn_curve
        self.sn_curve = sn_curve
        self._obj["cycles_to_failure"] = sn_curve.get_cycles(
            self._obj.stress_range
        )
        self._obj["pm_damage"] = np.divide(
            self._obj.count_cycle, self._obj.cycles_to_failure
        )
        return self._obj

    def des(self, slope: float, equivalent_cycles: float = 1e7) -> float:
        """Calculate the damage equivalent stress range (DES).

        Parameters
        ----------
        slope : float
            The SN curve slope.
        equivalent_cycles : float, optional
            The equivalent number of cycles, by default 1e7

        Returns
        -------
        float
            The DES.

        See Also
        --------
        calc_des
        """
        self.des_data["slope"] = slope
        self.des_data["equivalent_cycles"] = equivalent_cycles

        return calc_des(
            self._obj.stress_range,
            self._obj.count_cycle,
            slope,
            equivalent_cycles,
        )

    def dem(
        self,
        outer_radius: float,
        inner_radius: float,
        slope: float,
        equivalent_cycles: float = 1e7,
    ) -> float:
        """Calculate the damage equivalent stress range (DES).

        Parameters
        ----------
        slope : float
            The SN curve slope.
        equivalent_cycles : float, optional
            The equivalent number of cycles, by default 1e7

        Returns
        -------
        float
            The DES.

        See Also
        --------
        calc_des
        """
        self.dem_data["outer_radius"] = outer_radius
        self.dem_data["inner_radius"] = inner_radius
        self.dem_data["slope"] = slope
        self.dem_data["equivalent_cycles"] = equivalent_cycles

        return calc_dem(
            outer_radius,
            inner_radius,
            self._obj.stress_range,
            self._obj.count_cycle,
            slope,
            equivalent_cycles,
        )

    def plot_histogram(
        self,
        fig: Optional[matplotlib.figure.Figure] = None,
        ax: Optional[matplotlib.collections.PathCollection] = None,
        dens_func: Optional[Callable] = None,
        **kwargs,
    ) -> Tuple[
        matplotlib.figure.Figure, matplotlib.collections.PathCollection
    ]:
        """Plot the damage.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            The figure to plot on, by default None
        ax : matplotlib.collections.PathCollection, optional
            The axes to plot on, by default None
        kwargs
            Additional keyword arguments for the plot.

        Returns
        -------
        matplotlib.figure.Figure, matplotlib.collections.PathCollection
            The figure and axes.
        """
        fig, ax = make_axes(fig, ax)
        im = ax.scatter(
            self._obj.count_cycle,
            self._obj.stress_range,
            # edgecolors="#CCC",
            # linewidths=0.5,
            c=(
                dens_func(self._obj.pm_damage)
                if dens_func is not None
                else self._obj.pm_damage
            ),
            **kwargs,
        )
        cbar_label = (
            "PM damage"
            if dens_func is None
            else "PM damage(" + str(dens_func) + ")"
        )
        ax.set_xlabel("# of cycles")
        ax.set_ylabel("Stress range, MPa")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(cbar_label, rotation=270)
        cbar.ax.get_yaxis().labelpad = 10
        return fig, ax


try:
    # delete the accessor to avoid warning
    del pd.DataFrame.gassner  # type: ignore
except AttributeError:
    pass


@pd.api.extensions.register_dataframe_accessor("gassner")
class Gassner:
    """Accessor for the Gassner shift factor for variable amplitude."""

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj
        self.sn_curve = None

    @staticmethod
    def _validate(obj):
        """Validate the input DataFrame. Raise an error if the input
        DataFrame does not contain the right columns.
        """
        if {
            "shift_factor",
        }.issubset(obj.columns):
            e_msg = "'shift_factor' already calculated"
            raise AttributeError(e_msg)

        if not {"count_cycle", "mean_stress", "stress_range"}.issubset(
            obj.columns
        ):
            e_msg = (
                "Must have 'count_cycle', 'mean_stress' and 'stress_range'."
            )
            raise AttributeError(e_msg)

    def g(self, sn_curve: SNCurve):
        """Calculate the shift factor of the samples.

        The shift factor is a spectrum-dependent factor that allows
        to shift a Wohler curve and avoids having to calculate the
        damage using the :class:`~PalmgrenMiner` ruleevery time.

        Once the shift factor has been calculated, it can be applied
        to the Wohler SN curve. Then, accssing the curve through the
        maximum amplitude value of the spectrum will return the
        expected life for said spectrum.

        The shift factor for each cycle  in the spectrum is:

        .. math::

            G_j = \\frac{n_j}{N_S} \\,
            \\left({\\frac{\\sigma_{{max}_j} \\, \\sigma_{{alt}_j}}
            {\\max{\\sigma_{alt}} ^ {2}}}\\right)^{2}

        The shift factor then is:

        .. math::

            G = \\sum_{j=1}^{N_B} G_j

        Parameters
        ----------
        sn_curve : SNCurve
            The SNCurve object.

        Returns
        -------
        DataFrame
            The DataFrame with the shift factor per sample.
        """
        assert sn_curve.linear, "The SN curve must be linear"
        self._obj.sn_curve = sn_curve
        self.sn_curve = sn_curve
        stress_amp = self._obj.stress_range / 2
        max_amp = max(stress_amp)
        nu = np.divide(self._obj.count_cycle, sum(self._obj.count_cycle))
        self._obj["shift_factor"] = nu * np.power(
            np.divide(stress_amp, max_amp), self.sn_curve.slope
        )
        return self._obj


def calc_pm(
    stress_range: Union[float, pd.Series, np.ndarray],
    count_cycle: Union[float, pd.Series, np.ndarray],
    sn_curve: SNCurve,
) -> np.ndarray:
    """Calculates the damage using the Palmgren-Miner rule, i.e. the
    linear damage accumulation rule:

    .. math::

        D = \\sum_{j=1}^{N_{\\text{blocks}}} \\frac{n_j}{N_j} \\leq 1

    Parameters
    ----------
     stress_range : Union[pd.Series, np.ndarray]
        The stress range
    count_cycle : Union[pd.Series, np.ndarray]
        The number of cycles
    sn_curve : SNCurve
        The SNCurve object.

    Returns
    -------
    Union[float, np.ndarray]
        The damage according to the Palmgren-Miner rule.
    """
    stress_range, count_cycle = _check_param_couple_types(
        stress_range, count_cycle
    )
    return count_cycle / sn_curve.get_cycles(stress_range)


def calc_des(
    stress_range: Union[float, pd.Series, np.ndarray],
    count_cycle: Union[float, pd.Series, np.ndarray],
    slope: float,
    equivalent_cycles: float = 1e7,
) -> float:
    """Calculate the damage equivalent stress range (DES).

    Parameters
    ----------
    stress_range : Union[float, pd.Series, np.ndarray]
        The stress range
    count_cycle : Union[float, pd.Series, np.ndarray]
        The number of cycles
    slope : float
        The SN curve slope
    equivalent_cycles : float, optional
        The equivalent number of cycles, by default 1e7

    Returns
    -------
    float
        The DES.
    """
    stress_range, count_cycle = _check_param_couple_types(
        stress_range, count_cycle
    )
    des = np.power(
        np.sum(
            np.multiply(
                np.power(stress_range, float(slope)),
                count_cycle,
            )
            / equivalent_cycles
        ),
        1 / slope,
    )
    return des


def calc_dem(
    outer_radius: float,
    inner_radius: float,
    stress_range: Union[float, pd.Series, np.ndarray],
    count_cycle: Union[float, pd.Series, np.ndarray],
    slope: float,
    equivalent_cycles: float = 1e7,
) -> float:
    """Calculate the damage equivalent moment (DEM) in Nm.

    .. math::

        DEM = DES * I_c / r_o = DES * S

    with :math:`I_c` the area moment of inertia and :math:`r_i` the
    inner radius, or :math:`S` the Section modulus. The Damage
    Equivalent Stress Range (DES) is calculated using
    :func:`calc_des`

    Parameters
    ----------
    outer_radius : float
        The outer radius to calculate the DEM. Must be provided in m
    inner_radius : float
        The inner radius to calculate the DEM. Must be provided in m
    stress_range : Union[float, pd.Series, np.ndarray]
        The stress range
    count_cycle : Union[float, pd.Series, np.ndarray]
        The number of cycles
    slope : float
        The SN curve slope
    equivalent_cycles : float, optional
        The equivalent number of cycles, by default 1e7

    Raises
    ------
    ValueError
        If the inner radius is larger than the outer radius.

    Returns
    -------
    float
        The DEM in Nm.
    """
    if outer_radius <= inner_radius:
        raise ValueError("outer_radius must be greater than inner_radius")
    stress_range, count_cycle = _check_param_couple_types(
        stress_range, count_cycle
    )
    area_inertia = np.pi / 4 * (outer_radius**4 - inner_radius**4)
    des = calc_des(stress_range, count_cycle, slope, equivalent_cycles)

    return area_inertia * des * 1e6 / outer_radius


def get_pm(cycle_count: CycleCount, sn_curve: SNCurve) -> np.ndarray:
    """Calculates the damage using the Palmgren-Miner rule.

    Parameters
    ----------
    cycle_count : CycleCount
        The CycleCount object.
    sn_curve : SNCurve
        The SNCurve object.

    Returns
    -------
    np.ndarray
        The damage according to the Palmgren-Miner rule.
    """
    if cycle_count.unit != sn_curve.unit:
        raise ValueError(
            f"Units of cycle_count ({cycle_count.unit}) and "
            f"sn_curve ({sn_curve.unit}) do not match."
        )
    return calc_pm(cycle_count.stress_range, cycle_count.count_cycle, sn_curve)


def get_des(
    cycle_count: CycleCount, slope: float, equivalent_cycles: float = 1e7
) -> float:
    """Calculate the damage equivalent stress range (DES).

    Parameters
    ----------
    cycle_count : CycleCount
        The CycleCount object.
    slope : float
        The SN curve slope
    equivalent_cycles : float, optional
        The equivalent number of cycles, by default 1e7

    Returns
    -------
    float
        The DES.
    """
    return calc_des(
        cycle_count.stress_range,
        cycle_count.count_cycle,
        slope,
        equivalent_cycles,
    )


def get_dem(
    outer_radius: float,
    inner_radius: float,
    cycle_count: CycleCount,
    slope: float,
    equivalent_cycles: float = 1e7,
) -> float:
    """Calculate the damage equivalent moment (DEM) in Nm.

    .. math::

        DEM = DES * I_c / r_o = DES * S

    with :math:`I_c` the area moment of inertia and :math:`r_i` the
    inner radius, or :math:`S` the Section modulus. The Damage
    Equivalent Stress Range (DES) is calculated using
    :func:`calc_des`

    Parameters
    ----------
    outer_radius : float
        The outer radius to calculate the DEM. Must be provided in m
    inner_radius : float
        The inner radius to calculate the DEM. Must be provided in m
    stress_range : Union[pd.Series, np.ndarray]
        The stress range
    count_cycle : Union[pd.Series, np.ndarray]
        The number of cycles
    slope : float
        The SN curve slope
    equivalent_cycles : float, optional
        The equivalent number of cycles, by default 1e7

    Returns
    -------
    float
        The DEM in Nm.
    """
    return calc_dem(
        outer_radius,
        inner_radius,
        cycle_count.stress_range,
        cycle_count.count_cycle,
        slope,
        equivalent_cycles,
    )


def calc_manson_halford_exponents(
    cycles_to_failure: np.ndarray, exponent: float = 0.4
) -> np.ndarray:
    """Calculate the Mannson-Halford exponents :math:`e_{j, j+1}`.

    .. math::

        e{j, j+1} = \\left(\\frac{N_{j}}{N_{j+1}}\\right)^{exponent}

    Parameters
    ----------
    cycles_to_failure : np.ndarray
        The number of cycles to failure.
    exponent : float, optional
        The exponent, by default 0.4

    Returns
    -------
    np.ndarray
        The Mannson-Halford exponents.
    """
    return (
        np.hstack([cycles_to_failure[:-1] / cycles_to_failure[1:], [1]])
        ** exponent
    )


def calc_pavlou_exponents(
    stress_range: np.ndarray,
    ultimate_stress: float = 900,
    exponent: float = -0.75,
    use_dca: bool = False,
) -> np.ndarray:
    """Calculate the Pavlou exponents :math:`q(\\sigma_j)=e_{j, j+1}`.

    .. math::

        e_{j, j+1} = \\left(\\frac{\\Delta \\sigma_j / 2}{\\sigma_U}\\right)
        ^{exponent}

    where :math:`\\Delta \\sigma_j` is the stress amplitude, :math:`\\sigma_U`
    is the ultimate stress, :math:`\\Delta \\sigma` is the stress range and
    :math:`exponent` is the exponent.


    Parameters
    ----------
    stress_range : np.ndarray
        The stress range.
    ultimate_stress : float, optional
        The ultimate stress, by default 900
    exponent : float, optional
        The exponent, by default -0.75

    Returns
    -------
    np.ndarray
        The Pavlou exponents.
    """
    q_exp = (stress_range / 2 / ultimate_stress) ** exponent
    if use_dca:
        return q_exp
    return np.hstack([q_exp[1:] / q_exp[:-1], [1]])


def calc_si_jian_exponents(
    stress_range: np.ndarray,
) -> np.ndarray:
    """Calculate the Si-Jian et al. exponents as

    .. math::

        e_{j, j+1} = \\sigma_{j+1} / \\sigma_{j}

    where :math:`\\sigma_{j+1}` is the stress amplitude for the
    :math:`j`-th cycle.

    Parameters
    ----------
    stress_range : np.ndarray
        The stress range.

    Returns
    -------
    np.ndarray
        The Si-Jian et al. exponents.
    """
    return np.hstack([stress_range[1:] / stress_range[:-1], [1]])


def calc_theil_weights(
    stress_range: np.ndarray,
    sn_curve: SNCurve,
) -> np.ndarray:
    """Calculate the Theil weights for the Damage Curve Approach
    (DCA).

    .. math::

        w_j = \\frac{\\Delta\\sigma_j}{N_{f,j}}

    Parameters
    ----------
    stress_range : np.ndarray
        The stress range.
    sn_curve : SNCurve
        The py-fatigue SN Curve

    Returns
    -------
    np.ndarray
        Theil's model weights
    """
    return np.divide(stress_range, sn_curve.get_cycles(stress_range))


def _calc_damage_exponents(
    damage_rule: str,
    stress_range: np.ndarray,
    sn_curve: Optional[SNCurve] = None,
    **kwargs: dict,
):
    """Calculate damage exponents based on the specified damage rule.
    Parameters
    ----------
    damage_rule : str
        The name of the damage rule to use. Supported rules are:
        'pavlou', 'manson', 'leve', and 'si jian'.
    stress_range : array_like
        An array-like object containing the stress ranges.
    sn_curve : SNCurve, optional
        An object representing the S-N curve.  It must have a `get_cycles`
        method that accepts a stress range and returns the number of cycles
        to failure. This is only used for the 'manson' damage rule.
    **kwargs : dict, optional
        Additional keyword arguments required by specific damage rules.
        - For 'pavlou':
            - 'base_exponent' (float): The base exponent for the Pavlou
              damage rule. If not provided, a default value of -0.75 is
              used.
            - 'ultimate_stress' (float): The ultimate stress for the Pavlou
              damage rule. If not provided, a default value of 900 MPa is
              used.
        - For 'manson':
            - 'base_exponent' (float): The base exponent for the
              Manson-Halford damage rule. If not provided, a default value
              of 0.4 is used.
        - For 'leve':
            - 'base_exponent' (float): The base exponent for the Leve
              damage rule. If not provided, a default value of 2 is used.
    Returns
    -------
    numpy.ndarray
        An array of damage exponents calculated based on the specified
        damage rule.
    Raises
    ------
    ValueError
        If an unknown damage rule is specified.
    UserWarning
        If required keyword arguments are missing for a specific damage
        rule, a warning is issued and a default value is used.
        The warnings are formatted using `py_fatigue_formatwarning`.
    Notes
    -----
    The 'pavlou' damage rule requires 'base_exponent' and 'ultimate_stress'
    to be specified in kwargs. If not specified, default values are used
    and a warning is issued.
    The 'manson' damage rule requires 'base_exponent' to be specified in
    kwargs. If not specified, a default value is used and a warning is
    issued. The S-N curve object must have a `get_cycles` method.
    The 'leve' damage rule requires 'base_exponent' to be specified in
    kwargs. If not specified, a default value is used and a warning is
    issued.
    The 'si jian' damage rule does not require any additional keyword
    arguments.
    """
    ns = SimpleNamespace(**kwargs)
    if damage_rule.lower() == "pavlou":
        if "base_exponent" not in kwargs:
            w_msg = [
                "Pavlou damage rule requires 'base_exponent' ",
                "to be assigned.\nUsing preset value of -0.75.",
            ]
            warnings.warn("".join(w_msg), UserWarning)
            ns.base_exponent = -0.75
        if "ultimate_stress" not in kwargs:
            w_msg = [
                "Pavlou damage rule requires 'ultimate_stress' ",
                "to be assigned.\nUsing preset value of 900 MPa.",
            ]
            warnings.warn("".join(w_msg), UserWarning)
            ns.ultimate_stress = 900
        if "use_dca" not in kwargs:
            w_msg = [
                "Pavlou damage rule requires 'use_dca' (Damage Curve Approach)"
                "to be assigned.\nUsing preset value of true.",
            ]
            warnings.warn(" ".join(w_msg), UserWarning)
            ns.use_dca = False
        return calc_pavlou_exponents(
            stress_range, ns.ultimate_stress, ns.base_exponent, ns.use_dca
        )
    if "manson" in damage_rule.lower():
        if "base_exponent" not in kwargs:
            w_msg = [
                "Manson & Halford damage rule requires ",
                "'base_exponent' to be assigned.\n",
                "Using preset value of 0.4.",
            ]
            warnings.warn("".join(w_msg), UserWarning)
            ns.base_exponent = 0.4
            if not sn_curve:
                raise ValueError("sn_curve must be provided for 'manson'")
        return calc_manson_halford_exponents(
            sn_curve.get_cycles(stress_range), ns.base_exponent  # type: ignore
        )
    if damage_rule.lower() == "leve":
        if "base_exponent" not in kwargs:
            w_msg = [
                "Leve damage rule requires ",
                "'base_exponent' to be assigned.\n",
                "Using preset value of 2.",
            ]
            warnings.warn("".join(w_msg), UserWarning)
            ns.base_exponent = 2
        return ns.base_exponent * np.ones(len(stress_range))
    if "si jian" in damage_rule.lower():
        return calc_si_jian_exponents(stress_range)

    raise ValueError(f"Unknown damage rule: {damage_rule}")


def calc_nonlinear_damage(
    damage_rule: str,
    stress_range: np.ndarray,
    count_cycle: np.ndarray,
    sn_curve: SNCurve,
    **kwargs,
) -> float:
    """Calculate the fatigue damage using a nonlinear damage rule
    among the allowed ones:

    - 'Pavlou': Pavlou damage rule
    - 'Manson-Halford': Mannson-Halford damage rule
    - 'Si-Jian': Si-Jian et al damage rule
    - 'Leve': Leve damage rule

    .. warning::
        This function is not suited for variable amplitude loading.
        For variable amplitude loading, use
        :func:`calc_nonlinear_damage_with_dca`, as the fluctiations of the
        nonlinear damage exponent are controlled by the Damage Curve.

    The generic form of a nonlinear damage rule is:

    .. math::

        D = \\left(
            \\left( \\dots
                \\left(
                    \\left(
                        \\left(\\frac{n_1}{N_1}\\right)^{e_{1, 2}} +
                        \\frac{n_2}{N_2}
                    \\right)^{e_{2, 3}} +
                    \\frac{n_3}{N_3}
                \\right)^{e_{3, 4}} + \\dots + \\frac{n_{M-1}}{N_{M-1}}
            \\right)^{e_{M-1, M}} + \\dots + \\frac{n_M}{N_M}
        \\right)^{e_M}

    where :math:`n_j` is the number of cycles in the fatigue histogram
    at the :math:`j`-th cycle, :math:`N_j` is the number of cycles to
    failure at the :math:`j`-th cycle, :math:`e_{j, j+1}` is the exponent
    for the :math:`j`-th and :math:`j+1`-th cycles, :math:`M` is the
    number of load blocks in the fatigue spectrum.

    The formula is conveniently rewritten as pseudocode:

    .. code-block:: python
        :caption: pseudocode for the nonlinear damage rule

        # retrieve N_j using the fatigue histogram and SN curve
        # retrieve the exponents e_{j, j+1}
        #  calculate the damage
        D = 0
        for j in range(1, M+1):
            D = (D + n_j / N_j) ^ e_{j, j+1}

    Parameters
    ----------
    damage_rule : str
        The damage rule to use. Must be one of the following:
        'Pavlou', 'Manson-Halford', 'Si-Jian', 'Leve'.
    stress_range : np.ndarray
        The stress range.
    count_cycle : np.ndarray
        The number of cycles.
    sn_curve : SNCurve
        The SN curve.
    kwargs : dict
        The keyword arguments for the damage rule. The following
        keyword arguments are allowed:

        - 'base_exponent': The exponent for the damage rule.
        - 'ultimate_stress': The ultimate stress.

    Returns
    -------
    np.ndarray
        The cumulated damage.
    """
    damage_per_cycle = calc_pm(
        stress_range,
        count_cycle,
        sn_curve,
    )

    kwargs["use_dca"] = False
    damage_exp = _calc_damage_exponents(
        damage_rule, stress_range, sn_curve, **kwargs
    )

    total_damage = 0
    damage_array = np.empty(len(damage_per_cycle))
    for i, (d_i, e_i) in enumerate(zip(damage_per_cycle, damage_exp)):
        # total_damage = (total_damage + d_i) ** e_i - total_damage
        total_damage = (total_damage + d_i) ** e_i
        damage_array[i] = total_damage

    return damage_array  # type: ignore


def calc_nonlinear_damage_with_dca(
    damage_rule: str,
    stress_range: np.ndarray,
    count_cycle: np.ndarray,
    sn_curve: SNCurve,
    damage_bands: np.ndarray = np.array(
        [0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1]
    ),
    limit_damage: Union[float, int] = 1,
    logger: logging.Logger | None = None,
    plot: bool = False,
    **kwargs,
) -> tuple[
    np.ndarray,
    np.ndarray,
    matplotlib.figure.Figure | None,
    matplotlib.axes.Axes | None,
]:
    """Calculate the fatigue damage using a nonlinear damage rule based on the
    Damage Curve Approach (DCA) among the allowed ones:

    - 'Pavlou': Pavlou damage rule
    - 'Theil': Theil damage rule

    The DCA discreties the damage curve in multiple Damage bands
    :math:`\\Delta D_j` and calculates the damage for each band, depending
    on the damage value at the previous cycle. The damage is calculated
    as:

    .. math::

        D = \\sum_{j=1}^{n_b} \\Delta D_j

    where :math:`n_b` is the number of damage bands. In each damage band,
    the damage follows a weighted Palmgren-Miner sum, i.e.:

    .. math::

        \\Delta D_j = \\sum_{i=1}^{n_j} w_{i, j} \\frac{n_i}{N_i}

    where :math:`n_j` is the number of cycles in the fatigue histogram
    at the :math:`j`-th cycle, :math:`N_j` is the number of cycles to
    failure at the :math:`j`-th cycle, :math:`w_{i, j}` is the weight
    for the :math:`i`-th cycle in the :math:`j`-th damage band.

    Using the generic nonlinear damage accumulation formula:

    .. math::

        D(\\sigma) = \\left(\\frac{n}{N(\\sigma)}\\right)^{e(\\sigma)}

    and substituting it inside the weighted Palmgren-Miner sum, the
    equation of the weights can be extracted as:

    .. math::

        w_{i, j} = \\frac{D_j - D_{j-1}}{
            D_{j}^{1/e_{i, i}} - D_{j-1}^{1/e_{i, i}}
        }

    In the case of Pavlou's model, the exponents are:

    .. math::

        e_{j, j} = \\left(\\frac{\\Delta \\sigma_j / 2}
                                {\\sigma_U}\\right)^{exponent}

    While for Theil's method, the weights can be directly
    approximated as:

    .. math::

        w_j = 1 + \\frac{\\Delta\\sigma_j}{N_{f,j}}

    .. warning::
        If you want to use Theil's method in a production
        environment, it is recommended to use the
        :func:`calc_theil_cycles_to_failure` function, as it adheres
        to the original SN Cuve-based method.

    The formula is conveniently rewritten as pseudocode:

    .. code-block:: python
        :caption: pseudocode for the nonlinear damage rule

        # retrieve n_i, N_i using the fatigue histogram and SN curve
        # retrieve the exponents e_{j, j+1}
        # define the damage bands DB
        # calculate the damage
        D = 0
        for j in range(1, M+1):
            DB_j = np.digitize(D, DB, right=False)
            for i in range(1, n_j+1):
                if damage_rule == 'Pavlou':
                    w_ij = (DB[DB_j] - DB[DB_j - 1]) / \
                           ((DB[DB_j] ** (1 / e_i)) -
                            (DB[DB_j - 1] ** (1 / e_i)))
                elif damage_rule == 'Theil':
                    w_ij = stress_range_i / N_i + 1
                D += w_ij * n_i / N_i

    Parameters
    ----------
    damage_rule : str
        The damage rule to use. Must be one of the following:
        'Pavlou', 'Theil'.
    stress_range : np.ndarray
        The stress range.
    count_cycle : np.ndarray
        The number of cycles.
    sn_curve : SNCurve
        The SN curve.
    damage_bands : np.ndarray
        Damage bands for the Pavlou damage calculation, by default
        [0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1]
    limit_damage : float, optional
        Limit damage value, by default 1
    logger: logging.Logger, optional
        If a logger is defined, attach output to it
    plot: bool, by default False
        Whether to plot or not the cumulated damage
    kwargs : dict
        The keyword arguments for the damage rule. The following
        keyword arguments are allowed:

        - 'base_exponent': The exponent for the damage rule.
        - 'ultimate_stress': The ultimate stress.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, mpl.Figure | None, mpl.Axes | None]
        The cycle-to-cycle damage and the cumulated damage.
    """

    allowed_damage_rules = ["pavlou", "theil"]
    if damage_rule.lower() not in allowed_damage_rules:
        raise ValueError(
            f"Unknown damage rule: {damage_rule}. "
            f"Allowed damage rules are: {allowed_damage_rules}"
        )
    stress_range = np.asarray(stress_range, dtype=np.float64)
    count_cycle = np.asarray(count_cycle, dtype=np.float64)
    damage_bands = np.asarray(damage_bands, dtype=np.float64)
    pm_damage = calc_pm(
        stress_range,
        count_cycle,
        sn_curve,
    )

    if damage_rule.lower() == "theil":
        weights = calc_theil_weights(stress_range, sn_curve)
        nl_damage = (1 + weights) * pm_damage
        if not plot:
            return nl_damage, np.cumsum(nl_damage), None, None
        fig, ax = _plot_damage_accumulation(
            np.cumsum(nl_damage), np.cumsum(pm_damage), limit_damage
        )
        return nl_damage, np.cumsum(nl_damage), fig, ax

    kwargs["use_dca"] = True
    dmg_exp = _calc_damage_exponents(
        damage_rule, stress_range, sn_curve, **kwargs
    )

    nl_damage = pm_damage.copy()
    cumsum_nl_dmg = np.zeros_like(nl_damage)
    cur_dmg_band = 0
    for i in range(len(pm_damage)):  # pylint: disable=C0200
        prev_dmg_band = cur_dmg_band if i >= 1 else 0
        # fmt: off
        cur_dmg_band = np.digitize(cumsum_nl_dmg[i - 1], damage_bands,
                                   right=False) if i >= 1 else 0
        # cur_dmg_band = min(cur_dmg_band, len(damage_bands) - 1)
        if cur_dmg_band >= len(damage_bands) - 1:
            cur_dmg_band = len(damage_bands) - 1
        if (
            cur_dmg_band != prev_dmg_band
            and len(damage_bands) < 20
            and logger is not None
        ):
            if cur_dmg_band == 1:
                logger.info("Damage band change")
            logger.info(f"‣ at cycle {i}:\n"
                        f"                • from {prev_dmg_band} to "
                        f"{cur_dmg_band} ({damage_bands[cur_dmg_band - 1]}"
                        f" < D ≤ {damage_bands[cur_dmg_band]})\n"
                        f"                • current damage value: "
                        f"{cumsum_nl_dmg[i-1]}\033[0m")

        # Damage weight for the current cycle
        w_ij: float = (
            (damage_bands[cur_dmg_band] - damage_bands[cur_dmg_band - 1])
            / ((damage_bands[cur_dmg_band] ** (1 / dmg_exp[i]))
                - (damage_bands[cur_dmg_band - 1] ** (1 / dmg_exp[i])))
        )
        # Update the damage value for the current cycle
        nl_damage[i] = w_ij * pm_damage[i] if i >= 1 else pm_damage[0]
        cumsum_nl_dmg[i] = (
            nl_damage[i] + cumsum_nl_dmg[i - 1] if i >= 1 else nl_damage[0]
        )
        if i >= 1 and cumsum_nl_dmg[i] > limit_damage:
            w_msg = f"Damage value exceeds {limit_damage} at index {i}"
            warnings.warn(w_msg, UserWarning)
            break
        # fmt: on

    # Edit nl_damage and cumsum_nl_dmg so that all the valus of nldamage[i+1:]
    # are 0 and all the value of cumsum_nl_dmg[i+1:] = cumsum_nl_dmg[i]
    nl_damage[i + 1 :] = 0
    cumsum_nl_dmg[i + 1 :] = cumsum_nl_dmg[i]
    if not plot:
        return nl_damage, cumsum_nl_dmg, None, None
    fig, ax = make_axes()
    cumsum_pm_dmg = np.cumsum(pm_damage)
    fig, ax = _plot_damage_accumulation(
        cumsum_nl_dmg, cumsum_pm_dmg, limit_damage, fig, ax
    )
    return nl_damage, cumsum_nl_dmg, fig, ax


def get_nonlinear_damage(
    damage_rule: str,
    cycle_count: CycleCount,
    sn_curve: SNCurve,
    **kwargs,
) -> float:
    """Calculate the fatigue damage using a nonlinear damage rule.

    Parameters
    ----------
    damage_rule : str
        The damage rule to use. Must be one of the following:
        'Pavlou', 'Manson-Halford', 'Si-Jian', 'Leve'.
    cycle_count : CycleCount
        The cycle count object.
    sn_curve : SNCurve
        The SN curve object.
    kwargs : dict
        The keyword arguments for the damage rule. The following
        keyword arguments are allowed:

        - 'base_exponent': The exponent for the damage rule.
        - 'ultimate_stress': The ultimate stress.

    Returns
    -------
    float
        The cumulated damage.
    """
    if cycle_count.unit != sn_curve.unit:
        e_msg = "The units of the cycle count and SN curve must be the same."
        raise ValueError(e_msg)
    return calc_nonlinear_damage(
        damage_rule,
        cycle_count.stress_range,
        cycle_count.count_cycle,
        sn_curve,
        **kwargs,
    )


def get_nonlinear_damage_with_dca(
    damage_rule: str,
    cycle_count: CycleCount,
    sn_curve: SNCurve,
    damage_bands: np.ndarray,
    limit_damage: Union[float, int] = 1,
    logger: logging.Logger | None = None,
    plot: bool = False,
    **kwargs,
) -> tuple[
    np.ndarray,
    np.ndarray,
    matplotlib.figure.Figure | None,
    matplotlib.axes.Axes | None,
]:
    """Calculate the fatigue damage using a nonlinear damage rule based on the
    Damage Curve Approach (DCA) among the allowed ones:

    - 'Pavlou': Pavlou damage rule
    - 'Theil': Theil damage rule

    Refer to :func:`calc_nonlinear_damage_with_dca` for the
    complete documentation with mathematical details.

    Parameters
    ----------
    damage_rule : str
        The damage rule to use. Must be one of the following:
        'Pavlou', 'Theil'.
    cycle_count : CycleCount
        The cycle count object.
    sn_curve : SNCurve
        The SN curve object.
    damage_bands : np.ndarray
        The damage bands.
    limit_damage : Union[float, int], optional
        The limit damage, by default 1
    logger : logging.Logger, optional
        The logger object, by default None
    kwargs : dict
        The keyword arguments for the damage rule. The following
        keyword arguments are allowed:

        - 'base_exponent': The exponent for the damage rule.
        - 'ultimate_stress': The ultimate stress.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The cycle-to-cycle damage and the cumulated damage.
    """
    if cycle_count.unit != sn_curve.unit:
        e_msg = "The units of the cycle count and SN curve must be the same."
        raise ValueError(e_msg)
    kwargs["use_dca"] = True
    return calc_nonlinear_damage_with_dca(
        damage_rule,
        cycle_count.stress_range,
        cycle_count.count_cycle,
        sn_curve,
        damage_bands,
        limit_damage,
        logger,
        plot=plot,
        **kwargs,
    )


def find_sn_curve_intersection(
    slope: np.ndarray,
    intercept: np.ndarray,
    endurance: float,
    weight: float,
    res_stress: float,
    n_min: float,
    n_max: float,
    tol=1e-6,
):
    """
    Solve the following nonlinear equation:

    .. math::

        \\left(\\frac{a}{N}\\right)^{\\frac{1}{m}} = w \\cdot N + b

    where:

    - a is the intercept of the S-N curve,
    - m is the slope of the S-N curve,
    - w is the weight for the stress range,
    - b is the residual stress range,
    - N is the number of cycles to failure,

    for N in the range [:math:`n_{min}`, :math:`n_{max}`].
    The equation is linked to the S-N curve definition, therefore it can be
    as complex as the S-N curve itself, i.e. log-linear, log-bilinear, etc.

    Parameters
    ----------
    intercept : float
        The intercept of the S-N curve.
    slope : float
        The slope of the S-N curve.
    endurance : float
        The endurance limit.
    weight : float
        The weight for the stress range.
    res_stress : float
        The residual stress range.
    n_min : float, optional
        The minimum number of cycles to failure, by default 1e0.
    n_max : float, optional
        The maximum number of cycles to failure, by default 1e10.
    tol : float, optional
        The tolerance for the bisection method, by default 1e-6.

    Returns
    -------
    float
        The number of cycles to failure.

    Raises
    ------
    ValueError
        If the bisection method fails to find a solution.
    """
    return numba_bisect(
        __jit_sn_curve_residuals,
        n_min,
        n_max,
        tol,
        1000,
        slope,
        intercept,
        endurance,
        weight,
        res_stress,
    )
    # from ..utils import python_bisect
    # from ..material.sn_curve import __sn_curve_residuals
    # return python_bisect(
    #     __sn_curve_residuals,
    #     n_min,
    #     n_max,
    #     tol,
    #     1000,
    #     slope, intercept, endurance, weight, res_stress
    # )


@nb.njit(
    fastmath=True,
    cache=True,
)
def _calc_theil_sn_damage(stress_range, count_cycle, cycles_to_failure):
    """
    Calculate the cycles to failure using the Theil's method for variable
    amplitude loading.
    Perform a fatigue life prediction under a sequence of stress range blocks,
    taking into account the interaction between different stress levels. It
    uses Theil's method to determine the effective stress range and accumulated
    damage, and an S-N curve to predict the number of cycles to failure.

    Mathematically, the method can be described as follows. Provided a sequence
    of stress range blocks :math:`\\Delta\\sigma_i` and the corresponding
    number of cycles :math:`n_i`, after initializing all the cumulative
    variables to zero, the effective stress range is calculated as:

    .. math::

        N_i & = \\text{SN Curve}\\left(\\Delta\\sigma_i \\right) \\\\
        w_i & = \\frac{\\Delta\\sigma_i}{N_i} \\\\
        \\Delta\\sigma_{eff,i} & = \\Delta\\sigma_{cumsum,i-1}
                                 - w_i \\cdot n_{cumsum, i-1} \\\\
        n_{cumsum,i} & = n_{cumsum,i-1} + n_i \\\\
        \\Delta\\sigma_{cumsum,i} & = w_i \\cdot n_{cumsum,i}
                                    + \\Delta\\sigma_{eff,i}

    where:

    - :math:`\\Delta\\sigma_{eff}` is the effective stress range,
    - :math:`\\Delta\\sigma_{cumsum}` is the cumulative stress range,
    - :math:`w` is the damage weight for the current cycle,
    - :math:`n_{cumsum}` is the cumulative number of cycles,
    - :math:`n` is the number of cycles for the current block.

    The method is applied to each block in the sequence, and the cumulative
    number of cycles is updated at the end of each block. The process continues
    until the number of cycles to failure is determined. If failure does not
    occur until the end of the sequence, the number of cycles to failure is
    estimated by "extrapolating" the last block, i.e., by calculating the
    intersection of the S-N curve with the effective stress range of the last
    block through the bisection method.

    Parameters
    ----------
    stress_range : np.ndarray
        A numpy array containing the stress range for each block (MPa).
    count_cycle : np.ndarray
        A numpy array containing the number of cycles for each block.
    sn_curve : SNCurve
        An object representing the S-N curve of the material.
        It must have attributes `intercept` (float) and `slope` (float).
    plot : bool, optional
        If True, a plot of the damage accumulation is generated. Defaults to
        False.
    Returns
    -------
    tuple
        A tuple containing:

        - n_to_failure (float): The number of cycles to failure. If failure
          occurs within a block, the cumulative number of cycles at the end
          of that block is returned.
        - history (list): A list of tuples, where each tuple contains:

          - n_segment (np.ndarray): An array of cycle numbers for the
            segment.
          - stress_range_segment (np.ndarray): An array of corresponding
            stress ranges for the segment.
          - label (str): A label for the segment, indicating the block
            number and stress range.

    Raises
    ------
    ValueError
        If the fatigue life solver fails to find a solution within a block,
        indicating that failure has occurred within that block.
    Notes
    -----
    - The function assumes that the stress range and count cycle arrays have
      the same length.
    - The Theil's method is used to calculate the effective stress range, which
      takes into account the sequence of stress levels.
    - The S-N curve is used to relate the stress range to the number of cycles
      to failure.
    """

    history = []

    # Start at 0 cycles and 0 stress range
    n_cumsum = 0.0
    stress_range_cumsum = 0.0  # effective stress range at the block end

    # Process predetermined blocks (blocks 1 to L-1)
    w_ij = np.divide(stress_range, cycles_to_failure)
    for i in nb.prange(stress_range.size):  # pylint: disable=E1133
        current_stress_range = stress_range_cumsum - w_ij[i] * n_cumsum
        num_points = int(np.ceil(count_cycle[i]))
        n_end = n_cumsum + count_cycle[i]
        n_segment = np.logspace(
            np.log10(n_cumsum + 1), np.log10(n_end), num_points
        )
        stress_range_segment = w_ij[i] * n_segment + current_stress_range
        # label = f"Block {i+1} (r={stress_range[i]} MPa)"
        history.append((n_segment, stress_range_segment))
        n_cumsum = n_end
        stress_range_cumsum = w_ij[i] * n_cumsum + current_stress_range

    return history, stress_range_cumsum, n_cumsum


def calc_theil_sn_damage(
    stress_range, count_cycle, sn_curve: SNCurve, to_failure: bool = False
) -> tuple[tuple[str, Any, Any], ...]:
    """
    Calculate the cycles to failure using the Theil's method for variable
    amplitude loading.
    Perform a fatigue life prediction under a sequence of stress range blocks,
    taking into account the interaction between different stress levels. It
    uses Theil's method to determine the effective stress range and accumulated
    damage, and an S-N curve to predict the number of cycles to failure.

    Mathematically, the method can be described as follows. Provided a sequence
    of stress range blocks :math:`\\Delta\\sigma_i` and the corresponding
    number of cycles :math:`n_i`, after initializing all the cumulative
    variables to zero, the effective stress range is calculated as:

    .. math::

        N_i & = \\text{SN Curve}\\left(\\Delta\\sigma_i \\right) \\\\
        w_i & = \\frac{\\Delta\\sigma_i}{N_i} \\\\
        \\Delta\\sigma_{eff,i} & = \\Delta\\sigma_{cumsum,i-1}
                                 - w_i \\cdot n_{cumsum, i-1} \\\\
        n_{cumsum,i} & = n_{cumsum,i-1} + n_i \\\\
        \\Delta\\sigma_{cumsum,i} & = w_i \\cdot n_{cumsum,i}
                                    + \\Delta\\sigma_{eff,i}

    where:

    - :math:`\\Delta\\sigma_{eff}` is the effective stress range,
    - :math:`\\Delta\\sigma_{cumsum}` is the cumulative stress range,
    - :math:`w` is the damage weight for the current cycle,
    - :math:`n_{cumsum}` is the cumulative number of cycles,
    - :math:`n` is the number of cycles for the current block.

    The method is applied to each block in the sequence, and the cumulative
    number of cycles is updated at the end of each block. The process continues
    until the number of cycles to failure is determined. If failure does not
    occur until the end of the sequence, the number of cycles to failure is
    estimated by "extrapolating" the last block, i.e., by calculating the
    intersection of the S-N curve with the effective stress range of the last
    block through the bisection method.

    Parameters
    ----------
    stress_range : np.ndarray
        A numpy array containing the stress range for each block (MPa).
    count_cycle : np.ndarray
        A numpy array containing the number of cycles for each block.
    sn_curve : SNCurve
        An object representing the S-N curve of the material.
        It must have attributes `intercept` (float) and `slope` (float).
    Returns
    -------
    tuple
        A tuple containing:

        - n_to_failure (float): The number of cycles to failure. If failure
          occurs within a block, the cumulative number of cycles at the end
          of that block is returned.
        - history (list): A list of tuples, where each tuple contains:

          - n_segment (np.ndarray): An array of cycle numbers for the
            segment.
          - stress_range_segment (np.ndarray): An array of corresponding
            stress ranges for the segment.
          - label (str): A label for the segment, indicating the block
            number and stress range.

    Raises
    ------
    ValueError
        If the fatigue life solver fails to find a solution within a block,
        indicating that failure has occurred within that block.
    Notes
    -----
    - The function assumes that the stress range and count cycle arrays have
      the same length.
    - The Theil's method is used to calculate the effective stress range, which
      takes into account the sequence of stress levels.
    - The S-N curve is used to relate the stress range to the number of cycles
      to failure.
    """
    history, stress_range_cumsum, n_cumsum = _calc_theil_sn_damage(
        count_cycle=count_cycle,
        stress_range=stress_range,
        cycles_to_failure=sn_curve.get_cycles(stress_range),
    )

    labels = [
        f"Block {i+1} (r={stress_range[i]} MPa)"
        for i in range(len(stress_range))
    ]
    # tuple([(label, *tup) for tup, label in zip(history, labels)])
    hist_dict = {
        label: (n_segment, stress_range_segment)
        for (n_segment, stress_range_segment), label in zip(history, labels)
    }
    if to_failure:
        w_final = stress_range[-1] / sn_curve.get_cycles(stress_range[-1])
        # print("w_final", w_final)
        current_stress_range = stress_range_cumsum - w_final * n_cumsum
        # print("current_stress_range", current_stress_range)
        # Solve modified Basquin for total cycles at final block:
        n_final = find_sn_curve_intersection(
            sn_curve.slope,
            sn_curve.intercept,
            sn_curve.endurance,
            current_stress_range,
            w_final,
            n_min=n_cumsum,
            n_max=1e15,
        )
        n_final_array = np.logspace(np.log10(n_cumsum), np.log10(n_final), 20)
        stress_range_final = w_final * n_final_array + current_stress_range
        # n_cumsum += n_final
        # stress_range_cumsum = w_final * n_cumsum + current_stress_range
        # Append to the last label block
        hist_dict[labels[-1]] = (
            np.append(history[-1][0], n_final_array),
            np.append(history[-1][1], stress_range_final),
        )
    # fmt: off
    return tuple((label, *tup) for tup, label in zip(hist_dict.values(),
                                                      hist_dict.keys()))
    # fmt: on
