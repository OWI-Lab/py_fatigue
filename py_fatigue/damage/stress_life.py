"""The module :mod:`py_fatigue.damage.stress_life` contains all the
damage models related to the stress-life approach.
"""

# Packages from standard library
from types import SimpleNamespace
from typing import Callable, Optional, Tuple, Union
import warnings

# Packages from external libraries
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# py-fatigue imports
from py_fatigue.cycle_count.cycle_count import CycleCount
from py_fatigue.material.sn_curve import SNCurve
from py_fatigue.utils import make_axes
from py_fatigue.styling import py_fatigue_formatwarning
from py_fatigue.material.sn_curve import _check_param_couple_types

try:
    # delete the accessor to avoid warning
    del pd.DataFrame.miner  # type: ignore
except AttributeError:
    pass


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
            c=dens_func(self._obj.pm_damage)
            if dens_func is not None
            else self._obj.pm_damage,
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
        The outer radius to calculate the DEM. Must be provided in mm
    inner_radius : float
        The inner radius to calculate the DEM. Must be provided in mm
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
        The outer radius to calculate the DEM. Must be provided in mm
    inner_radius : float
        The inner radius to calculate the DEM. Must be provided in mm
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
    float
        The cumulated damage.
    """
    damage_per_cycle = calc_pm(
        stress_range,
        count_cycle,
        sn_curve,
    )
    allowed_rules = [
        "pavlou",
        "manson",
        "manson and halford",
        "manson halford",
        "leve",
        "si jian",
        "si jian et al",
    ]
    if damage_rule not in allowed_rules:
        e_msg = f"damage_rule must be one of {allowed_rules}"
        raise ValueError(e_msg)

    ns = SimpleNamespace(**kwargs)
    if damage_rule.lower() == "pavlou":
        if "base_exponent" not in kwargs:
            w_msg = [
                "Pavlou damage rule requires 'base_exponent' ",
                "to be assigned.\nUsing preset value of -0.75.",
            ]
            warnings.formatwarning = py_fatigue_formatwarning
            warnings.warn("".join(w_msg), UserWarning)
            ns.base_exponent = -0.75
        if "ultimate_stress" not in kwargs:
            w_msg = [
                "Pavlou damage rule requires 'ultimate_stress' ",
                "to be assigned.\nUsing preset value of 900 MPa.",
            ]
            warnings.formatwarning = py_fatigue_formatwarning
            warnings.warn("".join(w_msg), UserWarning)
            ns.ultimate_stress = 900
        damage_exp = calc_pavlou_exponents(
            stress_range, ns.ultimate_stress, ns.base_exponent
        )
    if "manson" in damage_rule.lower():
        if "base_exponent" not in kwargs:
            w_msg = [
                "Manson & Halford damage rule requires ",
                "'base_exponent' to be assigned.\n",
                "Using preset value of 0.4.",
            ]
            warnings.formatwarning = py_fatigue_formatwarning
            warnings.warn("".join(w_msg), UserWarning)
            ns.base_exponent = 0.4
        damage_exp = calc_manson_halford_exponents(
            sn_curve.get_cycles(stress_range), ns.base_exponent
        )
    if damage_rule.lower() == "leve":
        if "base_exponent" not in kwargs:
            w_msg = [
                "Leve damage rule requires ",
                "'base_exponent' to be assigned.\n",
                "Using preset value of 2.",
            ]
            warnings.formatwarning = py_fatigue_formatwarning
            warnings.warn("".join(w_msg), UserWarning)
            ns.base_exponent = 2
        damage_exp = ns.base_exponent * np.ones(len(damage_per_cycle))
    if "si jian" in damage_rule.lower():
        damage_exp = calc_si_jian_exponents(stress_range)

    total_damage = 0
    damage_array = np.empty(len(damage_per_cycle))
    for i, (d_i, e_i) in enumerate(zip(damage_per_cycle, damage_exp)):
        total_damage = (total_damage + d_i) ** e_i
        damage_array[i] = total_damage

    return damage_array[-1]


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
