"""
The :mod:`py_fatigue.msc.corrections` module collects all the
mean stress corrections implemented in py_fatigue.
classes.
"""

from typing import Union

import matplotlib.pyplot as plt
import numpy as np

import py_fatigue.cycle_count.histogram as ht


def dnvgl_mean_stress_correction(
    mean_stress: np.ndarray,
    stress_amplitude: np.ndarray,
    detail_factor: Union[float, int] = 0.8,
    plot: bool = False,
) -> np.ndarray:
    """Calculates the mean stress correction according to par.
    2.5 of`DNVGL-RP-C203 <https://bit.ly/3dUZ1OY>`_ which includes
    an attenuation factor :math:`p` for the stress ranges if the
    following cases:

    * base material without significant residual stresses \
    :math:`\\rightarrow p = 0.6`. This option neglects fully \
    compressive cycles.
    * welded connections without significant residual stresses \
    :math:`\\rightarrow p = 0.8`. This option multiplies the \
    stress range of fully compressive cycles by p.

    Given that the stress ranges are :math:`\\Delta \\sigma`,
    the corrected stress ranges are:

    .. math::

        \\Delta \\sigma_{corr} = f_m \\cdot \\Delta \\sigma

    where:

    .. math::

        f_m = \\frac{\\sigma_{max} + p \\cdot \\vert \\sigma_{min}
        \\vert}{\\sigma_{max} + \\vert\\sigma_{min} \\vert}

    with:

    .. math::

        \\frac{\\sigma_a}{\\vert \\sigma_m \\vert} \\leq 1, \\quad
        \\sigma_a = \\frac{\\sigma_{max} - \\sigma_{min}}{2} \\,\\land
        \\, \\sigma_m = \\frac{\\sigma_{max} + \\sigma_{min}}{2}

    Parameters
    ----------
    mean_stress : np.ndarray
        The mean stress of the fatigue test (:math:`\\sigma_m`).
    stress_amplitude : np.ndarray
        The stress amplitude of the fatigue test (:math:`\\sigma_a`).
    detail_factor : float, optional
        mean stress attenuation factor (:math:`p`), defaults to 0.8
        in welded connections without significant residual stresses.
    plot : bool, optional
        If True, the mean stress correction is plotted, defaults to
        False.

    Returns
    -------
    np.ndarray
        The mean stress corrected ranges
        (:math:`\\Delta \\sigma_{corr}`).
    """

    max_stress = mean_stress + stress_amplitude
    min_stress = mean_stress - stress_amplitude

    f_m = (max_stress + detail_factor * np.abs(min_stress)) / (
        max_stress + np.abs(min_stress)
    )

    f_m[f_m > 1] = 1
    f_m[min_stress > 0] = 1
    if detail_factor == 0.8:
        f_m[f_m < detail_factor] = detail_factor
    elif detail_factor == 0.6:
        f_m[f_m < detail_factor] = 0
    else:
        raise ValueError(
            f"Detail factor = {str(detail_factor)} is not allowed in "
            + "DNVGL-RP-C203. Only 0.6 and 0.8 are permitted."
        )

    stress_range_corr = 2 * f_m * stress_amplitude

    # NOTE: next plot has been included for testing purposes
    if plot:  # pragma: no cover
        # from matplotlib.ticker import MaxNLocator
        _, axes = ht.make_axes()
        xp, yp = zip(*sorted(zip(min_stress / max_stress, f_m)))
        axes.plot(
            xp,
            yp,
            linestyle=":",
            linewidth=1,
            color="#000000",
            marker=".",
            markersize=6,
            markerfacecolor="#000000",
            markeredgewidth=0.5,
            markeredgecolor="#666666",
        )
        plt.rcParams.update({"mathtext.default": "regular"})

        axes.set_xlabel("load ratio, $\\sigma_{min}/ \\sigma_{max}$")
        axes.set_ylabel("$f_m$")
        axes.spines["right"].set_visible(False)
        axes.spines["top"].set_visible(False)
        axes.set_xlim([-20, 20])

    return stress_range_corr


def walker_mean_stress_correction(
    mean_stress: np.ndarray,
    stress_amplitude: np.ndarray,
    gamma: float = 0.5,
    plot: bool = False,
) -> np.ndarray:
    """Calculates the mean stress correction according to Walker model.

    The correction is given by:

    .. math::

        \\Delta \\sigma_{corr} = {\\sigma_{max}} ^ {(1 - \\gamma)} \\,
        \\sigma_{alt} ^ {\\gamma}

    with:

    .. math::

        \\sigma_{max} = \\sigma_{mean} + \\sigma_{alt}

    Parameters
    ----------
    mean_stress : np.ndarray
        The mean stress of the fatigue test (:math:`\\sigma_{mean}`).
    stress_amplitude : np.ndarray
        The stress amplitude of the fatigue test (:math:`\\sigma_{alt}`).
    gamma : float, optional
        The gamma Walker exponent (:math:`\\gamma`), defaults to 0.5.
    plot : bool, optional
        If True, the mean stress correction is plotted, defaults to
        False.

    Returns
    -------
    np.ndarray
        The mean stress corrected ranges (:math:`\\Delta \\sigma_{corr}`).
    """

    max_stress = mean_stress + stress_amplitude

    stress_range_corr = np.nan_to_num(
        (max_stress ** (1 - gamma)) * (stress_amplitude**gamma)
    )

    # NOTE: next plot has been included for testing purposes
    if plot:  # pragma: no cover
        # from matplotlib.ticker import MaxNLocator
        _, axes = ht.make_axes()
        xp, yp = zip(*sorted(zip(2 * stress_amplitude, stress_range_corr)))
        axes.plot(
            xp,
            yp,
            linestyle=":",
            linewidth=1,
            color="#000000",
            marker=".",
            markersize=6,
            markerfacecolor="#000000",
            markeredgewidth=0.5,
            markeredgecolor="#666666",
        )
        plt.rcParams.update({"mathtext.default": "regular"})

        axes.set_xlabel("uncorrected stress range, $\\Delta \\sigma$")
        axes.set_ylabel("MSC stress range, , $\\Delta \\sigma_{MSC}$")
        axes.spines["right"].set_visible(False)
        axes.spines["top"].set_visible(False)

    return stress_range_corr


def swt_mean_stress_correction(
    mean_stress: np.ndarray,
    stress_amplitude: np.ndarray,
    plot: bool = False,
) -> np.ndarray:
    """Calculates the mean stress correction according to
    Smith-Watson-Topper model. It is equivalent to the Walker model
    with gamma = 0.5.

    The correction is given by:

    .. math::

        \\Delta \\sigma_{corr} = \\sqrt{\\sigma_{max} \\, \\sigma_{alt}}

    with:

    .. math::

        \\sigma_{max} = \\sigma_{mean} + \\sigma_{alt}

    Parameters
    ----------
    mean_stress : np.ndarray
        The mean stress of the fatigue test (:math:`\\sigma_{mean}`).
    stress_amplitude : np.ndarray
        The stress amplitude of the fatigue test (:math:`\\sigma_{alt}`).
    plot : bool, optional
        If True, the mean stress correction is plotted, defaults to
        False.

    Returns
    -------
    np.ndarray
        The mean stress corrected ranges (:math:`\\Delta \\sigma_{corr}`).
    """

    return walker_mean_stress_correction(
        mean_stress, stress_amplitude, gamma=0.5, plot=plot
    )
