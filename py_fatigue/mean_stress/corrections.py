"""
The :mod:`py_fatigue.msc.corrections` module collects all the
mean stress corrections implemented in py_fatigue.
classes.
"""

from __future__ import annotations
import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np

# from scipy.optimize import fsolve

from ..utils import make_axes, compile_specialized_newton, numba_newton
from ..styling import py_fatigue_formatwarning

warnings.formatwarning = py_fatigue_formatwarning


def dnvgl_mean_stress_correction(
    mean_stress: np.ndarray,
    stress_amplitude: np.ndarray,
    detail_factor: float | int = 0.8,
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

        \\frac{\\sigma_{amp}}{\\vert \\sigma_m \\vert} \\leq 1, \\quad
        \\sigma_{amp} = \\frac{\\sigma_{max} - \\sigma_{min}}{2} \\,\\land
        \\, \\sigma_m = \\frac{\\sigma_{max} + \\sigma_{min}}{2}

    Parameters
    ----------
    mean_stress : np.ndarray
        The mean stress of the fatigue test (:math:`\\sigma_m`).
    stress_amplitude : np.ndarray
        The stress amplitude of the fatigue test (:math:`\\sigma_{amp}`).
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
        _, axes = make_axes()
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
        axes.set_xlim((-20, 20))

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
        \\sigma_{amp} ^ {\\gamma}

    with:

    .. math::

        \\sigma_{max} = \\sigma_{mean} + \\sigma_{amp}

    Parameters
    ----------
    mean_stress : np.ndarray
        The mean stress of the fatigue test (:math:`\\sigma_{mean}`).
    stress_amplitude : np.ndarray
        The stress amplitude of the fatigue test (:math:`\\sigma_{amp}`).
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
        _, axes = make_axes()
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
        axes.set_ylabel("MSC stress range, $\\Delta \\sigma_{MSC}$")
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

        \\Delta \\sigma_{corr} = \\sqrt{\\sigma_{max} \\cdot \\sigma_{amp}}

    with:

    .. math::

        \\sigma_{max} = \\sigma_{mean} + \\sigma_{amp}

    Parameters
    ----------
    mean_stress : np.ndarray
        The mean stress of the fatigue test (:math:`\\sigma_{mean}`).
    stress_amplitude : np.ndarray
        The stress amplitude of the fatigue test (:math:`\\sigma_{amp}`).
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


def goodman_haigh_mean_stress_correction(  # pylint: disable=R0912 # noqa: C901,E501
    amp_in: np.ndarray | list[float],
    mean_in: np.ndarray | list[float],
    r_out: float | np.ndarray | list[float],
    ult_s: float,
    correction_exponent: float,
    initial_guess: np.ndarray | None = None,
    plot: bool = False,
    logger: logging.Logger | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    This function extends the standard Goodman-Haigh correction formula which
    computes the corrected stress amplitude at zero mean stress (:math:`R=-1`)
    for a given set of input stress amplitudes and mean stresses. 

    .. math::

        \\sigma_{amp, \\text{out}} =  \\frac{\\sigma_{amp, \\text{in}}}
        {1 - \\left(\\frac{\\sigma_{mean, \\text{in}}}{\\sigma_{ult}}
        \\right)^{n}}

    The correction exponent :math:`n` is a material property that can
    be obtained from experimental data.

    The current implementation extends the standard Goodman-Haigh formula by
    correcting the stress amplitude from an initial load ratio :math:`R_{in}`
    to any other load ratio :math:`R_{out}`, as long as the material's
    ultimate strength is known.
    The corrected stress amplitude :math:`\\sigma_{amp, \\text{out}}` is
    computed using the following implicit nonlinear equation:

    .. math::

        \\sigma_{amp, \\text{out}} =  \\frac{1}{ \\frac{1 - \\left(
            \\frac{(1 + R_{\\text{in}}) \\cdot \\sigma_{amp, \\text{in}}}
            {(1 - R_{\\text{in}}) \\cdot \\sigma_{ult}}
            \\right)^{n}}{\\sigma_{amp, \\text{in}}}
            + \\sigma_{amp, \\text{out}}^{n-1} \\cdot \\left(
            \\frac{(1 + R_{\\text{out}})}{(1 - R_{\\text{out}})
            \\cdot \\sigma_{ult}} \\right)^{n}}

    where:

    * :math:`R`\\ :sub:`in`\\  = :math:`\\sigma_{min} / \\sigma_{max} =
      (\\sigma_{mean} - \\sigma_{amp}) / (\\sigma_{mean} + \\sigma_{amp})` is
      the input stress ratio,
    * :math:`R`\\ :sub:`out`\\  is the output stress ratio,
    * :math:`\\sigma_{amp}`\\ :sub:`in`\\  is the input amplitude,
    * :math:`\\sigma_{amp}`\\ :sub:`out`\\  is the output amplitude,
    * :math:`\\sigma_{ult}` is the ultimate strength of the material, and
    * :math:`n` is the correction exponent.

    The solver is based on the Newton-Raphson method, which is implemented
    internally at :func:`py_fatigue.utils.numba_newton`,
    :func:`py_fatigue.utils.compile_specialized_newton`, and
    :func:`py_fatigue.mean_stress.corrections.__goodman_equation`.

    Parameters
    ----------
    amp_in : np.ndarray | list
        Input amplitude values.
    mean_in : np.ndarray | list
        Input mean stress values.
    r_out : float | np.ndarray
        Output stress ratio values.
    ult_s : float
        Ultimate strength of the material.
    correction_exponent : float
        Mean stress correction exponent (n).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Computed amp_out values, and corresponding mean_out values.

    See Also
    --------
    :func:`py_fatigue.utils.numba_newton`,
    :func:`py_fatigue.utils.compile_specialized_newton`,
    :func:`py_fatigue.mean_stress.corrections.__goodman

    Raises
    ------
    ValueError
        If the input arrays have different shapes or if there are negative
        stress amplitude values in amp_in.
    Warning
        If the number of points in the input arrays is too large for plotting.

    Examples
    --------
    >>> import numpy as np
    >>> from py_fatigue.mean_stress.corrections import (
    ...     goodman_haigh_mean_stress_correction
    ... )
    >>> amp_in = np.array([100, 200, 300])
    >>> mean_in = np.array([0, 0, 0])
    >>> r_out = np.array([-1, 0, 0.5, 0.75, 0.9])
    >>> ult_s = 1000
    >>> correction_exponent = 9
    >>> amp_out, mean_out = goodman_haigh_mean_stress_correction(
    ...     amp_in, mean_in, r_out, ult_s, correction_exponent, plot=True
    ... )

    """
    # Convert inputs to numpy arrays for element-wise operations
    if isinstance(amp_in, list):
        amp_in = np.asarray(amp_in, dtype=np.float64)
    if isinstance(mean_in, list):
        mean_in = np.asarray(mean_in, dtype=np.float64)
    if np.isscalar(r_out):
        r_out = np.array([r_out], dtype=np.float64)
    r_out = np.asarray(r_out, dtype=np.float64)
    # Check that the input arrays have the same shape
    if amp_in.shape != mean_in.shape:
        raise ValueError("amp_in and mean_in must have the same shape")
    # Check that all values in amp_in are positive
    if np.any(amp_in < 0):
        raise ValueError("You have negative stress amplitude values in amp_in")
    # Check that, if the product of the size of amp_in and r_out is greater
    # than 1,000,000, the user has set plot to False
    if amp_in.size * r_out.size > 1_000_000 and plot:
        if logger is not None:
            # fmt: off
            logger.warning("\033[1mThe number of points in the input arrays "
                           "is too large for plotting\033[0m")
            logger.info("Setting plot to False to avoid matplotlib errors")
        warnings.warn(
            "The number of points in the input arrays is too large "
            "for plotting. Setting plot to False to avoid "
            "matplotlib errors"
        )

        # fmt: on
        plot = False
    # Compute r_in from amp_in and mean_in
    r_in = (mean_in - amp_in) / (mean_in + amp_in)
    # mean_in = amp_in * (1 + r_in) / (1 - r_in)
    # Initial guess for amp_out set to Goodman Correction for r_out = - 1
    if initial_guess is None:
        initial_guess = amp_in / (1 - (mean_in / ult_s) ** correction_exponent)
    # Set initial guess equal to zero wherever r_in > 1
    initial_guess[r_in > 1] = 0

    # Solve for each amp_out using fsolve
    amp_out = np.zeros((len(r_out), len(initial_guess)))
    mean_out = np.zeros((len(r_out), len(initial_guess)))
    # amp_out = []
    # mean_out = []
    for r_out_val in r_out:
        # NOTE: Special cases for r_out = -1 and r_out = 0
        # NOTE: These cases are solved analytically to improve performance
        if r_out_val == -1:
            # amp_out.append(
            #     amp_in / (1 - (mean_in / ult_s) ** correction_exponent)
            # )
            # mean_out.append(np.zeros_like(amp_in))
            amp_out[r_out == -1, :] = amp_in / (
                1 - (mean_in / ult_s) ** correction_exponent
            )
            mean_out[r_out == -1, :] = np.zeros_like(amp_in)
            continue
        amp_out_fsolve = []
        for i in range(len(initial_guess)):
            # Solve the implicit equation using fsolve
            # fmt: off
            if mean_in[i] + amp_in[i] > 0:
                sol = numba_newton(__jit_goodman_equation, initial_guess[i],
                                    1E-6, 1000, amp_in[i], r_in[i],
                                    r_out_val, ult_s, correction_exponent)
                # sol = fsolve(__goodman_equation, x0=initial_guess[i],
                #              args=(amp_in[i], r_in[i], r_out_val, ult_s,
                #                    correction_exponent), xtol=1E-6)
            else:
                sol = amp_in[i]
            # Check if fsolve converged or not
            if sol is None:
                if logger is not None:
                    logger.warning(f"fsolve did not converge for amp_in="
                                   f"{amp_in[i]}, r_in={r_in[i]}, r_out="
                                   f"{r_out_val}")
                amp_out_fsolve.append(np.nan)
            else:
                amp_out_fsolve.append(sol if sol < ult_s else ult_s)
            # fmt: on
        # amp_out.append(amp_out_fsolve)
        # mean_out.append(
        #     np.array(amp_out_fsolve) * (1 + r_out_val) / (1 - r_out_val)
        # )
        amp_out[r_out == r_out_val, :] = np.asarray(amp_out_fsolve)
        mean_out[r_out == r_out_val, :] = (
            amp_out[r_out == r_out_val, :] * (1 + r_out_val) / (1 - r_out_val)
        )

    # Sort r_out, as well as amp_out and mean_out (by r_out)
    if np.all(r_out[:-1] <= r_out[1:]):
        srt_r_out = r_out.copy()
        srt_amp_out = amp_out.copy()
        srt_mean_out = mean_out.copy()
    else:
        srt_idx = np.argsort(r_out)
        srt_r_out = r_out[srt_idx]
        srt_amp_out = amp_out[srt_idx, :]
        srt_mean_out = mean_out[srt_idx, :]
    if plot:
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # First subplot: amp_out vs r_out
        # fmt: off
        for i in range(len(amp_in)):
            ax1.plot(srt_r_out, srt_amp_out[:, i],
                    marker='.', markersize=2, linewidth=0.5,
                    label=f"$amp_{{in}}={np.round(amp_in[i], 1)}, "
                        f"r_{{in}}={np.round(r_in[i], 2)}$")
        ax1.plot(r_in, amp_in,
                'o', color='yellow', markersize=4, markeredgecolor='black',
                markeredgewidth=0.5, label='Original Data')
        ax1.set_xlabel("load ratio, $\\sigma_{min}/ \\sigma_{max}$")
        ax1.set_ylabel("MSC stress amplitude, $\\Delta \\sigma_{MSC} / 2$")
        ax1.set_title("Goodman Correction - Amplitude vs R")
        ax1.set_xlim(right=1.25, left=max(-5, min(r_in) - 0.25))
        ax1.set_ylim(top=min(max(amp_in) * 1.5, ult_s))
        ax1.grid(True)

        # Second subplot: mean_out vs amp_out
        for i in range(len(amp_in)):
            ax2.plot(srt_mean_out[:, i], srt_amp_out[:, i],
                    marker='.', markersize=2, linewidth=0.5,
                    label=f"$amp_{{in}}={np.round(amp_in[i], 1)}, "
                        f"r_{{in}}={np.round(r_in[i], 2)}$")
        # stress_amplitude * (1 + load_ratio) / (1 - load_ratio)
        ax2.plot(amp_in * (1 + r_in) / (1 - r_in), amp_in,
                'o', color='yellow', markersize=4, markeredgecolor='black',
                markeredgewidth=0.5, label='Original Data')

        # Set limits to maximum stress amplitude and ultimate strength
        ax2.set_xlim(right=ult_s * 1.1)
        ax2.set_ylim(top=min(max(amp_in) * 1.5, ult_s))
        # draw lines at constant load ratio for better visualization

        for r in [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]:
            sample_mean_out = np.linspace(0, ult_s * i, 2)
            ax2.plot(sample_mean_out, (1 - r) / (1 + r) * sample_mean_out, ':',
                    color='#AAAAAA', linewidth=0.75)
        ax2.set_xlabel("MSC mean stress, $\\sigma_{m,MSC}$")
        ax2.set_ylabel("MSC stress amplitude, $\\Delta \\sigma_{MSC} / 2$")
        ax2.set_title("Goodman Correction - Amplitude vs Mean")
        handles, labels = ax2.get_legend_handles_labels()
        ax2.grid(False)

        # Adjust layout and show legend
        plt.tight_layout()
        if len(handles) < 31:
            fig.legend(handles, labels, bbox_to_anchor=(0.5, -0.05),
                       loc='upper center', ncol=4)
        # fmt: on
        plt.show()
    return amp_out, mean_out


def __goodman_equation(
    amp_out_val: float,
    amp_in_val: float,
    r_in_val: float,
    r_out_val: float,
    ult_s: float,
    correction_exponent: float,
) -> float:
    """Implicit equation to solve for amp_out.

    .. math::

        \\sigma_{amp, \\text{out}} =  \\frac{1}{ \\frac{1 - \\left(
            \\frac{(1 + R_{\\text{in}}) \\cdot \\sigma_{amp, \\text{in}}}
            {(1 - R_{\\text{in}}) \\cdot \\sigma_{ult}}
            \\right)^{n}}{\\sigma_{amp, \\text{in}}}
            + \\sigma_{amp, \\text{out}}^{n-1} \\cdot \\left(
            \\frac{(1 + R_{\\text{out}})}{(1 - R_{\\text{out}})
            \\cdot \\sigma_{ult}} \\right)^{n}}
    where:
    * :math:`R`\\ :sub:`in`\\  = :math:`\\sigma_{min} / \\sigma_{max} =
      (\\sigma_{mean} - \\sigma_{amp}) / (\\sigma_{mean} + \\sigma_{amp})` is
      the input stress ratio,
    * :math:`R`\\ :sub:`out`\\  is the output stress ratio,
    * :math:`\\sigma_{amp}`\\ :sub:`in`\\  is the input amplitude,
    * :math:`\\sigma_{amp}`\\ :sub:`out`\\  is the output amplitude,
    * :math:`\\sigma_{ult}` is the ultimate strength of the material, and
    * :math:`n` is the correction exponent.
    Parameters
    ----------
    amp_out_val : float
        The output amplitude value.
    amp_in_val : float
        The input amplitude value.
    r_in_val : float
        The input load ratio value.
    r_out_val : float
        The output load ratio value.
    ult_s : float
        The ultimate strength of the material.
    correction_exponent : float
        The correction exponent value.
    Returns
    -------
    float
        The difference between the left-hand side and right-hand side of the
        equation.
    """
    assert r_in_val != 1, "Input load ratio equals 1, meaning static load."
    # if r_in_val == 1:
    # fmt: off
    # raise ValueError(f"Input load ratio equals 1 at amp_in={amp_in_val}"
    #                   ", meaning static load. Revise input data.")
    # fmt: on
    trm1 = (1 + r_in_val) / (1 - r_in_val) * amp_in_val / ult_s
    trm2 = (1 + r_out_val) / (1 - r_out_val) / ult_s

    lhs = amp_out_val
    # fmt: off
    rhs = (1 - trm1**correction_exponent) / amp_in_val + \
            amp_out_val**(correction_exponent - 1) * trm2**correction_exponent
    # fmt: on
    return lhs - rhs**-1


__jit_goodman_equation = compile_specialized_newton(__goodman_equation)
