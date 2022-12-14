# -*- coding: utf-8 -*-

"""The :mod:`py_fatigue.material.paris_curve` module collects all
functions related to :term:`Paris' law` definition.
The main class is ParisCurve.
"""

# Standard "from" imports
from collections.abc import Collection
from typing import Any, Optional, Tuple, Union

# Standard imports
import abc

import io

# import itertools
# import warnings

# Non-standard "from" imports
# from plotly.express import colors as px_colors

# Non-standard imports
import matplotlib.figure
import matplotlib.axis
import matplotlib.pyplot as plt
import numba as nb
import numpy as np

# import plotly.graph_objs as go
# import plotly.io as pio

# py_fatigue imports
from py_fatigue.utils import make_axes
from py_fatigue.material.sn_curve import (
    _set_color,
    _get_name,
    _check_param_couple_types,
    ensure_array,
)


class AbstractCrackGrowthCurve(metaclass=abc.ABCMeta):
    """Abstract Paris curve.
    Concrete subclasses should define methods:

        _ `_repr_svg_`
        - `format_name`
        - `get_knee_sif`
        - `get_knee_growth_rate`
        - `get_sif`
        - `get_growth_rate`
    """

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    # Eight is reasonable in this case.

    def __init__(
        self,
        slope: Union[int, float, list, np.ndarray],
        intercept: Union[int, float, list, np.ndarray],
        threshold: Union[int, float] = 0,
        critical: Union[int, float] = np.inf,
        environment: Optional[str] = None,
        curve: Optional[str] = None,
        norm: Optional[str] = None,
        unit_string: str = "MPa √mm",
        color: Union[str, None] = None,
    ) -> None:
        """Define crack growth rate curve (Paris' law).
        See class docstring for more information.

        Parameters
        ----------
        slope : Union[int, float, list, np.ndarray]
            Paris curve slope
        intercept : Union[int, float, list, np.ndarray]
            crack growth rate axis intercept
        threshold : Union[int, float]
            propagation threshold, below which crack growth rate is null
        critical : Union[int, float]
            critical propagation stress intensity factor, imminent failure
        environment : str, optional
            Paris' law envirnoment, by default None
        curve : str, optional
            Paris' law category, by default None
        norm : str, optional
            Paris' law norm, by default None
        unit_string : str, optional
            units, by default "MPa"
        color : str, optional
            RGBS or HEX string for color, by default None
        """
        self.color, self.id = _set_color(color)
        self.norm = norm
        self.environment = environment
        self.curve = curve
        self.name = _get_name(curve, environment, norm)
        self.__slope, self.__intercept = _check_param_couple_types(
            slope, intercept
        )
        self.__linear = len(self.slope) == 1
        self.__threshold = threshold
        self.__critical = critical
        self.__unit = unit_string

    def __str__(self) -> str:
        """Crack growth curve __str__ method

        Returns:
            str: Crack growth curve string description
        """
        return self.name

    def _repr_svg_(self):
        """SVG representation of the crack growth curve instance

        Returns:
            str: the SVG object
        """
        # def _repr_png_(self) -> None:
        # """PNG representation of the crack growth curve instance

        # Returns:
        #     None
        # """

    def __eq__(self, other: object) -> bool:
        """Compare different crack growth curves and assert that they
        contain same attributes values
        """
        if self.__class__ != other.__class__:
            return False

        def compare_exact(first, second):
            """Return whether two dicts of arrays are exactly equal"""
            if first.keys() != second.keys():
                return False
            return all(
                np.array_equal(first[key], second[key]) for key in first
            )

        return compare_exact(self.__dict__, other.__dict__)

    @property
    def linear(self) -> bool:
        """Preventing attribute modification outside of constructor

        Returns:
            bool: Is it a linear Paris' law?
        """
        return self.__linear

    @property
    def slope(self) -> np.ndarray:
        """Preventing attribute modification outside of constructor

        Returns:
            np.ndarray: Slope
        """
        return self.__slope

    @property
    def intercept(self) -> np.ndarray:
        """Preventing attribute modification outside of constructor

        Returns:
            np.ndarray: Intercept
        """
        return self.__intercept

    @property
    def threshold(self) -> Union[int, float]:
        """Preventing attribute modification outside of constructor

        Returns:
            Union[int, float]: Threshold SIF
        """
        return self.__threshold

    @property
    def critical(self) -> Union[int, float]:
        """Preventing attribute modification outside of constructor

        Returns:
            Union[int, float]: Critical SIF
        """
        return self.__critical

    @property
    def unit(self) -> str:
        """Preventing attribute modification outside of constructor

        Returns:
            unit: Units of measure, by default "MPa"
        """
        return self.__unit

    @abc.abstractmethod
    def format_name(self, html_format: bool = False) -> None:
        """Reformat ParisCurve name.

        Parameters
        ----------
        html_format : bool, optional
            Choose whether the name format shall be HTML or not,
            by default True
        """

    @property
    @abc.abstractmethod
    def threshold_growth_rate(self) -> float:
        """Calculates the crack growth rate at threshold, if threshold
        SIF is defined.

        Returns
        -------
        float
            threshold crack growth rate
        """

    @property
    @abc.abstractmethod
    def critical_growth_rate(self) -> float:
        """Calculates the crack growth rate at critical, if critical
        SIF is defined.

        Returns
        -------
        float
            critical crack growth rate
        """

    @abc.abstractmethod
    def get_knee_growth_rate(
        self,
        check_knee: Optional[Collection] = None,
        significant_digits: int = 2,
    ) -> np.ndarray:
        """Calculates the crack growth rate at the knee, if the Paris'
        law is more than linear.

        Parameters
        ----------
        check_knee : Collection, optional
            Iterable of SIF values to check for the knee, by default None
        significant_digits : int, optional
            Number of significant digits to round the knee to, by default 2

        Returns
        -------
        1darray
            knee crack growth rate
        """

    @abc.abstractmethod
    def get_knee_sif(
        self,
        check_knee: Optional[Collection] = None,
        significant_digits: int = 2,
    ) -> np.ndarray:
        """Calculates the SIF at the knee, if the Paris' law is more
        than linear.

        Parameters
        ----------
        check_knee : iterable, optional
            SIF values to check the knee SIF against, by default None
        significant_digits : int, optional
            number of significant digits to check the knee SIF against,
            by default 2

        Returns
        -------
        np.ndarray
            knee SIF
        """

    @abc.abstractmethod
    def get_growth_rate(
        self, sif_range: Union[int, float, list, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Return cycles value(s) for stress range(s).

        Parameters
        ---------
        sif_range: int or float or list or 1darray
           :term:`SIF range(s)<SIF range>` to find the corresponding
           :term:`crack growth rate(s)<Crack growth rate>` for.

        Returns
        -------
        float or 1darray
            :term:`crack growth rate(s)<Crack growth rate>` for the
            :term:`SIF range(s)<SIF range>` indicating `sif_range`.
        """

    @abc.abstractmethod
    def get_sif(
        self, growth_rate: Union[int, float, list, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Return :term:`SIF range(s)<SIF range>` for the endurance(s) N.

        Parameters
        ---------
        cycles : int or float or list or 1darray
            :term:`crack growth rate(s)<Crack growth rate>` to find the
            corresponding :term:`SIF range(s)<SIF range>` for.

        Returns
        -------
        float or 1darray
            :term:`SIF range(s)<SIF range>` for corresponding
            :term:`crack growth rate(s)<Crack growth rate>`.
        """


class ParisCurve(AbstractCrackGrowthCurve):
    """Define a crack growth curve inthe form of Paris' law that can
    have an arbitrary number of slopes/intercepts as well as a
    threshold and critical stress intensity factor. For example:

    >>> #
    >>> #         ^                log da/dN - log ΔK
    >>> #         │                                *
    >>> #         │                                *
    >>> #         │                                *
    >>> #         │                                *
    >>> #         │                             *  .
    >>> #         │                          *  │  .
    >>> #         │                       *     │ m2
    >>> #         │                    *        │  .
    >>> #         │                 *-----------┘  .
    >>> #         │              *                 .
    >>> #         │           .*│                  .
    >>> #         │        . *  │ m1               .
    >>> #         │     .  *    │                  .
    >>> #         │  .   *------┘                  .
    >>> # (da/dN)1+    *                           .
    >>> #         │  . *                           .
    >>> #         │.   *                           .
    >>> # (da/dN)2+----|---------------------------|--------------->
    >>> #             ΔK0                         ΔKc
    >>> #                          Number of cycles

    The slope-intercept couples `((m1, log_a1), (m2, log_a2))`, and the
    endurance value `(Ne)` are the parameters necessary to fully describe this
    trilinear Paris' law. If the endurance is not set, it defaults to Inf.

    Example
    -------

    >>> from py_fatigue import ParisCurve
    >>> import numpy as np

    >>> # Define a Paris' law with 5 slopes and 5 interceps
    >>> SIF = np.linspace(1,2500, 300)
    >>> SLOPE_5 = np.array([2.88, 5.1, 8.16, 5.1, 2.88])
    >>> INTERCEPT_5 = np.array([1E-16, 1E-20, 1E-27, 1E-19, 1E-13])
    >>> THRESHOLD = 20
    >>> CRITICAL = 2000

    >>> pc = ParisCurve(slope=SLOPE_5, intercept=INTERCEPT_5,
                      threshold=THRESHOLD, critical=CRITICAL, norm="The norm",
                      environment="Environment", curve="nr. 4")
    >>> pc_.get_knee_sif()
    array([ 63.35804993, 193.9017369 , 411.50876026, 504.31594872])

    """

    def _repr_svg_(self) -> str:
        fig, ax = self.plot()
        # fig.dpi = 100
        fig.set_figheight(5)
        fig.set_figwidth(7)
        max_x = np.max(ax.lines[0].get_xdata())
        max_y = np.max(ax.lines[0].get_ydata())
        cg_unit = "[L][CYCLE]$^{-1}$"
        if "√" in self.unit:
            split_unit = self.unit.split("√")
            cg_unit = "/".join((split_unit[1], "cycle"))

        ann_1 = f"{self.name},\nSlope: {self.slope},\nIntercept:"
        ann_2 = f"{self.intercept},\nThreshold: {self.threshold:.2f},"
        ann_3 = f"{cg_unit},\nCritical: {self.critical:.2f}, {self.unit}"
        annotation = " ".join((ann_1, ann_2, ann_3))
        ax.text(
            1.5 * max_x,
            max_y,
            annotation,
            horizontalalignment="left",
            verticalalignment="top",
            color="#435580",
            bbox={
                "facecolor": "#FFF",
                "alpha": 0.95,
                "pad": 10,
                "edgecolor": "#0E76BC",
            },
            fontsize=8,
            fontweight="semibold",
        )
        if not self.linear:
            knee_zip = zip(self.get_knee_sif(), self.get_knee_growth_rate())
            for j, (knee_sif, knee_gr) in enumerate(knee_zip):
                annotation = (
                    f"Knee {j +1}: ({round(knee_sif, 2)}, {knee_gr:.2E})"
                )
                ax.plot(knee_sif, knee_gr, "o", color="#C40000")
                for item in (
                    [ax.title, ax.xaxis.label, ax.yaxis.label]
                    + ax.get_xticklabels()
                    + ax.get_yticklabels()
                ):
                    item.set_fontsize(10)
                ax.text(
                    1.1 * knee_sif,
                    self.get_growth_rate(knee_sif),
                    annotation,
                    horizontalalignment="left",  # if j % 2 == 0 else "left",
                    verticalalignment="center",
                    fontsize=8,
                    color="#C40000",
                )
        if self.critical < np.inf:
            annotation = (
                f"Critical: ({round(self.critical, 2)}, "
                + f"{self.critical_growth_rate:.2E})"
            )
            ax.plot(
                self.critical, self.critical_growth_rate, "d", color="#435580"
            )
            ax.text(
                0.9 * self.critical,
                self.critical_growth_rate,
                annotation,
                horizontalalignment="right",
                verticalalignment="center",
                fontsize=8,
                color="#435580",
            )
        if self.threshold > 0:
            annotation = (
                f"Threshold: ({round(self.threshold, 2)}, "
                + f"{self.threshold_growth_rate:.2E})"
            )
            ax.plot(
                self.threshold,
                self.threshold_growth_rate,
                "d",
                color="#435580",
            )
            ax.text(
                1.1 * self.threshold,
                self.threshold_growth_rate,
                annotation,
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=8,
                color="#435580",
            )
        fig.tight_layout()

        output = io.BytesIO()
        fig.savefig(output, format="svg")
        data = output.getvalue()  # .encode('utf-8') doesn't change anything
        plt.close(fig)
        return data.decode("utf-8")

    def format_name(self, html_format: bool = False) -> None:
        if not html_format:
            self.name = self.name.replace("<br> ", "\n")
        else:
            self.name = self.name.replace("\n", "<br> ")

    @property
    def threshold_growth_rate(self) -> float:
        """Calculates the crack growth rate at threshold, if threshold
        SIF is defined.

        Returns
        -------
        float
            threshold crack growth rate
        """
        return self.get_growth_rate(self.threshold)

    @property
    def critical_growth_rate(self) -> float:
        """Calculates the crack growth rate at critical, if critical
        SIF is defined.

        Returns
        -------
        float
            critical crack growth rate
        """
        return self.get_growth_rate(self.critical)

    def get_knee_growth_rate(
        self,
        check_knee: Optional[Collection] = None,
        significant_digits: int = 2,
    ) -> np.ndarray:
        """Calculates the crack growth rate at the knee, if the Paris'
        law is more than linear.

        Parameters
        ----------
        check_knee : Iterable, optional
            Iterable of SIF values to check for the knee, by default None
        significant_digits : int, optional
            Number of significant digits to round the knee to, by default 2

        Returns
        -------
        np.ndarray
            knee crack growth rate
        """
        knee_growth_rate = np.empty(self.intercept.size - 1, dtype=np.float64)
        if not self.linear:
            for i in range(self.intercept.size - 1):
                m_i = self.slope[i + 1] / (self.slope[i + 1] - self.slope[i])
                m_i_p_1 = self.slope[i] / (self.slope[i + 1] - self.slope[i])
                knee_growth_rate[i] = (
                    self.intercept[i] ** m_i / self.intercept[i + 1] ** m_i_p_1
                )
            if check_knee is not None:
                try:
                    iter(check_knee)
                except TypeError:
                    check_knee = np.asarray([check_knee])
                else:
                    check_knee = np.asarray(check_knee)
                if len(check_knee) != len(self.slope) - 1:  # type: ignore
                    raise ValueError(
                        f"{len(self.slope) - 1} knee points expected"
                    )
                check_knee_growth_rate = self.get_growth_rate(check_knee)
                _ = [
                    np.testing.assert_approx_equal(  # type: ignore
                        c_k_s, k_s, significant=significant_digits
                    )
                    for c_k_s, k_s in zip(
                        check_knee_growth_rate, knee_growth_rate
                    )
                ]
        else:
            if check_knee is not None:
                raise ValueError("0 knee points expected")
        return knee_growth_rate

    def get_knee_sif(
        self,
        check_knee: Optional[Collection] = None,
        significant_digits: int = 2,
    ) -> np.ndarray:
        """Calculates the SIF at the knee, if the Paris' law is more
        than linear.

        Parameters
        ----------
        check_knee : iterable, optional
            SIF values to check the knee SIF against, by default None
        significant_digits : int, optional
            number of significant digits to check the knee SIF against,
            by default 2

        Returns
        -------
        np.ndarray
            knee SIF
        """
        knee_sif = np.empty(self.intercept.size - 1, dtype=np.float64)
        if not self.linear:
            if self.intercept.size > 1:
                for i in range(self.intercept.size - 1):
                    knee_sif[i] = (
                        self.intercept[i] / self.intercept[i + 1]
                    ) ** (1 / (self.slope[i + 1] - self.slope[i]))
            if check_knee is not None:
                try:
                    len(check_knee)
                except TypeError:
                    check_knee = np.asarray([check_knee])
                else:
                    check_knee = np.asarray(check_knee)
                if len(check_knee) != len(self.slope) - 1:  # type: ignore
                    raise ValueError(
                        f"{len(self.slope) - 1} knee points expected"
                    )
                _ = [
                    np.testing.assert_approx_equal(  # type: ignore
                        c_k_s, k_s, significant=significant_digits
                    )
                    for c_k_s, k_s in zip(check_knee, knee_sif)  # type: ignore
                ]
        else:
            if check_knee is not None:
                raise ValueError("0 knee points expected")
        return knee_sif

    @ensure_array
    def get_growth_rate(
        self, sif_range: Union[int, float, list, np.ndarray]
    ) -> Union[float, np.ndarray]:
        return _calc_growth_rate(
            sif_range,
            self.slope,
            self.intercept,
            self.threshold,
            self.critical,
        )

    @ensure_array
    def get_sif(
        self, growth_rate: Union[int, float, list, np.ndarray]
    ) -> Union[float, np.ndarray]:
        return _calc_sif(
            growth_rate,
            self.slope,
            self.intercept,
            self.threshold,
            self.critical,
        )

    def plot(
        self,
        sif: Optional[list] = None,
        growth_rate: Optional[list] = None,
        dataset_name: Optional[str] = None,
        dataset_color: str = "#000",
        fig: Optional[matplotlib.figure.Figure] = None,
        ax: Optional[matplotlib.axes.Axes] = None,  # type: ignore
        **kwargs: Any,
    ) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:  # type: ignore

        """Use matplotlib to plot the Paris' law and a da/dN vs ΔK
        history dataset.

        Example
        -------

        Use matplotlib to plot the Paris' law and a da/dN vs ΔK history

        >>> fig, ax = pc.plot()

        Parameters
        ----------
        sif : list, optional
            SIF value(s) in the dataset,
            by default None
        growth_rate : list, optional
            :term:`Crack growth rate(s)<Crack growth rate>` in the dataset,
            by default None
        dataset_name : str, optional
            history dataset name, by default None
        dataset_color : str, optional
            history dataset color, by default "#000"
        fig : matplotlib.figure.Figure, optional
            figure object, by default None
        ax : matplotlib.axes.Axes, optional
            axis object, by default None
        **kwargs : Any, optional
            additional keyword arguments

        Returns
        -------
        matplotlib.figure.Figure, matplotlib.axes.Axes
            The figure and axes.
        """
        fig, ax = make_axes(fig, ax)
        sif_plot, growth_rate_plot = _paris_curve_data_points(self)

        # plt.figure(figsize=(6, 2.5))
        ax.loglog(  # type: ignore
            sif_plot,
            growth_rate_plot,
            label=self.name,
            color=self.color,
            **kwargs,
        )
        if sif is not None and growth_rate is not None:
            sif, growth_rate = _check_param_couple_types(sif, growth_rate)
            ax.loglog(  # type: ignore
                sif,
                growth_rate,
                label=dataset_name,
                color=dataset_color,
                linewidth=1,
                marker="o",
                markersize=4,
            )

        plt.grid(which="both", linestyle=":")
        ax.set_xlabel(f"SIF range (ΔK), {self.unit}")  # type: ignore
        if "√" in self.unit:
            split_unit = self.unit.split("√")
            y_unit = "/".join((split_unit[1], "cycle"))
        else:
            y_unit = "[L][CYCLE]$^{-1}$"
        ax.set_ylabel(f"Crack growth rate (da/dN), {y_unit}")  # type: ignore

        return fig, ax


@nb.njit(
    # 'float64[::1](float64[::1], float64[::1], float64[::1])',
    fastmath=False,
    parallel=True,
)
def _calc_growth_rate(sif, slope, intercept, threshold, critical):
    # pylint: disable=not-an-iterable
    assert intercept.size > 0 and intercept.size == slope.size
    assert np.min(sif) >= 0
    assert 0 <= threshold < critical <= np.inf

    knees_sif = np.empty(intercept.size - 1, dtype=np.float64)
    if intercept.size > 1:
        for i in nb.prange(intercept.size - 1):
            knees_sif[i] = (intercept[i] / intercept[i + 1]) ** (
                1 / (slope[i + 1] - slope[i])
            )
    knees_sif = np.hstack(
        (
            np.array([0.9999999999 * threshold]),
            knees_sif,
            np.array([critical / 0.9999999999]),
        )
    )
    e_msg = (
        "Knee(s) not in between threshold and critical SIF."
        + "\nCheck the definitions of slope, intercept, threshold, critical."
    )
    assert np.all(np.diff(knees_sif) > 0), e_msg
    # print("knees_sif:", knees_sif)
    idx = np.digitize(sif, knees_sif, right=False) - 1
    # print("idx:", idx)
    the_growth_rate = np.empty(sif.size, dtype=np.float64)
    for i in nb.prange(sif.size):
        if sif[i] < threshold:  # below threshold
            the_growth_rate[i] = 0
            continue
        if sif[i] > critical:  # above threshold
            the_growth_rate[i] = np.inf
            continue
        the_growth_rate[i] = intercept[idx[i]] * sif[i] ** slope[idx[i]]

    return the_growth_rate


@nb.njit(
    # 'float64[::1](float64[::1], float64[::1], float64[::1])',
    fastmath=False,
    parallel=True,
)
def _calc_sif(growth_rate, slope, intercept, threshold, critical):
    # pylint: disable=not-an-iterable
    assert intercept.size > 0 and intercept.size == slope.size
    assert np.min(growth_rate) >= 0
    assert 0 <= threshold < critical <= np.inf

    knees_growth_rate = np.empty(intercept.size - 1, dtype=np.float64)
    if intercept.size > 1:
        for i in nb.prange(intercept.size - 1):
            m_i = slope[i + 1] / (slope[i + 1] - slope[i])
            m_i_p_1 = slope[i] / (slope[i + 1] - slope[i])
            knees_growth_rate[i] = (
                intercept[i] ** m_i / intercept[i + 1] ** m_i_p_1
            )

    knees_growth_rate = np.hstack(
        (
            np.array([intercept[0] * (0.9999999999 * threshold) ** slope[0]]),
            knees_growth_rate,
            np.array([intercept[-1] * (critical / 0.9999999999) ** slope[-1]]),
        )
    )
    e_msg = (
        "Knee(s) not in between threshold and critical SIF."
        + "\nCheck the definitions of slope, intercept, threshold, critical."
    )
    assert np.all(np.diff(knees_growth_rate) > 0), e_msg
    # print("knees_growth_rate:", knees_growth_rate)
    idx = np.digitize(growth_rate, knees_growth_rate, right=False) - 1
    # print("idx:", idx)
    the_sif = np.empty(growth_rate.size, dtype=np.float64)
    nr_knees = intercept.size - 1
    for i in nb.prange(growth_rate.size):
        if idx[i] <= 0:  # below threshold
            the_sif[i] = threshold
            continue
        if idx[i] >= nr_knees:  # above threshold
            the_sif[i] = critical
            continue
        the_sif[i] = (growth_rate[i] / intercept[idx[i]]) ** (
            1 / slope[idx[i]]
        )

    return the_sif


def _paris_curve_data_points(pc: ParisCurve) -> tuple:
    """Draw Paris' curve data points, including possible knees and
    threshold/critical SIF.

    Parameters
    ----------
    pc : ParisCurve
        Instance of :class:`py_fatigue.ParisCurve`

    Returns
    -------
    tuple
        tuple of arrays
    """

    # if pc.threshold == 0 and pc.critical == np.inf:
    threshold: Union[int, float] = 1.0
    critical: Union[int, float] = 1000.0
    if pc.threshold > 0 and pc.critical < np.inf:
        threshold = pc.threshold
        critical = pc.critical
    if pc.threshold == 0 and pc.critical < np.inf:
        threshold = pc.critical / 100
        critical = pc.critical
    if pc.threshold > 0 and pc.critical == np.inf:
        threshold = pc.threshold
        critical = pc.threshold * 100

    sif_plot = np.logspace(
        np.log10(threshold), np.log10(0.999999999999 * critical), 20
    )  # 0.999999999999 to avoid rounding errors

    if not pc.linear:
        sif_plot = np.sort(np.append(sif_plot, np.array(pc.get_knee_sif())))
    growth_rate_plot = pc.get_growth_rate(sif_plot)
    if pc.threshold > 0:
        growth_rate_plot = np.hstack(
            [growth_rate_plot[0] / 10, growth_rate_plot]
        )
        sif_plot = np.hstack([pc.threshold, sif_plot])
    if pc.critical < np.inf:
        growth_rate_plot = np.hstack(
            [growth_rate_plot, growth_rate_plot[-1] * 10]
        )
        # Again, 0.999999999999 to avoid rounding errors
        sif_plot = np.hstack([sif_plot, 0.999999999999 * pc.critical])
    return sif_plot, growth_rate_plot
