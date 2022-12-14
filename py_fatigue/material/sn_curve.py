# -*- coding: utf-8 -*-

"""The :mod:`py_fatigue.material.sn_curve` module collects all functions
related to :term:`SN Curve` definition.
The main and only class is SNCurve.
"""

# Standard "from" imports
from collections.abc import Collection
from typing import (
    Any,
    Callable,
    Optional,
    Sized,
    Tuple,
    Union,
)

# Standard imports
import abc
import io
import itertools
import warnings

# Non-standard "from" imports
from functools import wraps
from plotly.express import colors as px_colors

# Non-standard imports
import matplotlib.figure
import matplotlib.axis
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio

# py_fatigue imports
from py_fatigue.utils import make_axes

COLOR_LIST = px_colors.qualitative.Alphabet
PLOTLY_FONT_FAMILY = "Roboto"
pio.templates.default = "none"
__all__ = ["SNCurve"]


# Decorator
def check_iterable(function: Callable) -> Callable:
    """Decorator checking whether function *args are iterable.

    Parameters
    ----------
    function : Callable
        Generic function

    Returns
    -------
    Callable
    Generic function output
    """

    @wraps(function)
    def wrapper(*args):
        iter_args = []
        for arg in args:
            try:
                iter(arg)
            except TypeError:
                iter_args.append(np.asarray([arg], dtype=float))
            else:
                iter_args.append(np.asarray(arg, dtype=float))
        return function(*iter_args)

    return wrapper


@check_iterable
def _check_param_couple_types(
    par_1: Sized,
    par_2: Sized,
) -> tuple:
    """Check whether a couple of input parameters for a function are
    compatible in terms of type and/or size.

    Parameters
    ----------
    par_1 : Iterable
        1st parameter
    par_2 : Iterable
        2nd parameter

    Returns
    -------
    None

    Raises
    ------
    ValueError
       Params lengths must match
    """
    if len(par_1) != len(par_2):
        raise ValueError(
            "Params lengths must match. Got:", len(par_1), len(par_2)
        )
    return par_1, par_2


# Decorator
def check_str(function: Callable) -> Callable:
    """Decorator that checks whether a variable is string or NoneType.

    Parameters
    ----------
    function : Callable
        Input function

    Returns
    -------
    Input function output
    """

    @wraps(function)
    def wrapper(*args):
        str_args = []
        for arg in args:
            str_args.append(str(arg or ""))
        return function(*str_args)

    return wrapper


@check_str
def _get_name(curve: str, environment: str, norm: str):
    """Defines the SN curve name

    Parameters
    ----------
    curve : str
    environment : str
    norm : str

    Returns
    -------
    str
        SN curve name
    """
    return f"Norm: {norm},\nCurve: {curve},\nEnvironment: {environment}"


# Decorator
def color_setter(func: Callable):
    """Counts how many times the SN Curve class has been called and sets the
    SN curve color accordingly.

    Parameters
    ----------
    func : Callable

    Returns
    -------
    Callable
    """

    class Wrapper:
        """The Wrapper class defines the color and calls attributes that will
        be assigned to SN curve.
        """

        color: Optional[str]

        def __init__(self):
            # Wrapper is callable because it has the `__call__` method.
            # Not being a plain function allows you to explicitly
            # define the function attributes
            self.calls = 0
            self.cycle_color_list = itertools.cycle(COLOR_LIST)

        def __call__(self, *args, **kwargs):
            self.calls += 1
            self.color = next(self.cycle_color_list)
            # print(f'wrapper called {wrapper.calls} times')
            return func(*args, **kwargs)

    return Wrapper()


@color_setter
def _set_color(color: Union[str, None]) -> tuple:
    """Sets the SN curve color.

    Parameters
    ----------
    color : str
        Color variable, can be RGBA or HEX string format.

    Returns
    -------
    tuple
        Color and SN curve call id
    """

    if color is None:
        color = _set_color.color
        my_id = _set_color.calls
    else:
        my_id = None
    return color, my_id


# Decorator
def ensure_array(method: Callable) -> Callable:
    """Ensures that the input variable of a class method is an array.

    Parameters
    ----------
    method : Callable
        Input method

    Returns
    -------
    Callable
        Input method output
    """

    @wraps(method)
    def wrapper(self, x):
        try:
            iter(x)
        except TypeError:
            xm = np.asarray([x])
            ym = method(self, xm)[0]
        else:
            xm = np.asarray(x)
            ym = method(self, xm)
        return ym

    return wrapper


class AbstractSNCurve(metaclass=abc.ABCMeta):
    """Abstract SN curve.
    Concrete subclasses should define methods:

        _ `_repr_svg_`
        - `endurance_stress`
        - `format_name`
        - `get_knee_cycles`
        - `get_knee_stress`
        - `get_cycles`
        - `get_stress`
    """

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    # Eight is reasonable in this case.

    def __init__(
        self,
        slope: Union[int, float, list, np.ndarray],
        intercept: Union[int, float, list, np.ndarray],
        endurance: Union[int, float] = np.inf,
        environment: Optional[str] = None,
        curve: Optional[str] = None,
        norm: Optional[str] = None,
        unit_string: str = "MPa",
        color: Optional[str] = None,
    ) -> None:
        """Define stress-life (SN) curve.
        See class docstring for more information.

        Parameters
        ----------
        slope : Union[int, float, list, np.ndarray]
            SN curve slope
        intercept : Union[int, float, list, np.ndarray]
            Stress axis intercept
        endurance : Union[int, float], optional
            endurance number of cycles, by default np.inf
        environment : str, optional
            SN curve envirnoment, by default None
        curve : str, optional
            SN curve category, by default None
        norm : str, optional
            SN curve norm, by default None
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
        self.__endurance = endurance
        _ = self.endurance_stress
        self.__unit = unit_string

    def __str__(self) -> str:
        """SN curve __str__ method

        Returns:
            str: SN curve string description
        """
        return self.name

    @abc.abstractmethod
    def _repr_svg_(self) -> str:
        """SVG representation of the SN curve instance

        Returns:
            str: the SVG object
        """
        # def _repr_png_(self) -> None:
        # """PNG representation of the SN curve instance

        # Returns:
        #     None
        # """

    def __eq__(self, other: object) -> bool:
        """Compare different SN curves and assert that they contain same
        attributes values
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
            bool: Is it a linear SN curve?
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
    def endurance(self) -> Union[int, float]:
        """Preventing attribute modification outside of constructor

        Returns:
            Union[int, float]: Endurance
        """
        return self.__endurance

    @property
    def unit(self) -> str:
        """Preventing attribute modification outside of constructor

        Returns:
            unit: Units of measure, by default "MPa"
        """
        return self.__unit

    @property
    @abc.abstractmethod
    def endurance_stress(self) -> float:
        """Calculates the endurance stress, if endurance is defined.

        Returns
        -------
        float
            endurance stress

        Raises
        ------
        ValueError
            Endurance (XXX cycles) is not compatible with SN curve knees
            (list). Consider changing params
        """

    @abc.abstractmethod
    def format_name(self, html_format: bool = False) -> None:
        """Reformat SNCurve name.

        Parameters
        ----------
        html_format : bool, optional
            Choose whether the name format shall be HTML or not,
            by default True
        """

    @abc.abstractmethod
    def get_knee_cycles(
        self,
        check_knee: Optional[Collection] = None,
        significant_digits: int = 2,
    ) -> np.ndarray:
        """Calculate the knee cycles.

        Parameters
        ----------
        check_knee : Collection, optional
            Collection of knee stress values, by default None
        significant_digits : int, optional
            Number of significant digits, by default 2

        Returns
        -------
        1darray:
            Knee cycles
        """

    @abc.abstractmethod
    def get_knee_stress(
        self,
        check_knee: Optional[Collection] = None,
        significant_digits: int = 2,
    ) -> np.ndarray:
        """Return stress at the knee(s).

        Arguments
        ---------
        check_knee : int or float or list or 1darray, optional
            Cycles value(s) to check against, defaults to Inf
        significant_digits : int, optional
            Assertion accuracy, defaults to 2

        Returns
        -------
        1darray
            Stress at the knee(s)
        """

    @abc.abstractmethod
    def get_cycles(
        self, stress_range: Union[int, float, list, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Return cycles value(s) for stress range(s).

        Arguments
        ---------
        stress_range: int or float or list or 1darray
           :term:`stress range(s)<Stress ranges>` to find the corresponding
           number of cycles value(s) for.

        Returns
        -------
        float or 1darray
            number of cycles value(s) for the
            :term:`stress range(s)<Stress ranges>` `stress_range`.
        """

    @abc.abstractmethod
    def get_stress(
        self, cycles: Union[int, float, list, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Return stress range(s) for the endurance(s) N.

        Arguments
        ---------
        cycles : int or float or list or 1darray
            number of cycles value(s) to find the corresponding
            :term:`stress range(s)<Stress ranges>` for.

        Returns
        -------
        float or 1darray
            :term:`Stress range(s)<Stress ranges>` for
            number of cycles value(s) `cycles`.
        """


class SNCurve(AbstractSNCurve):
    """Define an SN curve that can have an arbitrary number of
    slopes/intercepts and an endurance. For example:

    >>> #
    >>> #         ^                log N - log S
    >>> # log(a1) +
    >>> #         │*
    >>> #         │ *
    >>> #         │  *
    >>> #         │   *  m1
    >>> #         │    *---+
    >>> # log(a2) +     *  |
    >>> #         │  .   * | 1
    >>> #     S   │     . *|
    >>> #     t   │        *       m2
    >>> #     r   │           *--------+
    >>> #     e   │              *     | 1
    >>> #     s   │                 *  |
    >>> #     s   │                    *
    >>> #         │                        *  *  *  *  *  *  *  *
    >>> #         │
    >>> #         │
    >>> #         │────────────────────────|────────────────────────>
    >>> #                                  Ne
    >>> #                          Number of cycles

    The slope-intercept couples `((m1, log_a1), (m2, log_a2))`, and the
    endurance value `(Ne)` are the parameters necessary to fully describe this
    trilinear SN curve. If the endurance is not set, it defaults to Inf.

    Example
    -------

    >>> from py_fatigue import SNCurve
    >>> import numpy as np

    First we create an SN curve with the following properties:

    >>> w3a = SNCurve(
    >>>     [3, 5],
    >>>     [10.970, 13.617],
    >>>     norm='DNVGL-RP-C203',
    >>>     curve='W3',
    >>>     environment='Air',
    >>> )

    Then we can plot the SN curve defined using plotly

    >>> data, layout = w3a.plotly()
    >>> fig = go.Figure(data=data, layout=layout)
    >>> fig.show
    """

    def _repr_svg_(self) -> str:
        fig, ax = self.plot()
        fig.dpi = 600
        max_x = np.max(ax.lines[0].get_xdata())
        max_y = np.max(ax.lines[0].get_ydata())
        annotation = (
            self.name
            + ","
            + "\nSlope: "
            + str(self.slope)
            + ","
            + "\nIntercept: "
            + str(self.intercept)
        )
        ax.text(
            max_x,
            max_y,
            annotation,
            horizontalalignment="right",
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
            knee_zip = zip(self.get_knee_cycles(), self.get_knee_stress())
            for j, (knee_cycles, knee_stress) in enumerate(knee_zip):
                annotation = (
                    f"Knee {j +1}: "
                    + f"({round(knee_cycles, 0):.2E}, "
                    + f"{round(knee_stress, 2)})"
                )
                ax.plot(knee_cycles, knee_stress, "o", color="#C40000")
                ax.text(
                    10 ** (0.98 * self.intercept[j])
                    * np.power(knee_stress, -self.slope[j]),
                    knee_stress,
                    annotation,
                    horizontalalignment="right",  # if j % 2 == 0 else "left",
                    verticalalignment="center",
                    fontsize=8,
                    color="#C40000",
                )
        if self.endurance < np.inf:
            x_reduced = 10 ** (0.98 * self.intercept[-1]) * np.power(
                self.endurance_stress, -self.slope[-1]
            )
            annotation = (
                "Endurance: "
                + f"({round(self.endurance, 0):.2E}, "
                + f"{round(self.endurance_stress, 2)})"
            )
            ax.plot(
                self.endurance, self.endurance_stress, "d", color="#435580"
            )
            ax.text(
                x_reduced,
                self.endurance_stress,
                annotation,
                horizontalalignment="right",
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

    @property
    def endurance_stress(self) -> float:
        if not self.linear:
            if np.max(self.get_knee_cycles()) > self.endurance:
                raise ValueError(
                    f"Endurance ({self.endurance} cycles) is ",
                    "lower than SN curve knee(s) ",
                    f"({self.get_knee_cycles()} cycles). ",
                    "\nConsider changing params",
                )
            return self.get_stress(self.endurance)
        return 0.0

    def format_name(self, html_format: bool = False) -> None:
        if not html_format:
            self.name = self.name.replace("<br> ", "\n")
        else:
            self.name = self.name.replace("\n", "<br> ")

    def get_knee_stress(
        self,
        check_knee: Optional[Collection] = None,
        significant_digits: int = 2,
    ) -> np.ndarray:
        if not self.linear:
            knee_stress = np.asarray(
                [
                    np.power(
                        10 ** (self.intercept[j] - _a),
                        1 / (self.slope[j] - _m),
                    )
                    for j, (_m, _a) in enumerate(
                        zip(self.slope[1:], self.intercept[1:])
                    )
                ]
            )
            if check_knee is not None:
                try:
                    iter(check_knee)  # type: ignore
                except TypeError:
                    check_knee = np.asarray([check_knee])
                else:
                    check_knee = np.asarray(check_knee)
                assert isinstance(check_knee, Collection)
                if len(check_knee) != len(self.slope) - 1:  # type: ignore
                    raise ValueError(
                        f"{len(self.slope) - 1} knee points expected"
                    )
                check_knee_stress = self.get_stress(check_knee)
                _ = [
                    np.testing.assert_approx_equal(  # type: ignore
                        c_k_s, k_s, significant=significant_digits
                    )
                    for c_k_s, k_s in zip(check_knee_stress, knee_stress)
                ]
            return knee_stress
        if check_knee is not None:
            raise ValueError("0 knee points expected")
        return np.array([])

    def get_knee_cycles(
        self,
        check_knee: Optional[Collection] = None,
        significant_digits: int = 2,
    ) -> np.ndarray:
        knee_stress = self.get_knee_stress()
        if len(knee_stress) > 0:
            knee = np.asarray(
                [
                    10 ** self.intercept[j] * np.power(k_s_r, -self.slope[j])
                    for j, k_s_r in enumerate(knee_stress)
                ]
            )
            if check_knee is not None:
                try:
                    iter(check_knee)
                except TypeError:
                    check_knee = np.asarray([check_knee])
                else:
                    check_knee = np.asarray(check_knee)
                assert isinstance(check_knee, Collection)
                if len(check_knee) != len(self.slope) - 1:  # type: ignore
                    raise ValueError(
                        f"{len(self.slope) - 1} knee points expected"
                    )
                _ = [
                    np.testing.assert_approx_equal(  # type: ignore
                        c_k, k, significant=significant_digits
                    )
                    for c_k, k in zip(check_knee, knee)  # type: ignore
                ]
            return knee
        if check_knee is not None:
            raise ValueError("0 knee points expected")
        return np.array([])

    @ensure_array
    def get_cycles(
        self, stress_range: Union[int, float, list, np.ndarray]
    ) -> Union[float, np.ndarray]:
        # m, a = self.slope, self.intercept
        # # List comprehension approach
        # np.minimum(np.max(
        #     [10 ** _a * np.power(stress_range.astype(float), -_m) \
        #         for _m, _a in zip(m, a)],
        #     axis=0
        # ), self.endurance)
        # # Numpy broadcasting approach
        # np.minimum(np.max(
        #     np.power(10, a) * np.power(
        #         stress_range.astype(float)[:, None], -m
        #     ).reshape(
        #         v1.shape[0],-1),
        #     axis=1
        # ), self.endurance)

        return _calc_cycles(
            stress_range, self.slope, self.intercept, self.endurance
        )

    @ensure_array
    def get_stress(
        self, cycles: Union[int, float, list, np.ndarray]
    ) -> Union[float, np.ndarray]:
        # m, a = self.slope, self.intercept
        # endurance_stress = np.power(10 ** a / self.endurance, 1 / m)
        # # List comprehension approach
        # np.maximum(np.max(
        #     [10 ** _a / cycles, -_m) \
        #         for _m, _a in zip(m, a)],
        #     axis=0
        # ), endurance_stress)
        # # Numpy broadcasting approach
        # np.maximum(np.max(
        #     np.power(10 ** a / cycles.astype(float)[:, None], 1 / m).reshape(
        #         cycles.shape[0],-1),
        #         axis=1
        # ), endurance_stress)
        return _calc_stress(cycles, self.slope, self.intercept, self.endurance)

    def n(  # pylint: disable=invalid-name
        self, sigma: Union[int, float, list, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Equivalent of :func:`~SNCurve.get_cycles()`. Added for backwards
        compatibility. It will be removed in a future release.

        Parameters
        ---------
        sigma: int or float or list or 1darray
           :term:`stress range(s)<Stress ranges>` to find the corresponding
           number of cycles value(s) for.

        Returns
        -------
        float or 1darray
            number of cycles value(s) for the
            :term:`stress range(s)<Stress ranges>` `sigma`.
        """

        warnings.warn(
            category=FutureWarning,
            message=(
                "This function is added for backwards compatibility only and"
                + " will be removed in a future release of py_fatigue"
            ),
        )
        return self.get_cycles(sigma)

    def sigma(
        self,
        n: Union[int, float, list, np.ndarray],  # pylint: disable=invalid-name
    ) -> Union[float, np.ndarray]:
        """Equivalent of :func:`SNCurve.get_stress()`. Added for backwards
        compatibility. It will be removed in a future release

        Arguments
        ---------
        n: int or float or list or 1darray
           number of cycles value(s) to find the corresponding
           :term:`stress range(s)<Stress ranges>` for.

        Returns
        -------
        float or 1darray
            :term:`stress range(s)<Stress ranges>` for the
            number of cycles value(s) `n`.
        """
        warnings.warn(
            category=FutureWarning,
            message=(
                "This function is added for backwards compatibility only and"
                + " will be removed in a future release of py_fatigue"
            ),
        )
        return self.get_stress(n)

    def plotly(
        self,
        cycles: Optional[list] = None,
        stress_range: Optional[list] = None,
        dataset_name: Optional[str] = None,
        dataset_color: str = "#000",
    ) -> Tuple[list, dict]:
        """Use plotly to plot the SN curve and a stress-cycles history dataset.

        Example
        -------

        Use plotly to plot the SN curve

        >>> data, layout = sncurve.plotly()
        >>> fig = go.Figure(data=data, layout=layout)
        >>> fig.show

        Parameters
        ----------
        cycles : list, optional
            number of cycles value(s) in the dataset,
            by default None
        stress_range : list, optional
            :term:`Stress range(s)<Stress ranges>` in the dataset,
            by default None
        dataset_name : str, optional
            history dataset name, by default None
        dataset_color : str, optional
            history dataset color, by default "#000"

        Returns
        -------
        Tuple[list, dict]
            data, layout
        """

        n_plot = np.asarray(
            [10**_ * np.linspace(1, 10, 10) for _ in range(4, 9)]
        ).reshape(1, 50)[0]
        if self.endurance < np.inf:
            n_plot = np.sort(
                np.append(
                    n_plot, np.array([self.endurance, 10 * self.endurance])
                )
            )
        if not self.linear:
            n_plot = np.sort(
                np.append(n_plot, np.array(self.get_knee_cycles()))
            )
        sigma_plot = self.get_stress(n_plot)
        data = [
            go.Scattergl(
                x=n_plot,
                y=sigma_plot,
                name=self.name,
                line=dict(color=self.color, width=1),
            )
        ]

        if cycles is not None and stress_range is not None:
            cycles, stress_range = _check_param_couple_types(
                cycles,
                stress_range,
            )
            # print(cycles, stress_range)
            data.append(
                go.Scattergl(
                    x=cycles,
                    y=stress_range,
                    name=dataset_name,
                    mode="lines+markers",
                    line=dict(color=dataset_color, width=1),
                    marker=dict(color=dataset_color, size=4),
                )
            )
        layout = go.Layout(
            xaxis=dict(
                title="Number of cycles",
                linecolor="#000",
                type="log",
                tickformat=".0f",
            ),
            yaxis=dict(
                title="Stress range, " + self.unit,
                linecolor="#000",
                type="log",
                tickformat=".0f",
            ),
            font=dict(family="Roboto", size=14, color="#000"),
            legend=dict(font=dict(family="Roboto", size=12, color="#000")),
        )

        return data, layout

    def plot(
        self,
        cycles: Optional[list] = None,
        stress_range: Optional[list] = None,
        dataset_name: Optional[str] = None,
        dataset_color: str = "#000",
        fig: Optional[matplotlib.figure.Figure] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        **kwargs: Any,
    ) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:

        """Use plotly to plot the SN curve and a stress-cycles history dataset.

        Example
        -------

        Use plotly to plot the SN curve

        >>> fig, ax = sncurve.plot()

        Parameters
        ----------
        cycles : list, optional
            number of cycles value(s) in the dataset,
            by default None
        stress_range : list, optional
            :term:`Stress range(s)<Stress ranges>` in the dataset,
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
        n_plot, sigma_plot = _sn_curve_data_points(self)

        # plt.figure(figsize=(6, 2.5))
        ax.loglog(
            n_plot, sigma_plot, label=self.name, color=self.color, **kwargs
        )
        if cycles is not None and stress_range is not None:
            cycles, stress_range = _check_param_couple_types(
                cycles,
                stress_range,
            )
            ax.loglog(
                cycles,
                stress_range,
                label=dataset_name,
                color=dataset_color,
                linewidth=1,
                marker="o",
                markersize=4,
            )

        plt.grid(which="both", linestyle=":")
        ax.set_xlabel("Number of cycles")
        ax.set_ylabel("Stress range, " + self.unit)

        return fig, ax


@nb.njit(
    # 'float64[::1](float64[::1], float64[::1], float64[::1])',
    fastmath=False,
    parallel=True,
)
def _calc_cycles(stress, slope, intercept, endurance):
    # pylint: disable=not-an-iterable
    assert intercept.size > 0 and intercept.size == slope.size
    assert np.min(stress) >= 0
    log10 = np.log(10)
    the_cycles = np.empty(stress.size, dtype=np.float64)
    for i in nb.prange(stress.size):
        log_stress = np.log(stress[i])
        max_i = intercept[0] * log10 - slope[0] * log_stress
        for j in range(1, len(intercept)):
            value = intercept[j] * log10 - slope[j] * log_stress
            if value > max_i:
                max_i = value
        the_cycles[i] = np.exp(max_i)
    if endurance < np.inf:
        the_cycles[the_cycles > endurance] = np.inf
    return the_cycles


# TODO: Check whether, using the knee, is it possible to generalise further
# TODO: the concept of SN curve.
# @nb.njit(
#     # 'float64[::1](float64[::1], float64[::1], float64[::1])',
#     fastmath=False,
#     parallel=True,
# )
# def _calc_cycles(stress, slope, intercept, endurance, knee):
#     # pylint: disable=not-an-iterable
#     assert intercept.size > 0 and intercept.size == slope.size
#     assert np.min(stress) > 0
#     log10 = np.log(10)
#     the_cycles = np.empty(stress.size, dtype=np.float64)
#     for i in nb.prange(stress.size):
#         log_stress = np.log(stress[i])
#         the_cycles[i] = np.exp(intercept[0] * log10 - slope[0] * log_stress)
#         if knee is not None:
#             for j in range(0, len(intercept)):
#                 value = intercept[j] * log10 - slope[j] * log_stress
#                 if value > max_i:
#                     max_i = value
#             the_cycles[i] = np.exp(max_i)
#     if endurance < np.inf:
#         the_cycles[the_cycles > endurance] = np.inf
#     return the_cycles


@nb.njit(
    # 'float64[::1](float64[::1], float64[::1], float64[::1])',
    fastmath=False,
    parallel=True,
)
def _calc_stress(cycles, slope, intercept, endurance):
    # pylint: disable=not-an-iterable
    assert intercept.size > 0 and intercept.size == slope.size
    assert np.min(cycles) > 0
    log10 = np.log(10)
    the_stress = np.empty(cycles.size, dtype=np.float64)
    for i in nb.prange(cycles.size):
        log_cycles = np.log(cycles[i])
        max_i = (intercept[0] * log10 - log_cycles) / slope[0]
        for j in range(1, len(intercept)):
            value = (intercept[j] * log10 - log_cycles) / slope[j]
            if value > max_i:
                max_i = value
        the_stress[i] = np.exp(max_i)
    if endurance < np.inf:
        endurance_stress = np.exp(
            (intercept[-1] * log10 - np.log(endurance)) / slope[-1]
        )
        the_stress[the_stress < endurance_stress] = endurance_stress
    return the_stress


def _sn_curve_data_points(sn: SNCurve) -> tuple:
    """Draw SN curve data points, including possible knees and enduarance.

    Parameters
    ----------
    sn : SNCurve
        Instance of :class:`py_fatigue.SNCurve`

    Returns
    -------
    tuple
        tuple of arrays
    """
    n_plot = np.asarray(
        [10**_ * np.linspace(1, 10, 10) for _ in range(4, 9)]
    ).reshape(1, 50)[0]
    if sn.endurance < np.inf:
        n_plot = np.sort(
            np.append(n_plot, np.array([sn.endurance, 10 * sn.endurance]))
        )
    if not sn.linear:
        n_plot = np.sort(np.append(n_plot, np.array(sn.get_knee_cycles())))
    sigma_plot = sn.get_stress(n_plot)
    return n_plot, sigma_plot
