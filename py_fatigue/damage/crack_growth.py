"""The module :mod:`py_fatigue.damage.crack_growth` contains all the
damage models related to the crack growth approach.
"""

# Packages from the Python Standard Library
from typing import Dict, Tuple, Type, TypeVar

# Packages from third party libraries
from numba.experimental import jitclass
import numba as nb
import numpy as np
import pandas as pd

# Local packages
from py_fatigue.cycle_count.cycle_count import CycleCount
from py_fatigue.material.crack_growth_curve import ParisCurve
from py_fatigue.utils import split, to_numba_dict
from py_fatigue.geometry import AbstractCrackGeometry
from py_fatigue.geometry.cylinder import f_hol_cyl_01


try:
    # delete the accessor to avoid warning
    del pd.DataFrame.cg  # type: ignore
except AttributeError:
    pass

ACG = TypeVar("ACG", bound=AbstractCrackGeometry)

spec = [
    ("stress_range", nb.float64[::1]),
    ("count_cycle", nb.float64[:]),
    ("slope", nb.float64[::1]),
    ("intercept", nb.float64[::1]),
    ("threshold", nb.float64),
    ("critical", nb.float64),
    ("crack_type", nb.types.string),
    ("crack_geometry", nb.types.DictType(nb.types.unicode_type, nb.float64)),
    ("crack_depth", nb.float64[::1]),
    ("sif", nb.float64[::1]),
    ("geometry_factor", nb.float64[::1]),
    ("final_cycles", nb.float64),
    ("failure", nb.boolean),
]


@jitclass(spec)
class CalcCrackGrowth:
    """Crack growth object. This object is used to calculate the crack
    growth rate and the crack size. The class has been implemented
    using the Numba library to speed up the calculations.

    The crack size is numerically integrated over the crack growth
    curve. In this example we will consider the Paris' law, but the
    function can be used with any other crack growth curve.

    .. math::

        \\begin{align}
        \\begin{cases}
            \\frac{da}{dN} & = C \\left( \\Delta K \\right)^{m} \\\\
            \\Delta K & = Y(a) \\cdot \\Delta \\sigma \\sqrt{\\pi a} \\\\
        \\end{cases}
        \\end{align}

    From the equations above, the crack growths rate can be expressed as a
    function of the stress range. Considering the Paris' law, as in this
    example, gives:

    .. math::

        \\frac{da}{dN} = C \\left( Y(a) \\cdot \\Delta \\sigma \\sqrt{\\pi a}
            \\right)^{m}

    se this crack growth rate together with the cycle step size to
    determine the incremental growth of the crack for this iteration:

    .. math::

        \\Delta a = \\frac{da}{dN} \\cdot \\Delta N

    The new crack size is then calculated by adding the incremental growth
    to the previous crack size:

    .. math::

        a_{j+1} = a_{j} + \\Delta a

    where :math:`a_{j}` is the crack size at the :math:`j`-th cycle.

    The crack is grown iteratively either until the failure condition
    is met or until the stress history is over.

    If failure occurs, the total number of cycles that it took to grow the
    crack to the critical size becomes the predicted life of the part.


    Parameters
    ----------
    stress_range : np.ndarray
        Stress range array.
    count_cycle : np.ndarray
        Number of cycles array.
    slope : np.ndarray
        Slope array.
    intercept : np.ndarray
        Intercept array.
    threshold : float
        Threshold value.
    critical : float
        Critical value.
    crack_geometry : dict
        Crack geometry dictionary. It must contain the following keys:
        - "initial_depth": float
        - "_id": str



    Returns
    -------
    CalcCrackGrowth
        Crack growth object.
    """

    def __init__(
        self,
        stress_range: np.ndarray,
        count_cycle: np.ndarray,
        slope: np.ndarray,
        intercept: np.ndarray,
        threshold: float,
        critical: float,
        crack_type: str,
        crack_geometry: dict,
    ):

        assert intercept.size > 0 and intercept.size == slope.size
        assert np.min(stress_range) >= 0.0
        assert stress_range.size > 0 and stress_range.size == count_cycle.size
        self.stress_range = stress_range
        self.count_cycle = count_cycle
        self.slope = slope
        self.intercept = intercept
        self.threshold = threshold
        self.critical = critical
        self.crack_type = crack_type
        self.crack_geometry = crack_geometry
        cs, sif, factor, failure = self.calc_crack_depth()
        self.crack_depth = cs
        self.sif = sif
        self.geometry_factor = factor
        self.final_cycles = sum(self.count_cycle[: len(self.crack_depth)])
        self.failure = failure

    def __str__(self) -> str:
        """String representation of the object."""
        if self.failure:
            return "Crack growth object: to failure."
        return "Crack growth object: to maximum number of cycles - no failure."

    @property
    def size(self) -> int:
        """Size of the problem."""
        return self.stress_range.size

    @property
    def get_knees_sif(self) -> np.ndarray:
        """Get the SIF values at the knees of the Paris curve.

        If the Paris curve hs more than one (slope, intercept) couple,
        the SIF values at the knees are calculated using the following
        equation:

        .. math::

            \\text{knee}_{i} = \\left(\\frac{\\text{intercept}_{i}}
                {\\text{intercept}_{i+1}}\\right)^{\\frac{1}{
                \\text{slope}_{i+1} - \\text{slope_{i}}}}
        """
        cg_crv_size = self.intercept.size
        knees_sif = np.empty(
            cg_crv_size - 1, dtype=np.float64
        )  # pylint: disable=E1133, C0301
        if cg_crv_size > 1:  # pylint: disable=E1133, C0301
            for i in nb.prange(  # pylint: disable=E1133, C0301
                cg_crv_size - 1
            ):
                knees_sif[i] = (self.intercept[i] / self.intercept[i + 1]) ** (
                    1 / (self.slope[i + 1] - self.slope[i])
                )
        knees_sif = np.hstack(
            (
                np.array([0.9999999999 * self.threshold]),
                knees_sif,
                np.array([self.critical / 0.9999999999]),
            )
        )
        e_msg = (
            "Knee(s) not in between threshold and critical SIF."
            + "\nCheck the definitions of slope, intercept, "
            + "threshold, critical."
        )
        assert np.all(np.diff(knees_sif) > 0), e_msg
        return knees_sif

    def calc_growth_rate(
        self, stress_, crack_depth_
    ) -> Tuple[float, float, float]:

        """Calculate the crack growth rate given the stress and the
        crack size using the Paris' law:

        .. math::

            \\frac{da}{dN} = \\text{intercept} \\, {\\Delta K)^{\\text{slope}}

        Parameters
        ----------
        stress_ : float
            Stress value.
        crack_depth_ : float
            Crack size value.

        Returns
        -------
        Tuple[float, float, float]
            Crack growth rate, stress intensity factor, and geometry factor.
        """

        sif, g_factor = get_sif(
            stress_, crack_depth_, self.crack_type, self.crack_geometry
        )
        knees = self.get_knees_sif
        if sif < self.threshold:
            the_growth_rate = 0.0
        if sif > self.critical:
            the_growth_rate = np.inf
        else:
            idx = np.digitize(np.asarray([sif]), knees, right=True) - 1
            the_growth_rate = (
                self.intercept[idx][0] * sif ** self.slope[idx][0]
            )

        return the_growth_rate, sif, g_factor

    def calc_crack_depth(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Calculate the crack size and the SIF values.

        Using the assigned stress ranges and count-cycles arrays, the final
        crack size is calculated. The method iterates over the stress ranges
        and count-cycles arrays and updates the crack size at each step.

        .. code-block:: python
            :caption: Pseudocode

            crack_depth = [crack_geometry["initial_depth"]]

            sifs = []
            factors = []
            for stress, count_cycle in zip(stress_range, count_cycle):
                for i in range(count_cycle):
                    sifs.append(get_sif(stress,
                                        crack_depth, self.crack_geometry))
                    factors.append(get_geometry_factor(crack_depth,
                                                       crack_type,
                                                       crack_geometry))
                    if sifs[-1] >= critical:
                        return crack_depth, sifs, factors, True
                    if sifs[-1] < threshold:
                        continue
                    crack_depth.append(crack_depth[-1] +
                                       calc_growth_rate(stress, crack_depth))
            return crack_depth, sifs, factors, False

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, bool]
            Crack size array, SIF array, geometric factor array, failure flag.
        """

        # Initialisation
        the_depth = np.empty(self.size + 1, dtype=np.float64)
        the_depth[0] = self.crack_geometry["initial_depth"]
        the_sif = np.empty(self.size, dtype=np.float64)
        the_factor = np.empty(self.size, dtype=np.float64)
        # Loop
        for i in nb.prange(self.size):  # pylint: disable=E1133
            # the_sif[i], the_factor[i] = get_sif(
            #     self.stress_range[i],
            #     the_depth[i],
            #     self.crack_type,
            #     self.crack_geometry,
            # )
            # if the_sif[i] >= self.critical:
            #     print("Critical SIF reached. Stopping calculation.")
            #     return the_depth[:i], the_sif[:i], the_factor[:i], True
            # if the_sif[i] < self.threshold:
            #     the_depth[i + 1] = the_depth[i]
            #     continue
            incr_depth, the_sif[i], the_factor[i] = self.calc_growth_rate(
                self.stress_range[i], the_depth[i]
            )
            cur_depth = the_depth[i] + incr_depth * self.count_cycle[i]
            if the_factor[i] == 0.0:
                print(
                    "Crack depth is greater than or equal to the thickness."
                    "Stopping calculation."
                )
                return the_depth[:i], the_sif[:i], the_factor[:i], True
            if the_sif[i] >= self.critical:
                print("Critical SIF reached. Stopping calculation.")
                return the_depth[:i], the_sif[:i], the_factor[:i], True
            if the_sif[i] < self.threshold:
                the_depth[i + 1] = the_depth[i]
                continue
            if cur_depth / the_depth[i] > 10:
                w_msg = (
                    "Crack size increased by more than 10x in the"
                    + " last iteration. Stopping calculation."
                )
                print(w_msg)
                return the_depth[:i], the_sif[:i], the_factor[:i], True
            the_depth[i + 1] = cur_depth
        print("Fatigue spectrum applied w/o failure. Stopping calculation")
        return the_depth[:-1], the_sif, the_factor, False


@nb.njit(
    nb.float64(
        nb.float64,
        nb.types.unicode_type,
        nb.types.DictType(nb.types.unicode_type, nb.float64),
    ),
    fastmath=True,
    cache=True,
)
def get_geometry_factor(
    crack_depth: float,
    crack_type: str,
    crack_geometry: dict,  # pylint: disable=W0613, C0301
) -> float:
    """Get the geometric factor. This should be a function of the crack
    size. The default value is 1.0."""
    # Placeholder for future implementation
    if crack_type == "HOL_CYL_01":
        return f_hol_cyl_01(crack_depth, crack_geometry)
    if crack_type == "INF_SUR_00":
        return 1.0

    e_msg = "Geometric factor not implemented for assigned crack type."
    raise NotImplementedError(e_msg)


@nb.njit(
    nb.types.UniTuple(nb.float64, 2)(
        nb.float64,
        nb.float64,
        nb.types.unicode_type,
        nb.types.DictType(nb.types.unicode_type, nb.float64),
    ),
    fastmath=True,
    cache=True,
)
def get_sif(
    stress_: float,
    crack_depth_: float,
    crack_type: str,
    crack_geometry: Dict[list, float],
) -> Tuple[float, float]:
    """Get the SIF value given the stress and the crack size. The
    method calculates the SIF value according to the following
    equation:

    .. math::

        SIF = Y(a) \\cdot \\Delta \\sigma \\cdot \\sqrt{\\pi a}

    Parameters
    ----------
    stress_ : float
        Stress value.
    crack_depth_ : float
        Crack size value.
    crack_type : str
        Crack type.
    crack_geometry : dict
        Crack geometry dictionary.

    Returns
    -------
    Tuple[float, float]
        SIF value, geometric factor.
    """
    y_1 = get_geometry_factor(crack_depth_, crack_type, crack_geometry)
    return stress_ * np.sqrt(np.pi * crack_depth_) * y_1, y_1


def get_crack_growth(
    cycle_count: CycleCount,
    cg_curve: ParisCurve,
    crack_geometry: Type[ACG],
    express_mode: bool = False,
) -> CalcCrackGrowth:
    """Calculate the crack size by numerically integrating the crack growth
    curve. In this example we will consider the Paris' law, but the function
    can be used with any other crack growth curve.

    .. math::

        \\begin{align}
        \\begin{cases}
            \\frac{da}{dN} & = C \\left( \\Delta K \\right)^{m} \\\\
            \\Delta K & = Y(a) \\cdot \\Delta \\sigma \\sqrt{\\pi a} \\\\
        \\end{cases}
        \\end{align}

    From the equations above, the crack growths rate can be expressed as a
    function of the stress range. Considering the Paris' law, as in this
    example, gives:

    .. math::

        \\frac{da}{dN} = C \\left( Y(a) \\cdot \\Delta \\sigma \\sqrt{\\pi a}
            \\right)^{m}

    se this crack growth rate together with the cycle step size to determine
    the incremental growth of the crack for this iteration:

    .. math::

        \\Delta a = \\frac{da}{dN} \\cdot \\Delta N

    The new crack size is then calculated by adding the incremental growth to
    the previous crack size:

    .. math::

        a_{j+1} = a_{j} + \\Delta a

    where :math:`a_{j}` is the crack size at the :math:`j`-th cycle.

    The crack is grown iteratively either until the failure condition is met
    or until the stress history is over.

    If failure occurs, the total number of cycles that it took to grow the
    crack to the critical size becomes the predicted life of the part.

    Parameters
    ----------
    cycle_count : py_fatigue.CycleCount
        Cycle count object.
    cg_curve : py_fatigue.ParisCurve
        The crack growth curve
    crack_geometry : Type[ACG]
        The initial crack geometry
    express_mode : bool, optional
        If True, the crack growth is calculated using the express mode. This
        mode is faster but 'slightly' less accurate. The default is False.

    Returns
    -------
    CalcCrackGrowth
        The crack growth calculation object
    """
    # Handle the stress ranges and cycle counts as they might be clustered
    # in a matrix and we need to iterate over single or "little" cycles
    # within the same stress range block
    if cycle_count.unit not in cg_curve.unit:
        e_msg = (
            f"Cycle count unit ({cycle_count.unit}) and crack growth "
            f"curve unit ({cg_curve.unit}) are not compatible."
        )
        raise ValueError(e_msg)
    if np.max(cycle_count.count_cycle) > 1:
        if express_mode:
            splts = [
                10 ** int(np.log10(cycle)) for cycle in cycle_count.count_cycle
            ]
        else:
            splts = [int(cycle) for cycle in cycle_count.count_cycle]
        splt_cyc = [
            split(int(cycle), n) if int(cycle) > 0 else [cycle]
            for cycle, n in zip(cycle_count.count_cycle, splts)
        ]
        splt_len = [len(_) for _ in splt_cyc]

        count_cycle = np.asarray(
            [float(item) for sublist in splt_cyc for item in sublist]
        )
        stress_range = np.repeat(cycle_count.stress_range, splt_len)
    else:
        stress_range = cycle_count.stress_range
        count_cycle = cycle_count.count_cycle

    # Handle the geometry

    crack_type = str(crack_geometry._id)
    if crack_type not in ["HOL_CYL_01", "INF_SUR_00"]:
        raise ValueError("Unsupported crack geometry")
    geometry = to_numba_dict(crack_geometry.__dict__)
    cg = CalcCrackGrowth(
        stress_range,
        count_cycle,
        cg_curve.slope,
        cg_curve.intercept,
        float(cg_curve.threshold),
        float(cg_curve.critical),
        crack_type,
        geometry,
    )
    # crack_geometry.geometry_factor = cg.geometry_factor
    # crack_geometry.crack_depth_ = cg.crack_depth
    # crack_geometry.sif_ = cg.sif
    # crack_geometry.count_cycle_ = np.cumsum(
    #     cg.count_cycle[: len(cg.crack_depth)]
    # )
    # crack_geometry.final_cycles_ = cg.final_cycles
    return cg  # crack_geometry


@pd.api.extensions.register_dataframe_accessor("cg")
class CrackGrowth:
    """Accessor for the crack growth calculation

    Parameters
    ----------
    pandas_obj : pd.DataFrame
        The pandas dataframe object that will be extended
        with the crack growth calculation results. The original dataframe must
        have the following columns:
        - stress_range
        - count_cycle
        - mean_stress
        The dataframe will be extended with the following columns:
        - crack_depth
        - sif
        - cumul_cycles
        - geometry_factor
    """

    def __init__(self, pandas_obj):
        # self._validate(pandas_obj)
        self._obj = pandas_obj
        self.cg_curve = None
        self.crack_geometry = None
        self.final_cycles = None

    @staticmethod
    def _validate(obj):
        """Validate the input DataFrame. Raise an error if the input
        DataFrame does not contain the right columns.
        """
        if {
            "crack_depth",
        }.issubset(obj.columns):
            e_msg = "'crack_depth' already calculated"
            raise AttributeError(e_msg)

        if not {"count_cycle", "mean_stress", "stress_range"}.issubset(
            obj.columns
        ):
            e_msg = (
                "Must have 'count_cycle', 'mean_stress' and 'stress_range'."
            )
            raise AttributeError(e_msg)

    def calc_growth(
        self,
        cg_curve: ParisCurve,
        crack_geometry: Type[ACG],
        express_mode: bool = False,
    ):
        """Calculate the crack_propagation of the given crack geometry.

        Parameters
        ----------
        cg_curve : py_fatigue.ParisCurve
            The crack growth curve
        crack_geometry : Type[ACG]
            The initial crack geometry
        express_mode : bool, optional
            If True, the crack propagation runs 'express', by default False

        Returns
        -------
        DataFrame
            The DataFrame with the results of the crack_growth analysis.
        """

        self._validate(self._obj)

        # Handle the stress ranges and cycle counts as they might be clustered
        # in a matrix and we need to iterate over single or "little" cycles
        # within the same stress range block

        if np.max(self._obj["count_cycle"]) > 1:
            if express_mode:
                splts = [
                    10 ** int(np.log10(cycle))
                    for cycle in self._obj["count_cycle"].values
                ]
            else:
                splts = [
                    int(cycle) for cycle in self._obj["count_cycle"].values
                ]
            splt_cyc = [
                split(int(cycle), n) if int(cycle) > 0 else [cycle]
                for cycle, n in zip(self._obj["count_cycle"].values, splts)
            ]
            splt_len = [len(_) for _ in splt_cyc]

            count_cycle = np.asarray(
                [float(item) for sublist in splt_cyc for item in sublist]
            )
            stress_range = np.repeat(
                self._obj["stress_range"].values, splt_len
            )
        else:
            stress_range = self._obj["stress_range"].values
            count_cycle = self._obj["count_cycle"].values

        # Handle the geometry
        crack_type = str(crack_geometry._id)
        if crack_type not in ["HOL_CYL_01", "INF_SUR_00"]:
            raise ValueError("Unsupported crack geometry")
        geometry = to_numba_dict(crack_geometry.__dict__)

        cg_ = CalcCrackGrowth(
            stress_range,
            count_cycle,
            cg_curve.slope,
            cg_curve.intercept,
            float(cg_curve.threshold),
            float(cg_curve.critical),
            crack_type,
            geometry,
        )

        self._obj.final_cycles = cg_.final_cycles
        self.final_cycles = cg_.final_cycles

        cumul_cycle = np.cumsum(cg_.count_cycle[: len(cg_.crack_depth)])
        # manage the fact that clustered cycles are split in cg analysys
        # while the df keeps them clustered
        d_l = len(self._obj) - len(cg_.crack_depth)
        if d_l >= 0:
            self._obj["crack_depth"] = np.concatenate(
                (cg_.crack_depth, np.full(d_l, np.nan))
            )
            self._obj["sif"] = np.concatenate((cg_.sif, np.full(d_l, np.nan)))
            self._obj["cumul_cycle"] = np.concatenate(
                (cumul_cycle, np.full(d_l, np.nan))
            )
            self._obj["geometry_factor"] = np.concatenate(
                (cg_.geometry_factor, np.full(d_l, np.nan))
            )
        else:
            idx = np.asarray(self._obj["count_cycle"].cumsum(), dtype=int) - 1
            self._obj["crack_depth"] = cg_.crack_depth[idx]
            self._obj["sif"] = cg_.sif[idx]
            self._obj["cumul_cycle"] = cumul_cycle[idx]
            self._obj["geometry_factor"] = cg_.geometry_factor[idx]
        self._obj.cg_curve = cg_curve
        self.cg_curve = cg_curve
        self._obj.crack_geometry = crack_geometry
        self.crack_geometry = crack_geometry
        return self._obj
