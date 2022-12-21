# -*- coding: utf-8 -*-

r"""The :mod:`py_fatigue.py_fatigue` module collects all functions related to
:term:`cycle-counting(s)<Cycle-counting>` definition.
The main class is CycleCount.
"""

# Standard imports
from __future__ import annotations
import copy
import datetime as dt
import itertools
from dataclasses import dataclass
from time import sleep
from types import SimpleNamespace
from typing import (
    Any,
    Sequence,
    cast,
    Callable,
    Iterable,
    Optional,
    Union,
    NewType,
    Tuple,
    TypeVar,
)

# import io
# import itertools
import warnings

# Non-standard imports
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from py_fatigue.cycle_count import calc_rainflow
import py_fatigue.cycle_count.histogram as ht
import py_fatigue.utils as pfu
import py_fatigue.mean_stress.corrections as msc
from py_fatigue.mean_stress import MeanStress
from py_fatigue.stress_range import StressRange

from py_fatigue.styling import TermColors, py_fatigue_formatwarning

__all__ = ["CycleCount"]


NDT = NewType("NDT", dt.datetime)  # non-aware datetime
ADT = NewType("ADT", dt.datetime)  # timezone aware datetime
DatetimeLike = TypeVar("DatetimeLike", ADT, NDT)


def _assess_json_keys(data: dict) -> dict:
    """Assess the keys in a JSON file.

    Parameters
    ----------
    data : dict
        Dictionary to assess

    Returns
    -------
    dict
        Dictionary with the same keys as the input
    """

    allowed = (
        "bin_lb",
        "range_bin_lower_bound",
        "bin_width",
        "range_bin_width",
        "mean_bin_lower_bound",
        "mean_bin_width",
        "res",
        "no_sm_c",
        "nr_small_cycles",
        "res_sig",
        "hist",
        "lg_c",
        "as_dur",
    )

    for k in data:
        if k not in allowed:
            raise KeyError(f"{k} is not an allowed key")
    if "range_bin_lower_bound" in data:
        if "bin_lb" in data:
            key_err = "'bin_lb' and 'range_bin_lower_bound' are both present"
            raise KeyError(key_err)
    if "bin_lb" in data:
        data["range_bin_lower_bound"] = data.pop("bin_lb")
        data["range_bin_width"] = data.pop("bin_width")

    if "mean_bin_lower_bound" not in data:
        data["mean_bin_lower_bound"] = None

    if "no_sm_c" in data and "nr_small_cycles" not in data:
        data["nr_small_cycles"] = data.pop("no_sm_c")

    # try:
    #     data["res"]
    # except KeyError as key_error:
    #     raise KeyError("'res' is not defined in data") from key_error
    # try:
    #     data["lg_c"]
    # except KeyError as key_error:
    #     raise KeyError("'lg_c' is not defined in data") from key_error
    try:
        data["res_sig"]
    except KeyError:
        data["res_sig"] = None

    return data


def _build_input_data_from_json(  # noqa: C901
    data: dict,
    timestamp: Union[ADT, NDT] = ADT(dt.datetime.now(dt.timezone.utc)),
    round_decimals: int = 4,
    name: Optional[str] = None,
    mean_stress_corrected: str = "No",
    lffd_solved: bool = False,
) -> dict:
    """Function returning the same data it receives"""
    data = _assess_json_keys(data)
    the_hist = np.empty(0)
    range_bin_centers = np.empty(0)
    mean_bin_centers = np.empty(0)
    if "hist" not in data:
        data["hist"] = []
    if (
        len(data["hist"]) == 0
        and len(data["res"]) == 0
        and len(data["lg_c"]) == 0
    ):
        raise ValueError("'hist' is empty")
    for k, v in data.items():
        if k != "hist":
            data[k] = np.array(v) if isinstance(v, list) else v
    for hst in data["hist"]:
        hst = np.array(hst)
    # A) Case without mean stress
    # Adding hist cycles
    if len(data["hist"]) > 0 and not isinstance(data["res"][0], Iterable):
        the_hist = np.asarray(data["hist"])
        zero_hist_counts = np.where(the_hist == 0)[0]
        range_bin_centers = np.around(
            np.linspace(
                data["range_bin_lower_bound"] + data["range_bin_width"] / 2,
                data["range_bin_lower_bound"]
                + data["range_bin_width"] / 2
                + data["range_bin_width"] * (len(data["hist"]) - 1),
                len(data["hist"]),
            ),
            round_decimals,
        )
        the_hist = np.delete(the_hist, zero_hist_counts)
        range_bin_centers = np.delete(range_bin_centers, zero_hist_counts)
    # Adding large cycles
    if len(data["lg_c"]) > 0 and not isinstance(
        data["res"][0],
        Iterable  # isinstance(
        # data["lg_c"][0], (float, int, np.int16, np.int32)
    ):
        the_hist = np.hstack([the_hist, np.ones(len(data["lg_c"]))])
        range_bin_centers = np.hstack([range_bin_centers, data["lg_c"]])
    # Adding residual cycles
    if len(data["res"]) > 0 and not isinstance(data["res"][0], Iterable):
        the_hist = np.hstack([the_hist, 0.5 * np.ones(len(data["res"]))])
        range_bin_centers = np.hstack([range_bin_centers, data["res"]])
    # B) Case with mean stress
    if len(data["hist"]) > 0 and isinstance(data["hist"][0], Iterable):
        the_hist = np.asarray(
            list(itertools.chain.from_iterable(data["hist"]))
        )
        zero_hist_counts = np.where(the_hist == 0)[0]
        for idx in range(len(data["hist"])):
            len_mean_bin = len(data["hist"][idx])
            new_ranges = np.around(
                np.linspace(
                    data["range_bin_lower_bound"]
                    + data["range_bin_width"] / 2,
                    data["range_bin_lower_bound"]
                    + data["range_bin_width"] / 2
                    + data["range_bin_width"] * (len_mean_bin - 1),
                    len_mean_bin,
                ),
                round_decimals,
            )
            range_bin_centers = np.hstack([range_bin_centers, new_ranges])
            new_means = (
                data["mean_bin_lower_bound"]
                + idx * data["mean_bin_width"] * np.ones(len_mean_bin)
                + data["mean_bin_width"] / 2
            )
            mean_bin_centers = np.hstack([mean_bin_centers, new_means])

        the_hist = np.delete(the_hist, zero_hist_counts)
        range_bin_centers = np.delete(range_bin_centers, zero_hist_counts)
        mean_bin_centers = np.delete(mean_bin_centers, zero_hist_counts)
    # Adding large cycles
    if len(data["lg_c"]) > 0 and isinstance(data["lg_c"][0], Iterable):
        the_hist = np.hstack([the_hist, np.ones(len(data["lg_c"][:, 0]))])
        mean_bin_centers = np.hstack([mean_bin_centers, data["lg_c"][:, 0]])
        range_bin_centers = np.hstack([range_bin_centers, data["lg_c"][:, 1]])
    # Adding residual cycles
    if len(data["res"]) > 0 and isinstance(data["res"][0], Iterable):
        the_hist = np.hstack([the_hist, 0.5 * np.ones(len(data["res"][:, 0]))])
        mean_bin_centers = np.hstack([mean_bin_centers, data["res"][:, 0]])
        range_bin_centers = np.hstack([range_bin_centers, data["res"][:, 1]])

    df = {
        "count_cycle": the_hist,
        "mean_stress": mean_bin_centers
        if "mean_bin_width" in data
        else np.zeros(len(range_bin_centers)),
        "stress_range": range_bin_centers,
        "range_bin_lower_bound": data["range_bin_lower_bound"],
        "range_bin_width": data["range_bin_width"],
        "_mean_bin_lower_bound": data["mean_bin_lower_bound"]
        if "mean_bin_lower_bound" in data
        else None,
        "mean_bin_width": data["mean_bin_width"]
        if "mean_bin_width" in data
        else 10,
        "nr_small_cycles": data["nr_small_cycles"],
        "residuals_sequence": data["res_sig"],
        "timestamp": timestamp,
        "name": name,
        "mean_stress_corrected": mean_stress_corrected,
        "lffd_solved": lffd_solved,
    }
    return df


@dataclass(repr=False)
class CycleCount:
    """Cycle-counting class.

    The class heavily relies on the :mod:`py_fatigue.rainflow` and
    :mod:`py_fatigue.histogram` modules.

    CycleCount instances should be defined from either of the
    following two class methods:

        - :func:`~CycleCount.from_rainflow`
        - :func:`~CycleCount.from_timeseries`
    """

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-public-methods
    # pylint: disable=too-many-arguments
    # 16 is reasonable in this case.

    count_cycle: np.ndarray[Any, np.dtype[np.float64]]
    stress_range: np.ndarray[Any, np.dtype[np.float32]]
    mean_stress: np.ndarray[Any, np.dtype[np.float64]]
    timestamp: Union[ADT, NDT] = ADT(dt.datetime.now(dt.timezone.utc))
    unit: str = "MPa"
    range_bin_lower_bound: float = 0.2
    range_bin_width: float = 0.05
    _mean_bin_lower_bound: Optional[float] = None
    mean_bin_width: float = 10
    nr_small_cycles: float = 0
    residuals_sequence: np.ndarray[Any, Any] = np.empty(0)
    _min_max_sequence: np.ndarray[Any, Any] = np.empty(0)
    lffd_solved: bool = False
    mean_stress_corrected: str = "No"
    stress_concentration_factor: float = 1.0
    name: Optional[str] = None
    _time_sequence: Optional[np.ndarray] = None

    def __post_init__(self):
        if not np.isnan(np.min(self.mean_stress)):
            if self._mean_bin_lower_bound is None:
                self._mean_bin_lower_bound = (
                    round(min(self.mean_stress) / self.mean_bin_width)
                    * self.mean_bin_width
                    - self.mean_bin_width / 2
                )
            if self.time_sequence is None:
                self.time_sequence = np.asarray(self.timestamp)

    @classmethod
    def from_rainflow(
        cls,
        data: dict,
        timestamp: Union[ADT, NDT] = ADT(dt.datetime.now(dt.timezone.utc)),
        round_decimals: int = 4,
        name: Optional[str] = None,
        mean_stress_corrected: str = "No",
        lffd_solved: bool = False,
    ) -> "CycleCount":
        """Create a cycle-counting object from rainflow cycles.
        Parameters
        ----------
        *args
            Positional arguments passed to `py_fatigue.rainflow.rainflow`.
        **kwargs
            Keyword arguments passed to `py_fatigue.rainflow.rainflow`.
        Returns
        -------
        CycleCount
            A cycle-counting object.
        """
        df = _build_input_data_from_json(
            data,
            timestamp,
            round_decimals,
            name,
            mean_stress_corrected,
            lffd_solved,
        )

        return cls(**df)

    @classmethod
    def from_timeseries(
        cls,
        data: np.ndarray,
        time: Optional[np.ndarray] = None,
        timestamp: Union[ADT, NDT] = ADT(dt.datetime.now(dt.timezone.utc)),
        range_bin_lower_bound: float = 0.2,
        range_bin_width: float = 0.05,
        mean_bin_lower_bound: Optional[float] = None,
        mean_bin_width: float = 10,
        name: Optional[str] = None,
    ) -> "CycleCount":
        """Generate a cycle-count from a timeseries.

        Parameters
        ----------
        *args
            Positional arguments passed to `_build_input_data_from_json`.
        **kwargs
            Keyword arguments passed to `_build_input_data_from_json`.
        """
        cc = calc_rainflow(data=data, time=time, extended_output=True)
        stress_range = 2 * cc[0][:, 0]
        mean_stress = cc[0][:, 1]
        count_cycle = cc[0][:, 2]
        residuals_sequence = cc[1]
        df = {
            "count_cycle": count_cycle,
            "mean_stress": mean_stress,
            "stress_range": stress_range,
            "timestamp": timestamp,
            "range_bin_lower_bound": range_bin_lower_bound,
            "range_bin_width": range_bin_width,
            "_mean_bin_lower_bound": mean_bin_lower_bound,
            "mean_bin_width": mean_bin_width,
            "residuals_sequence": residuals_sequence,
            "name": name,
        }
        return cls(**df)

    # ! Instance managed attributes
    @property
    def mean_bin_lower_bound(self) -> float:
        """The stress values.

        Returns
        -------
        values : np.ndarray
        """
        return cast(float, self._mean_bin_lower_bound)

    @mean_bin_lower_bound.setter
    def mean_bin_lower_bound(self, value: float) -> None:
        self._mean_bin_lower_bound = value

    @property
    def time_sequence(self) -> np.ndarray:
        """The sequence of timestamps possibly added.

        Returns
        -------
        time_sequence : np.ndarray
        """
        if self._time_sequence is None:
            return np.asarray([self.timestamp])
        return self._time_sequence

    @property
    def min_max_sequence(self) -> np.ndarray:
        """The minimum and maximum of the :term:`residuals<Residuals>`
        (half-cycles) sequence.

        Returns
        -------
        np.ndarray, shape=(2, )
            The minimum and maximum of the half-cycles sequence.
        """
        if self.residuals_sequence is None:
            return self._min_max_sequence
        if (
            len(self._min_max_sequence) == 0
            and len(self.residuals_sequence) >= 2
        ):
            return np.hstack(
                [
                    min(self.residuals_sequence),
                    max(self.residuals_sequence),
                ]
            )
        return self._min_max_sequence

    @property
    def residuals(self) -> np.ndarray:
        """Extracts the :term:`residuals<Residuals>`, also known as half
        cycles, from mean-range couples of the cycle-counting object.

        Returns
        -------
        np.ndarray, shape (n, 2)
            Array containing mean stresses and stress ranges of the residuals.
            - residual_mean_stresses = array[:, 0]
            - residual_stress_ranges = array[:, 1]
        """
        _, idx = pfu.split_full_cycles_and_residuals(self.count_cycle)
        return np.asarray([self.mean_stress[idx], self.stress_range[idx]]).T

    @property
    def full_cycles(self) -> np.ndarray:
        """Extracts the full cycles only from mean-range couples.

        Returns
        -------
        np.ndarray, shape (n, 2)
            Array containing mean stresses and stress ranges of the full
            cycles.
            - full_cycles_mean_stresses = array[:, 0]
            - full_cycles_stress_ranges = array[:, 1]
        """
        full_mean = pfu.calc_full_cycles(self.mean_stress, self.count_cycle)
        full_range = pfu.calc_full_cycles(self.stress_range, self.count_cycle)
        return np.asarray([full_mean, full_range]).T

    @property
    def half_cycles(self) -> np.ndarray:
        """Symlink to residuals

        See also
        --------
        :func:`py_fatigue.cycle_count.CycleCount.residuals`
        """
        return self.residuals

    @property
    def mean_bin_upper_bound(self) -> float:
        """Returns the upper bound of the mean stress bin edges.

        Returns
        -------
        float
            The upper bound of the mean stress bins (highest mean bin edge).
        """
        return pfu.bin_upper_bound(
            self.mean_bin_lower_bound,
            self.mean_bin_width,
            max(self.full_cycles[:, 0]),
        )

    @property
    def range_bin_upper_bound(self) -> float:
        """Returns the upper bound of the stress range bins.

        Returns
        -------
        float
            The upper bound of the stress range bins (highest range bin edge).
        """
        return pfu.bin_upper_bound(
            self.range_bin_lower_bound,
            self.range_bin_width,
            max(self.full_cycles[:, 1]),
        )

    @property
    def bin_centers(self) -> tuple:
        """Extracts the bin centers from the cycle-counting object.

        Returns
        -------
        tuple
            Tuple containing mean stresses and stress ranges of the bin
            centers.
            - bin_centers_mean_stresses = tuple[0]
            - bin_centers_stress_ranges = tuple[1]
        """
        mean_bin_centers = (
            pfu.calc_bin_edges(
                self.mean_bin_lower_bound,
                self.mean_bin_upper_bound,
                self.mean_bin_width,
            )[:-1]
            + self.mean_bin_width / 2
        )

        range_bin_centers = (
            pfu.calc_bin_edges(
                self.range_bin_lower_bound,
                self.range_bin_upper_bound,
                self.range_bin_width,
            )[:-1]
            + self.range_bin_width / 2
        )
        return mean_bin_centers, range_bin_centers

    @property
    def bin_edges(self) -> tuple:
        """Extracts the bin edges from the cycle-counting object.

        Returns
        -------
        ht.tuple
            Tuple containing mean stresses and stress ranges of the bin edges.
            - bin_edges_mean_stresses = tuple[0]
            - bin_edges_stress_ranges = tuple[1]
        """
        mean_bin_edges = pfu.calc_bin_edges(
            self.mean_bin_lower_bound,
            self.mean_bin_upper_bound,
            self.mean_bin_width,
        )

        range_bin_edges = pfu.calc_bin_edges(
            self.range_bin_lower_bound,
            self.range_bin_upper_bound,
            self.range_bin_width,
        )
        return mean_bin_edges, range_bin_edges

    @property
    def stress_amplitude(self) -> np.ndarray:
        """Extracts the stress amplitude from the cycle-counting object.

        Returns
        -------
        np.ndarray
            Array containing the stress amplitude.
        """
        return self.stress_range / 2

    @property
    def min_stress(self) -> np.ndarray:
        """Extracts the min stress from the cycle-counting object.

        Returns
        -------
        np.ndarray
            Array containing the min stresses.
        """
        return self.mean_stress - self.stress_amplitude

    @property
    def max_stress(self) -> np.ndarray:
        """Extracts the max stress from the cycle-counting object.

        Returns
        -------
        np.ndarray
            Array containing the max stresses.
        """
        return self.mean_stress + self.stress_amplitude

    @property
    def statistical_moments(self) -> tuple[float, float, float]:
        """Calculate the spectral moments from the cycle-counting object.

        Returns
        -------
        tuple[float, float, float]
            Tuple containing the spectral moments.
            - mean = tuple[0]
            - coefficient of variance = tuple[1]
            - skewness = tuple[2]
        """
        stress_range = np.hstack(
            [self.half_cycles[:, 1], self.full_cycles[:, 1]]
        )
        m_1 = np.mean(stress_range)
        std_dev = np.sqrt(np.mean((stress_range - np.mean(stress_range)) ** 2))
        m_2 = std_dev / m_1
        m_3 = np.mean((stress_range - np.mean(stress_range)) ** 3) / (m_1**3)

        return m_1, m_2, m_3

    # ! Representations
    def as_dict(
        self,
        damage_tolerance_for_binning: float = 0.001,
        damage_exponent: float = 5.0,
        round_decimals: int = 4,
        max_consecutive_zeros: int = 32,
        legacy_export: bool = False,
        debug_mode=False,
    ) -> dict:
        """Convert the cycle-counting object to a dictionary.

        Parameters
        ----------
        damage_tolerance_for_binning : float, optional
            Tolerance for large-cycles calculation when binning,
            by default 0.001
        damage_exponent : float, optional
            SN curve exponent used to define large cycles. Higher means more
            conservative, by default 5.0
        max_consecutive_zeros : int, optional
            maximum number of consecutive zeros before saving a cycle
            to large cycles array. This quantity is introduced to save
            storage space, by default 8
        round_decimals : int, optional
            Number of decimals after comma, by default 4

        Returns
        -------
        dict
            Dictionary containing the following keys:
            - "hist"
            - "range_bin_lower_bound"
            - "range_bin_width"
            - "mean_bin_lower_bound"
            - "mean_bin_width"
            - "nr_small_cycles"
            - "res_sig"
            - "res"
            - "lg_c"

        See also
        --------
        :func:`py_fatigue.cycle_count.rainflow_binner`
        """
        stress_ranges = StressRange(
            _counts=self.count_cycle,
            _values=self.stress_range,
            bin_width=self.range_bin_width,
            _bin_lb=self.range_bin_lower_bound,
        )

        mean_stresses = MeanStress(
            _counts=self.count_cycle,
            _values=self.mean_stress,
            bin_width=self.mean_bin_width,
            _bin_lb=self.mean_bin_lower_bound,
        )

        output_dict = ht.rainflow_binner(
            stress_ranges,
            mean_stresses if not legacy_export else None,
            damage_tolerance_for_binning=damage_tolerance_for_binning,
            damage_exponent=damage_exponent,
            max_consecutive_zeros=max_consecutive_zeros,
            round_decimals=round_decimals,
            debug_mode=debug_mode,
        )
        output_dict["res_sig"] = np.asarray(
            np.round(self.residuals_sequence, round_decimals)
        ).tolist()
        return output_dict

    def _repr_html_(self) -> str:  # pragma: no cover
        """HTML representation of the cycle-counting object.

        Returns
        -------
        str
        """
        df = pd.DataFrame()
        df["Cycle counting object"] = [
            f"largest full stress range, {self.unit}",
            f"largest stress range, {self.unit}",
            "number of full cycles",
            "number of residuals",
            "number of small cycles",
            "stress concentration factor",
            "residuals resolved",
            "mean stress-corrected",
        ]

        df[self.name] = [
            max(self.full_cycles[:, 1])
            if max(self.count_cycle) > 0.5
            and max(self.full_cycles[:, 1]) > self.range_bin_lower_bound
            else None,
            max(self.stress_range),
            int(sum(self.count_cycle) - int(len(self.residuals[:, 1])) / 2),
            int(len(self.residuals[:, 1])),
            int(self.nr_small_cycles),
            "N/A"
            if self.stress_concentration_factor == 1
            else self.stress_concentration_factor,
            bool(self.lffd_solved),
            self.mean_stress_corrected,
        ]
        df = df.set_index("Cycle counting object")

        return df._repr_html_()  # pylint: disable=protected-access

    def __str__(self) -> str:
        """String representation of the cycle-counting object.

        Returns
        -------
        str
        """
        strng = f"CC_{self.name}"
        if len(self.time_sequence) == 1:
            strng = "".join(
                (
                    f"CC_{self.name} (",
                    f"{self.timestamp.strftime('%d %b %Y, %H:%M')} - ",
                    "",
                )
            )
        if len(self.time_sequence) > 1:
            strng = "".join(
                (
                    f"CC_ {self.name} (from ",
                    f"{self.time_sequence[0].strftime('%d %b %Y, %H:%M')} to ",
                    f"{self.time_sequence[-1].strftime('%d %b %Y, %H:%M')})",
                )
            )
        return str(strng)

    def to_df(self) -> pd.DataFrame:  # pragma: no cover
        """
        Represent the :class:`CycleCount` instance as a pandas
        dataframe.

        Returns
        -------
        :class:`pd.DataFrame`
            dataframe containing relevant fatigue information
        """

        df = pd.DataFrame.from_dict(
            {
                "count_cycle": self.count_cycle,
                "mean_stress": self.mean_stress,
                "stress_range": self.stress_range,
            }
        )
        df._metadata = {
            "name": self.name,
            "timestamp": self.timestamp,
            "time_sequence": self.time_sequence,
            "residuals_sequence": self.residuals_sequence,
            "mean_stress_corrected": self.mean_stress_corrected,
            "stress_concentration_factor": self.stress_concentration_factor,
            "nr_small_cycles": self.nr_small_cycles,
            "lffd_solved": self.lffd_solved,
            "unit": self.unit,
        }

        return df

    # ! Operators
    def __eq__(self, other: object) -> bool:
        """Equality operation for :class:`CycleCount` instances.

        Parameters
        ----------
        other : object
            object to compare to

        Returns
        -------
        bool
            True if the objects are equal, False otherwise
        """
        if not isinstance(other, CycleCount):
            return False

        return pfu.compare(self.__dict__, other.__dict__)

    def __add__(  # fgh - noqa: _C901_
        self, other: "CycleCount"
    ) -> "CycleCount":
        """Sum operation for :class:`CycleCount` instances.

        Parameters
        ----------
        other : Union[CycleCount, None]
            CycleCount object to be added to the current object.

        Returns
        -------
        CycleCount
            Summed cycle-count object.

        Raises
        ------
        TypeError
            Trying to add a non-CycleCount object.
        TypeError
            Trying to add a CycleCount object with different names.
        ValueError
            Timestamps must be defined for CycleCount sum.
        UserWarning
            Different bin widths.
        TypeError
            Timestamps must be sorted ascending.
        UserWarning
            Adding-up instances having different stress concentration factors.
        TypeError
            residuals_sequence is not iterable.
        UserWarning
            No residuals_sequence attribute found.
        """

        # Running initial checks
        if not _cycle_count_add_checks(self, other):
            return self

        # Performing actual sum
        # added_counts = np.hstack([self.count_cycle, other.count_cycle])
        # added_range = np.hstack([self.stress_range, other.stress_range])
        # added_mean = np.hstack([self.mean_stress, other.mean_stress])
        (
            added_mean_bin_lower_bound,
            added_mean_bin_width,
            added_range_bin_lower_bound,
            added_range_bin_width,
        ) = _handling_different_bins_in_sum(self, other)

        # added_nr_small_cycles = self.nr_small_cycles + other.nr_small_cycles
        # added_timestamp = np.hstack([self.timestamp, other.timestamp])
        added_scf = np.average(
            [
                self.stress_concentration_factor,
                other.stress_concentration_factor,
            ]
        )
        # added_as_duration = self.as_duration + other.as_duration
        # ? Residuals sequence management
        # 1) Residuals sequence is defined in both self and other
        if hasattr(self, "residuals_sequence") and hasattr(
            other, "residuals_sequence"
        ):
            # 1a) Residuals sequence is ok in both self and other
            if isinstance(self.residuals_sequence, Iterable) and isinstance(
                other.residuals_sequence, Iterable
            ):
                added_res_seq = np.hstack(
                    [self.residuals_sequence, other.residuals_sequence]
                )
                added_min_max = np.hstack(
                    [self.min_max_sequence, other.min_max_sequence]
                )
            # 1b) Residuals sequence is ok in self, but not in other
            elif isinstance(
                self.residuals_sequence, Iterable
            ) and not isinstance(other.residuals_sequence, Iterable):
                w_msg = "".join(
                    (
                        f"No residuals_sequence found on {TermColors.CRED}",
                        f"{other.timestamp.strftime('%d %B %Y, %H:%M')}",
                        f"{TermColors.CYELLOW2}.",
                    )
                )
                warnings.formatwarning = py_fatigue_formatwarning
                warnings.warn(w_msg, UserWarning)
                added_res_seq = self.residuals_sequence
                added_min_max = self.min_max_sequence
            # 1c) Residuals sequence is ok in other, but not in self
            elif not isinstance(
                self.residuals_sequence, Iterable
            ) and isinstance(other.residuals_sequence, Iterable):
                w_msg = "".join(
                    (
                        f"No residuals_sequence found on {TermColors.CRED}",
                        f"{self.timestamp.strftime('%d %B %Y, %H:%M')}",
                        f"{TermColors.CYELLOW2}.",
                    )
                )
                warnings.formatwarning = py_fatigue_formatwarning
                warnings.warn(w_msg, UserWarning)
                added_res_seq = other.residuals_sequence
                added_min_max = other.min_max_sequence
            # 1d) Residuals sequence is neither defined in self nor in other
            else:
                w_msg = "No residuals_sequence attribute found."
                warnings.formatwarning = py_fatigue_formatwarning
                warnings.warn(w_msg, UserWarning)
                added_res_seq = np.empty(0)
                added_min_max = np.empty(0)
        # 2) Residuals sequence is defined in self but not in other
        elif hasattr(self, "residuals_sequence") and not hasattr(
            other, "residuals_sequence"
        ):
            w_msg = "".join(
                (
                    f"No residuals_sequence found on {TermColors.CRED}",
                    f"{other.timestamp.strftime('%d %B %Y, %H:%M')}",
                    f"{TermColors.CYELLOW2}.",
                )
            )
            warnings.formatwarning = py_fatigue_formatwarning
            warnings.warn(w_msg, UserWarning)
            added_res_seq = self.residuals_sequence
            added_min_max = self.min_max_sequence
        # 3) Residuals sequence is defined in other but not in self
        elif not hasattr(self, "residuals_sequence") and hasattr(
            other, "residuals_sequence"
        ):
            w_msg = "".join(
                (
                    f"No residuals_sequence found on {TermColors.CRED}",
                    f"{self.timestamp.strftime('%d %B %Y, %H:%M')}",
                    f"{TermColors.CYELLOW2}.",
                )
            )
            warnings.formatwarning = py_fatigue_formatwarning
            warnings.warn(w_msg, UserWarning)
            added_res_seq = other.residuals_sequence
            added_min_max = other.min_max_sequence
        # 4) Neither self nor other have Residuals sequence
        else:
            w_msg = "No residuals_sequence attribute found."
            warnings.formatwarning = py_fatigue_formatwarning
            warnings.warn(w_msg, UserWarning)
            added_res_seq = np.empty(0)
            added_min_max = np.empty(0)

        added_cc = CycleCount(
            count_cycle=np.hstack([self.count_cycle, other.count_cycle]),
            mean_stress=np.hstack([self.mean_stress, other.mean_stress]),
            stress_range=np.hstack([self.stress_range, other.stress_range]),
            range_bin_lower_bound=added_range_bin_lower_bound,
            range_bin_width=added_range_bin_width,
            _mean_bin_lower_bound=added_mean_bin_lower_bound,
            mean_bin_width=added_mean_bin_width,
            nr_small_cycles=self.nr_small_cycles + other.nr_small_cycles,
            residuals_sequence=added_res_seq,
            _min_max_sequence=added_min_max,
            lffd_solved=False,
            mean_stress_corrected="No",
            stress_concentration_factor=float(added_scf),
            timestamp=min(self.timestamp, other.timestamp),
            name=self.name,
            _time_sequence=np.unique(
                np.hstack([self.time_sequence, other.time_sequence])
            ),
        )
        return added_cc

    def __radd__(self, other):
        """
        Reverse add function.
        """
        if other is None or other == 0 or np.isnan(other):
            return self
        return self.__add__(other)

    def __mul__(self, other: Union[float, int]):
        """Left multiplication of :class:`CycleCount` \
        instance by a scalar stress multiplication factor .

        Parameters
        ----------
        other : Union[float, int]
            stress concentration factor

        Raises
        ------
        TypeError: Multiplication by a scalar is the only thing \
        allowed

        Returns
        -------
        :class:`CycleCount`
            :class:`CycleCount` instance with \
        "raised" stresses and stress concentration factor != 1
        """
        return _multiplication_by_scalar(self, other)

    def __rmul__(self, other):
        """Left multiplication of :class:`CycleCount` \
        instance by a scalar stress multiplication factor.

        Parameters
        ----------
        other : Union[float, int]
            stress concentration factor

        Raises
        ------
        TypeError: Multiplication by a scalar is the only thing \
        allowed

        Returns
        -------
        :class:`CycleCount`
            :class:`CycleCount` instance with \
        "raised" stresses and stress concentration factor != 1
        """
        return _multiplication_by_scalar(self, other)

    def solve_lffd(self, solve_mode: str = "Residuals") -> "CycleCount":
        """Resolve the LFFD using Marsh et al. (2016) method.

        Parameters
        ----------
        solve_mode : str
            Solve mode. Options are: "Residuals" and "Min-Max".

        Returns
        -------
        :class:`CycleCount`
            :class:`CycleCount` instance with :term:`LFFD` resolved.
        """
        return _solve_lffd(self, solve_mode)

    def resolve_residuals(self) -> "CycleCount":
        """Resolve the residuals. It is a symlink to :meth:`solve_lffd`.

        Returns
        -------
        :class:`CycleCount`
            :class:`CycleCount` instance with :term:`LFFD` resolved.

        See also
        --------
        :meth:`solve_lffd`
        """
        return self.solve_lffd()

    def mean_stress_correction(
        self,
        correction_type: str = "DNVGL-RP-C203",
        plot: bool = False,
        enforce_pulsating_load: bool = False,
        **kwargs,
    ) -> "CycleCount":
        """Calculates the mean stress correction returning a new
        :class:`CycleCount` instance with corrected stress ranges
        at :math:`R=\\sigma_{min}/\\sigma_{max}=0`.

        Parameters
        ----------
        correction_type : str, optional
            Type of mean stress correction, by default "DNVGL-RP-C203"
        plot : bool, optional
            Plot the mean stress corrected history, by default False
        enforce_pulsating_load : bool, optional
            Choosing to enforce load ratio equals zero prior
            correction (useful in legacy analyses), by default False

        `correction_types` : list
            * *dnvgl-rp-c203*
            * *Goodman* (not included yet)

        **kwargs : dict
            *detail_factor* : float
                See DNVGL-RP-C203, by default 0.8
            *yield_strength* : float
                Yielding stress amplitude

        Returns
        -------
        CycleCount
            :class:`CycleCount` instance with corrected stress ranges

        Raises
        ------
        ValueError
            `correction_type` must be one of `correction_types`.
        ValueError
            Mean stress correction will not be applied because the
            initial mean stress array is made of zeros. Set
            enforce_load_ratio to True to proceed.
        ValueError
            This correction type is not supported.

        See also
        --------
        :func:`py_fatigue.mean_stress.corrections.dnvgl_mean_stress_correction`
        """
        correction_types = [
            "dnvgl-rp-c203",
            "walker",
            "swt",
            "smith-watson-topper",
        ]
        if correction_type.lower() not in correction_types:
            raise ValueError(
                f"'correction_type' must be one of {correction_types}"
            )
        if self.mean_stress_corrected != "No":
            w_msg = "".join(
                (
                    "Cyclecount instance has already been corrected using ",
                    f"{TermColors.CBEIGE2}{self.mean_stress_corrected}",
                    TermColors.CYELLOW2,
                    ".",
                )
            )
            warnings.formatwarning = py_fatigue_formatwarning
            warnings.warn(w_msg, UserWarning)
            return self

        self_ = copy.deepcopy(self) if self.lffd_solved else self.solve_lffd()

        if np.all((self_.mean_stress == 0)) and not enforce_pulsating_load:
            e_msg = "".join(
                (
                    f"{correction_type} mean stress correction will not be ",
                    "applied because the mean stress array is made of zeros. ",
                    "Set 'enforce_pulsating_load=True' to override mean ",
                    "stresses to pulsating loading (zero load ratio).",
                )
            )
            raise ValueError(e_msg)
        if np.all((self_.mean_stress == 0)) and enforce_pulsating_load:
            self_.mean_stress = self_.stress_amplitude

        ns = SimpleNamespace(**kwargs)
        if correction_type.lower() == ("dnvgl-rp-c203"):
            if "detail_factor" not in kwargs:
                w_msg = "".join(
                    (
                        "No detail factor provided for DNVGL-RP-C203 ",
                        "mean stress correction. \nPassing detail_factor=0.8 ",
                        "This value is suitable for welded connections ",
                        "without significant residual stresses.",
                    )
                )
                warnings.formatwarning = py_fatigue_formatwarning
                warnings.warn(w_msg, UserWarning)
                ns.detail_factor = 0.8
            stress_ranges_corr = msc.dnvgl_mean_stress_correction(
                self_.mean_stress,
                self_.stress_amplitude,
                ns.detail_factor,
                plot,
            )
            mean_stress_corr = copy.deepcopy(stress_ranges_corr / 2)

        if correction_type.lower() == ("walker"):
            if "gamma" not in kwargs:
                w_msg = "".join(
                    (
                        "No gamma exponent provided for Walker ",
                        "mean stress correction. \nPassing gamma=0.5. ",
                        "This value is suitable for many materials ",
                        "and corresponds to applying SWT correction.",
                    )
                )
                warnings.formatwarning = py_fatigue_formatwarning
                warnings.warn(w_msg, UserWarning)
                ns.gamma = 0.5
            stress_ranges_corr = msc.walker_mean_stress_correction(
                self_.mean_stress,
                self_.stress_amplitude,
                ns.gamma,
                plot,
            )
            mean_stress_corr = np.zeros(len(self_.mean_stress))

        if correction_type.lower() in ["swt", "smith-watson-topper"]:
            stress_ranges_corr = msc.swt_mean_stress_correction(
                self_.mean_stress,
                self_.stress_amplitude,
                plot,
            )
            mean_stress_corr = np.zeros(len(self_.mean_stress))

        cc_msc = CycleCount(
            count_cycle=self_.count_cycle,
            stress_range=stress_ranges_corr,
            mean_stress=mean_stress_corr,
            timestamp=self_.timestamp,
            range_bin_lower_bound=self_.range_bin_lower_bound,
            range_bin_width=self_.range_bin_width,
            _mean_bin_lower_bound=self_.mean_bin_lower_bound,
            mean_bin_width=self_.mean_bin_width,
            mean_stress_corrected="".join(
                f"{correction_type.upper()}: {ns.__dict__}"
            ),
            lffd_solved=self_.lffd_solved,
            name=self_.name,
            nr_small_cycles=self_.nr_small_cycles,
            residuals_sequence=self_.residuals_sequence,
            stress_concentration_factor=self_.stress_concentration_factor,
            _time_sequence=self_.time_sequence,
        )

        if len(self.time_sequence) == 1:
            w_msg = "".join(
                (
                    "If mean stress correction is performed in the contest ",
                    "of long-term fatigue analysis, please perform the sum ",
                    "of multiple CycleCount instances prior mean stress ",
                    "correction (MSC).\nIn fact, applying the MSC before ",
                    "summing into long-term CycleCount instance results in ",
                    "non-conservative life estimates, as after MSC, ",
                    "low-frequency fatigue cannot be estimated accurately.",
                )
            )

            warnings.formatwarning = py_fatigue_formatwarning
            warnings.warn(w_msg, UserWarning)

        return cc_msc

    # ! Plotting methods
    def plot_histogram(
        self,
        plot_type: str = "min-max",
        mode: str = "mountain",
        bins: Union[int, Sequence[int]] = 50,
        bin_range: Optional[np.ndarray] = None,
        normed: bool = False,
        weights: Optional[np.ndarray] = None,  # np.histogram2d args
        fig: Optional[matplotlib.figure.Figure] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        dens_func: Optional[Callable] = None,
        **kwargs: dict,
    ) -> tuple:  # pragma: no cover
        """Plot the rainflow histogram of the :term:`Cycle-counting` instance
        with min and max stress on the x-y axes.

        Parameters
        ----------
        plot_type : str
            Type of plot to be generated, default is "min-max". Possible \
            values are: ["min-max", "mean-range"]
        mode : [None | 'mountain' | 'valley' | 'clip']
        Possible values are:
            - None : The points are plotted as one scatter object, in the
                order in-which they are specified at input
            - 'mountain' : The points are sorted/plotted in the order of
                the number of points in their 'bin'. This means that points
                in the highest density will be plotted on-top of others. This
                cleans-up the edges a bit, the points near the edges will
                overlap
            - 'valley' : The reverse order of 'mountain'. The low density
                bins are plotted on top of the high ones

        bins : int or array_like or [int, int] or [array, array]
            The bin specification, by default 50 bins are used:
                - If int, the number of bins for the two dimensions
                    (nx=ny=bins)
                - If array_like, the bin edges for the two dimensions
                    (x_edges = y_edges = bins)
                - If [int, int], the number of bins in each dimension
                    (n_x, n_y = bins)
                - If [array, array], the bin edges in each dimension
                    (x_edges, y_edges = bins)
                - A combination [int, array] or [array, int], where int
                    is the number of bins and array is the bin edges

        bin_range : array_like, shape(2,2), optional
            The leftmost and rightmost edges of the bins along each dimension
            (if not specified explicitly in the `bins` parameters):
            ``[[x_min, x_max], [y_min, y_max]]``. \
            All values outside of this range will be considered outliers
            and not tallied in the histogram.
            The default value is None, in which case all values are tallied
        normed : bool, optional
            If False, returns the number of samples in each bin. If True,
            returns the bin density ``bin_count / sample_count / bin_area``,
            default is False.
        weights : array_like, shape(N,), optional
            An array of values ``w_i`` weighing each sample ``(y_i, x_i)``.
            Weights are normalized to 1 if `normed` is True. If `normed` is
            False, the values of the returned histogram are equal to the sum of
            the weights belonging to the samples falling into each bin.
            The default is None, which gives each sample equal weight
        fig : matplotlib.figure.Figure, optional
            a Figure instance to add axes into
        dens_func : function or callable, optional
            A function that modifies (inputs and returns) the dens
            values (e.g., np.log10). The default is to not modify the
            values
        kwargs : these keyword arguments are all passed on to scatter

        Returns
        -------
        figure, paths : (
            `~matplotlib.figure.Figure`,
            `~matplotlib.axes.Axes`
        )
            The figure with scatter instance.
        """
        x = np.hstack([self.full_cycles[:, 0], self.residuals[:, 0]])
        y = np.hstack([self.full_cycles[:, 1], self.residuals[:, 1]])
        plot_types = [
            "min-max",
            "mean-range",
            "counts-range",
            "counts-range-cumsum",
            "counts-range-2D",
        ]
        if plot_type not in plot_types:
            e_msg = f"Invalid plot type. Must be one of {plot_types}."
            raise ValueError(e_msg)
        if plot_type == "min-max":
            x = x - y / 2
            y = x + y / 2
            bins = bins if bins is not None else 50
        if "counts-range" in plot_type:
            sor = sorted(
                zip(self.stress_range, self.count_cycle),
                reverse=True,
            )

            xy = np.asarray(list(map(list, sor)))
            x = xy[:, 1]
            y = xy[:, 0]
            bins = bins if bins is not None else self.bin_edges
            fig, axes = ht.make_axes(fig=fig, ax=ax)
            im = axes.scatter(
                np.cumsum(x) if "cumsum" in plot_type else x,
                y,
                # edgecolors="#CCC",
                # linewidths=0.5,
                c=x,
                **kwargs,
            )
            # axes.set_yscale("log")
            # axes.set_xscale("log")
        if plot_type in ("min-max", "mean-range"):
            fig, axes, im = ht.xy_hist2d(
                x=x,
                y=y,
                mode=mode,
                bins=bins,
                bin_range=bin_range,
                normed=normed,
                weights=weights,
                dens_func=dens_func,
                fig=fig,
                ax=ax,
                **kwargs,
            )
            axes.axis("equal")
        cbar_label = (
            "# of cycles"
            if dens_func is None
            else "# of cycles(" + str(dens_func) + ")"
        )
        axes.set_xlabel(
            f"Min stress, {self.unit}"
            if plot_type == "min-max"
            else f"Mean stress, {self.unit}"
            if plot_type == "mean-range"
            else "# of cycles"
        )
        axes.set_ylabel(
            f"Max stress, {self.unit}"
            if plot_type == "min-max"
            else f"Range, {self.unit}"
        )
        cbar = plt.colorbar(im, ax=axes)
        cbar.set_label(cbar_label, rotation=270)
        cbar.ax.get_yaxis().labelpad = 10

        return fig, axes

    def plot_residuals_sequence(
        self,
        fig: Optional[matplotlib.figure.Figure] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        **kwargs,
    ) -> Tuple[
        matplotlib.figure.Figure, matplotlib.axes.Axes
    ]:  # pragma: no cover
        """Plot the residuals of the stress-strain curve.

        Parameters
        ----------
        fig : Optional[matplotlib.figure.Figure], optional
            a Figure instance to add axes into, by default None
        ax : Optional[matplotlib.axes.Axes], optional
            an Axes instance to add axes into, by default None

        Returns
        -------
        Tuple[
            matplotlib.figure.Figure,
            matplotlib.axes.Axes
        ]
            The figure with plot instance.
        """
        _, res_res_seq, res_res_idx = calc_rainflow(
            data=np.asarray(self.residuals_sequence),
            extended_output=True,
        )

        fig, axes = ht.make_axes(fig=fig, ax=ax)
        axes.plot(self.residuals_sequence, "#003366", ls=":", **kwargs)
        axes.plot(res_res_idx, res_res_seq, "#FF0080", **kwargs)
        axes.set_xlabel("Residuals sequence")
        axes.set_ylabel("Residuals")
        axes.grid(visible=True, which="major", color="#CCCCCC", linestyle="-")
        axes.minorticks_on()
        axes.grid(visible=True, which="minor", color="#E7E6DD", linestyle=":")
        axes.xaxis.grid(False, which="minor")
        axes.spines["right"].set_visible(False)
        axes.spines["top"].set_visible(False)
        return fig, axes

    def plot_half_cycles_sequence(
        self,
        fig: Optional[matplotlib.figure.Figure] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        **kwargs,
    ) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """A symlink to plot_residuals_sequence.

        Parameters
        ----------
        fig : Optional[matplotlib.figure.Figure], optional
            a Figure instance to add axes into, by default None
        ax : Optional[matplotlib.axes.Axes], optional
            an Axes instance to add axes into, by default None

        Returns
        -------
        Tuple[
            matplotlib.figure.Figure,
            matplotlib.axes.Axes
        ]
            The figure with plot instance.
        """
        return self.plot_residuals_sequence(fig=fig, ax=ax, **kwargs)


def _multiplication_by_scalar(
    self_: "CycleCount", other_: Union[float, int]
) -> "CycleCount":
    """Multiplication of :class:`CycleCount` by a scalar factor

    Parameters
    ----------
    self_ : CycleCount
        CycleCount instance
    other_ : Union[float, int]
        Factor to multiply the instance by

    Returns
    -------
    CycleCount
        CycleCount instance with "raised" stresses and stress concentration
        factor != 1

    Raises
    ------
    TypeError
        Multiplication by a scalar is the only thing allowed
    """
    if not isinstance(other_, (float, int)):
        e_msg = "Multiplication by a scalar SCF is the only thing allowed"
        raise TypeError(e_msg)

    if isinstance(other_, int):
        other_ = float(other_)

    if self_.stress_concentration_factor != 1.0:
        w_msg = "".join(
            (
                f"SCF already defined and set to {TermColors.CGREEN2}",
                f"{self_.stress_concentration_factor}{TermColors.CYELLOW2}.",
            )
        )
        warnings.formatwarning = py_fatigue_formatwarning
        warnings.warn(w_msg, UserWarning)

    mul_range = other_ * self_.stress_range
    mul_mean = other_ * self_.mean_stress
    mul_res_seq = (
        other_ * np.array(self_.residuals_sequence)
        if self_.residuals_sequence is not None
        else np.empty(0)
    )
    mul_min_max = (
        other_ * np.array(self_.min_max_sequence)
        if self_.min_max_sequence is not None
        else np.empty(0)
    )
    mul_mean_bin = (
        other_ * self_.mean_bin_lower_bound
        if self_.mean_bin_lower_bound is not None
        else None
    )
    mul_scf = other_ * self_.stress_concentration_factor

    mul_cc = CycleCount(
        count_cycle=self_.count_cycle,
        mean_stress=mul_mean,
        stress_range=mul_range,
        range_bin_lower_bound=other_ * self_.range_bin_lower_bound,
        range_bin_width=other_ * self_.range_bin_width,
        _mean_bin_lower_bound=mul_mean_bin,
        mean_bin_width=other_ * self_.mean_bin_width,
        residuals_sequence=mul_res_seq,
        _min_max_sequence=mul_min_max,
        lffd_solved=self_.lffd_solved,
        mean_stress_corrected=self_.mean_stress_corrected,
        stress_concentration_factor=mul_scf,
        timestamp=self_.timestamp,
        name=str(self_.name) + ", SCF=" + str(other_),
    )

    # mul_cc.stress_concentration_factor = (
    #     other_ * self_.stress_concentration_factor
    # )
    return mul_cc


def _cycle_count_add_checks(
    self_: CycleCount,
    other_: CycleCount,
) -> bool:
    """Check if the two cycle count instances are compatible for addition.

    Parameters
    ----------
    self_ : CycleCount
        First cycle count instance
    other_ : Optional[CycleCount]
        Second cycle count instance

    Returns
    -------
    bool
        True if the two cycle count instances are compatible for addition
    """
    checks_ok = True
    # ! 1) TypeError
    # Trying to add to a non CycleCount object instance
    if self_.__class__.__name__ != other_.__class__.__name__:
        if (
            other_ is None
            or other_ == 0
            or isinstance(other_, float)
            and np.isnan(other_)
        ):
            w_msg = "".join(
                (
                    "Trying to add to a non CycleCount object instance ",
                    f"after {self_.timestamp.strftime('%d %B %Y, %H:%M')}.",
                )
            )
            warnings.formatwarning = py_fatigue_formatwarning
            warnings.warn(w_msg, UserWarning)
            return False
        raise TypeError(
            "Trying to add to a non CycleCount object instance",
            type(self_),
            type(other_),
        )
    # ! 2) TypeError
    # Trying to add-up different parameters.
    if self_.name != other_.name:
        w_msg = "".join(
            (
                "Summing different parameters. "
                f"({self_.name} + {other_.name})"
            )
        )
        warnings.formatwarning = py_fatigue_formatwarning
        warnings.warn(w_msg, UserWarning)

    # ! 4) UserWarning
    # Different bin widths
    _bin_widths_add_check(self_.range_bin_width, other_.range_bin_width)
    _bin_widths_add_check(self_.mean_bin_width, other_.mean_bin_width, "mean")

    # ! 5) TypeError
    # Timestamps must be sorted ascending.
    if self_.timestamp is not None and other_.timestamp is not None:
        if np.max(self_.time_sequence) > np.min(other_.time_sequence):
            e_msg = (
                np.min(other_.time_sequence).strftime("%d %b, %Y, %H:%M")
                + " predates "
                + np.max(self_.time_sequence).strftime("%d %b, %Y, %H:%M")
                + ". Timestamps must be sorted ascending."
            )
            raise TypeError(e_msg)

    # ! 6) UserWarning
    # Adding-up instances having different stress concentration factors.
    if self_.stress_concentration_factor != other_.stress_concentration_factor:
        w_msg = "".join(
            (
                "Adding-up instances having different SCFs (",
                f"{TermColors.CGREEN}{self_.stress_concentration_factor} and ",
                f"{other_.stress_concentration_factor}{TermColors.CYELLOW2}",
                ").",
            )
        )
        warnings.formatwarning = py_fatigue_formatwarning
        warnings.warn(w_msg, UserWarning)

    return checks_ok


def _bin_widths_add_check(
    self_: float, other_: float, bin_range: str = "range"
) -> None:
    """Check if the two bin ranges are compatible for addition.

    Parameters
    ----------
    self_ : float
        First bin range
    other_ : float
        Second bin range
    bin_range : str, optional
        Bin range type. Default is "range".

    Returns
    -------
    None
    """

    if self_ != other_:
        w_msg = "".join(
            (
                f"Different bin {bin_range} widths. ({TermColors.CGREEN}",
                f"{self_} and {other_}{TermColors.CYELLOW2}).",
            )
        )
        warnings.formatwarning = py_fatigue_formatwarning
        warnings.warn(w_msg, UserWarning)


def _handling_different_bins_in_sum(
    self_: CycleCount,
    other_: "CycleCount",
) -> tuple:
    """Handle the different bin ranges for addition.

    Parameters
    ----------
    self_ : CycleCount
        First cycle count instance
    other_ : CycleCount
        Second cycle count instance

    Returns
    -------
    tuple
        Tuple containing the bin low bound and widths for mean and range
    """
    if self_.mean_bin_width is not None:
        if other_.mean_bin_width is not None:
            added_mean_bin_lower_bound = np.min(
                [self_.mean_bin_lower_bound, other_.mean_bin_lower_bound]
            )
            added_mean_bin_width = np.max(
                [self_.mean_bin_width, other_.mean_bin_width]
            )
        else:
            added_mean_bin_lower_bound = self_.mean_bin_lower_bound
            added_mean_bin_width = self_.mean_bin_width
    else:
        added_mean_bin_lower_bound = None
        added_mean_bin_width = None

    added_range_bin_lower_bound = np.min(
        [self_.range_bin_lower_bound, other_.range_bin_lower_bound]
    )
    added_range_bin_width = np.max(
        [self_.range_bin_width, other_.range_bin_width]
    )

    return (
        added_mean_bin_lower_bound,
        added_mean_bin_width,
        added_range_bin_lower_bound,
        added_range_bin_width,
    )


def _solve_lffd(
    self_: CycleCount, solve_mode: str = "Residuals"
) -> CycleCount:
    """Retrieve the LFFD for the cycle count instance.

    Parameters
    ----------
    self_ : CycleCount
        Cycle count instance
    solve_mode : str, optional
        Solve mode. Default is "Residuals".

    Returns
    -------
    CycleCount
        Cycle count instance with the LFFD solved.
    """

    ret_self, res_seq = _lffd_checks(self_, solve_mode)
    if ret_self:
        return self_

    # NOTE: next plot has been included for testing purposes

    res_rf, res_res_seq, _ = calc_rainflow(
        data=res_seq,
        extended_output=True,
    )

    cc_res = CycleCount(
        count_cycle=res_rf[:, 2],
        stress_range=2 * res_rf[:, 0],
        mean_stress=res_rf[:, 1],
        timestamp=self_.timestamp,
        _mean_bin_lower_bound=self_.mean_bin_lower_bound,
        mean_bin_width=self_.mean_bin_width,
        range_bin_lower_bound=self_.range_bin_lower_bound,
        range_bin_width=self_.range_bin_width,
        nr_small_cycles=self_.nr_small_cycles,
        residuals_sequence=res_res_seq,
        _min_max_sequence=self_.min_max_sequence
        if "residuals" in solve_mode.lower()
        else np.asarray([min(res_res_seq), max(res_res_seq)]),
        lffd_solved=True,
        mean_stress_corrected=self_.mean_stress_corrected,
        name=self_.name,
        _time_sequence=self_.time_sequence,
        stress_concentration_factor=self_.stress_concentration_factor,
    )

    if max(self_.count_cycle) <= 0.5:
        return cc_res

    cc_no_res = CycleCount(
        count_cycle=self_.count_cycle[self_.count_cycle > 0.5],
        stress_range=self_.stress_range[self_.count_cycle > 0.5],
        mean_stress=self_.mean_stress[self_.count_cycle > 0.5],
        timestamp=self_.timestamp,
        _mean_bin_lower_bound=self_.mean_bin_lower_bound,
        mean_bin_width=self_.mean_bin_width,
        range_bin_lower_bound=self_.range_bin_lower_bound,
        range_bin_width=self_.range_bin_width,
        residuals_sequence=np.empty(0),
        _min_max_sequence=np.empty(0),
        lffd_solved=True,
        mean_stress_corrected=self_.mean_stress_corrected,
        name=self_.name,
        _time_sequence=np.array(np.max(self_.time_sequence)),
        stress_concentration_factor=self_.stress_concentration_factor,
    )
    cc_solved = cc_res + cc_no_res
    cc_solved.lffd_solved = True
    return cc_solved


def _lffd_checks(self_: CycleCount, solve_mode: str) -> tuple:
    """Run checks on the :term:`LFFD` (Low-Frequency Fatigue Dynamics)
    retrieve method and sets the residuals sequence either to residuals
    or to min-max (approximated).

    Parameters
    ----------
    self_ : CycleCount
        The CycleCount object to check.
    solve_mode : str
        The solve mode to use.

    Returns
    -------
    tuple
        A tuple containing the following:
        - bool, True if the checks are passed, False otherwise.
        - np.ndarray, the residuals or min-max sequence.

    Raises
    ------
    ValueError
        If the solve mode is not supported.
    """

    solve_modes = [
        "min-max",
        "min/max",
        "min max",
        "residuals",
    ]
    if solve_mode.lower() not in solve_modes:
        e_msg = f"Invalid solve mode. Must be one of {solve_modes}."
        raise ValueError(e_msg)

    return_self = False
    if self_.lffd_solved:
        w_msg = "Residuals already resolved. Nothing to do."
        warnings.formatwarning = py_fatigue_formatwarning
        warnings.warn(w_msg, UserWarning)
        return_self = True
    if (
        hasattr(self_, "residuals_sequence")
        and self_.residuals_sequence is not None
    ):
        if "residuals" in solve_mode.lower():
            res_sequence = self_.residuals_sequence
        if "min" in solve_mode.lower() or "max" in solve_mode.lower():
            res_sequence = self_.min_max_sequence
    else:
        w_msg = "'residuals_sequence' is not available. Nothing to do."
        warnings.formatwarning = py_fatigue_formatwarning
        warnings.warn(w_msg, UserWarning)
        return_self = True
    if len(res_sequence) < 3:
        w_msg = "".join(
            (
                f"lffd_solve() not possible when{TermColors.CGREEN2}",
                f"{TermColors.CBOLD}len{TermColors.CEND}{TermColors.CYELLOW2}("
                f"residuals_sequence) < 3.\n Returning {TermColors.CBEIGE2}",
                f"{TermColors.CBOLD}'self'{TermColors.CEND}.",
            )
        )
        warnings.formatwarning = py_fatigue_formatwarning
        warnings.warn(w_msg, UserWarning)
        return_self = True
    return return_self, res_sequence


def pbar_sum(cc_list: Sequence[CycleCount]) -> CycleCount:
    """Sum a list of CycleCount objects using a progress bar.

    Parameters
    ----------
    cc_list : list
        The list of CycleCount objects to sum.

    Returns
    -------
    py_fatigue.CycleCount
        The sum of the CycleCount objects.
    """

    if len(cc_list) == 1:
        return cc_list[0]

    chunk_size = int(np.sqrt(len(cc_list)))
    cc_list = [cc for cc in cc_list if cc is not None]

    sleep(0.15)
    chunked_cc_list = list(pfu.chunks(cc_list, chunk_size))
    l_c = len(chunked_cc_list)

    cc_sum: CycleCount = chunked_cc_list[0]  # type: ignore
    cc_sum_list: list[CycleCount] = []

    for j, chunked_cc_sublist in enumerate(chunked_cc_list):
        k = j + 1
        # try:
        blocks = int(k / l_c * 32)
        # except ZeroDivisionError:
        #     blocks = int(k / l_c * 32)
        underscores = 32 - blocks
        pbar = f"{k * len(chunked_cc_sublist)}/{len(cc_list)} -> "
        pbar += TermColors.CBLUE2 + "|" + "".join(["█"] * blocks)
        pbar += "".join(["_"] * underscores) + "|" + TermColors.CEND
        # try:
        pbar += f" -> { min(int(k / l_c * 100), 100)}% completed... "
        # except ZeroDivisionError:
        #     pbar += f" -> { min(int(k / l_c * 100), 100)}% completed... "
        print(pbar, end="\r")

        if len(chunked_cc_sublist) == 1:
            cc_sum_list.append(chunked_cc_sublist[0])
            cc_sum = (
                cc_sum_list[-1]
                if len(cc_sum_list) == 1
                else cc_sum + cc_sum_list[-1]
            )
            continue
        cc_sum_list.append(chunked_cc_sublist[0])
        for cc in chunked_cc_sublist[1::]:
            if cc is None:
                continue
            cc_sum_list[-1] += cc
        cc_sum = (
            cc_sum_list[-1]
            if len(cc_sum_list) == 1
            else cc_sum + cc_sum_list[-1]
        )
    return cc_sum
