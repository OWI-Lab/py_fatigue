"""
The :mod:`py_fatigue.utils` module collects all the utility functions
and classes.
"""
# Packages from the Python Standard Library
from __future__ import annotations
from dataclasses import dataclass
from typing import (
    Any,
    Generator,
    Tuple,
    cast,
    Callable,
    ClassVar,
    Generic,
    NamedTuple,
    Optional,
    TypeVar,
    Union,
)
import copy

# Packages from non-standard libraries
from pydantic.fields import ModelField
import matplotlib
import matplotlib.pyplot as plt
import numba as nb
import numpy as np


# __all__ = [
#     "inplacify",
#     "calc_full_cycles",
#     "calc_half_cycles",
#     "split_full_cycles_and_residuals",
#     "FatigueStress",
#     "make_axes",
#     "calc_bin_edges",
#     "bin_upper_bound",
#     "chunks",

# ]

# Local imports


def inplacify(method: Callable) -> Callable:
    """
    Make a method inplace.

    Parameters
    ----------
    method : Callable
        The method to make inplace.

    Returns
    -------
    Callable
        The inplace method.
    """

    def wrap(self, *args, **kwargs):
        inplace = kwargs.pop("inplace", False)
        if inplace:
            return method(self, *args, **kwargs)
        return method(copy.copy(self), *args, **kwargs)

    return wrap


def make_axes(
    fig: Optional[matplotlib.figure.Figure] = None,  # type: ignore
    ax: Optional[matplotlib.axes.Axes] = None,  # type: ignore
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:  # type: ignore
    """Check if a figure and axes are provided, and if not, create them.

    Parameters
    ----------
    fig : Optional[matplotlib.figure.Figure], optional
        The figure instance, by default None
    ax : Optional[matplotlib.axes.Axes], optional
        The axes instance, by default None

    Returns
    -------
    tuple
        The figure and axes instances

    Raises
    ------
    TypeError
        If fig is not a matplotlib.figure.Figure instance
    """
    if fig is None:
        fig, axes = plt.subplots()
    else:
        if not isinstance(fig, matplotlib.figure.Figure):  # type: ignore
            raise TypeError("fig must be a matplotlib.figure.Figure instance")
        if ax is None:
            axes = fig.gca()
        else:
            axes = ax
    return fig, axes


class IntInt(NamedTuple):
    """Represents a pair of integers, namely the number of bins for
    mean and ranges

    Parameters
    ----------
    NamedTuple : NamedTuple
    """

    mean_bin_nr: int
    range_bin_nr: int


class ArrayArray(NamedTuple):
    """Represents a pair of arrays, namely the mean and range bin edge
    values

    Parameters
    ----------
    NamedTuple : NamedTuple
    """

    mean_bin_edges: np.ndarray
    range_bin_edges: np.ndarray


class IntArray(NamedTuple):
    """Represents an integer and an array, namely the number of bins
    for mean and range bin edge values

    Parameters
    ----------
    NamedTuple : NamedTuple
    """

    mean_bin_nr: int
    range_bin_edges: np.ndarray


class ArrayInt(NamedTuple):
    """Represents an array and an integer, namely the number of bins
    for range and mean bin edge values

    Parameters
    ----------
    NamedTuple : NamedTuple
    """

    mean_bin_edges: np.ndarray
    range_bin_edges: int


UnionOfIntArray = Union[
    int, np.ndarray, IntInt, ArrayArray, IntArray, ArrayInt
]


@dataclass(repr=False)
class FatigueStress:
    """Fatigue stress class. Mean stress, range, maximum, and minimum
    can inherit from this class that defines the following properties:

        - `bin_edges`: the bin edges for the histogram
        - `bin_centers`: the bin centers for the histogram
        - `full`: the full cycles
        - `full_counts`: the counts of the full cycles
        - `half`: the half cycles
        - `half_counts`: the counts of the half cycles
        - `bins_idx`: the index of the full cycles bins
        - `binned_values`: the binned full cycles
    """

    category: ClassVar[str]
    _counts: np.ndarray
    _values: np.ndarray
    bin_width: float
    _bin_lb: Optional[float] = None
    _bin_ub: Optional[float] = None
    # @properties
    # bin_edges: np.ndarray
    # bin_centers: np.ndarray
    # full: np.ndarray
    # half: np.ndarray
    # half_counts: np.ndarray
    # bins_idx: np.ndarray
    # binned_values: np.ndarray

    def __post_init__(self):
        if len(self._counts) == 0 or len(self._values) == 0:
            raise ValueError("No data provided")
        if len(self._counts) != len(self._values):
            raise ValueError("counts and values must have the same length")
        self._counts = np.asarray(self._counts)
        self._values = np.asarray(self._values)
        if self._bin_lb is None:
            self._bin_lb = (
                round(min(self._values) / self.bin_width) * self.bin_width
                - self.bin_width / 2
            )
        if self._bin_ub is None:
            self._bin_ub = bin_upper_bound(
                self._bin_lb,
                self.bin_width,
                max(self.full) if len(self.full) > 0 else max(self._values),
            )

    @property
    def bin_lb(self) -> float:
        """The lower bound of the bins. Casting resolves mypy error"""
        return cast(float, self._bin_lb)

    @bin_lb.setter
    def bin_lb(self, value: float) -> None:
        self._bin_lb = value

    @property
    def bin_ub(self) -> float:
        """The upper bound of the bins. Casting resolves mypy error"""
        return cast(float, self._bin_ub)

    @bin_ub.setter
    def bin_ub(self, value: float) -> None:
        self._bin_ub = value

    @property
    def counts(self) -> np.ndarray:
        """The number of cycles.

        Returns
        -------
        counts : np.ndarray
        """
        return self._counts

    # val setter function
    @counts.setter
    def counts(self, val: np.ndarray) -> None:
        """Setter for the counts.

        Parameters
        ----------
        val : np.ndarray
            The new counts.

        Returns
        -------
        counts : np.ndarray
        """
        self._counts = val

    @property
    def values(self) -> np.ndarray:
        """The stress values.

        Returns
        -------
        values : np.ndarray
        """
        return self._values

    # val setter function
    @values.setter
    def values(self, val: np.ndarray) -> None:
        """Setter for the values.

        Parameters
        ----------
        val : np.ndarray
            The new values.
        """
        self._values = val

    @property
    def bin_edges(self) -> np.ndarray:
        """The bin edges.

        Returns
        -------
        bin_edges : np.ndarray
        """
        return calc_bin_edges(self.bin_lb, self.bin_ub, self.bin_width)

    @property
    def bin_centers(self):
        """The bin centers.

        Returns
        -------
        bin_centers : np.ndarray
        """
        return self.bin_edges[:-1] + self.bin_width / 2

    @property
    def full(self):
        """The full cycles stresses.

        Returns
        -------
        full : np.ndarray
        """
        return calc_full_cycles(self.values, self.counts)

    @property
    def half(self):
        """The half cycles stresses.

        Returns
        -------
        half : np.ndarray
        """
        return calc_half_cycles(self.values, self.counts)

    @property
    def bins_idx(self):
        """The index of the full cycles bins.

        Returns
        -------
        bins_idx : np.ndarray
        """
        digitized = np.digitize(self.full, self.bin_edges, right=True)
        digitized[np.where(digitized > 0)] -= 1
        return digitized

    @property
    def binned_values(self):
        """The binned full cycles.

        Returns
        -------
        binned_values : np.ndarray
        """
        return self.bin_centers[self.bins_idx]


def calc_bin_edges(
    bin_lb: float, bin_ub: float, bin_width: float
) -> np.ndarray:
    """Calculate the bin edges for a fatigue histogram using the given
    bin width, lower bound and upper bound.

    Parameters
    ----------
    bin_lb : float
        Lowest bin edge value
    bin_ub : float
        Uppest bin edge value
    bin_width : float
        Bin width

    Returns
    -------
    np.ndarray
        Bin edges
    """
    bin_ub = bin_upper_bound(bin_lb, bin_width, bin_ub)

    n_bins = int(np.round((bin_ub + bin_width - bin_lb) / bin_width))

    bin_edges = np.around(
        np.linspace(start=bin_lb, stop=bin_ub, num=n_bins),
        decimals=5,
    )
    return bin_edges


def bin_upper_bound(
    bin_lower_bound: float, bin_width: float, max_val: float
) -> float:
    """Returns the upper bound of the bin edges, given the lower bound,
    bin width and maximum value of the array. If the maximum value is
    not a multiple of the bin width, the upper bound is rounded up to
    the next multiple of the bin width.
    Also, if the maximum value is smaller than the lower bound, the
    upper bound is set to the lower bound.

    Parameters
    ----------
    bin_lower_bound : float
        Left edge of the first bin
    bin_width : float
        Bin width
    max_val : float
        Maximum value of the array to be binned

    Returns
    -------
    float
        The upper bound of the bin edges
    """
    return bin_lower_bound + bin_width * np.ceil(
        (max_val - bin_lower_bound) / bin_width
    )


def split_full_cycles_and_residuals(count_cycle: np.ndarray) -> tuple:
    """
    Returns the :term:`stress ranges<Stress Ranges>` or
    :term:`mean stresses<Mean stress>` from only the full cycles.
    """

    splt = np.asarray(np.modf(count_cycle)).T
    full_idx = np.where((splt[:, 1] % 1 == 0) & (splt[:, 1] > 0))[0]
    half_idx = np.where((splt[:, 0] == 0.5) & (splt[:, 0] > 0))[0]
    # zero_idx = np.where((splt[:, 0] == 0) & (splt[:, 1] == 0))[0]
    # full_idx = np.sort(np.hstack([full_idx, zero_idx]))
    # half_idx = np.sort(np.hstack([half_idx, zero_idx]))
    return full_idx, half_idx


def calc_full_cycles(
    stress: np.ndarray, count_cycle: np.ndarray, debug_mode: bool = False
) -> np.ndarray:
    """
    Returns the :term:`stress ranges<Stress Ranges>` or
    :term:`mean stresses<Mean stress>` from only the full cycles.
    """
    full_idx, _ = split_full_cycles_and_residuals(count_cycle)
    # integer_count = np.delete(count_cycle, half_idx).astype(int)
    integer_count = count_cycle[full_idx].astype(int)
    integer_stress = stress[full_idx]
    if debug_mode:
        print("full cycles idx", full_idx)
        print("count_cycle[full_idx]", integer_count)
        print("stress[full_idx]", integer_stress)
        print("len integer_count", len(integer_count))
        print("len integer_stress", len(integer_stress))
    return np.repeat(integer_stress, integer_count)


def calc_half_cycles(
    stress: np.ndarray, count_cycle: np.ndarray, debug_mode: bool = False
) -> np.ndarray:
    """
    Returns the :term:`stress ranges<Stress Ranges>` or
    :term:`mean stresses<Mean stress>` from only the half cycles.
    """
    _, half_idx = split_full_cycles_and_residuals(count_cycle)

    half_stress = stress[half_idx]
    if debug_mode:
        half_count = count_cycle[half_idx]
        print("half cycles idx", half_idx)
        print("count_cycle[half_idx]", half_count)
        print("len half_count", len(half_count))
        print("len half_stress", len(half_stress))
    return half_stress


def compare(a: Any, b: Any) -> bool:
    """Compare two objects.

    Parameters
    ----------
    a : Any
    b : Any

    Returns
    -------
    bool
        True if the a and b are equal, False otherwise.
    """
    # print(f"a: {a}, b: {b}")
    if not hasattr(a, "__iter__") or isinstance(a, str):
        return a == b

    try:
        if not len(a) == len(b):
            return False
        if isinstance(a, np.ndarray):
            return np.array_equal(a, b)
        if isinstance(a, dict):
            return all(compare(v, b[k]) for k, v in a.items())
        return all(compare(aa, bb) for aa, bb in zip(a, b))
    except (TypeError, KeyError):
        return False


def split(what: int, by: int):
    """Split an integer into a list of integers.

    >>> split(5, 2)
    [2, 3]
    >>> split(20, 7)
    [3, 3, 3, 3, 3, 3, 2]

    Parameters
    ----------
    what : int
        The integer to split
    by : int
        The number of integers to split the integer into

    Returns
    -------
    list
        The list of integers
    """
    return [what // by + 1] * (what % by) + [what // by] * (by - what % by)


JSON_ENCODERS = {np.ndarray: lambda arr: arr.tolist()}

DType = TypeVar("DType")


class TypedArray(np.ndarray, Generic[DType]):
    """Wrapper class for numpy arrays that stores and validates type
    information. This can be used in place of a numpy array, but when
    used in a pydantic BaseModel or with pydantic.validate_arguments,
    its dtype will be *coerced* at runtime to the declared type.
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, val, field: ModelField):
        """Validate the value of the field."""
        dtype_field = field.sub_fields[0]  # type: ignore
        actual_dtype = dtype_field.type_.__args__[0]
        # If numpy cannot create an array with the request dtype,
        # an error will be raised and correctly bubbled up.
        np_array = np.array(val, dtype=actual_dtype)
        return np_array


def to_numba_dict(
    data: dict, key_type: type = str, val_type: type = float
) -> nb.types.DictType:
    """Converts a dictionary to a numba typed dict, provided the output
    key and value types.

    Parameters
    ----------
    data : dict
        The dictionary to be converted.
    key_type : type, optional
        The key type, by default str
    val_type : type, optional
        The type of the values, by default float

    Returns
    -------
    nb.types.DictType
        The numba typed dictionary.
    """
    dct = nb.typed.Dict.empty(
        key_type=nb.types.string,
        value_type=nb.float64,
    )
    for key, value in data.items():
        if not isinstance(value, val_type) or not isinstance(key, key_type):
            continue
        dct[key] = value
    return dct


def chunks(lst: list[Any], n: int) -> Generator[list[Any], None, None]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]
