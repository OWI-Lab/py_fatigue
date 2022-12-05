from contextlib import nullcontext
from typing import Any, Callable, Tuple
from unittest.mock import Mock
import numpy as np
import pytest

import py_fatigue.cycle_count.histogram as ht
import py_fatigue.utils as pfu
from tests.test_utils import EXPECTED_1


def expected(x) -> Tuple[Any, nullcontext]:
    """Mocking function for pytest.mark.parametrize. Expected value"""
    return x, nullcontext()


def error(*args: Any, **kwargs: Any) -> None:
    """Mocking function for pytest.mark.parametrize. Error handling"""
    return Mock(), pytest.raises(*args, **kwargs)


X_1 = np.array([0, 1, 2, 1, 0, 2, 0])
Y_1 = np.array([1, 1, 4, 5, 1, 4, 4])
EXPECTED_1 = np.array([2.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0])
X_2 = np.array([0, 1, 2, 1, 0, 2, 0])
Y_2 = np.array([1, 5, 4, 5, 1, 4, 4])
EXPECTED_2 = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0])
X_3 = X_1
Y_3 = np.hstack([Y_1, [1]])
X_4 = X_1
Y_4 = np.where(Y_1 > 20, 0, Y_1)
mh_names = "x, y, expected, error"
mh_data = [
    (X_1, Y_1, *expected(EXPECTED_1)),
    (X_2, Y_2, *expected(EXPECTED_2)),
    (X_3, Y_3, *error(ValueError, match="same length")),
]


@pytest.mark.parametrize(mh_names, mh_data)
def test_map_hist(
    x: np.ndarray, y: np.ndarray, expected: np.ndarray, error: Callable
) -> None:
    """Test map_hist function. The function should map an histogram
    to a 1D array of the same length of the input X and Y arrays.
    """
    with error:
        h, b_e = ht.make_histogram(x, y)
        assert np.allclose(ht.map_hist(x, y, h, b_e), expected)


SIGNAL_BR = np.array([-3, 1, -1, 5, -1, 5, -1, 0, -4, 2, 1, 4, 1, 4, 3, 4, 2])
# fmt: off
EXPECTED_BR = {"nr_small_cycles": 0, "range_bin_lower_bound": 0.5,
               "range_bin_width": 1, "mean_bin_lower_bound": -1,
               "mean_bin_width": 1,
               "hist": [[1.0, 1.0], [], [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0], [1.0]],
               "lg_c": [],
               "res": [[1.0, 8.0], [0.5, 9.0], [0.0, 8.0], [3.0, 2.0]],
               "res_sig": [-3, 5, -4, 4, 2]}
br_names = ",".join(
    [
        "sig", "range_bin_lb", "range_bin_width", "mean_bin_lb",
        "mean_bin_width", "damage_tolerance_for_binning",
        "max_consetutive_zeros", "expected", "error",
    ]
)
# fmt: on

br_data = [
    (SIGNAL_BR, 0.5, 1, -1, 1, 1, 16, *expected(EXPECTED_BR)),
]
@pytest.mark.parametrize(br_names, br_data)
def test_binned_rainflow(
    sig: np.ndarray,
    range_bin_lb: float,
    range_bin_width: float,
    mean_bin_lb: float,
    mean_bin_width: float,
    damage_tolerance_for_binning: float,
    max_consetutive_zeros: int,
    expected: np.ndarray,
    error: Callable
) -> None:
    """Test binned_rainflow function. The function should bin an array
    of rainflow counts into a histogram.
    """
    with error:
        dct = ht.binned_rainflow(
            sig, 
            range_bin_lower_bound=range_bin_lb,
            range_bin_width=range_bin_width,
            mean_bin_lower_bound=mean_bin_lb,
            mean_bin_width=mean_bin_width,
            damage_tolerance_for_binning=damage_tolerance_for_binning,
            max_consetutive_zeros=max_consetutive_zeros,
        )
        for k, _ in expected.items():
            if k not in dct.keys():
                assert False
        pfu.compare(dct, expected)


# @given(number=hy.integers())
# def test_map_hist(number):
#     assert isinstance(number, int)
