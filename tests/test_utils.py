# from hypothesis import  given, strategies as hy
# Standard imports
from contextlib import nullcontext
from hypothesis import given, strategies as hy
from typing import Any, List, Callable, Tuple
import unittest
from unittest.mock import Mock

import os
import pytest
import warnings

# Non-standard imports
from numpy.typing import ArrayLike
import numpy as np

os.environ["NUMBA_DISABLE_JIT"] = "1"
import py_fatigue.utils as pu

pu.UnionOfIntArray


@hy.composite
def same_len_lists(draw, min_value=2, max_value=50):

    n = draw(hy.integers(min_value=min_value, max_value=max_value))
    fixed_length_list = hy.lists(hy.integers(), min_size=n, max_size=n)

    return (draw(fixed_length_list), draw(fixed_length_list))


def expected(x) -> Tuple[Any, nullcontext]:
    """Mocking function for pytest.mark.parametrize. Expected value"""
    return x, nullcontext()


def error(*args: Any, **kwargs: Any) -> None:
    """Mocking function for pytest.mark.parametrize. Error handling"""
    return Mock(), pytest.raises(*args, **kwargs)


COUNTS_1 = np.array([1, 2, 6, 8, 7, 7, 14, 2, 9, 4, 3], dtype=float)
EXPECTED_1 = (
    np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int64),
    np.array([], dtype=np.int64),
)
COUNTS_2 = np.insert(COUNTS_1, [0, 1, 4], 0.5)
EXPECTED_2 = (
    np.array([1, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13], dtype=np.int64),
    np.array([0, 2, 6], dtype=np.int64),
)
COUNTS_3 = np.insert(COUNTS_1, [-1, -1, -1], 0.5)
EXPECTED_3 = (
    np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13], dtype=np.int64),
    np.array([10, 11, 12], dtype=np.int64),
)
COUNTS_4 = np.append(COUNTS_1, [0.5, 0.5, 0.5, 0.5, 0.5])
EXPECTED_4 = (
    np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int64),
    np.array([11, 12, 13, 14, 15], dtype=np.int64),
)
COUNTS_5 = np.hstack([COUNTS_1, [np.nan]])
EXPECTED_5 = EXPECTED_1
sfcar_names = "counts, expected, error"
sfcar_data = [
    (COUNTS_1, *expected(EXPECTED_1)),
    (COUNTS_2, *expected(EXPECTED_2)),
    (COUNTS_3, *expected(EXPECTED_3)),
    (COUNTS_4, *expected(EXPECTED_4)),
    # (COUNTS_5, *expected(EXPECTED_5)),
]


@pytest.mark.parametrize(sfcar_names, sfcar_data)
def test_split_full_cycles_and_residuals(
    counts: np.ndarray,
    expected: tuple,
    error: Callable,
) -> None:
    """Test split_full_cycles_and_residuals function"""
    with error:
        calculated_output = pu.split_full_cycles_and_residuals(counts)
        assert np.all(calculated_output[0] == expected[0])
        assert np.all(calculated_output[1] == expected[1])


bub_names = "bin_lower_bound, bin_width, max_val, expected, error"
bub_data = [
    (-10, 5, 100.5, *expected(105.0)),
    (60, 6, -20, *expected(-18.0)),
    (60, 6, 60, *expected(60.0)),
    (0, 1, 99, *expected(99.0)),
    (0, 1, 98.9, *expected(99.0)),
]


@pytest.mark.parametrize(bub_names, bub_data)
def test_bin_upper_bound(
    bin_lower_bound: float,
    bin_width: float,
    max_val: float,
    expected: float,
    error: Callable,
):
    """Test calc_bin_edges function"""
    with error:
        calculated_output = pu.bin_upper_bound(
            bin_lower_bound, bin_width, max_val
        )
        assert np.all(calculated_output == expected)


# fmt: off
cbe_names = "bin_lb, bin_ub, bin_width, expected, error"
cbe_data = [
    (1,  2, 3, *expected(np.array([1., 4.]))),
    (1, -1, 3, *expected(np.array([1.]))),
    (1, 10, 1, *expected(np.array(
        [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.]))
    ),
    (1, 10.1, 1, *expected(np.array(
        [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.]))
    ),
    (1, 10.1, 0.5, *expected(np.array(
        [ 1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ,  4.5,  5. ,  5.5,  6. ,
        6.5,  7. ,  7.5,  8. ,  8.5,  9. ,  9.5, 10. , 10.5]))
    ),
    (1.1,10.1, 0.2, *expected(np.array(
      [ 1.1,  1.3,  1.5,  1.7,  1.9,  2.1,  2.3,  2.5,  2.7,  2.9,  3.1,
        3.3,  3.5,  3.7,  3.9,  4.1,  4.3,  4.5,  4.7,  4.9,  5.1,  5.3,
        5.5,  5.7,  5.9,  6.1,  6.3,  6.5,  6.7,  6.9,  7.1,  7.3,  7.5,
        7.7,  7.9,  8.1,  8.3,  8.5,  8.7,  8.9,  9.1,  9.3,  9.5,  9.7,
        9.9, 10.1]))
    ),
]
# fmt: on
@pytest.mark.parametrize(cbe_names, cbe_data)
def test_calc_bin_edges(
    bin_lb: float,
    bin_ub: float,
    bin_width: float,
    expected: np.ndarray,
    error: Callable,
):
    """Test calc_bin_edges function"""
    with error:
        calculated_output = pu.calc_bin_edges(bin_lb, bin_ub, bin_width)
        assert np.all(calculated_output == expected)


chc_names = "count_cycle, expected, error"
chc_data = [
    (COUNTS_1, *expected(np.array([], dtype=int))),
    (COUNTS_2, *expected(np.array([1, 3, 7], dtype=int))),
    (COUNTS_3, *expected(np.array([11, 12, 13], dtype=int))),
    (COUNTS_4, *expected(np.array([12, 13, 14, 15, 16], dtype=int))),
]


@pytest.mark.parametrize(chc_names, chc_data)
def test_calc_half_cycles(
    count_cycle: np.ndarray,
    expected: np.ndarray,
    error: Callable,
):
    """Test calc_half_cycles function"""
    with error:
        calculated_output = pu.calc_half_cycles(
            np.arange(1, len(count_cycle) + 1, 1), count_cycle
        )
        assert np.all(calculated_output == expected)


cfc_names = "count_cycle, expected, error"
# fmt: off
cfc_data = [
    (COUNTS_1, *expected(np.array(
        [ 1,  2,  2,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,
          5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,
          7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  9,  9,  9,  9,
          9,  9,  9,  9,  9, 10, 10, 10, 10, 11, 11, 11]))
    ),
    (COUNTS_2, *expected(np.array(
        [ 2,  4,  4,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,
          8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,  9, 10, 10, 10,
         10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 12, 12, 12, 12,
         12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14]))
    ),
    (COUNTS_3, *expected(np.array(
        [ 1,  2,  2,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,
          5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,
          7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  9,  9,  9,  9,
          9,  9,  9,  9,  9, 10, 10, 10, 10, 14, 14, 14]))
    ),
    (COUNTS_4, *expected(np.array(
        [ 1,  2,  2,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,
          5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,
          7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  9,  9,  9,  9,
          9,  9,  9,  9,  9, 10, 10, 10, 10, 11, 11, 11]))
    ),
]
# fmt: on
@pytest.mark.parametrize(cfc_names, cfc_data)
def test_calc_full_cycles(
    count_cycle: np.ndarray,
    expected: np.ndarray,
    error: Callable,
):
    """Test calc_full_cycles function"""
    with error:
        calculated_output = pu.calc_full_cycles(
            np.arange(1, len(count_cycle) + 1, 1), count_cycle
        )
        assert np.all(calculated_output == expected)


class TestFatigueStress(unittest.TestCase):
    """Test the FatigueStress class"""

    # def setUp(self):
    #     """
    #     Set up for some of the tests.
    #     """

    @given(same_len_lists())
    def test_faulty_fatigue_stress(self, lists: Tuple[List[int], List[int]]):
        """Assert that an error is raised when the counts and values
        have different lengths
        """
        vals, cts = lists
        with self.assertRaises(ValueError) as ve:
            pu.FatigueStress(_counts=cts, _values=vals[1:], bin_width=0.5)
            self.assertEqual(
                "counts and values must have the same length",
                str(ve.exception)
            )

    @given(same_len_lists())
    def test_empty_fatigue_stress(self, lists: Tuple[List[int], List[int]]):
        """Assert that an error is raised when the counts or values is empty
        """
        vals, cts = lists
        with self.assertRaises(ValueError) as ve:
            pu.FatigueStress(_counts=cts, _values=[], bin_width=0.5)
            self.assertEqual(
                "No data provided",
                str(ve.exception)
            )
        with self.assertRaises(ValueError) as ve:
            pu.FatigueStress(_counts=[], _values=vals, bin_width=0.5)
            self.assertEqual(
                "No data provided",
                str(ve.exception)
            )


@pytest.mark.parametrize(
    "x, y, expected_slopes, expected_intercepts",
    [
        (
            [1, 2, 3],
            [2, 4, 8],
            np.array([2.0, 4.0]),
            np.array([0.0, -4.0]),
        ),
        (
            [1, 10, 100],
            [10, 100, 1000],
            np.array([10.0, 10.0]),
            np.array([0.0, 0.0]),
        ),
    ],
)
def test_calc_slope_intercept(
    x: ArrayLike,
    y: ArrayLike,
    expected_slopes: ArrayLike,
    expected_intercepts: ArrayLike,
) -> None:
    """Test calc_slope_intercept function"""
    calculated_slopes, calculated_intercepts = pu.calc_slope_intercept(x, y)
    assert np.allclose(calculated_slopes, expected_slopes)
    assert np.allclose(calculated_intercepts, expected_intercepts)


@pytest.mark.parametrize(
    "input_value, expected_output",
    [
        (5, np.array([5])),
        ([1, 2, 3], np.array([1, 2, 3])),
        (np.array([4, 5, 6]), np.array([4, 5, 6])),
    ],
)
def test_ensure_array(input_value, expected_output):
    """Test ensure_array decorator"""

    class TestClass:
        @pu.ensure_array
        def method(self, x):
            return x

    test_instance = TestClass()
    assert np.array_equal(test_instance.method(input_value), expected_output)