# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

# Standard imports
import os
import unittest
from contextlib import nullcontext
from typing import Any, List, Callable, Tuple
from unittest.mock import Mock

# Non-standard imports
import numpy as np
import matplotlib.pyplot as plt
import pytest
from hypothesis import given, strategies as hy
from numpy.typing import ArrayLike

# Project imports
import py_fatigue.utils as pu

os.environ["NUMBA_DISABLE_JIT"] = "1"


@hy.composite
def same_len_lists(draw, min_value=2, max_value=50):
    """Generate two lists of the same length with integers"""
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
    (60, 6, -20, *expected(60.0)),
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


def test_check_iterable_decorator():
    """Test check_iterable decorator"""

    @pu.check_iterable
    def test_func(a, b):
        return a + b

    # Test with scalars
    result = test_func(1, 2)
    assert np.array_equal(result, np.array([3.0]))

    # Test with lists
    result = test_func([1, 2], [3, 4])
    assert np.array_equal(result, np.array([4.0, 6.0]))


def test_check_str_decorator():
    """Test check_str decorator"""

    @pu.check_str
    def test_func(a, b):
        return f"{a}_{b}"

    result = test_func("hello", "world")
    assert result == "hello_world"

    result = test_func(None, "test")
    assert result == "_test"


def test_inplacify_decorator():
    """Test inplacify decorator"""
    from dataclasses import dataclass

    @dataclass
    class TestClass:
        value: int = 0

        @pu.inplacify
        def increment(self):
            self.value += 1
            return self

    # Test inplace=False (default)
    obj = TestClass(value=5)
    new_obj = obj.increment()
    assert obj.value == 5  # Original unchanged
    assert new_obj.value == 6  # New object incremented

    # Test inplace=True
    obj = TestClass(value=5)
    result = obj.increment(inplace=True)
    assert obj.value == 6  # Original changed
    assert result is obj  # Same object returned


def test_make_axes():
    """Test make_axes function"""
    import matplotlib.pyplot as plt

    # Test with no arguments
    fig, ax = pu.make_axes()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)

    # Test with provided figure
    fig_input = plt.figure()
    fig, ax = pu.make_axes(fig=fig_input)
    assert fig is fig_input
    assert isinstance(ax, plt.Axes)
    plt.close(fig)

    # Test with invalid figure type
    with pytest.raises(TypeError, match="fig must be a matplotlib.figure.Figure"):
        pu.make_axes(fig="not_a_figure")


def test_to_numba_dict():
    """Test to_numba_dict function"""
    import numba as nb
    data = {
        "key1": 1.0,
        "key2": 2.0,
        "key3": 3.0
    }

    result = pu.to_numba_dict(data)
    # The next line fails since os.environ["NUMBA_DISABLE_JIT"] = "1"
    if os.getenv("NUMBA_DISABLE_JIT") != "1":
        assert isinstance(result, (nb.typed.Dict, nb.typed.typeddict.Dict))
    else:
        assert isinstance(result, dict)
    assert result["key1"] == 1.0
    assert result["key2"] == 2.0
    assert result["key3"] == 3.0


def test_to_numba_dict_with_invalid_types():
    """Test to_numba_dict with invalid types"""
    import numba as nb

    data = {
        "key1": 1.0,
        "key2": "invalid",  # Wrong type - will be skipped
        123: 3.0,  # Wrong key type - will be skipped
    }

    result = pu.to_numba_dict(data)
    if os.getenv("NUMBA_DISABLE_JIT") != "1":
        assert isinstance(result, (nb.typed.Dict, nb.typed.typeddict.Dict))
    else:
        assert isinstance(result, dict)
    assert result["key1"] == 1.0
    assert "key2" not in result  # Should be skipped
    assert len(result) == 1  # Only valid entries


def test_compare_function():
    """Test compare function"""
    # Test with equal values
    assert pu.compare(1, 1)
    assert pu.compare("test", "test")
    assert pu.compare([1, 2, 3], [1, 2, 3])
    assert pu.compare({"a": 1, "b": 2}, {"a": 1, "b": 2})

    # Test with numpy arrays
    assert pu.compare(np.array([1, 2, 3]), np.array([1, 2, 3]))

    # Test with unequal values
    assert not pu.compare(1, 2)
    assert not pu.compare([1, 2], [1, 2, 3])
    assert not pu.compare({"a": 1}, {"b": 1})

    # Test with nested structures
    assert pu.compare({"a": [1, 2]}, {"a": [1, 2]})
    assert not pu.compare({"a": [1, 2]}, {"a": [1, 3]})

    # Test with different types that cause exceptions
    assert not pu.compare({"a": 1}, [1, 2])


def test_split_function():
    """Test split function"""
    result = pu.split(5, 2)
    assert result == [3, 2]

    result = pu.split(20, 7)
    assert result == [3, 3, 3, 3, 3, 3, 2]

    result = pu.split(10, 3)
    assert result == [4, 3, 3]

    # Edge cases
    result = pu.split(1, 1)
    assert result == [1]

    result = pu.split(0, 5)
    assert result == [0, 0, 0, 0, 0]


def test_chunks_function():
    """Test chunks function"""
    test_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = list(pu.chunks(test_list, 3))
    assert result == [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]

    result = list(pu.chunks(test_list, 5))
    assert result == [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]

    # Edge cases
    result = list(pu.chunks([], 3))
    assert result == []

    result = list(pu.chunks([1, 2], 5))
    assert result == [[1, 2]]


def test_named_tuples():
    """Test NamedTuple classes"""
    # Test IntInt
    int_int = pu.IntInt(mean_bin_nr=5, range_bin_nr=10)
    assert int_int.mean_bin_nr == 5
    assert int_int.range_bin_nr == 10

    # Test ArrayArray
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])
    array_array = pu.ArrayArray(mean_bin_edges=arr1, range_bin_edges=arr2)
    assert np.array_equal(array_array.mean_bin_edges, arr1)
    assert np.array_equal(array_array.range_bin_edges, arr2)

    # Test IntArray
    int_array = pu.IntArray(mean_bin_nr=5, range_bin_edges=arr2)
    assert int_array.mean_bin_nr == 5
    assert np.array_equal(int_array.range_bin_edges, arr2)

    # Test ArrayInt
    array_int = pu.ArrayInt(mean_bin_edges=arr1, range_bin_edges=10)
    assert np.array_equal(array_int.mean_bin_edges, arr1)
    assert array_int.range_bin_edges == 10


def test_fatigue_stress_properties():
    """Test FatigueStress additional properties"""
    counts = np.array([1.0, 2.0, 0.5, 3.0, 0.5])
    values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    fs = pu.FatigueStress(_counts=counts, _values=values, bin_width=5.0)

    # Test binned_values
    binned = fs.binned_values
    assert isinstance(binned, np.ndarray)
    assert len(binned) > 0

    # Test bins_idx
    bins_idx = fs.bins_idx
    assert isinstance(bins_idx, np.ndarray)
    assert len(bins_idx) == len(fs.full)

    # Test full and half properties
    full = fs.full
    half = fs.half
    assert isinstance(full, np.ndarray)
    assert isinstance(half, np.ndarray)
    assert len(half) == 2  # Two 0.5 counts


def test_fatigue_stress_setters():
    """Test FatigueStress property setters"""
    counts = np.array([1.0, 2.0, 3.0])
    values = np.array([10.0, 20.0, 30.0])
    fs = pu.FatigueStress(_counts=counts, _values=values, bin_width=5.0)

    # Test bin_lb setter
    new_lb = 5.0
    fs.bin_lb = new_lb
    assert fs.bin_lb == new_lb

    # Test bin_ub setter
    new_ub = 50.0
    fs.bin_ub = new_ub
    assert fs.bin_ub == new_ub

    # Test counts setter
    new_counts = np.array([2.0, 3.0, 4.0])
    fs.counts = new_counts
    assert np.array_equal(fs.counts, new_counts)

    # Test values setter
    new_values = np.array([15.0, 25.0, 35.0])
    fs.values = new_values
    assert np.array_equal(fs.values, new_values)


def test_py_bisect():
    """Test py_bisect function"""
    def test_func(x):
        return x**2 - 4

    root = pu.py_bisect(test_func, 0, 5, tol=1e-6, mxiter=100)
    assert np.isclose(root, 2.0, atol=1e-5)

    # Test with function that has immediate solution
    def simple_func(x):
        return x - 2

    root = pu.py_bisect(simple_func, 2, 3, tol=1e-6, mxiter=100)
    assert np.isclose(root, 2.0, atol=1e-6)


def test_py_newton():
    """Test py_newton function"""
    def test_func(x):
        return x**2 - 4

    root = pu.py_newton(test_func, x0=1.0, tol=1e-6, mxiter=100)
    assert np.isclose(root, 2.0, atol=1e-5)


def test_numba_bisect():
    """Test numba_bisect wrapper"""
    def test_func(x):
        return x**2 - 4

    root = pu.numba_bisect(test_func, 0, 5, tol=1e-6, mxiter=100)
    assert np.isclose(root, 2.0, atol=1e-5)


def test_numba_newton():
    """Test numba_newton wrapper"""
    def test_func(x):
        return x**2 - 4

    root = pu.numba_newton(test_func, x0=1.0, tol=1e-6, mxiter=100)
    assert np.isclose(root, 2.0, atol=1e-5)


def test_compile_specialized_bisect():
    """Test compile_specialized_bisect function"""
    def test_func(x):
        return (x**2 - 4,)

    compiled_bisect = pu.compile_specialized_bisect(test_func)
    root = compiled_bisect(0, 5, tol=1e-6, mxiter=100)
    assert np.isclose(root, 2.0, atol=1e-5)


def test_compile_specialized_newton():
    """Test compile_specialized_newton function"""
    def test_func(x):
        return x**2 - 4

    compiled_newton = pu.compile_specialized_newton(test_func)
    root = compiled_newton(x0=1.0, tol=1e-6, mxiter=100)
    assert np.isclose(root, 2.0, atol=1e-5)


def test_custom_formatter():
    """Test CustomFormatter class"""
    formatter = pu.CustomFormatter()

    # Create log records for different levels
    import logging

    # Test DEBUG level
    record_debug = logging.LogRecord(
        name="test", level=logging.DEBUG, pathname="test.py",
        lineno=1, msg="Debug message", args=(), exc_info=None
    )
    formatted_debug = formatter.format(record_debug)
    assert "Debug message" in formatted_debug
    assert "🐞" in formatted_debug

    # Test INFO level
    record_info = logging.LogRecord(
        name="test", level=logging.INFO, pathname="test.py",
        lineno=1, msg="Info message", args=(), exc_info=None
    )
    formatted_info = formatter.format(record_info)
    assert "Info message" in formatted_info
    assert "ℹ️" in formatted_info

    # Test WARNING level
    record_warning = logging.LogRecord(
        name="test", level=logging.WARNING, pathname="test.py",
        lineno=1, msg="Warning message", args=(), exc_info=None
    )
    formatted_warning = formatter.format(record_warning)
    assert "Warning message" in formatted_warning
    assert "⚠️" in formatted_warning

    # Test ERROR level
    record_error = logging.LogRecord(
        name="test", level=logging.ERROR, pathname="test.py",
        lineno=1, msg="Error message", args=(), exc_info=None
    )
    formatted_error = formatter.format(record_error)
    assert "Error message" in formatted_error
    assert "⛔" in formatted_error

    # Test CRITICAL level
    record_critical = logging.LogRecord(
        name="test", level=logging.CRITICAL, pathname="test.py",
        lineno=1, msg="Critical message", args=(), exc_info=None
    )
    formatted_critical = formatter.format(record_critical)
    assert "Critical message" in formatted_critical
    assert "🆘" in formatted_critical


def test_calc_slope_intercept_edge_cases():
    """Test calc_slope_intercept function error cases"""
    # Test with mismatched lengths
    with pytest.raises(AssertionError):
        pu.calc_slope_intercept([1, 2], [1, 2, 3])

    # Test with too few points
    with pytest.raises(AssertionError):
        pu.calc_slope_intercept([1], [1])


def test_split_full_cycles_and_residuals_edge_cases():
    """Test split_full_cycles_and_residuals with edge cases"""
    # All zeros
    counts = np.array([0.0, 0.0, 0.0])
    full_idx, half_idx = pu.split_full_cycles_and_residuals(counts)
    assert len(full_idx) == 0
    assert len(half_idx) == 0

    # Mix of zeros and values
    counts = np.array([0.0, 1.0, 0.5, 0.0, 2.0])
    full_idx, half_idx = pu.split_full_cycles_and_residuals(counts)
    assert 1 in full_idx
    assert 4 in full_idx
    assert 2 in half_idx

    # Only half cycles
    counts = np.array([0.5, 0.5, 0.5])
    full_idx, half_idx = pu.split_full_cycles_and_residuals(counts)
    assert len(full_idx) == 0
    assert len(half_idx) == 3


def test_bin_upper_bound_edge_cases():
    """Test bin_upper_bound with edge cases"""
    # Max value exactly on bin edge
    result = pu.bin_upper_bound(0, 1, 5)
    assert result == 5.0

    # Max value between bin edges
    result = pu.bin_upper_bound(0, 1, 5.5)
    assert result == 6.0

    # Negative values
    result = pu.bin_upper_bound(-10, 2, -5)
    assert result == -4.0

    # Max value below lower bound, should return lower bound
    result = pu.bin_upper_bound(10, 2, 5)
    assert result == 10.0


def test_calc_bin_edges_edge_cases():
    """Test calc_bin_edges with edge cases"""
    # Single bin
    result = pu.calc_bin_edges(0, 1, 1)
    assert len(result) == 2

    # Very small bin width
    result = pu.calc_bin_edges(0, 1, 0.1)
    assert len(result) == 11

    # Negative range
    result = pu.calc_bin_edges(-5, -1, 1)
    expected_length = int(np.round((-1 + 1 - (-5)) / 1))
    assert len(result) == expected_length


def test_calc_half_cycles_edge_cases():
    """Test calc_half_cycles with edge cases"""
    # All full cycles
    count_cycle = np.array([1.0, 2.0, 3.0])
    stress = np.arange(1, len(count_cycle) + 1, 1)
    result = pu.calc_half_cycles(stress, count_cycle)
    assert len(result) == 0

    # Mixed full and half cycles
    count_cycle = np.array([1.0, 0.5, 2.0, 0.5])
    stress = np.arange(1, len(count_cycle) + 1, 1)
    result = pu.calc_half_cycles(stress, count_cycle)
    assert len(result) == 2
    assert np.array_equal(result, [2, 4])


def test_calc_full_cycles_edge_cases():
    """Test calc_full_cycles with edge cases"""
    # All half cycles
    count_cycle = np.array([0.5, 0.5, 0.5])
    stress = np.arange(1, len(count_cycle) + 1)
    result = pu.calc_full_cycles(stress, count_cycle)
    assert len(result) == 0

    # Large count values
    count_cycle = np.array([3.0, 2.0, 1.0])
    stress = np.array([1.0, 2.0, 3.0])
    result = pu.calc_full_cycles(stress, count_cycle)
    expected = np.array([1, 1, 1, 2, 2, 3])
    assert np.array_equal(result, expected)

    # Test debug mode
    result = pu.calc_full_cycles(stress, count_cycle, debug_mode=True)
    assert np.array_equal(result, expected)


def test_calc_half_cycles_debug_mode():
    """Test calc_half_cycles with debug mode"""
    count_cycle = np.array([1.0, 0.5, 2.0, 0.5])
    stress = np.array([1.0, 2.0, 3.0, 4.0])
    result = pu.calc_half_cycles(stress, count_cycle, debug_mode=True)
    assert len(result) == 2


def test_fatigue_stress_bin_properties():
    """Test FatigueStress bin-related properties"""
    counts = np.array([1.0, 2.0, 3.0])
    values = np.array([10.0, 20.0, 30.0])
    fs = pu.FatigueStress(_counts=counts, _values=values, bin_width=5.0, _bin_lb=5.0, _bin_ub=35.0)

    # Test bin_edges
    edges = fs.bin_edges
    assert isinstance(edges, np.ndarray)
    assert len(edges) > 0

    # Test bin_centers
    centers = fs.bin_centers
    assert isinstance(centers, np.ndarray)
    assert len(centers) == len(edges) - 1


def test_fatigue_stress_auto_bounds():
    """Test FatigueStress with automatic bound calculation"""
    counts = np.array([1.0, 2.0, 3.0])
    values = np.array([10.0, 20.0, 30.0])
    fs = pu.FatigueStress(_counts=counts, _values=values, bin_width=5.0)

    # Should automatically calculate bounds
    assert fs.bin_lb is not None
    assert fs.bin_ub is not None
    assert fs.bin_ub > fs.bin_lb


def test_plot_damage_accumulation():
    """Test _plot_damage_accumulation function"""
    cumsum_nl_dmg = np.array([0.1, 0.3, 0.6, 0.9, 1.2])
    cumsum_pm_dmg = np.array([0.05, 0.2, 0.5, 0.8, 1.1])
    limit_damage = 1.0

    fig, ax = pu._plot_damage_accumulation(cumsum_nl_dmg, cumsum_pm_dmg, limit_damage)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


@pytest.mark.parametrize(
    "x, y, expected_slopes, expected_intercepts",
    [
        (
            [0, 1, 2],
            [1, 2, 3],
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
        ),
        (
            [1, 3, 5],
            [2, 6, 10],
            np.array([2.0, 2.0]),
            np.array([0.0, 0.0]),
        ),
    ],
)
def test_calc_slope_intercept_additional_cases(
    x: list,
    y: list,
    expected_slopes: np.ndarray,
    expected_intercepts: np.ndarray,
) -> None:
    """Test calc_slope_intercept function with additional cases"""
    calculated_slopes, calculated_intercepts = pu.calc_slope_intercept(x, y)
    assert np.allclose(calculated_slopes, expected_slopes)
    assert np.allclose(calculated_intercepts, expected_intercepts)


def test_typed_array():
    """Test TypedArray class"""
    # This is a complex class that requires pydantic validation
    # Test basic functionality
    arr = np.array([1, 2, 3])
    typed_arr = pu.TypedArray.__new__(pu.TypedArray, arr)
    assert isinstance(typed_arr, np.ndarray)


def test_fatigue_stress_edge_cases():
    """Test FatigueStress edge cases"""
    # Test with empty full cycles
    counts = np.array([0.5, 0.5])  # Only half cycles
    values = np.array([10.0, 20.0])
    fs = pu.FatigueStress(_counts=counts, _values=values, bin_width=5.0)

    # Should handle empty full cycles gracefully
    full = fs.full
    assert len(full) == 0

    # bins_idx should handle empty full cycles
    bins_idx = fs.bins_idx
    assert len(bins_idx) == 0

    # binned_values should handle empty full cycles
    binned = fs.binned_values
    assert len(binned) == 0


def test_json_encoders():
    """Test JSON_ENCODERS"""
    import json

    # Test numpy array encoding
    arr = np.array([1, 2, 3])
    encoded = pu.JSON_ENCODERS[np.ndarray](arr)
    assert encoded == [1, 2, 3]

    # Test in actual JSON encoding
    data = {"array": arr}
    json_str = json.dumps(data, default=lambda x: pu.JSON_ENCODERS.get(type(x), str)(x))
    assert "[1, 2, 3]" in json_str