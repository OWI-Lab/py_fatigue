# -*- coding: utf-8 -*-

r"""The following tests are based on the rainflow algorithm 
implementation according to the ASTM standard E1049-85 [1]_,
in its 2017 version.

.. _[1]: https://www.astm.org/e1049-85r17.html
"""


# from hypothesis import  given, strategies as hy
# Standard imports
from contextlib import nullcontext
from typing import Any, Union, Callable, Tuple
from unittest.mock import Mock

import os
import pytest
import warnings

# Non-standard imports
import numpy as np

os.environ["NUMBA_DISABLE_JIT"] = "1"
import py_fatigue.cycle_count.rainflow as rf


def expected(x) -> Tuple[Any, nullcontext]:
    """Mocking function for pytest.mark.parametrize. Expected value"""
    return x, nullcontext()


def error(*args: Any, **kwargs: Any) -> None:
    """Mocking function for pytest.mark.parametrize. Error handling"""
    return Mock(), pytest.raises(*args, **kwargs)


test_names = "input_1, input_2, expected, error"
test_data = [
    ([1], [], *expected([1])),
    ([0], [1, 2], *expected([0, 1, 2])),
    ([0, 1], 0, *error(TypeError, match="can only concatenate")),
]


@pytest.mark.parametrize(test_names, test_data)
def test_concatenate(
    input_1: list,
    input_2: list,
    expected: list,
    error: Callable,
) -> None:
    """This test is here to show the potential of pytest.mark.parametrize
    combined with unittest.mock.Mock to handle multiple tests at once even if
    some exceptions might occur.
    The test verifies list concatenation.

    Parameters
    ----------
    input_1 : list
        list 1
    input_2 : list
        list 2
    expected : list
        Expected result
    error : Callable
        Error handling

    Raises
    ------
    Exception
        Exception is raised if the error handling raises an exception.
    """
    with error:
        calculated_output = input_1 + input_2
        np.testing.assert_allclose(calculated_output, expected)


@pytest.mark.parametrize(
    "x, v, kind, expected, error",
    [
        ([0, 1, -1, 1], 0, "u", *expected([0, 2])),
        ([0, 1, -1, 1], 0, "d", *expected([1])),
        ([0, 1, 3], 0, "d", *expected([])),
        ([0, -1, -3], 0, "u", *expected([])),
        ([-0.2, -0.1, -0.3, 0, 1, -1], 0, None, *expected([3, 4])),
        (
            [-0.2, np.nan, -0.3, 0, 1, -1],
            0,
            None,
            *error(ValueError, match="data contain NaN"),
        ),
    ],
)
def test_findcross_simple(
    x: list, v: float, kind: Union[str, None], expected: list, error: Callable
) -> None:
    """Testing the findcross function.

    Parameters
    ----------
    x : list
        List of values to be analysed.
    v : float
        level v.
    kind : Union[string, none]
        defines type of crossing returned. Possible options are
        'd'  : downcrossings only
        'u'  : upcrossings only
        None : All crossings will be returned
    desired : list
        Expected output.

    See also
    --------
    rf.findcross
    """
    with error:
        np.testing.assert_allclose(rf.findcross(x, v, kind), expected)


def test_findcross() -> None:
    """Testing the findcross function. We expect to identify correctly the
    indices of the upcrossings and downcrossings for the defined level v.

    See also
    --------
    rf.findcross
    """
    np.testing.assert_allclose(rf.findcross([0, 0, 1, -1, 1], 0), [1, 2, 3])
    np.testing.assert_allclose(rf.findcross([0, 1, -1, 1], 0), [0, 1, 2])

    t = np.linspace(0, 7 * np.pi, 250)
    x = np.sin(t)
    ind = rf.findcross(x, 0.75)
    np.testing.assert_allclose(ind, [9, 25, 80, 97, 151, 168, 223, 239])
    ind_u = rf.findcross(x, 0.75, "u")
    np.testing.assert_allclose(ind_u, [9, 80, 151, 223])
    ind_d = rf.findcross(x, 0.75, "d")
    np.testing.assert_allclose(ind_d, [25, 97, 168, 239])
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        rf.findcross([])
        # Verify some things
        assert len(w) == 1  # 1 warning
        assert issubclass(w[0].category, UserWarning)
        assert "crossings found in x" in str(w[0].message)
    with pytest.raises(ValueError) as ve:
        np.testing.assert_allclose(rf.findcross([0, 1, 3], 0, "asfghjkl"), [])
    assert "Unknown cycle-crossing definition!" in ve.value.args[0]


def test_findextrema() -> None:
    """Testing the findextrema function. We expect to identify correctly the
    minima and maxima of a vector.

    See also
    --------
    rf.findextrema
    """
    t = np.linspace(0, 7 * np.pi, 250)
    x = np.sin(t)
    ind = rf.findextrema(x)
    np.testing.assert_allclose(ind, [18, 53, 89, 125, 160, 196, 231])


def test_findtp() -> None:
    """Testing the find turning points function (findtp).

    See also
    --------
    rf.rainflow
    rf.findtp
    """
    import py_fatigue.cycle_count.rainflow as rf

    tp_threshold = 0.3
    t = np.linspace(0, 7 * np.pi, 250)
    x = np.sin(t) + 0.1 * np.sin(50 * t)
    ind = np.hstack([0, rf.findextrema(x), 249])
    # fmt: off
    np.testing.assert_allclose(
        ind,
        [  0,   1,   3,   4,   6,   7,   9,  11,  13,  14,  16,  18,  19,
        21,  23,  25,  26,  28,  29,  31,  33,  35,  36,  38,  39,  41,
        43,  45,  46,  48,  50,  51,  53,  55,  56,  58,  60,  61,  63,
        65,  67,  68,  70,  71,  73,  75,  77,  78,  80,  81,  83,  85,
        87,  88,  90,  92,  93,  95,  97,  99, 100, 102, 103, 105, 107,
       109, 110, 112, 113, 115, 117, 119, 120, 122, 124, 125, 127, 129,
       131, 132, 134, 135, 137, 139, 141, 142, 144, 145, 147, 149, 151,
       152, 154, 156, 157, 159, 161, 162, 164, 166, 167, 169, 171, 173,
       174, 176, 177, 179, 181, 183, 184, 186, 187, 189, 191, 193, 194,
       196, 198, 199, 201, 203, 205, 206, 208, 209, 211, 213, 215, 216,
       218, 219, 221, 223, 225, 226, 228, 230, 231, 233, 235, 237, 238,
       240, 241, 243, 245, 247, 248, 249],
    )
    # fmt: on
    _, tp = t[ind], x[ind]
    idx = rf.findtp(x)
    tp_2 = x[idx]
    np.testing.assert_allclose(tp, tp_2)
    # fmt: off
    truth = [
         0.32484554,  0.60329418,  0.53123749,  0.80278069,  0.72644823,
         0.98764942,  0.85997662,  1.08753972,  0.91869732,  1.07387743,
         0.8601374 ,  0.98178875,  0.76105663,  0.84713811,  0.52982459,
         0.61608977,  0.31876041, -0.30918883, -0.55299245, -0.51168064,
        -0.81113506, -0.70838817, -0.98969837, -0.87909019, -1.0616873 ,
        -0.89950458, -1.07206545, -0.89481234, -1.0167659 , -0.7455551 ,
        -0.852961  , -0.56019981, -0.60749918, -0.30507263, -0.37069363,
         0.56134608,  0.47858707,  0.80363185,  0.72081618,  0.95262794,
         0.84065041,  1.06438217,  0.91478426,  1.09550837,  0.87946353,
         1.01251385,  0.76022273,  0.84493278,  0.58256657,  0.65693807,
         0.31222428,  0.38614209, -0.54762387, -0.47761034, -0.75777485,
        -0.68940454, -0.95914474, -0.83674518, -1.07562624, -0.91864572,
        -1.07940458, -0.87582121, -1.00737202, -0.79810615, -0.8876521 ,
        -0.57861007, -0.66864123, -0.35247912, -0.37928475,  0.496636  ,
         0.46075031,  0.76635478,  0.66806761,  0.95907372,  0.85957563,
         1.04740349,  0.89554418,  1.07849396,  0.90667651,  1.04016907,
         0.77942286,  0.89166956,  0.60611396,  0.6599374 ,  0.36430669,
         0.43109291, -0.50462438, -0.4250993 , -0.75730568, -0.67754055,
        -0.92000805, -0.81743808, -1.05095363, -0.90691929, -1.0995006 ,
        -0.89513098, -1.03364224, -0.79059311, -0.88406121, -0.63139759,
        -0.70834311, -0.36957205, -0.44614672,  0.49014759,  0.42189424,
         0.71009649,  0.64923963,  0.92711005,  0.80985911,  1.05981514,
         0.9146325 ,  1.08094452,  0.8876392 ,  1.02927349,  0.83182171,
         0.92501331,  0.62475154,  0.71879492,  0.40761235,  0.43865593
    ]
    # fmt: on
    np.testing.assert_allclose(tp[np.abs(tp) > tp_threshold], truth)

    rfs = rf.findrfc_astm(tp)
    rfs_2 = rf.findrfc_astm(tp_2)
    np.testing.assert_allclose(rfs, rfs_2)

    x_2 = [np.random.random()]
    err_msg = "findtp should return an empty list if no tp"
    assert len(rf.findtp(x_2)) == 0, err_msg


test_names = "trng_pts, time, expected"
test_data = [
    ([-1, -2, 1, -3, 5, -1, 3, -4, 4, -3, 1, -2, 3, 2, 6, 5], None),
    (
        [-1, -2, 1, -3, 5, -1, 3, -4, 4, -3, 1, -2, 3, 2, 6, 5],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ),
]


@pytest.mark.parametrize(
    "trng_pts, time",
    [
        ([-1, -2, 1, -3, 5, -1, 3, -4, 4, -3, 1, -2, 3, 2, 6, 5], None),
        (
            [-1, -2, 1, -3, 5, -1, 3, -4, 4, -3, 1, -2, 3, 2, 6, 5],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        ),
    ],
)
def test_rainflow(trng_pts: list, time: Union[list, None]) -> None:
    """Testing the rainflow function and epecting the results of the
    example given in the :term:`rainflow<Rainflow>` documentation.

    Parameters
    ----------
    trng_pts : list
        Turning points
    time : Union[list, None]
        Dummy time array

    See also
    --------
    rf.rainflow
    """
    rf_out = rf.rainflow(trng_pts, time=time, extended_output=True)
    rf_out_2 = rf.rainflow(trng_pts, time=time, extended_output=False)
    np.testing.assert_allclose(
        rf_out[0][:, 0],
        [0.5, 1.5, 2.0, 2.0, 4.0, 1.5, 0.5, 3.5, 4.5, 5.0, 0.5],
    )
    np.testing.assert_allclose(
        rf_out[0][:, 1],
        [-1.5, -0.5, -1.0, 1.0, 1.0, -0.5, 2.5, 0.5, 0.5, 1.0, 5.5],
    )
    np.testing.assert_allclose(
        rf_out[0][:, 2],
        [0.5, 0.5, 0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5],
    )
    np.testing.assert_allclose(
        rf_out[0][:, 3],
        [0.0, 1.0, 2.0, 5.0, 3.0, 10.0, 12.0, 8.0, 4.0, 7.0, 14.0],
    )
    np.testing.assert_allclose(
        rf_out[0][:, 4],
        [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 6.0, 14.0, 2.0],
    )
    for j in range(4):
        np.testing.assert_allclose(rf_out[0][:, j], rf_out_2[:, j])

    np.testing.assert_allclose(rf_out[1], [-1, -2,  1, -3,  5, -4,  6,  5])
    np.testing.assert_allclose(rf_out[2], [0, 1, 2, 3, 4, 7, 14, 15])
    # np.testing.assert_allclose(rf_out[3], [2, 3, 4, 5, 8, 16])

    with pytest.raises(ValueError) as te:
        rf.rainflow(trng_pts, time=np.array([0, 1, 2]))
    assert "time and data must have the same length" in te.value.args[0]
    with pytest.raises(TypeError) as te:
        rf.rainflow(1, time=3)
    assert "data shall be either numpy.1darray or list" in te.value.args[0]


os.environ["NUMBA_DISABLE_JIT"] = "0"
