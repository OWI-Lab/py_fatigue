import numpy as np
import pytest

from py_fatigue.damage.stress_life import find_sn_curve_intersection
from py_fatigue.mean_stress.corrections import goodman_haigh_mean_stress_correction


def test_goodman_haigh_unsorted_r_out_branch():
    amp_in = np.array([100.0, 150.0])
    mean_in = np.array([10.0, 20.0])
    r_out = np.array([0.2, -1.0])  # intentionally unsorted

    amp_out, mean_out = goodman_haigh_mean_stress_correction(
        amp_in,
        mean_in,
        r_out,
        ult_s=900.0,
        correction_exponent=1.0,
    )

    assert amp_out.shape == (2, 2)
    assert mean_out.shape == (2, 2)


def test_goodman_clip_path_with_non_linear_exponent():
    amp_in = np.array([100.0, 200.0])
    mean_in = np.array([5.0, 10.0])
    r_out = np.array([-1.0, 0.0])

    amp_out, mean_out = goodman_haigh_mean_stress_correction(
        amp_in,
        mean_in,
        r_out,
        ult_s=500.0,
        correction_exponent=2.0,
    )

    assert np.all(np.isfinite(amp_out))
    assert np.all(amp_out >= 0.0)
    assert np.all(amp_out <= 500.0)
    assert np.all(np.isfinite(mean_out))


def test_find_sn_curve_intersection_wrapper_executes():
    out = find_sn_curve_intersection(
        np.array([3.0, 5.0]),
        np.array([12.0, 15.0]),
        1e8,
        120.0,
        0.0,
        1e2,
        1e9,
    )

    assert np.isfinite(out)
    assert out > 0.0
