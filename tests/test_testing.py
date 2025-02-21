import pytest
import numpy as np
from py_fatigue.testing import get_sampled_time

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_get_sampled_time_valid():
    fs = 100.0
    duration = 1.0
    start = 0.0
    expected = np.arange(start, start + duration, 1 / fs)
    result = get_sampled_time(fs, duration, start)
    np.testing.assert_array_equal(result, expected)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_get_sampled_time_negative_start():
    with pytest.raises(ValueError, match="Start time must be positive."):
        get_sampled_time(100.0, 1.0, start=-1.0)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_get_sampled_time_negative_duration():
    with pytest.raises(ValueError, match="Duration must be positive."):
        get_sampled_time(100.0, -1.0)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_get_sampled_time_zero_fs():
    with pytest.raises(ValueError, match="Sampling frequency must be positive."):
        get_sampled_time(0.0, 1.0)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_get_sampled_time_duration_too_short():
    with pytest.raises(ValueError, match="The signal duration is too short for the sampling frequency."):
        get_sampled_time(100.0, 0.001)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_get_sampled_time_duration_not_multiple_of_fs():
    fs = 100.0
    duration = 1.005
    start = 0.0
    with pytest.warns(UserWarning, match="The signal duration is not a multiple of the sampling frequency."):
        result = get_sampled_time(fs, duration, start)
    expected = np.arange(start, start + duration, 1 / fs)
    np.testing.assert_array_almost_equal(result, expected)