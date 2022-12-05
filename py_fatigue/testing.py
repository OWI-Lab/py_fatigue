from __future__ import annotations
from typing import Any
import warnings

import numpy as np
import numpy.typing as npt

from py_fatigue.styling import py_fatigue_formatwarning


def get_sampled_time(
    fs: float, duration: float, start: float = 0.0
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    Get the time vector of the signal.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    duration : float
        Duration of the signal in s.
    start : float, optional
        Start time of the signal in s. The default is 0.0.

    Returns
    -------
    np.ndarray
        Time vector of the signal.
    """
    if start < 0:
        raise ValueError("Start time must be positive.")
    if duration < 0:
        raise ValueError("Duration must be positive.")
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive.")
    if duration < 1 / fs:
        raise ValueError(
            "The signal duration is too short for the sampling frequency."
        )
    if np.absolute(duration % fs) > 1e-3:
        w_msg = (
            "The signal duration is not a multiple of the sampling frequency."
            f"The signal will last {duration - (duration % (1 / fs))} s."
        )
        warnings.formatwarning = py_fatigue_formatwarning
        warnings.warn(w_msg, UserWarning)

    return np.arange(start, start + duration, 1 / fs)


def get_random_data(
    t: np.ndarray[Any, np.dtype[np.float64]],
    seed: int = 2,
    random_type: str = "normal",
    min_: float = -10.0,
    range_: float = 50.0,
    **kwargs: Any,
) -> npt.NDArray:
    """
    Get random data from a given array.

    Parameters
    ----------
    t : array_like
        Array from which to get data (time).
    seed : int
        Seed for the random generator.
    random_type : str, optional
        Type of random data to generate. The default is "normal". See
        `numpy.random` for more information.
    min_ : float, optional
        Minimum value of the random data. The default is -10.0.
    range_ : float, optional
        Range of the random data. The default is 50.0.
    **kwargs : Any
        Keyword arguments for the random generators, see:
        https://numpy.org/doc/stable/reference/random/generator.html

    Returns
    -------
    array_like
        Random data.
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    y = getattr(rng, random_type)(**kwargs, size=len(t))
    y = ((y - y.min()) / (y.max() - y.min())) * range_ + min_
    return y
