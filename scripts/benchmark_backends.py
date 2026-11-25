"""Measure representative Py-Fatigue calls with either installed backend.

Run this file from outside a checkout with the desired interpreter. Use a
fresh NUMBA_CACHE_DIR for the v2 cold measurement. The inputs are synthetic.
The peak memory figure uses the Unix resource API.
"""

from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
import json
import resource
from statistics import median
from time import perf_counter

import numpy as np

started = perf_counter()
from py_fatigue import CycleCount, ParisCurve, SNCurve, __version__
from py_fatigue.damage.crack_growth import get_crack_growth
from py_fatigue.damage.stress_life import calc_theil_sn_damage, get_pm
from py_fatigue.geometry import InfiniteSurface
from py_fatigue.mean_stress.corrections import (
    goodman_haigh_mean_stress_correction,
)

import_seconds = perf_counter() - started


def run():
    """Measure first and warm calls and collect results for comparison."""
    epoch = datetime(2024, 1, 1, tzinfo=timezone.utc)
    signal = np.random.default_rng(7).normal(0, 20, 50_000)
    cycle_count = CycleCount.from_timeseries(signal[:500], timestamp=epoch)
    sn_curve = SNCurve(3.0, 12.0)
    paris = ParisCurve(3.0, 1e-12)
    cases = {
        "long_rainflow": lambda: CycleCount.from_timeseries(
            signal, timestamp=epoch
        ),
        "sn_scalar": lambda: sn_curve.get_stress(100_000.0),
        "sn_batch": lambda: sn_curve.get_cycles(np.linspace(1, 300, 2048)),
        "paris_batch": lambda: paris.get_growth_rate(np.linspace(1, 80, 2048)),
        "miner": lambda: get_pm(cycle_count, sn_curve),
        "theil": lambda: calc_theil_sn_damage(
            np.array([100.0, 120.0]), np.array([40.0, 80.0]), sn_curve
        ),
        "goodman": lambda: goodman_haigh_mean_stress_correction(
            np.array([20.0, 40.0]),
            np.array([100.0, 105.0]),
            r_out=np.array([0.5]),
            ult_s=400,
            correction_exponent=2.0,
        ),
        "legacy_growth": lambda: get_crack_growth(
            cycle_count, paris, InfiniteSurface(initial_depth=0.8)
        ),
    }
    timings = {}
    compare = {}
    for name, call in cases.items():
        with redirect_stdout(StringIO()):
            before = perf_counter()
            result = call()
            first = perf_counter() - before
            warm = []
            for _ in range(7):
                before = perf_counter()
                call()
                warm.append(perf_counter() - before)
        timings[name] = {"first_sec": first, "warm_sec": median(warm)}
        if name == "long_rainflow":
            result = [
                result.stress_range,
                result.mean_stress,
                result.count_cycle,
                result.residuals_sequence,
            ]
        if name == "legacy_growth":
            result = [
                result.crack_depth,
                result.sif,
                result.geometry_factor,
                result.final_cycles,
            ]
        if name == "theil":
            result = [(cycles, stress) for _, cycles, stress in result]
        compare[name] = result
    return {
        "version": __version__,
        "import_seconds": import_seconds,
        "peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "timings": timings,
        "comparison": compare,
    }


if __name__ == "__main__":

    def encode(value):
        if isinstance(value, (np.ndarray, np.generic)):
            return value.tolist()
        raise TypeError(f"Cannot encode {type(value).__name__}")

    print(json.dumps(run(), default=encode))
