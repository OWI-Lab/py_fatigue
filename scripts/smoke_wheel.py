"""Check public workflows against an installed native distribution."""

from importlib.metadata import distribution
from importlib.util import find_spec
import math

from py_fatigue import CycleCount, SNCurve
from py_fatigue.crack_growth import (
    CenterThroughCrack,
    LoadBlock,
    NasgroCurve,
    simulate_crack_growth,
)
from py_fatigue.cycle_count.rainflow import findcross_indices

import numpy as np

installed = distribution("py-fatigue")
assert all(
    not dependency.lower().startswith(("numba", "llvmlite"))
    for dependency in (installed.requires or [])
)
license_files = set(installed.metadata.get_all("License-File", []))
assert {"LICENSE", "THIRD_PARTY_NOTICES"} <= license_files
(notice_file,) = [
    path for path in installed.files or () if path.name == "THIRD_PARTY_NOTICES"
]
notices = notice_file.read_text(encoding="utf-8")
assert notices is not None
assert (
    "Toshiki Teramura" in notices and "PyO3 Project and Contributors" in notices
)
assert find_spec("numba") is None
np.testing.assert_array_equal(
    findcross_indices(np.array([-1, 1, -1], dtype=np.int8)),
    [0, 1],
)
signal = np.array([4, 7, 2, 10, 5, 9, 3, 4, 2, 12, 5, 11], dtype=float)
assert CycleCount.from_timeseries(signal).count_cycle.size > 0
assert SNCurve(3.0, 12.0).get_cycles(100.0)[0] == 1e6
assert CenterThroughCrack(100.0, 10.0).stress_intensity(
    20.0
) > 20.0 * math.sqrt(math.pi * 10.0)

geometry = CenterThroughCrack(100.0, 10.0)
curve = NasgroCurve(1e-12, 2.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 3.0, 0.6)
growth = simulate_crack_growth(
    geometry,
    curve,
    [LoadBlock(0.0, 20.0, 10.0)],
    max_cycle_step=1,
    length_tolerance=1e-7,
    cycle_tolerance=1e-4,
    max_trials=100,
)
assert growth.termination == "spectrum_exhausted"
assert growth.history[-1].sizes[0] > geometry.half_length
