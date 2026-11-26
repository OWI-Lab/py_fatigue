"""Data and calculations for the FAC-1 and MERC-3 handbook lessons."""

from math import ceil

import numpy as np
from numpy.typing import NDArray

from py_fatigue import CycleCount, ParisCurve
from py_fatigue.damage import get_crack_growth
from py_fatigue.damage.crack_growth import CalcCrackGrowth
from py_fatigue.geometry import InfiniteSurface, TabulatedGeometryFactor

# Figure MERC-3.1, 7075-T6 Al at R = 0, with lengths in inches.
MERC3_COEFFICIENT = 2.5e-9
MERC3_EXPONENT = 3.6
MERC3_STRESS_RANGE = 10.0  # ksi
MERC3_INITIAL_LENGTH = 0.05
MERC3_FINAL_LENGTH = 1.25

# Figure MERC-3.4, digitized (crack length in inches, beta).
MERC3_DIGITIZED = np.array(
    [
        (0.050, 2.646),
        (0.075, 2.220),
        (0.100, 1.968),
        (0.150, 1.596),
        (0.200, 1.392),
        (0.300, 1.164),
        (0.400, 1.020),
        (0.500, 0.918),
        (0.600, 0.882),
        (0.700, 0.882),
        (0.800, 0.894),
        (0.900, 0.918),
        (1.000, 1.002),
        (1.050, 1.086),
        (1.100, 1.206),
        (1.150, 1.428),
        (1.180, 1.662),
        (1.200, 1.992),
        (1.210, 2.196),
        (1.220, 2.496),
        (1.230, 3.024),
        (1.235, 3.156),
    ]
)
MERC3_PUBLISHED_CYCLES = {
    "Reference": 47_400,
    "Increase large β": 45_500,
    "Increase small β": 35_800,
}


def merc3_constant_beta(
    beta: float,
) -> tuple[float, CalcCrackGrowth]:
    """Compare Eq. (11) with the native constant-factor Paris solver."""
    effective_stress = beta * MERC3_STRESS_RANGE
    power = 1 - MERC3_EXPONENT / 2
    exact_cycles = (MERC3_FINAL_LENGTH**power - MERC3_INITIAL_LENGTH**power) / (
        power
        * MERC3_COEFFICIENT
        * (effective_stress * np.sqrt(np.pi)) ** MERC3_EXPONENT
    )
    curve = ParisCurve(
        MERC3_EXPONENT,
        MERC3_COEFFICIENT,
        critical=effective_stress * np.sqrt(np.pi * MERC3_FINAL_LENGTH),
        unit_string="ksi √in",
    )
    cycles = CycleCount(
        stress_range=np.array([effective_stress]),
        mean_stress=np.array([effective_stress / 2]),
        count_cycle=np.array([ceil(1.01 * exact_cycles)], dtype=float),
        unit="ksi",
    )
    result = get_crack_growth(
        cycles, curve, InfiniteSurface(initial_depth=MERC3_INITIAL_LENGTH)
    )
    return float(exact_cycles), result


def merc3_geometry_factors(
    length: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """Interpolate the reference and perturbed MERC-3 geometry factors."""
    source_length, source_beta = MERC3_DIGITIZED.T
    # Hold the last digitized beta to the published end length of 1.25 in.
    beta = np.interp(length, source_length, source_beta)
    return {
        "Reference": beta,
        "Increase large β": beta * np.interp(beta, [1, 3], [1, 1.1]),
        "Increase small β": beta * np.interp(beta, [1, 3], [1.1, 1]),
    }


def paris_reference_cycles(
    length: NDArray[np.float64],
    delta_k: NDArray[np.float64],
    coefficient: float,
    exponent: float,
) -> NDArray[np.float64]:
    """Integrate 1 / (C ΔKⁿ) independently of the package solver."""
    rate = coefficient * delta_k**exponent
    if np.any(~np.isfinite(rate)) or np.any(rate <= 0):
        raise ValueError("The reference history needs finite positive rates.")
    inverse = 1 / rate
    increments = np.diff(length) * (inverse[:-1] + inverse[1:]) / 2
    return np.r_[0.0, np.cumsum(increments)]


def native_paris_growth(
    geometry: TabulatedGeometryFactor,
    curve: ParisCurve,
    stress_ranges: NDArray[np.float64],
) -> CalcCrackGrowth:
    """Run Py-Fatigue on the handbook inputs, one R = 0 cycle per step."""
    cycles = CycleCount(
        stress_range=stress_ranges,
        mean_stress=stress_ranges / 2,
        count_cycle=np.ones_like(stress_ranges),
        unit="ksi",
    )
    return get_crack_growth(cycles, curve, geometry)


def crack_history(
    length: NDArray[np.float64],
    reference_cycles: NDArray[np.float64],
    load_fraction: NDArray[np.float64],
    exponent: float,
) -> tuple[NDArray[np.float64], bool]:
    """Invert a reference Paris history for proportionally scaled loads.

    This applies to a single-slope Paris law with no threshold, toughness
    limit, or load interaction. Return the history through the first
    cycle that reaches the final grid length, or through the whole
    spectrum if it ends first.
    """
    equivalent = np.r_[0.0, np.cumsum(load_fraction**exponent)]
    first_crossing = np.searchsorted(equivalent, reference_cycles[-1])
    return (
        np.interp(equivalent[: first_crossing + 1], reference_cycles, length),
        first_crossing < equivalent.size,
    )


def merc3_beta_histories(
    length: NDArray[np.float64],
) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Return beta and cumulative cycles on the supplied crack-length grid."""
    histories = {}
    for case, factor in merc3_geometry_factors(length).items():
        delta_k = MERC3_STRESS_RANGE * factor * np.sqrt(np.pi * length)
        histories[case] = factor, paris_reference_cycles(
            length, delta_k, MERC3_COEFFICIENT, MERC3_EXPONENT
        )
    return histories


# Tables FAC-1.1 and FAC-1.2. Each array contains the bypass, bearing, and
# separately rounded total K_I rows, in ksi √in.
FAC1_CRACK_RATIOS = np.array([0.09, 0.23, 0.45, 0.90, 1.35])
FAC1_SIF = {
    "NASGRO TC03": np.array(
        [
            [13.09, 15.36, 16.00, 18.28, 20.50],
            [3.63, 3.76, 3.21, 2.79, 2.60],
            [16.72, 19.12, 19.21, 21.08, 23.10],
        ]
    ),
    "FRANC2D coarse": np.array(
        [
            [13.24, 15.18, 16.33, 18.38, 20.44],
            [3.63, 3.67, 3.31, 2.82, 2.59],
            [16.87, 18.85, 19.64, 21.20, 23.02],
        ]
    ),
    "FRANC2D refined": np.array(
        [
            [13.16, 15.37, 16.47, 18.41, 20.57],
            [3.62, 3.73, 3.36, 2.85, 2.63],
            [16.77, 19.10, 19.83, 21.27, 23.19],
        ]
    ),
}


def fac1_totals() -> dict[str, NDArray[np.float64]]:
    """Reconstruct combined Mode I intensities by superposition."""
    return {
        label: bypass + bearing
        for label, (bypass, bearing, _) in FAC1_SIF.items()
    }
