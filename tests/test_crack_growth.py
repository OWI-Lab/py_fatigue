"""Reference and integration tests for ordered plate crack propagation."""

from dataclasses import replace
import math

import pytest

from py_fatigue import CycleCount, SNCurve
from py_fatigue.crack_growth import (
    CenterThroughCrack,
    LoadBlock,
    NasgroCurve,
    SemiEllipticalSurfaceCrack,
    simulate_crack_growth,
)
from py_fatigue.stress_life import CycleCount as SharedCycleCount
from py_fatigue.stress_life import SNCurve as StressLifeCurve


def a357_curve() -> NasgroCurve:
    """Convert the NASGRO 3.0 paper's Figure 1 fit to MPa and mm."""
    intensity = 6.894757293 * math.sqrt(25.4)
    return NasgroCurve(
        coefficient=25.4 * 0.623e-8 / intensity**2.180,
        exponent=2.180,
        p=0.5,
        q=2.0,
        threshold_zero=9.69 * intensity,
        threshold_fanning=3.873,
        toughness=27.7 * intensity,
        constraint=1.0,
        peak_to_flow=0.6,
    )


def test_existing_classes_keep_identity():
    assert SharedCycleCount is CycleCount
    assert StressLifeCurve is SNCurve


def test_center_crack_finite_width_reference():
    # NASA TM X-72709 center-cracked tension specimen and secant correction.
    geometry = CenterThroughCrack(half_width=50.0, half_length=25.0)
    intensity = 40.0 * math.sqrt(math.pi * 25.0 * math.sqrt(2.0))
    assert geometry.stress_intensity(40.0) == pytest.approx(intensity)


def test_surface_crack_finite_element_reference():
    # NASA TP-1578 Table I, tension, a/c=0.6, a/t=0.4 and c/b=0.2.
    a = 6.0
    shape = 0.6
    geometry = SemiEllipticalSurfaceCrack(50.0, 15.0, a, 10.0)
    depth, surface = geometry.stress_intensity(1.0)
    scale = math.sqrt(math.pi * a / (1 + 1.464 * shape**1.65))
    # The report compares fit error against the maximum K in the case.
    absolute_tolerance = 0.05 * 1.145
    assert depth / scale == pytest.approx(1.145, abs=absolute_tolerance)
    assert surface / scale == pytest.approx(1.015, abs=absolute_tolerance)


def test_nasgro_published_curve_and_threshold():
    curve = a357_curve()
    intensity = 6.894757293 * math.sqrt(25.4)
    # Figure 1, fitted R=0.4 curve at delta K=10 ksi sqrt(in). The
    # logarithmic plot supports an approximate, rather than tabular, value.
    observed_rate = curve.get_growth_rate(
        100.0, (10 / 0.6 - 10) * intensity, 10 / 0.6 * intensity
    )
    assert 1e-6 < observed_rate / 25.4 < 5e-6
    assert curve.get_growth_rate(100.0, 0.0, intensity) == 0.0
    with pytest.raises(ValueError, match="Load"):
        curve.get_growth_rate(100.0, 0.0, curve.toughness)


@pytest.mark.parametrize(
    "geometry, dimensions",
    [
        (CenterThroughCrack(50.0, 5.0), 1),
        (SemiEllipticalSurfaceCrack(50.0, 15.0, 6.0, 10.0), 2),
    ],
)
def test_ordered_loads_and_convergence(geometry, dimensions):
    curve = a357_curve()
    schedule = [LoadBlock(20.0, 115.0, 200.5), LoadBlock(0.0, 120.0, 150.0)]
    controls = dict(
        length_tolerance=1e-6, cycle_tolerance=1e-4, max_trials=10000
    )
    coarse = simulate_crack_growth(
        geometry, curve, schedule, max_cycle_step=1.0, **controls
    )
    fine = simulate_crack_growth(
        geometry, curve, schedule, max_cycle_step=0.5, **controls
    )
    reverse = simulate_crack_growth(
        geometry, curve, schedule[::-1], max_cycle_step=0.5, **controls
    )
    assert fine.termination == "spectrum_exhausted"
    assert fine.final_cycles == 350.5
    assert len(fine.history[0].sizes) == dimensions
    for initial, final in zip(fine.history[0].sizes, fine.history[-1].sizes):
        assert final > initial
    if dimensions == 2:
        # Independent RK4 calculation of TP-1578 and NASGRO 3.0,
        # using 4,000 steps per block and the fixture's A357 fit.
        assert fine.history[-1].sizes == pytest.approx(
            (6.012803427721083, 10.006956211847994), abs=2e-6
        )
    assert coarse.history[-1].sizes == pytest.approx(
        fine.history[-1].sizes, abs=2e-6
    )
    assert [point.block for point in fine.history] == sorted(
        point.block for point in fine.history
    )
    assert any(
        a != b
        for a, b in zip(fine.history[-1].sizes, reverse.history[-1].sizes)
    )


def test_budget_and_initial_toughness():
    geometry = CenterThroughCrack(50.0, 5.0)
    curve = a357_curve()
    controls = dict(
        max_cycle_step=1.0,
        length_tolerance=1e-6,
        cycle_tolerance=1e-4,
        max_trials=1,
    )
    loads = [LoadBlock(0.0, 80.0, 10.0)]
    with pytest.raises(RuntimeError, match="budget"):
        simulate_crack_growth(geometry, curve, loads, **controls)
    immediate = simulate_crack_growth(
        geometry, replace(curve, toughness=1.0), loads, **controls
    )
    assert immediate.termination == "toughness"
    assert immediate.final_cycles == 0


def test_center_crack_life_against_separated_integral():
    """Compare the terminal cycle count to independent quadrature."""
    start, end, half_width = 5.0, 6.0, 50.0
    maximum, minimum = 20.0, 18.0
    coefficient = 1e-5
    target_toughness = maximum * math.sqrt(
        math.pi * end / math.cos(math.pi * end / (2 * half_width))
    )
    curve = NasgroCurve(
        coefficient=coefficient,
        exponent=2.0,
        p=0.0,
        q=0.0,
        threshold_zero=0.0,
        threshold_fanning=0.0,
        toughness=target_toughness,
        constraint=3.0,
        peak_to_flow=0.6,
    )
    # At R=0.9 these closure inputs give f=R. With n=2 and p=q=0,
    # integrating the reciprocal growth rate gives the life to the cutoff.
    def integrand(length):
        return math.cos(math.pi * length / (2 * half_width)) / (
            math.pi * length * coefficient * (maximum - minimum) ** 2
        )

    segments = 10_000
    width = (end - start) / segments
    expected = width * math.fsum(
        integrand(start + (i + 0.5) * width) for i in range(segments)
    )
    result = simulate_crack_growth(
        CenterThroughCrack(half_width, start),
        curve,
        [LoadBlock(minimum, maximum, 2 * expected)],
        max_cycle_step=0.25,
        length_tolerance=1e-7,
        cycle_tolerance=1e-5,
        max_trials=20_000,
    )
    assert result.termination == "toughness"
    assert result.final_cycles == pytest.approx(expected, rel=2e-4)
    assert result.history[-1].sizes[0] == pytest.approx(end, abs=1e-6)
