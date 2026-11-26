"""Reference and integration tests for ordered plate crack propagation."""

from dataclasses import replace
import math

import numpy as np
import pytest

from py_fatigue import CycleCount, SNCurve
from py_fatigue.crack_growth import (
    AxialCylinderThroughCrack,
    CircumferentialCylinderSurfaceCrack,
    CircumferentialCylinderThroughCrack,
    SphereThroughCrack,
    CenterThroughCrack,
    EdgeThroughCrack,
    HoleBoreSurfaceCrack,
    LoadBlock,
    NasgroCurve,
    RolledBoltThreadSurfaceCrack,
    RoundBarSurfaceCrack,
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


@pytest.mark.parametrize(
    "radius, depth, half_length, reference",
    [
        (20.0, 5.0, 6.25, [1.106, 1.155]),
        (40.0, 8.0, 8.0, [1.072, 1.381]),
        (30.0, 3.5, 5.0, [1.113, 1.053875]),
    ],
)
def test_sc05_external_pipe_tension(
    radius: float, depth: float, half_length: float, reference: list[float]
) -> None:
    """Reproduce NASA TM-87594 Table 1 and an eight-corner midpoint."""
    crack = CircumferentialCylinderSurfaceCrack(
        radius, 10.0, depth, half_length
    )
    q = 1 + 1.464 * (depth / half_length) ** 1.65
    scale = 30.0 * math.sqrt(math.pi * depth / q)
    assert crack.stress_intensity(30.0) == pytest.approx(
        [factor * scale for factor in reference]
    )


@pytest.mark.parametrize(
    "depth, reference", [(0.5, 0.84), (2.5, 0.62), (5.0, 1.0)]
)
def test_sc08_rolled_thread_tension(depth: float, reference: float) -> None:
    """Use the minor diameter in NLR-TP-99313 Fig. 2 and Table 2."""
    crack = RolledBoltThreadSurfaceCrack(12.0, 10.0, depth)
    assert crack.stress_intensity(20.0) == pytest.approx(
        20.0 * math.sqrt(math.pi * depth) * reference
    )


def test_sc11_single_bore_crack_matches_published_curve() -> None:
    """Compare the two tips to the wide-plate NASA TM-85793 Fig. 9(c)."""
    crack = HoleBoreSurfaceCrack(1e6, 20.0, 20.0, 5.0, 5.0, 1)
    scale = math.sqrt(math.pi * 5.0 / 2.464)
    radial, bore = crack.stress_intensity(1.0)
    # The plot and fitted equations differ slightly at the two endpoints.
    assert radial / scale == pytest.approx(1.8, abs=0.11)
    assert bore / scale == pytest.approx(2.75, abs=0.11)
    two = replace(crack, crack_count=2)
    # Equation 64: when a/t=c/r=0.5, the product a*c/(t*r) is 0.25.
    factor = math.sqrt((4 / math.pi + 0.125) / (4 / math.pi + 0.25))
    assert crack.stress_intensity(1.0) == pytest.approx(
        [factor * value for value in two.stress_intensity(1.0)]
    )


@pytest.mark.parametrize(
    "crack",
    [
        CircumferentialCylinderSurfaceCrack(20.0, 10.0, 1.0, 1.25),
        CircumferentialCylinderSurfaceCrack(20.0, 10.0, 4.0, 7.0),
        RolledBoltThreadSurfaceCrack(12.0, 10.0, 5.01),
        RolledBoltThreadSurfaceCrack(10.0, 12.0, 1.0),
        HoleBoreSurfaceCrack(80.0, 12.0, 12.0, 4.0, 6.0, 1),
        HoleBoreSurfaceCrack(80.0, 12.0, 12.0, 2.2, 2.5, 3),
    ],
)
def test_other_surface_cases_reject_invalid_geometry(
    crack: (
        CircumferentialCylinderSurfaceCrack
        | RolledBoltThreadSurfaceCrack
        | HoleBoreSurfaceCrack
    ),
) -> None:
    with pytest.raises(ValueError, match="Geometry"):
        crack.stress_intensity(25.0)


@pytest.mark.parametrize(
    "crack",
    [
        CircumferentialCylinderSurfaceCrack(20.0, 10.0, 4.0, 5.0),
        RolledBoltThreadSurfaceCrack(12.0, 10.0, 1.5),
        HoleBoreSurfaceCrack(80.0, 12.0, 12.0, 2.2, 2.5, 2),
    ],
)
def test_ordered_surface_growth_and_profiles(
    crack: (
        CircumferentialCylinderSurfaceCrack
        | RolledBoltThreadSurfaceCrack
        | HoleBoreSurfaceCrack
    ),
) -> None:
    curve = NasgroCurve(1e-10, 2.0, 0.5, 1.0, 25.0, 0.0, 500.0, 2.0, 0.3)
    loads = [LoadBlock(10.0, 70.0, 1_000), LoadBlock(20.0, 90.0, 1_000)]
    result = simulate_crack_growth(
        crack,
        curve,
        loads,
        max_cycle_step=10.0,
        length_tolerance=1e-6,
        cycle_tolerance=1e-4,
        max_trials=400,
    )
    assert result.termination == "spectrum_exhausted"
    assert result.final_cycles == 2_000
    first, last = result.history[0], result.history[-1]
    assert first.k_max == pytest.approx(
        np.atleast_1d(crack.stress_intensity(70.0))
    )
    assert all(end > start for start, end in zip(first.sizes, last.sizes))
    fig, ax = crack.plot(result=result)
    try:
        initial, final = ax.lines[-2:]
        assert initial.get_label() == "Initial crack"
        assert final.get_label() == "Final accepted crack"
        assert not np.allclose(
            initial.get_xdata(), final.get_xdata(), equal_nan=True
        )
    finally:
        from matplotlib import pyplot as plt

        plt.close(fig)


def test_sc11_stops_on_the_admissible_side_of_the_aspect_limit() -> None:
    crack = HoleBoreSurfaceCrack(80.0, 12.0, 12.0, 2.2, 2.5, 2)
    curve = NasgroCurve(1e-10, 2.0, 0.5, 1.0, 25.0, 0.0, 500.0, 2.0, 0.3)
    loads = [LoadBlock(10.0, 70.0, 50_000), LoadBlock(20.0, 90.0, 25_005)]
    result = simulate_crack_growth(
        crack,
        curve,
        loads,
        max_cycle_step=10.0,
        length_tolerance=1e-6,
        cycle_tolerance=1e-4,
        max_trials=2 * sum(math.ceil(block.cycles / 10.0) for block in loads),
    )
    assert result.termination == "geometry_limit"
    assert 50_000 < result.final_cycles < 75_005
    depth, bore_span = result.history[-1].sizes
    assert bore_span / depth <= 2.0
    assert bore_span / depth == pytest.approx(2.0, abs=1e-6)
    replace(crack, depth=depth, half_length=bore_span).stress_intensity(90.0)


@pytest.mark.parametrize(
    "fraction, reference", [(0.1, 1.1892), (0.4, 2.1114), (0.8, 11.955)]
)
def test_edge_crack_uniform_tension_reference(
    fraction: float, reference: float
) -> None:
    """Compare the Tada fit to NASA NTRS 19860020138 Table 3."""
    crack = EdgeThroughCrack(50.0, 50.0 * fraction)
    factor = crack.stress_intensity(1.0) / math.sqrt(math.pi * crack.length)
    assert factor == pytest.approx(reference, rel=0.01)


def test_edge_crack_grows_and_checks_the_solution_domain() -> None:
    crack = EdgeThroughCrack(50.0, 5.0)
    result = simulate_crack_growth(
        crack,
        a357_curve(),
        [LoadBlock(20.0, 115.0, 150.0)],
        max_cycle_step=1.0,
        length_tolerance=1e-6,
        cycle_tolerance=1e-4,
        max_trials=150,
    )
    assert result.termination == "spectrum_exhausted"
    assert result.final_cycles == 150.0
    assert result.history[-1].sizes[0] > crack.length
    assert result.dimension_names == ("Length",)
    with pytest.raises(ValueError, match="Geometry"):
        EdgeThroughCrack(50.0, 41.0).stress_intensity(1.0)


def test_through_cracks_net_section_in_membrane_tension() -> None:
    assert CenterThroughCrack(50.0, 5.0).net_section_stress(90.0) == 100.0
    # Edge: 90 mm ligament, 5 mm load eccentricity and 0.74 of the
    # outer fibre distance. These are per unit plate thickness.
    area = 90.0
    bending = (100.0 * 100.0 * 5.0) * (0.74 * 45.0) / (area**3 / 12)
    crack = EdgeThroughCrack(100.0, 10.0)
    assert crack.net_section_stress(100.0) == pytest.approx(
        10000.0 / area + bending
    )
    with pytest.raises(ValueError, match="Load"):
        crack.net_section_stress(-1.0)


def test_surface_crack_net_section_against_section_integrals() -> None:
    """Independently integrate the removed half-ellipse's section moments."""
    width, thickness, depth, half_length = 20.0, 10.0, 4.0, 4.0
    points, weights = np.polynomial.legendre.leggauss(24)
    angle = (points + 1) * math.pi / 4
    z = depth * np.sin(angle)
    d_area = (
        weights * math.pi / 4 * 2 * half_length * depth * np.cos(angle) ** 2
    )
    area = width * thickness - d_area.sum()
    centroid = (width * thickness**2 / 2 - d_area @ z) / area
    inertia = width * thickness**3 / 3 - d_area @ z**2 - area * centroid**2
    force = 100.0 * width * thickness
    moment = force * (centroid - thickness / 2)
    expected = force / area + moment * 0.87 * centroid / inertia
    crack = SemiEllipticalSurfaceCrack(width / 2, thickness, depth, half_length)
    assert crack.net_section_stress(100.0) == pytest.approx(expected)


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
    with pytest.raises(
        RuntimeError,
        match=r"1-trial budget in load block 1 after 1 cumulative cycles.*max_trials",
    ):
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
    coarse = simulate_crack_growth(
        CenterThroughCrack(half_width, start),
        curve,
        [LoadBlock(minimum, maximum, 2 * expected)],
        max_cycle_step=20.0,
        length_tolerance=1e-3,
        cycle_tolerance=1.0,
        max_trials=20_000,
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
    assert coarse.termination == "toughness"
    assert abs(result.final_cycles - expected) < abs(
        coarse.final_cycles - expected
    )
    assert 0 < result.final_cycles < 2 * expected
    assert result.final_cycles == pytest.approx(expected, rel=2e-4)
    assert result.history[-1].sizes[0] == pytest.approx(end, abs=1e-6)


def test_toughness_crossing_at_start_of_next_block():
    geometry = CenterThroughCrack(50.0, 5.0)
    curve = replace(a357_curve(), toughness=geometry.stress_intensity(60.0))
    result = simulate_crack_growth(
        geometry,
        curve,
        [LoadBlock(10.0, 10.0, 50.5), LoadBlock(10.0, 80.0, 100.0)],
        max_cycle_step=10.0,
        length_tolerance=1e-6,
        cycle_tolerance=1e-4,
        max_trials=100,
    )
    assert result.termination == "toughness"
    assert result.final_cycles == 50.5
    assert result.history[-1].sizes == (5.0,)


def test_cycle_resolution_error_is_distinct_from_trial_budget():
    geometry = CenterThroughCrack(50.0, 5.0)
    result_curve = a357_curve()
    with pytest.raises(RuntimeError, match="too small.*load block 2"):
        simulate_crack_growth(
            geometry,
            result_curve,
            [LoadBlock(1.0, 1.0, 2.0**54), LoadBlock(1.0, 1.0, 0.5)],
            max_cycle_step=2.0**54,
            length_tolerance=1e-6,
            cycle_tolerance=1e-4,
            max_trials=10,
        )


def test_sc07_tensile_intensity_against_flagro_reference() -> None:
    """NASA TM-102165 Table 5, RD1, in ksi and inches before conversion."""
    crack = RoundBarSurfaceCrack(diameter=0.90 * 25.4, depth=0.400 * 25.4)
    intensity = crack.stress_intensity(47.2 * 6.894757293)
    assert intensity / (6.894757293 * math.sqrt(25.4)) == pytest.approx(
        73.9, abs=0.05
    )


def test_sc07_propagates_depth_and_rejects_diameter_midpoint() -> None:
    crack = RoundBarSurfaceCrack(100.0, 4.0)
    result = simulate_crack_growth(
        crack,
        replace(a357_curve(), threshold_zero=0.0),
        [LoadBlock(20.0, 120.0, 150.0)],
        max_cycle_step=1.0,
        length_tolerance=1e-6,
        cycle_tolerance=1e-4,
        max_trials=150,
    )
    assert result.termination == "spectrum_exhausted"
    assert result.final_cycles == 150
    assert result.dimension_names == ("Depth",)
    assert result.history[-1].sizes[0] > crack.depth
    assert result.history[0].k_max == (crack.stress_intensity(120.0),)
    with pytest.raises(ValueError, match="Geometry"):
        RoundBarSurfaceCrack(100.0, 50.0).stress_intensity(120.0)


@pytest.mark.parametrize(
    "geometry, correction",
    [
        (SphereThroughCrack(100.0, 1.0, 10.0), 4.0**0.4),
        (AxialCylinderThroughCrack(100.0, 1.0, 10.0), math.sqrt(2.736)),
    ],
)
def test_through_shell_factors_at_unit_curvature(geometry, correction) -> None:
    """NASA CR-4706 A-27 and A-36 at c/sqrt(Rt) = 1."""
    assert geometry.stress_intensity(100.0) == pytest.approx(
        100.0 * math.sqrt(math.pi * 10.0) * correction
    )


@pytest.mark.parametrize(
    "radius_ratio, angle, reference", [(10.0, 10, 1.05), (40.0, 40, 2.16)]
)
def test_tc08_tension_factor_against_shell_finite_elements(
    radius_ratio: float, angle: float, reference: float
) -> None:
    """Compare Forman/Sanders factors with Rege, Table 8.2."""
    radius = 100.0
    half_length = radius * math.radians(angle)
    geometry = CircumferentialCylinderThroughCrack(
        radius, radius / radius_ratio, half_length, 3000.0, 0.3
    )
    factor = geometry.stress_intensity(1.0) / math.sqrt(math.pi * half_length)
    # The fit and the shell finite-element table are different solutions.
    assert factor == pytest.approx(reference, abs=0.02)


@pytest.mark.parametrize(
    "geometry, invalid",
    [
        (SphereThroughCrack(100.0, 1.0, 10.0), {"half_length": 30.1}),
        (AxialCylinderThroughCrack(100.0, 1.0, 10.0), {"half_length": 100.1}),
        (
            CircumferentialCylinderThroughCrack(100.0, 1.0, 10.0, 2500.0, 0.3),
            {"length": 1900.0},
        ),
    ],
)
def test_through_shell_growth_and_domain(geometry, invalid) -> None:
    result = simulate_crack_growth(
        geometry,
        replace(a357_curve(), threshold_zero=0.0),
        [LoadBlock(0.0, 40.0, 10.0), LoadBlock(10.0, 50.0, 10.0)],
        max_cycle_step=1.0,
        length_tolerance=1e-6,
        cycle_tolerance=1e-4,
        max_trials=40,
    )
    assert result.final_cycles == 20
    assert result.history[-1].sizes[0] > geometry.half_length
    assert result.history[0].k_max == (geometry.stress_intensity(40.0),)
    with pytest.raises(ValueError, match="Geometry"):
        replace(geometry, **invalid).stress_intensity(40.0)
