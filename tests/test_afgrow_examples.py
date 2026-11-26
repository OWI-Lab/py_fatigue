"""Regression checks against the USAF Damage Tolerant Design Handbook."""

import numpy as np
import pytest

from notebooks._examples.afgrow import (
    FAC1_CRACK_RATIOS,
    FAC1_SIF,
    MERC3_COEFFICIENT,
    MERC3_EXPONENT,
    MERC3_FINAL_LENGTH,
    MERC3_INITIAL_LENGTH,
    MERC3_PUBLISHED_CYCLES,
    MERC3_STRESS_RANGE,
    crack_history,
    fac1_totals,
    merc3_beta_histories,
    merc3_constant_beta,
    native_paris_growth,
    paris_reference_cycles,
)
from py_fatigue import ParisCurve
from py_fatigue.geometry import TabulatedGeometryFactor


def test_merc3_constant_beta_life() -> None:
    """The native Paris history reaches the length used in Eq. (11)."""
    reference, growth = merc3_constant_beta(1.0)
    increased, faster_growth = merc3_constant_beta(1.1)

    for exact, result in [(reference, growth), (increased, faster_growth)]:
        assert result.failure
        assert result.final_cycles == pytest.approx(exact, rel=5e-5)
        assert result.crack_depth[-1] == pytest.approx(
            MERC3_FINAL_LENGTH, abs=5e-4
        )
    assert increased / reference == pytest.approx(1.1**-3.6)


def test_merc3_digitized_beta_life() -> None:
    """Figure MERC-3.5 is an approximate check on the digitized input."""
    length = np.linspace(MERC3_INITIAL_LENGTH, MERC3_FINAL_LENGTH, 1201)
    histories = merc3_beta_histories(length)
    finer = merc3_beta_histories(
        np.linspace(MERC3_INITIAL_LENGTH, MERC3_FINAL_LENGTH, 2401)
    )

    curve = ParisCurve(MERC3_EXPONENT, MERC3_COEFFICIENT, unit_string="ksi √in")
    stress = np.full(
        max(MERC3_PUBLISHED_CYCLES.values()) * 2, MERC3_STRESS_RANGE
    )
    for case, published in MERC3_PUBLISHED_CYCLES.items():
        beta, cycles = histories[case]
        assert np.all(np.diff(cycles) > 0)
        assert cycles[-1] == pytest.approx(finer[case][1][-1], abs=1)
        assert cycles[-1] == pytest.approx(published, rel=0.02)
        geometry = TabulatedGeometryFactor(length[0], length, beta)
        native = native_paris_growth(geometry, curve, stress)
        assert native.termination == "geometry_limit"
        assert not native.failure
        assert native.final_cycles == pytest.approx(cycles[-1], abs=5)
        assert native.final_cycles == pytest.approx(published, rel=0.02)


@pytest.mark.parametrize("source", list(FAC1_SIF))
def test_fac1_mode_i_superposition(source: str) -> None:
    """Tables FAC-1.1 and FAC-1.2 round components and totals separately."""
    np.testing.assert_allclose(
        fac1_totals()[source], FAC1_SIF[source][2], atol=0.015, rtol=0
    )


def test_paris_growth_under_proportionally_scaled_loads() -> None:
    """At constant reference K, each cycle makes a known crack increment."""
    length = np.array([0.1, 1.1])
    reference = paris_reference_cycles(length, np.array([2.0, 2.0]), 0.125, 2.0)
    np.testing.assert_allclose(reference, [0.0, 2.0])

    fractions = np.array([0.5, 1.0, 0.5, 1.0, 1.0])
    history, reached = crack_history(length, reference, fractions, 2.0)
    assert reached
    np.testing.assert_allclose(history, [0.1, 0.225, 0.725, 0.85, 1.1])

    history, reached = crack_history(length, reference, fractions[:3], 2.0)
    assert not reached
    np.testing.assert_allclose(history, [0.1, 0.225, 0.725, 0.85])


@pytest.mark.parametrize("source", list(FAC1_SIF))
def test_fac1_variable_paris_life_matches_native(source: str) -> None:
    """Compare the two solvers under proportional bypass and bearing loads."""
    table_length = FAC1_CRACK_RATIOS
    length = np.union1d(
        np.linspace(table_length[0], table_length[-1], 1001), table_length
    )
    total = fac1_totals()[source]
    delta_k = np.interp(length, table_length, total)
    bypass_stress = 10.0
    geometry = TabulatedGeometryFactor(
        length[0], length, delta_k / (bypass_stress * np.sqrt(np.pi * length))
    )
    curve = ParisCurve(MERC3_EXPONENT, MERC3_COEFFICIENT, unit_string="ksi √in")
    load_fraction = np.tile([0.5, 0.75, 1.0], 20_000)
    reference = paris_reference_cycles(
        length, delta_k, MERC3_COEFFICIENT, MERC3_EXPONENT
    )
    history, reached = crack_history(
        length, reference, load_fraction, MERC3_EXPONENT
    )
    native = native_paris_growth(geometry, curve, bypass_stress * load_fraction)
    assert reached
    assert native.termination == "geometry_limit"
    assert native.final_cycles == pytest.approx(history.size - 1, abs=2)
    recorded = native.sif / load_fraction[: native.sif.size]
    np.testing.assert_allclose(
        recorded,
        np.interp(native.crack_depth, table_length, total),
        atol=3e-4,
        rtol=0,
    )
