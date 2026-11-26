"""Regression checks against the USAF Damage Tolerant Design Handbook."""

import numpy as np
import pytest

from notebooks._examples.afgrow import (
    FAC1_SIF,
    MERC3_FINAL_LENGTH,
    MERC3_INITIAL_LENGTH,
    MERC3_PUBLISHED_CYCLES,
    fac1_totals,
    merc3_beta_histories,
    merc3_constant_beta,
)


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

    for case, published in MERC3_PUBLISHED_CYCLES.items():
        _, cycles = histories[case]
        assert np.all(np.diff(cycles) > 0)
        assert cycles[-1] == pytest.approx(finer[case][1][-1], abs=1)
        assert cycles[-1] == pytest.approx(published, rel=0.02)


@pytest.mark.parametrize("source", list(FAC1_SIF))
def test_fac1_mode_i_superposition(source: str) -> None:
    """Tables FAC-1.1 and FAC-1.2 round components and totals separately."""
    np.testing.assert_allclose(
        fac1_totals()[source], FAC1_SIF[source][2], atol=0.015, rtol=0
    )
