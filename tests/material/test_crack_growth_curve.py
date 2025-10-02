# -*- coding: utf-8 -*-
"""Tests for the crack growth curve module."""

import os

import pytest
import numpy as np
import numba as nb
from numba.core.errors import TypingError
from py_fatigue.material.crack_growth_curve import ParisCurve, WalkerCurve
from py_fatigue.material.crack_growth_curve import _calc_growth_rate, _calc_sif

nb.config.DISABLE_JIT = True

def test_pure_paris_curve():
    paris_pure = ParisCurve(slope=3.1, intercept=1.2e-14)
    assert paris_pure.linear
    assert paris_pure.slope == 3.1
    assert paris_pure.intercept == 1.2e-14
    assert paris_pure.threshold == 0
    assert paris_pure.critical == np.inf
    assert paris_pure.threshold_growth_rate == paris_pure.get_growth_rate(paris_pure.threshold)
    assert np.isinf(paris_pure.critical_growth_rate)
    assert np.isinf(paris_pure.get_growth_rate(paris_pure.critical))
    assert len(paris_pure.get_knee_growth_rate()) == 0
    with pytest.raises(ValueError):
        paris_pure.get_knee_growth_rate(check_knee=True)
    with pytest.raises(ValueError):
        paris_pure.get_knee_sif(check_knee=True)
    assert paris_pure.__str__() == paris_pure.name
    paris_pure.format_name()
    assert paris_pure.name == paris_pure.name.replace("<br>", "\n")
    paris_pure.format_name(html_format=True)
    assert paris_pure.name == paris_pure.name.replace("\n", "<br>")
    with pytest.raises(ValueError):
        paris_pure.get_knee_growth_rate(check_knee=np.array([0]))
    with pytest.raises(ValueError):
        paris_pure.get_knee_sif(check_knee=np.array([0]))


def test_paris_curve_initialization():
    slope = [2.88, 5.1, 8.16, 5.1, 2.88]
    intercept = [1E-16, 1E-20, 1E-27, 1E-19, 1E-13]
    threshold = 20
    critical = 2000
    pc = ParisCurve(slope=slope, intercept=intercept, threshold=threshold, critical=critical)

    assert not pc.linear
    assert pc.__str__() == pc.name
    assert np.array_equal(pc.slope, slope)
    assert np.array_equal(pc.intercept, intercept)
    assert pc.threshold == threshold
    assert pc.critical == critical
    assert pc.get_growth_rate(threshold) == pc.threshold_growth_rate
    assert pc.get_growth_rate(critical) == pc.critical_growth_rate
    knee_gr = pc.get_knee_growth_rate()
    knee_sif = pc.get_knee_sif()
    assert len(knee_gr) == len(knee_sif)
    assert np.allclose(
        pc.get_knee_growth_rate(check_knee=knee_sif), knee_gr, rtol=1e-2
    )
    with pytest.raises(ValueError):
        pc.get_knee_growth_rate(check_knee=knee_sif[:-2])
    with pytest.raises(ValueError):
        pc.get_knee_growth_rate(check_knee=knee_sif[0])
    assert np.allclose(
        pc.get_knee_sif(check_knee=knee_sif), knee_sif, rtol=1e-2
    )
    with pytest.raises(ValueError):
        pc.get_knee_sif(check_knee=knee_gr[:-2])
    with pytest.raises(ValueError):
        pc.get_knee_sif(check_knee=knee_gr[0])
    # with pytest.raises(ValueError):
    #     pc.get_knee_sif(check_knee=knee_gr[:-2])
    assert np.all(knee_gr >= 0)
    assert np.all(knee_sif >= threshold)
    assert np.all(knee_sif <= critical)
    assert np.allclose(knee_gr, pc.get_growth_rate(knee_sif), rtol=1e-2)
    assert np.allclose(knee_sif, pc.get_sif(knee_gr), rtol=1e-2)
    assert np.allclose(
        pc.get_growth_rate(knee_sif),
        _calc_growth_rate(
            knee_sif, pc.slope, pc.intercept, pc.threshold, pc.critical
        ),
        rtol=1e-2
    )
    assert np.allclose(
        pc.get_sif(knee_gr),
        _calc_sif(
            knee_gr, pc.slope, pc.intercept, pc.threshold, pc.critical
        ),
        rtol=1e-2
    )

    with pytest.raises(TypingError):
        assert _calc_growth_rate(
            threshold - 1e-2 , pc.slope, pc.intercept, pc.threshold, pc.critical
        ) == 0
    assert _calc_growth_rate(
        np.array([threshold - 1e-2]),
        pc.slope,
        pc.intercept,
        pc.threshold,
        pc.critical
    ) == 0

    with pytest.raises(TypingError):
        assert _calc_growth_rate(
            critical + 1e-2, pc.slope, pc.intercept, pc.threshold, pc.critical
        ) == np.inf
    assert _calc_growth_rate(
        np.array([critical + 1e-2]),
        pc.slope,
        pc.intercept,
        pc.threshold,
        pc.critical
    ) == np.inf

def test_paris_curve_from_knee_points():
    knee_sif = [20, 100, 500, 1000]
    knee_growth_rate = [1E-10, 1E-7, 1E-6, 1E-4]
    pc = ParisCurve.from_knee_points(knee_sif=knee_sif, knee_growth_rate=knee_growth_rate)

    assert np.allclose(pc.get_knee_sif(), knee_sif[1:-1], rtol=1e-2)
    assert np.allclose(pc.get_knee_growth_rate(), knee_growth_rate[1:-1], rtol=1e-2)

def test_paris_curve_get_growth_rate():
    slope = [2.88, 5.1, 8.16, 5.1, 2.88]
    intercept = [1E-16, 1E-20, 1E-27, 1E-19, 1E-13]
    threshold = 20
    critical = 2000
    pc = ParisCurve(slope=slope, intercept=intercept, threshold=threshold, critical=critical)

    sif_range = np.linspace(20, 2000, 10)
    growth_rates = pc.get_growth_rate(sif_range)

    assert len(growth_rates) == len(sif_range)
    assert np.all(growth_rates >= 0)

def test_paris_curve_get_sif():
    slope = [2.88, 5.1, 8.16, 5.1, 2.88]
    intercept = [1E-16, 1E-20, 1E-27, 1E-19, 1E-13]
    threshold = 20
    critical = 2000
    pc = ParisCurve(slope=slope, intercept=intercept, threshold=threshold, critical=critical)

    growth_rate = np.logspace(-10, -4, 10)
    sif = pc.get_sif(growth_rate)

    assert len(sif) == len(growth_rate)
    assert np.all(sif >= threshold)
    assert np.all(sif <= critical)

def test_paris_curve_plot():
    slope = [2.88, 5.1, 8.16, 5.1, 2.88]
    intercept = [1E-16, 1E-20, 1E-27, 1E-19, 1E-13]
    threshold = 20
    critical = 2000
    pc = ParisCurve(slope=slope, intercept=intercept, threshold=threshold, critical=critical)

    fig, ax = pc.plot()
    assert fig is not None
    assert ax is not None

def test_pure_walker_curve():
    walker_pure = WalkerCurve(slope=3.1, intercept=1.2e-14, load_ratio=0.1,
                              walker_exponent=0.5)
    assert walker_pure.linear
    assert walker_pure.slope == 3.1
    assert walker_pure.intercept == 1.2e-14
    assert walker_pure.threshold == 0
    assert walker_pure.critical == np.inf
    assert walker_pure.walker_exponent == 0.5
    assert walker_pure.load_ratio == 0.1
    assert walker_pure.threshold_growth_rate == walker_pure.get_growth_rate(walker_pure.threshold)
    assert np.isinf(walker_pure.critical_growth_rate)
    assert np.isinf(walker_pure.get_growth_rate(walker_pure.critical))
    assert len(walker_pure.get_knee_growth_rate()) == 0
    with pytest.raises(ValueError):
        walker_pure.get_knee_growth_rate(check_knee=True)
    with pytest.raises(ValueError):
        walker_pure.get_knee_sif(check_knee=True)
    assert walker_pure.__str__() == walker_pure.name
    walker_pure.format_name()
    assert walker_pure.name == walker_pure.name.replace("<br>", "\n")
    walker_pure.format_name(html_format=True)
    assert walker_pure.name == walker_pure.name.replace("\n", "<br>")
    with pytest.raises(ValueError):
        walker_pure.get_knee_growth_rate(check_knee=np.array([0]))
    with pytest.raises(ValueError):
        walker_pure.get_knee_sif(check_knee=np.array([0]))


def test_walker_curve_initialization():
    slope = [2.88, 5.1, 8.16, 5.1, 2.88]
    intercept = [1E-16, 1E-20, 1E-27, 1E-19, 1E-13]
    threshold = 20
    critical = 2000
    wc = WalkerCurve(slope=slope, intercept=intercept, threshold=threshold,
                     critical=critical, walker_exponent=0.5, load_ratio=0.1)

    assert not wc.linear
    assert wc.__str__() == wc.name
    assert np.array_equal(wc.slope, slope)
    assert np.array_equal(wc.intercept, intercept)
    assert wc.threshold <= threshold
    assert wc.critical <= critical
    assert wc.get_growth_rate(wc.threshold) == wc.threshold_growth_rate
    assert wc.get_growth_rate(wc.critical) == wc.critical_growth_rate
    knee_gr = wc.get_knee_growth_rate()
    knee_sif = wc.get_knee_sif()
    assert len(knee_gr) == len(knee_sif)
    assert np.allclose(
        wc.get_knee_growth_rate(check_knee=knee_sif), knee_gr, rtol=1e-2
    )
    with pytest.raises(ValueError):
        wc.get_knee_growth_rate(check_knee=knee_sif[:-2])
    with pytest.raises(ValueError):
        wc.get_knee_growth_rate(check_knee=knee_sif[0])
    assert np.allclose(
        wc.get_knee_sif(check_knee=knee_sif), knee_sif, rtol=1e-2
    )
    with pytest.raises(ValueError):
        wc.get_knee_sif(check_knee=knee_gr[:-2])
    with pytest.raises(ValueError):
        wc.get_knee_sif(check_knee=knee_gr[0])
    # with pytest.raises(ValueError):
    #     wc.get_knee_sif(check_knee=knee_gr[:-2])
    assert np.all(knee_gr >= 0)
    assert np.all(knee_sif >= threshold)
    assert np.all(knee_sif <= critical)
    assert np.allclose(knee_gr, wc.get_growth_rate(knee_sif), rtol=1e-2)
    assert np.allclose(knee_sif, wc.get_sif(knee_gr), rtol=1e-2)
    assert np.allclose(
        wc.get_growth_rate(knee_sif),
        _calc_growth_rate(
            knee_sif, wc.slope, wc.walker_intercept, wc.threshold, wc.critical
        ),
        rtol=1e-2
    )
    assert np.allclose(
        wc.get_sif(knee_gr),
        _calc_sif(
            knee_gr, wc.slope, wc.walker_intercept, wc.threshold, wc.critical
        ),
        rtol=1e-2
    )

    with pytest.raises(TypingError):
        assert _calc_growth_rate(
            wc.threshold - 1e-2 , wc.slope, wc.intercept, wc.threshold, wc.critical
        ) == 0
    assert _calc_growth_rate(
        np.array([wc.threshold - 1e-2]),
        wc.slope,
        wc.intercept,
        wc.threshold,
        wc.critical
    ) == 0

    with pytest.raises(TypingError):
        assert _calc_growth_rate(
            critical + 1e-2, wc.slope, wc.intercept, wc.threshold, wc.critical
        ) == np.inf
    assert _calc_growth_rate(
        np.array([critical + 1e-2]),
        wc.slope,
        wc.intercept,
        wc.threshold,
        wc.critical
    ) == np.inf

def test_walker_curve_from_knee_points():
    knee_sif = [20, 100, 500, 1000]
    knee_growth_rate = [1E-10, 1E-7, 1E-6, 1E-4]
    wc = WalkerCurve.from_knee_points(knee_sif=knee_sif, knee_growth_rate=knee_growth_rate)

    assert np.allclose(wc.get_knee_sif(), knee_sif[1:-1], rtol=1e-2)
    assert np.allclose(wc.get_knee_growth_rate(), knee_growth_rate[1:-1], rtol=1e-2)

def test_walker_curve_get_growth_rate():
    slope = [2.88, 5.1, 8.16, 5.1, 2.88]
    intercept = [1E-16, 1E-20, 1E-27, 1E-19, 1E-13]
    threshold = 20
    critical = 2000
    wc = WalkerCurve(slope=slope, intercept=intercept, threshold=threshold,
                     critical=critical, walker_exponent=0.5, load_ratio=0.2)

    sif_range = np.linspace(20, 2000, 10)
    growth_rates = wc.get_growth_rate(sif_range)

    assert len(growth_rates) == len(sif_range)
    assert np.all(growth_rates >= 0)

def test_walker_curve_get_sif():
    slope = [2.88, 5.1, 8.16, 5.1, 2.88]
    intercept = [1E-16, 1E-20, 1E-27, 1E-19, 1E-13]
    threshold = 20
    critical = 2000
    wc = WalkerCurve(slope=slope, intercept=intercept, threshold=threshold,
                     critical=critical, walker_exponent=0.4, load_ratio=0.3)

    growth_rate = np.logspace(-10, -4, 10)
    sif = wc.get_sif(growth_rate)

    assert len(sif) == len(growth_rate)
    assert np.all(sif >= threshold)
    assert np.all(sif <= critical)

def test_walker_curve_plot():
    slope = [2.88, 5.1, 8.16, 5.1, 2.88]
    intercept = [1E-16, 1E-20, 1E-27, 1E-19, 1E-13]
    threshold = 20
    critical = 2000
    wc = WalkerCurve(slope=slope, intercept=intercept, threshold=threshold,
                     critical=critical, walker_exponent=0.5, load_ratio=0.1)

    fig, ax = wc.plot()
    assert fig is not None
    assert ax is not None

def test_walker_curve_multiple_load_ratios():
    """Assert that increasing the load ratios reduces the threshold and critical
    SIF"""

    slope = [2.88, 5.1, 8.16, 5.1, 2.88]
    intercept = [1E-16, 1E-20, 1E-27, 1E-19, 1E-13]
    threshold = 20
    critical = 2000
    wc_old = None
    w_exp = 0.5
    for r in [0, 0.1, 0.3, 0.5, 0.7, 0.9]:
        wc = WalkerCurve(slope=slope, intercept=intercept, threshold=threshold,
                         critical=critical, walker_exponent=w_exp, load_ratio=r)
        if wc_old is not None:
            assert np.all(wc.walker_intercept >= wc_old.walker_intercept)
            assert wc.critical <= wc_old.critical
        wc_old = wc

def test_walker_curve_change_load_ratios():
    """Assert that increasing the load ratios reduces the threshold and critical
    SIF"""
    import copy
    slope = [2.88, 5.1, 8.16, 5.1, 2.88]
    intercept = [1E-16, 1E-20, 1E-27, 1E-19, 1E-13]
    threshold = 20
    critical = 2000
    wc_old = None
    w_exp = 0.5
    wc = WalkerCurve(slope=slope, intercept=intercept, threshold=threshold,
                     critical=critical, walker_exponent=w_exp, load_ratio=0)
    for r in [0.1, 0.3, 0.5, 0.7, 0.9]:
        wc_old = copy.copy(wc)
        wc.load_ratio = r
        assert np.all(wc.walker_intercept >= wc_old.walker_intercept)
        assert wc.critical <= wc_old.critical

def test_walker_curve_change_exponent():
    """Assert that increasing the Walker exponent towards 1 moves the curve
    closer to the Paris curve"""
    import copy
    slope = [2.88, 5.1, 8.16, 5.1, 2.88]
    intercept = [1E-16, 1E-20, 1E-27, 1E-19, 1E-13]
    threshold = 20
    critical = 2000
    wc_old = None
    w_exp = 0.5
    wc = WalkerCurve(slope=slope, intercept=intercept, threshold=threshold,
                     critical=critical, walker_exponent=w_exp, load_ratio=0)
    for w_e in [0, 0.25, 0.5, 0.75, 1]:
        wc_old = copy.copy(wc)
        wc.walker_exponent = w_e
        assert np.all(wc.walker_intercept >= wc_old.walker_intercept)
        assert wc.critical >= wc_old.critical
        assert wc.threshold >= wc_old.threshold

    pc = ParisCurve(slope=slope, intercept=intercept, threshold=threshold,
                    critical=critical)

    # Assert that the two curves are the same when the exponent is 1
    assert wc.threshold == pc.threshold
    assert wc.critical == pc.critical
    assert np.allclose(wc.get_knee_sif(), pc.get_knee_sif(), rtol=1e-2)
    assert np.allclose(wc.get_knee_growth_rate(), pc.get_knee_growth_rate(),
                       rtol=1e-2)


nb.config.DISABLE_JIT = False
