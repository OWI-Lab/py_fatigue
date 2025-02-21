import pytest
import numpy as np
from py_fatigue.material.crack_growth_curve import ParisCurve

def test_paris_curve_initialization():
    slope = [2.88, 5.1, 8.16, 5.1, 2.88]
    intercept = [1E-16, 1E-20, 1E-27, 1E-19, 1E-13]
    threshold = 20
    critical = 2000
    pc = ParisCurve(slope=slope, intercept=intercept, threshold=threshold, critical=critical)

    assert np.array_equal(pc.slope, slope)
    assert np.array_equal(pc.intercept, intercept)
    assert pc.threshold == threshold
    assert pc.critical == critical

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
