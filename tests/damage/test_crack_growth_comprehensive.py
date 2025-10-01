# -*- coding: utf-8 -*-

r"""Comprehensive tests for the damage.crack_growth module.

This module tests all classes and functions in the crack growth module:
- CalcCrackGrowth numba JIT class
- get_crack_growth function
- CrackGrowth pandas DataFrame accessor
"""

# Packages from the Python Standard Library
import os
import sys
import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch

# Packages from non-standard libraries
from hypothesis import given, strategies as hy, assume
import matplotlib.pyplot as plt

# Local imports
import py_fatigue.damage as damage
from py_fatigue.damage.crack_growth import (
    CalcCrackGrowth,
    get_crack_growth
)
from py_fatigue.utils import to_numba_dict
import py_fatigue.geometry as geometry
import py_fatigue.material as material
from py_fatigue import CycleCount, ParisCurve

PROJECT_PATH = os.path.dirname(os.getcwd())
if not PROJECT_PATH in sys.path:
    sys.path.append(PROJECT_PATH)


class TestCalcCrackGrowth:
    """Test the CalcCrackGrowth numba JIT class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create test data arrays
        self.stress_range = np.array([100.0, 150.0, 200.0])
        self.count_cycle = np.array([1000.0, 500.0, 200.0])
        self.slope = np.array([3.0])
        self.intercept = np.array([1e-12])
        self.threshold = 5.0
        self.critical = 50.0
        self.crack_type = "INF_SUR_00"
        self.crack_geometry = to_numba_dict({"initial_depth": 1.0, "_id": "INF_SUR_00"})

    def test_calc_crack_growth_initialization(self):
        """Test basic initialization of CalcCrackGrowth."""
        calc = CalcCrackGrowth(
            self.stress_range,
            self.count_cycle,
            self.slope,
            self.intercept,
            self.threshold,
            self.critical,
            self.crack_type,
            self.crack_geometry
        )
        
        assert np.array_equal(calc.stress_range, self.stress_range)
        assert np.array_equal(calc.count_cycle, self.count_cycle)
        assert np.array_equal(calc.slope, self.slope)
        assert np.array_equal(calc.intercept, self.intercept)
        assert calc.threshold == self.threshold
        assert calc.critical == self.critical
        assert calc.crack_type == self.crack_type

    @given(
        stress_value=hy.floats(min_value=50.0, max_value=300.0),
        count_value=hy.floats(min_value=100.0, max_value=5000.0),
        slope_value=hy.floats(min_value=2.0, max_value=5.0),
        intercept_value=hy.floats(min_value=1e-15, max_value=1e-10)
    )
    def test_calc_crack_growth_property_based(self, stress_value, count_value, slope_value, intercept_value):
        """Test CalcCrackGrowth with property-based testing."""
        stress_range = np.array([stress_value])
        count_cycle = np.array([count_value])
        slope = np.array([slope_value])
        intercept = np.array([intercept_value])
        
        calc = CalcCrackGrowth(
            stress_range,
            count_cycle,
            slope,
            intercept,
            self.threshold,
            self.critical,
            self.crack_type,
            self.crack_geometry
        )
        
        assert calc.stress_range[0] == stress_value
        assert calc.count_cycle[0] == count_value
        assert calc.slope[0] == slope_value
        assert calc.intercept[0] == intercept_value

    def test_calc_crack_growth_size_property(self):
        """Test the size property of CalcCrackGrowth."""
        calc = CalcCrackGrowth(
            self.stress_range,
            self.count_cycle,
            self.slope,
            self.intercept,
            self.threshold,
            self.critical,
            self.crack_type,
            self.crack_geometry
        )
        
        assert calc.size == len(self.stress_range)

    def test_calc_crack_growth_string_representation(self):
        """Test string representation of CalcCrackGrowth."""
        calc = CalcCrackGrowth(
            self.stress_range,
            self.count_cycle,
            self.slope,
            self.intercept,
            self.threshold,
            self.critical,
            self.crack_type,
            self.crack_geometry
        )
        
        str_repr = str(calc)
        # Should contain information about the object
        assert "Crack growth object" in str_repr

    def test_calc_crack_growth_arrays_validation(self):
        """Test that CalcCrackGrowth validates array inputs correctly."""
        # Test with mismatched array sizes
        with pytest.raises(AssertionError):
            CalcCrackGrowth(
                np.array([100.0, 150.0]),  # 2 elements
                np.array([1000.0]),        # 1 element - mismatch
                self.slope,
                self.intercept,
                self.threshold,
                self.critical,
                self.crack_type,
                self.crack_geometry
            )

    def test_calc_crack_growth_negative_stress_validation(self):
        """Test that negative stress ranges are rejected."""
        with pytest.raises(AssertionError):
            CalcCrackGrowth(
                np.array([-100.0, 150.0]),  # Contains negative value
                self.count_cycle[:2],
                self.slope,
                self.intercept,
                self.threshold,
                self.critical,
                self.crack_type,
                self.crack_geometry
            )

    def test_calc_crack_growth_empty_arrays_validation(self):
        """Test that empty arrays are rejected."""
        with pytest.raises((AssertionError, ValueError)):
            CalcCrackGrowth(
                np.array([]),  # Empty array
                np.array([]),
                self.slope,
                self.intercept,
                self.threshold,
                self.critical,
                self.crack_type,
                self.crack_geometry
            )

    def test_calc_crack_growth_slope_intercept_mismatch(self):
        """Test that mismatched slope and intercept arrays are rejected."""
        with pytest.raises(AssertionError):
            CalcCrackGrowth(
                self.stress_range,
                self.count_cycle,
                np.array([3.0, 4.0]),  # 2 elements
                np.array([1e-12]),     # 1 element - mismatch
                self.threshold,
                self.critical,
                self.crack_type,
                self.crack_geometry
            )

    def test_calc_crack_growth_crack_depth_calculation(self):
        """Test that crack depth is calculated during initialization."""
        calc = CalcCrackGrowth(
            self.stress_range,
            self.count_cycle,
            self.slope,
            self.intercept,
            self.threshold,
            self.critical,
            self.crack_type,
            self.crack_geometry
        )
        
        # Should have calculated crack depths
        assert hasattr(calc, 'crack_depth')
        assert hasattr(calc, 'sif')
        assert hasattr(calc, 'geometry_factor')
        assert hasattr(calc, 'final_cycles')

    def test_calc_crack_growth_with_different_geometries(self):
        """Test CalcCrackGrowth with different initial crack depths."""
        geometry1 = to_numba_dict({"initial_depth": 0.5, "_id": "INF_SUR_00"})
        geometry2 = to_numba_dict({"initial_depth": 2.0, "_id": "INF_SUR_00"})
        
        calc1 = CalcCrackGrowth(
            self.stress_range,
            self.count_cycle,
            self.slope,
            self.intercept,
            self.threshold,
            self.critical,
            self.crack_type,
            geometry1
        )
        
        calc2 = CalcCrackGrowth(
            self.stress_range,
            self.count_cycle,
            self.slope,
            self.intercept,
            self.threshold,
            self.critical,
            self.crack_type,
            geometry2
        )
        
        # Different initial depths should lead to different results
        assert calc1.crack_geometry["initial_depth"] != calc2.crack_geometry["initial_depth"]


class TestGetCrackGrowthFunction:
    """Test the get_crack_growth function."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create test geometry
        self.geometry = geometry.InfiniteSurface(initial_depth=1.0)
        
        # Create test crack growth curve
        self.cg_curve = ParisCurve(slope=3.0, intercept=1e-12)
        
        # Create test cycle count
        self.cycle_count = CycleCount(
            count_cycle=np.array([1000.0, 500.0, 200.0]),
            stress_range=np.array([100.0, 150.0, 200.0]),
            mean_stress=np.array([0.0, 0.0, 0.0]),
        )

    def test_get_crack_growth_basic(self):
        """Test basic functionality of get_crack_growth."""
        result = get_crack_growth(
            cycle_count=self.cycle_count,
            cg_curve=self.cg_curve,
            crack_geometry=self.geometry
        )
        
        assert isinstance(result, CalcCrackGrowth)
        assert sum(result.count_cycle) == sum(self.cycle_count.count_cycle)

    def test_get_crack_growth_with_express_mode(self):
        """Test get_crack_growth with express mode enabled."""
        result_express = get_crack_growth(
            cycle_count=self.cycle_count,
            cg_curve=self.cg_curve,
            crack_geometry=self.geometry,
            express_mode=True
        )
        
        result_normal = get_crack_growth(
            cycle_count=self.cycle_count,
            cg_curve=self.cg_curve,
            crack_geometry=self.geometry,
            express_mode=False
        )
        
        # Both should be CalcCrackGrowth objects
        assert isinstance(result_express, CalcCrackGrowth)
        assert isinstance(result_normal, CalcCrackGrowth)

    def test_get_crack_growth_single_cycle(self):
        """Test get_crack_growth with single cycle count."""
        single_cycle = CycleCount(
            count_cycle=np.array([1.0]),
            stress_range=np.array([120.0]),
            mean_stress=np.array([0.0]),
        )
        
        result = get_crack_growth(
            cycle_count=single_cycle,
            cg_curve=self.cg_curve,
            crack_geometry=self.geometry
        )
        
        assert isinstance(result, CalcCrackGrowth)
        assert result.size == 1

    @given(
        stress_range=hy.floats(min_value=50.0, max_value=300.0),
        count_cycle=hy.floats(min_value=100.0, max_value=5000.0)
    )
    def test_get_crack_growth_property_based(self, stress_range, count_cycle):
        """Test get_crack_growth with property-based testing."""
        cycle_count = CycleCount(
            count_cycle=np.array([count_cycle]),
            stress_range=np.array([stress_range]),
            mean_stress=np.array([0.0]),
        )
        
        result = get_crack_growth(
            cycle_count=cycle_count,
            cg_curve=self.cg_curve,
            crack_geometry=self.geometry
        )
        
        assert isinstance(result, CalcCrackGrowth)
        assert result.size >= 1

    def test_get_crack_growth_with_different_geometries(self):
        """Test get_crack_growth with different geometry initial depths."""
        geometry1 = geometry.InfiniteSurface(initial_depth=0.5)
        geometry2 = geometry.InfiniteSurface(initial_depth=2.0)
        
        result1 = get_crack_growth(self.cycle_count, self.cg_curve, geometry1)
        result2 = get_crack_growth(self.cycle_count, self.cg_curve, geometry2)
        
        # Different initial depths should lead to different results
        assert result1.crack_geometry["initial_depth"] != result2.crack_geometry["initial_depth"]

    def test_get_crack_growth_unit_compatibility_check(self):
        """Test that get_crack_growth checks unit compatibility."""
        # Create a cycle count with incompatible units
        incompatible_cycle_count = CycleCount(
            count_cycle=np.array([1000.0]),
            stress_range=np.array([100.0]),
            mean_stress=np.array([0.0]),
            unit="ksi√in"  # Different from curve unit
        )
        
        # This should raise an error
        with pytest.raises(ValueError, match="not compatible"):
            get_crack_growth(
                cycle_count=incompatible_cycle_count,
                cg_curve=self.cg_curve,
                crack_geometry=self.geometry
            )

    def test_get_crack_growth_unsupported_geometry(self):
        """Test that unsupported geometries raise an error."""
        # Create a mock geometry with unsupported ID
        mock_geometry = Mock()
        mock_geometry._id = "UNSUPPORTED_GEO"
        mock_geometry.__dict__ = {"initial_depth": 1.0, "_id": "UNSUPPORTED_GEO"}
        
        with pytest.raises(ValueError, match="Unsupported crack geometry"):
            get_crack_growth(
                cycle_count=self.cycle_count,
                cg_curve=self.cg_curve,
                crack_geometry=mock_geometry
            )

    def test_get_crack_growth_large_cycle_counts(self):
        """Test get_crack_growth with large cycle counts that require splitting."""
        large_cycle_count = CycleCount(
            count_cycle=np.array([50000.0]),  # Large value that will be split
            stress_range=np.array([100.0]),
            mean_stress=np.array([0.0]),
        )
        
        result = get_crack_growth(
            cycle_count=large_cycle_count,
            cg_curve=self.cg_curve,
            crack_geometry=self.geometry
        )
        
        assert isinstance(result, CalcCrackGrowth)
        # Should have more elements due to splitting
        assert result.size > 1


class TestCrackGrowthDataFrameAccessor:
    """Test the CrackGrowth pandas DataFrame accessor."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create test DataFrame with required columns for crack growth
        self.df = pd.DataFrame({
            'stress_range': [50.0, 100.0, 150.0, 200.0],
            'count_cycle': [1000.0, 500.0, 300.0, 200.0],
            'mean_stress': [0.0, 0.0, 0.0, 0.0]
        })
        
        # Create test crack growth curve and geometry
        self.cg_curve = ParisCurve(slope=3.0, intercept=1e-12)
        self.geometry = geometry.InfiniteSurface(initial_depth=1.0)

    def test_crack_growth_accessor_available(self):
        """Test that the .cg accessor is available on DataFrames."""
        assert hasattr(self.df, 'cg')

    def test_crack_growth_accessor_calc_growth_exists(self):
        """Test that the calc_growth method exists in the accessor."""
        assert hasattr(self.df.cg, 'calc_growth')

    def test_crack_growth_accessor_calc_growth_basic(self):
        """Test basic calc_growth functionality of the accessor."""
        result = self.df.cg.calc_growth(
            cg_curve=self.cg_curve,
            crack_geometry=self.geometry
        )
        
        assert isinstance(result, pd.DataFrame)
        
        # Should have additional columns after calculation
        expected_columns = ['crack_depth', 'sif', 'cumul_cycle', 'geometry_factor']
        for col in expected_columns:
            assert col in result.columns

    def test_crack_growth_accessor_validation(self):
        """Test that accessor validates required columns."""
        # Test with missing columns
        incomplete_df = pd.DataFrame({
            'stress_range': [50.0, 100.0],
            # Missing 'count_cycle' and 'mean_stress'
        })
        
        with pytest.raises(AttributeError, match="Must have"):
            incomplete_df.cg.calc_growth(
                cg_curve=self.cg_curve,
                crack_geometry=self.geometry
            )

    def test_crack_growth_accessor_already_calculated_validation(self):
        """Test that accessor prevents recalculation."""
        # First calculation should work
        result = self.df.cg.calc_growth(
            cg_curve=self.cg_curve,
            crack_geometry=self.geometry
        )
        
        # Second calculation should raise error
        with pytest.raises(AttributeError, match="already calculated"):
            result.cg.calc_growth(
                cg_curve=self.cg_curve,
                crack_geometry=self.geometry
            )

    def test_crack_growth_accessor_with_express_mode(self):
        """Test accessor with express mode enabled."""
        result = self.df.cg.calc_growth(
            cg_curve=self.cg_curve,
            crack_geometry=self.geometry,
            express_mode=True
        )
        
        assert isinstance(result, pd.DataFrame)
        assert 'crack_depth' in result.columns

    @given(
        stress_ranges=hy.lists(hy.floats(min_value=10.0, max_value=500.0), min_size=2, max_size=10),
        count_cycles=hy.lists(hy.floats(min_value=100.0, max_value=5000.0), min_size=2, max_size=10)
    )
    def test_crack_growth_accessor_property_based(self, stress_ranges, count_cycles):
        """Test accessor with property-based testing."""
        assume(len(stress_ranges) == len(count_cycles))
        
        df = pd.DataFrame({
            'stress_range': stress_ranges,
            'count_cycle': count_cycles,
            'mean_stress': [0.0] * len(stress_ranges)
        })
        
        # Should not raise an error
        result = df.cg.calc_growth(
            cg_curve=self.cg_curve,
            crack_geometry=self.geometry
        )
        
        assert isinstance(result, pd.DataFrame)
        assert 'crack_depth' in result.columns

    def test_crack_growth_accessor_attributes_set(self):
        """Test that accessor sets attributes correctly."""
        result = self.df.cg.calc_growth(
            cg_curve=self.cg_curve,
            crack_geometry=self.geometry
        )
        
        # Check that attributes are set on both the accessor and DataFrame
        assert hasattr(result.cg, 'cg_curve')
        assert hasattr(result.cg, 'crack_geometry')
        assert hasattr(result.cg, 'final_cycles')
        
        assert result.cg.cg_curve is self.cg_curve
        assert result.cg.crack_geometry is self.geometry

    def test_crack_growth_accessor_with_large_cycles(self):
        """Test accessor with large cycle counts that require splitting."""
        large_df = pd.DataFrame({
            'stress_range': [100.0],
            'count_cycle': [50000.0],  # Large value
            'mean_stress': [0.0]
        })
        
        result = large_df.cg.calc_growth(
            cg_curve=self.cg_curve,
            crack_geometry=self.geometry
        )
        
        assert isinstance(result, pd.DataFrame)
        assert 'crack_depth' in result.columns

    def test_crack_growth_accessor_with_single_cycles(self):
        """Test accessor when all cycles are 1.0 or less."""
        single_df = pd.DataFrame({
            'stress_range': [100.0, 150.0],
            'count_cycle': [1.0, 0.5],  # Small values
            'mean_stress': [0.0, 0.0]
        })
        
        result = single_df.cg.calc_growth(
            cg_curve=self.cg_curve,
            crack_geometry=self.geometry
        )
        
        assert isinstance(result, pd.DataFrame)
        assert 'crack_depth' in result.columns


class TestIntegration:
    """Integration tests for the crack growth module."""

    def setup_method(self):
        """Set up test fixtures for integration tests."""
        self.geometry = geometry.InfiniteSurface(initial_depth=1.0)
        self.cg_curve = ParisCurve(slope=3.0, intercept=1e-12)
        self.cycle_count = CycleCount(
            count_cycle=np.array([1000.0, 500.0, 200.0]),
            stress_range=np.array([100.0, 150.0, 200.0]),
            mean_stress=np.array([0.0, 0.0, 0.0]),
        )

    def test_end_to_end_crack_growth_analysis(self):
        """Test complete end-to-end crack growth analysis."""
        # Get crack growth results using function
        result_func = get_crack_growth(
            cycle_count=self.cycle_count,
            cg_curve=self.cg_curve,
            crack_geometry=self.geometry
        )
        
        # Validate function results
        assert isinstance(result_func, CalcCrackGrowth)
        
        # Test pandas accessor approach
        df = pd.DataFrame({
            'stress_range': self.cycle_count.stress_range,
            'count_cycle': self.cycle_count.count_cycle,
            'mean_stress': self.cycle_count.mean_stress
        })
        
        result_accessor = df.cg.calc_growth(
            cg_curve=self.cg_curve,
            crack_geometry=self.geometry
        )
        
        assert isinstance(result_accessor, pd.DataFrame)
        assert 'crack_depth' in result_accessor.columns

    def test_module_exports(self):
        """Test that the damage module exports the expected functions."""
        assert hasattr(damage, 'get_crack_growth')
        
        # Test that we can import the function directly
        from py_fatigue.damage.crack_growth import get_crack_growth as imported_func
        assert imported_func is not None

    def test_calc_crack_growth_with_real_parameters(self):
        """Test CalcCrackGrowth with realistic engineering parameters."""
        # Realistic steel parameters
        stress_range = np.array([80.0, 120.0, 160.0])  # MPa
        count_cycle = np.array([10000.0, 5000.0, 2000.0])
        slope = np.array([3.0])  # Typical for steel
        intercept = np.array([1e-12])  # m/cycle
        threshold = 5.0  # MPa√m
        critical = 50.0  # MPa√m
        crack_type = "INF_SUR_00"
        crack_geometry = to_numba_dict({"initial_depth": 1e-3, "_id": "INF_SUR_00"})  # 1mm initial crack
        
        calc = CalcCrackGrowth(
            stress_range,
            count_cycle,
            slope,
            intercept,
            threshold,
            critical,
            crack_type,
            crack_geometry
        )
        
        # Should calculate realistic results
        assert calc.final_cycles > 0
        assert len(calc.crack_depth) > 0

    def test_pandas_accessor_registration(self):
        """Test that the pandas accessor is properly registered."""
        # Create any DataFrame
        df = pd.DataFrame({'a': [1, 2, 3]})
        
        # The .cg accessor should be available
        assert hasattr(df, 'cg')
        
        # But it should work properly only with crack growth data
        cg_df = pd.DataFrame({
            'stress_range': [100.0],
            'count_cycle': [1000.0],
            'mean_stress': [0.0]
        })
        
        assert hasattr(cg_df, 'cg')
        assert hasattr(cg_df.cg, 'calc_growth')

    @given(
        initial_depth=hy.floats(min_value=0.1, max_value=5.0),
        stress_range=hy.floats(min_value=50.0, max_value=200.0),
        count_cycle=hy.floats(min_value=100.0, max_value=5000.0)
    )
    def test_integration_property_based(self, initial_depth, stress_range, count_cycle):
        """Property-based integration test."""
        # Create geometry with given initial depth
        geo = geometry.InfiniteSurface(initial_depth=initial_depth)
        
        # Create cycle count
        cycle_count_obj = CycleCount(
            count_cycle=np.array([count_cycle]),
            stress_range=np.array([stress_range]),
            mean_stress=np.array([0.0]),
        )
        
        # Get crack growth results
        result = get_crack_growth(
            cycle_count=cycle_count_obj,
            cg_curve=self.cg_curve,
            crack_geometry=geo
        )
        
        # Validate basic properties
        assert isinstance(result, CalcCrackGrowth)
        assert result.crack_geometry["initial_depth"] == initial_depth

    def test_consistency_between_function_and_accessor(self):
        """Test consistency between get_crack_growth function and DataFrame accessor."""
        # Function approach
        result_func = get_crack_growth(
            cycle_count=self.cycle_count,
            cg_curve=self.cg_curve,
            crack_geometry=self.geometry
        )
        
        # DataFrame accessor approach
        df = pd.DataFrame({
            'stress_range': self.cycle_count.stress_range,
            'count_cycle': self.cycle_count.count_cycle,
            'mean_stress': self.cycle_count.mean_stress
        })
        
        result_df = df.cg.calc_growth(
            cg_curve=self.cg_curve,
            crack_geometry=self.geometry
        )
        
        # Both should use the same underlying calculation
        assert isinstance(result_func, CalcCrackGrowth)
        assert isinstance(result_df, pd.DataFrame)
        
        # The DataFrame approach should have stored the final_cycles
        assert hasattr(result_df.cg, 'final_cycles')
        assert result_df.cg.final_cycles > 0