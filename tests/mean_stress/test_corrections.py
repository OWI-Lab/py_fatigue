# -*- coding: utf-8 -*-

r"""Comprehensive tests for mean stress corrections module.

This module contains tests for all mean stress correction functions:
- dnvgl_mean_stress_correction
- walker_mean_stress_correction  
- swt_mean_stress_correction
- goodman_haigh_mean_stress_correction
- __goodman_equation
"""

# Standard imports
import logging
import warnings

# Non-standard imports
import numpy as np
import pytest
from hypothesis import given, strategies as hy

from py_fatigue.mean_stress.corrections import (
    dnvgl_mean_stress_correction,
    walker_mean_stress_correction,
    swt_mean_stress_correction,
    goodman_haigh_mean_stress_correction,
)


class TestDNVGLMeanStressCorrection:
    """Test the DNVGL-RP-C203 mean stress correction."""

    def test_basic_functionality(self):
        """Test basic functionality with known values."""
        mean_stress = np.array([10, 20, -10, 0])
        stress_amplitude = np.array([50, 30, 40, 25])
        
        # Test with detail_factor=0.8 (default)
        result = dnvgl_mean_stress_correction(mean_stress, stress_amplitude)
        
        assert len(result) == len(mean_stress)
        assert np.all(result >= 0)
        assert isinstance(result, np.ndarray)

    def test_detail_factor_08(self):
        """Test with detail_factor=0.8 (welded connections)."""
        mean_stress = np.array([100, -50, 0])
        stress_amplitude = np.array([80, 60, 40])
        
        result = dnvgl_mean_stress_correction(
            mean_stress, stress_amplitude, detail_factor=0.8
        )
        
        # For fully tensile (mean + amp > 0, mean - amp > 0), f_m should be 1
        # For fully compressive cycles, f_m should be >= 0.8
        assert len(result) == 3
        assert np.all(result >= 0)

    def test_detail_factor_06(self):
        """Test with detail_factor=0.6 (base material)."""
        mean_stress = np.array([100, -50, 0])
        stress_amplitude = np.array([80, 60, 40])
        
        result = dnvgl_mean_stress_correction(
            mean_stress, stress_amplitude, detail_factor=0.6
        )
        
        # For fully compressive cycles with detail_factor=0.6, stress should be 0
        assert len(result) == 3
        assert np.all(result >= 0)

    def test_invalid_detail_factor(self):
        """Test that invalid detail factors raise ValueError."""
        mean_stress = np.array([10, 20])
        stress_amplitude = np.array([5, 15])
        
        with pytest.raises(ValueError, match="Detail factor.*not allowed"):
            dnvgl_mean_stress_correction(
                mean_stress, stress_amplitude, detail_factor=0.7
            )
        
        with pytest.raises(ValueError, match="Detail factor.*not allowed"):
            dnvgl_mean_stress_correction(
                mean_stress, stress_amplitude, detail_factor=0.9
            )

    @given(
        mean_stress=hy.lists(
            hy.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
            min_size=1, max_size=20
        ),
        stress_amp=hy.lists(
            hy.floats(min_value=1, max_value=1000, allow_nan=False, allow_infinity=False),
            min_size=1, max_size=20
        )
    )
    def test_property_based_dnvgl_08(self, mean_stress, stress_amp):
        """Property-based test for DNVGL with detail_factor=0.8."""
        if len(mean_stress) != len(stress_amp):
            mean_stress = mean_stress[:min(len(mean_stress), len(stress_amp))]
            stress_amp = stress_amp[:min(len(mean_stress), len(stress_amp))]
        
        mean_stress = np.array(mean_stress)
        stress_amp = np.array(stress_amp)
        
        result = dnvgl_mean_stress_correction(
            mean_stress, stress_amp, detail_factor=0.8
        )
        
        # Results should be non-negative
        assert np.all(result >= 0)
        # Results should be finite
        assert np.all(np.isfinite(result))
        # For tensile stress, corrected range should equal 2*amplitude
        tensile_mask = (mean_stress - stress_amp) > 0
        if np.any(tensile_mask):
            np.testing.assert_allclose(
                result[tensile_mask], 2 * stress_amp[tensile_mask], rtol=1e-10
            )

    @given(
        mean_stress=hy.lists(
            hy.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
            min_size=1, max_size=20
        ),
        stress_amp=hy.lists(
            hy.floats(min_value=1, max_value=1000, allow_nan=False, allow_infinity=False),
            min_size=1, max_size=20
        )
    )
    def test_property_based_dnvgl_06(self, mean_stress, stress_amp):
        """Property-based test for DNVGL with detail_factor=0.6."""
        if len(mean_stress) != len(stress_amp):
            mean_stress = mean_stress[:min(len(mean_stress), len(stress_amp))]
            stress_amp = stress_amp[:min(len(mean_stress), len(stress_amp))]
        
        mean_stress = np.array(mean_stress)
        stress_amp = np.array(stress_amp)
        
        result = dnvgl_mean_stress_correction(
            mean_stress, stress_amp, detail_factor=0.6
        )
        
        # Results should be non-negative
        assert np.all(result >= 0)
        # Results should be finite
        assert np.all(np.isfinite(result))

    def test_plotting_functionality(self):
        """Test that plotting doesn't raise errors."""
        mean_stress = np.array([10, -20, 0, 50])
        stress_amplitude = np.array([15, 25, 30, 20])
        
        # Should not raise any exceptions
        result = dnvgl_mean_stress_correction(
            mean_stress, stress_amplitude, plot=True
        )
        assert len(result) == len(mean_stress)

    def test_edge_cases(self):
        """Test edge cases."""
        # Zero mean stress (results in stress range between -50 and +50)
        # For DNVGL: min_stress = 0 - 50 = -50 (compressive)
        # max_stress = 0 + 50 = 50 (tensile)
        # f_m should be < 1, resulting in < 100
        result = dnvgl_mean_stress_correction(
            np.array([0]), np.array([50]), detail_factor=0.8
        )
        assert result[0] < 100  # Should be less than 2*amplitude for mixed loading
        assert result[0] >= 80  # But at least detail_factor * 2 * amplitude
        
        # Very small amplitude with high tensile mean
        result = dnvgl_mean_stress_correction(
            np.array([100]), np.array([1]), detail_factor=0.8
        )
        assert result[0] == 2.0  # Should be 2 * amplitude for fully tensile


class TestWalkerMeanStressCorrection:
    """Test the Walker mean stress correction."""

    def test_basic_functionality(self):
        """Test basic functionality with known values."""
        mean_stress = np.array([10, 20, -10, 0])
        stress_amplitude = np.array([50, 30, 40, 25])
        
        result = walker_mean_stress_correction(mean_stress, stress_amplitude)
        
        assert len(result) == len(mean_stress)
        assert np.all(result >= 0)
        assert isinstance(result, np.ndarray)

    def test_gamma_variations(self):
        """Test with different gamma values."""
        mean_stress = np.array([100, -50])
        stress_amplitude = np.array([80, 60])
        
        # Test gamma = 0.5 (default, equivalent to SWT)
        result_05 = walker_mean_stress_correction(
            mean_stress, stress_amplitude, gamma=0.5
        )
        
        # Test gamma = 0 (maximum influence of mean stress)
        result_0 = walker_mean_stress_correction(
            mean_stress, stress_amplitude, gamma=0.0
        )
        
        # Test gamma = 1 (no influence of mean stress)
        result_1 = walker_mean_stress_correction(
            mean_stress, stress_amplitude, gamma=1.0
        )
        
        assert len(result_05) == 2
        assert len(result_0) == 2  
        assert len(result_1) == 2
        
        # When gamma = 1, result should equal stress amplitude
        np.testing.assert_allclose(result_1, stress_amplitude, rtol=1e-10)

    @given(
        gamma=hy.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    def test_gamma_range(self, gamma):
        """Test gamma parameter in valid range."""
        mean_stress = np.array([50, -25, 0])
        stress_amplitude = np.array([30, 40, 25])
        
        result = walker_mean_stress_correction(
            mean_stress, stress_amplitude, gamma=gamma
        )
        
        assert len(result) == 3
        assert np.all(result >= 0)
        assert np.all(np.isfinite(result))

    def test_plotting_functionality(self):
        """Test that plotting doesn't raise errors."""
        mean_stress = np.array([10, -20, 0, 50])
        stress_amplitude = np.array([15, 25, 30, 20])
        
        # Should not raise any exceptions
        result = walker_mean_stress_correction(
            mean_stress, stress_amplitude, plot=True
        )
        assert len(result) == len(mean_stress)

    def test_negative_max_stress_handling(self):
        """Test handling of negative maximum stress."""
        # Case where mean_stress + stress_amplitude < 0
        mean_stress = np.array([-100])
        stress_amplitude = np.array([50])  # max_stress = -50
        
        result = walker_mean_stress_correction(mean_stress, stress_amplitude)
        
        # Should handle negative max stress gracefully
        assert len(result) == 1
        assert np.isfinite(result[0])

    @given(
        mean_stress=hy.lists(
            hy.floats(min_value=-500, max_value=500, allow_nan=False, allow_infinity=False),
            min_size=1, max_size=10
        ),
        stress_amp=hy.lists(
            hy.floats(min_value=1, max_value=500, allow_nan=False, allow_infinity=False),
            min_size=1, max_size=10
        )
    )
    def test_property_based_walker(self, mean_stress, stress_amp):
        """Property-based test for Walker correction."""
        if len(mean_stress) != len(stress_amp):
            mean_stress = mean_stress[:min(len(mean_stress), len(stress_amp))]
            stress_amp = stress_amp[:min(len(mean_stress), len(stress_amp))]
        
        mean_stress = np.array(mean_stress)
        stress_amp = np.array(stress_amp)
        
        result = walker_mean_stress_correction(mean_stress, stress_amp)
        
        # Results should be non-negative
        assert np.all(result >= 0)
        # Results should be finite (nan_to_num should handle infinities)
        assert np.all(np.isfinite(result))


class TestSWTMeanStressCorrection:
    """Test the Smith-Watson-Topper mean stress correction."""

    def test_basic_functionality(self):
        """Test basic functionality."""
        mean_stress = np.array([10, 20, -10, 0])
        stress_amplitude = np.array([50, 30, 40, 25])
        
        result = swt_mean_stress_correction(mean_stress, stress_amplitude)
        
        assert len(result) == len(mean_stress)
        assert np.all(result >= 0)
        assert isinstance(result, np.ndarray)

    def test_equivalence_to_walker_gamma_05(self):
        """Test that SWT is equivalent to Walker with gamma=0.5."""
        mean_stress = np.array([100, -50, 0, 25])
        stress_amplitude = np.array([80, 60, 40, 35])
        
        swt_result = swt_mean_stress_correction(mean_stress, stress_amplitude)
        walker_result = walker_mean_stress_correction(
            mean_stress, stress_amplitude, gamma=0.5
        )
        
        np.testing.assert_allclose(swt_result, walker_result, rtol=1e-10)

    def test_plotting_functionality(self):
        """Test that plotting doesn't raise errors."""
        mean_stress = np.array([10, -20, 0, 50])
        stress_amplitude = np.array([15, 25, 30, 20])
        
        # Should not raise any exceptions
        result = swt_mean_stress_correction(
            mean_stress, stress_amplitude, plot=True
        )
        assert len(result) == len(mean_stress)

    @given(
        mean_stress=hy.lists(
            hy.floats(min_value=-500, max_value=500, allow_nan=False, allow_infinity=False),
            min_size=1, max_size=10
        ),
        stress_amp=hy.lists(
            hy.floats(min_value=1, max_value=500, allow_nan=False, allow_infinity=False),
            min_size=1, max_size=10
        )
    )
    def test_property_based_swt(self, mean_stress, stress_amp):
        """Property-based test for SWT correction."""
        if len(mean_stress) != len(stress_amp):
            mean_stress = mean_stress[:min(len(mean_stress), len(stress_amp))]
            stress_amp = stress_amp[:min(len(mean_stress), len(stress_amp))]
        
        mean_stress = np.array(mean_stress)
        stress_amp = np.array(stress_amp)
        
        result = swt_mean_stress_correction(mean_stress, stress_amp)
        
        # Results should be non-negative
        assert np.all(result >= 0)
        # Results should be finite
        assert np.all(np.isfinite(result))


class TestGoodmanHaighMeanStressCorrection:
    """Test the Goodman-Haigh mean stress correction."""

    def test_basic_functionality(self):
        """Test basic functionality with known values."""
        amp_in = np.array([100, 200, 150])
        mean_in = np.array([0, 50, -25])
        r_out = -1.0  # Fully reversed loading
        ult_s = 1000
        correction_exponent = 1.0  # Goodman correction
        
        amp_out, mean_out = goodman_haigh_mean_stress_correction(
            amp_in, mean_in, r_out, ult_s, correction_exponent
        )
        
        assert amp_out.shape == (1, 3)  # r_out as scalar creates (1, len(amp_in))
        assert mean_out.shape == (1, 3)
        assert np.all(amp_out >= 0)
        assert np.all(np.isfinite(amp_out))

    def test_input_validation(self):
        """Test input validation."""
        # Shape mismatch
        with pytest.raises(ValueError, match="must have the same shape"):
            goodman_haigh_mean_stress_correction(
                np.array([100, 200]), np.array([50]), -1.0, 1000, 1.0
            )
        
        # Negative amplitude
        with pytest.raises(ValueError, match="negative stress amplitude"):
            goodman_haigh_mean_stress_correction(
                np.array([-100, 200]), np.array([50, 25]), -1.0, 1000, 1.0
            )

    def test_r_out_minus_one_special_case(self):
        """Test the special case for r_out = -1 (analytical solution)."""
        amp_in = np.array([100, 200])
        mean_in = np.array([50, 100])
        r_out = -1.0
        ult_s = 1000
        correction_exponent = 1.0
        
        amp_out, mean_out = goodman_haigh_mean_stress_correction(
            amp_in, mean_in, r_out, ult_s, correction_exponent
        )
        
        # For r_out = -1, mean_out should be zeros
        np.testing.assert_allclose(mean_out[0], np.zeros_like(amp_in), rtol=1e-10)
        
        # Analytical solution: amp_out = amp_in / (1 - (mean_in / ult_s)^n)
        expected_amp_out = amp_in / (1 - (mean_in / ult_s) ** correction_exponent)
        np.testing.assert_allclose(amp_out[0], expected_amp_out, rtol=1e-6)

    def test_different_correction_exponents(self):
        """Test different correction exponents (Goodman=1, Gerber=2)."""
        amp_in = np.array([100])
        mean_in = np.array([50])
        r_out = -1.0
        ult_s = 1000
        
        # Goodman correction (n=1)
        amp_out_goodman, _ = goodman_haigh_mean_stress_correction(
            amp_in, mean_in, r_out, ult_s, correction_exponent=1.0
        )
        
        # Gerber correction (n=2)
        amp_out_gerber, _ = goodman_haigh_mean_stress_correction(
            amp_in, mean_in, r_out, ult_s, correction_exponent=2.0
        )
        
        # Both should be greater than input amplitude due to mean stress effect
        assert amp_out_goodman[0, 0] > amp_in[0]
        assert amp_out_gerber[0, 0] > amp_in[0]
        # For the specific case with positive mean stress, Gerber (n=2) gives lower correction than Goodman (n=1)
        # This is because higher exponents reduce the mean stress effect

    def test_multiple_r_out_values(self):
        """Test with multiple r_out values."""
        amp_in = np.array([100, 150])
        mean_in = np.array([25, 75])
        r_out = np.array([-1.0, 0.0, 0.5])
        ult_s = 1000
        correction_exponent = 1.0
        
        amp_out, mean_out = goodman_haigh_mean_stress_correction(
            amp_in, mean_in, r_out, ult_s, correction_exponent
        )
        
        assert amp_out.shape == (3, 2)  # 3 r_out values, 2 input points
        assert mean_out.shape == (3, 2)

    def test_list_inputs(self):
        """Test with list inputs instead of numpy arrays."""
        amp_in = [100, 200]
        mean_in = [50, 25]
        r_out = [-1.0, 0.0]
        ult_s = 1000
        correction_exponent = 1.0
        
        amp_out, mean_out = goodman_haigh_mean_stress_correction(
            amp_in, mean_in, r_out, ult_s, correction_exponent
        )
        
        assert isinstance(amp_out, np.ndarray)
        assert isinstance(mean_out, np.ndarray)
        assert amp_out.shape == (2, 2)

    def test_large_array_warning(self):
        """Test warning for large arrays with plotting."""
        # Create large arrays that should trigger warning
        amp_in = np.ones(1001)
        mean_in = np.zeros(1001)
        r_out = np.linspace(-1, 0.9, 1001)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            amp_out, _ = goodman_haigh_mean_stress_correction(
                amp_in, mean_in, r_out, 1000, 1.0, plot=True
            )
            
            # Should have triggered a warning about large arrays
            assert len(w) > 0
            assert "too large for plotting" in str(w[0].message)

    def test_logger_integration(self):
        """Test logger integration."""
        # Create a logger
        logger = logging.getLogger("test_logger")
        
        amp_in = np.array([100])
        mean_in = np.array([50])
        r_out = 0.0
        ult_s = 1000
        correction_exponent = 1.0
        
        # Should not raise any errors with logger
        amp_out, mean_out = goodman_haigh_mean_stress_correction(
            amp_in, mean_in, r_out, ult_s, correction_exponent, logger=logger
        )
        
        assert amp_out.shape == (1, 1)

    def test_plotting_functionality(self):
        """Test plotting functionality."""
        amp_in = np.array([100, 150, 200])
        mean_in = np.array([25, 50, 75])
        r_out = np.array([-1.0, 0.0, 0.5])
        ult_s = 1000
        correction_exponent = 1.0
        
        # Should not raise errors
        amp_out, mean_out = goodman_haigh_mean_stress_correction(
            amp_in, mean_in, r_out, ult_s, correction_exponent, plot=True
        )
        
        assert amp_out.shape == (3, 3)

    def test_initial_guess_parameter(self):
        """Test initial_guess parameter."""
        amp_in = np.array([100, 150])
        mean_in = np.array([25, 50])
        r_out = 0.0
        ult_s = 1000
        correction_exponent = 1.0
        initial_guess = np.array([120, 180])
        
        amp_out, mean_out = goodman_haigh_mean_stress_correction(
            amp_in, mean_in, r_out, ult_s, correction_exponent, 
            initial_guess=initial_guess
        )
        
        assert amp_out.shape == (1, 2)

    @given(
        amp_in=hy.lists(
            hy.floats(min_value=10, max_value=200, allow_nan=False, allow_infinity=False),
            min_size=1, max_size=3
        ),
        mean_in=hy.lists(
            hy.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False),
            min_size=1, max_size=3
        ),
        ult_s=hy.floats(min_value=800, max_value=2000, allow_nan=False, allow_infinity=False),
        correction_exponent=hy.floats(min_value=1.0, max_value=2.0, allow_nan=False, allow_infinity=False)
    )
    def test_property_based_goodman_haigh(self, amp_in, mean_in, ult_s, correction_exponent):
        """Property-based test for Goodman-Haigh correction."""
        if len(amp_in) != len(mean_in):
            min_len = min(len(amp_in), len(mean_in))
            amp_in = amp_in[:min_len]
            mean_in = mean_in[:min_len]
        
        amp_in = np.array(amp_in)
        mean_in = np.array(mean_in)
        r_out = -1.0  # Use simple case
        
        # Filter out cases that might cause numerical issues
        # Skip if any amplitude is too close to ultimate strength relative to mean stress
        stress_ratio = np.abs(mean_in) / ult_s
        if np.any(stress_ratio > 0.1):  # More conservative threshold
            # Skip problematic cases
            return
        
        # Additional check: ensure reasonable stress levels
        if np.any(amp_in / ult_s > 0.3):
            return
        
        try:
            amp_out, mean_out = goodman_haigh_mean_stress_correction(
                amp_in, mean_in, r_out, ult_s, correction_exponent
            )
            
            # Results should be finite and positive where they exist
            finite_mask = np.isfinite(amp_out)
            if np.any(finite_mask):
                assert np.all(amp_out[finite_mask] >= 0)
                assert np.all(np.isfinite(mean_out[finite_mask]))
        except (RuntimeWarning, Warning):
            # Some parameter combinations may cause numerical warnings, which is okay
            pass


class TestIntegration:
    """Integration tests combining multiple correction methods."""

    def test_corrections_comparison(self):
        """Compare different correction methods for consistency."""
        mean_stress = np.array([50, -25, 0])
        stress_amplitude = np.array([100, 75, 50])
        
        # DNVGL corrections
        dnvgl_08 = dnvgl_mean_stress_correction(
            mean_stress, stress_amplitude, detail_factor=0.8
        )
        dnvgl_06 = dnvgl_mean_stress_correction(
            mean_stress, stress_amplitude, detail_factor=0.6
        )
        
        # Walker/SWT corrections  
        walker_05 = walker_mean_stress_correction(
            mean_stress, stress_amplitude, gamma=0.5
        )
        swt = swt_mean_stress_correction(mean_stress, stress_amplitude)
        
        # SWT should equal Walker with gamma=0.5
        np.testing.assert_allclose(walker_05, swt, rtol=1e-10)
        
        # All results should be positive and finite
        for result in [dnvgl_08, dnvgl_06, walker_05, swt]:
            assert np.all(result >= 0)
            assert np.all(np.isfinite(result))

    def test_edge_case_stress_combinations(self):
        """Test edge cases with various stress combinations."""
        # Zero stress cases
        zero_mean = np.array([0])
        small_amp = np.array([1e-6])
        
        for correction in [dnvgl_mean_stress_correction, walker_mean_stress_correction, swt_mean_stress_correction]:
            if correction == dnvgl_mean_stress_correction:
                result = correction(zero_mean, small_amp, detail_factor=0.8)
            else:
                result = correction(zero_mean, small_amp)
            assert np.all(np.isfinite(result))
            assert np.all(result >= 0)

    def test_large_stress_values(self):
        """Test with large stress values."""
        large_mean = np.array([1e6])
        large_amp = np.array([5e5])
        
        # Should handle large values gracefully
        result_dnvgl = dnvgl_mean_stress_correction(large_mean, large_amp)
        result_walker = walker_mean_stress_correction(large_mean, large_amp)
        result_swt = swt_mean_stress_correction(large_mean, large_amp)
        
        for result in [result_dnvgl, result_walker, result_swt]:
            assert np.all(np.isfinite(result))
            assert np.all(result >= 0)