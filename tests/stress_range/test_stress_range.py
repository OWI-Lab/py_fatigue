# -*- coding: utf-8 -*-

r"""The following tests are meant to assess the correct behavior of the
StressRange class.
"""


# from hypothesis import  given, strategies as hy
# Standard imports
import unittest

# Non-standard imports
import numpy as np

from py_fatigue.stress_range import StressRange


class TestStressRange(unittest.TestCase):
    """Test the StressRange class."""

    # inputs
    values = np.array([2.0, 6.0, 6.0, 6.0, 1.0, 1.0, 3.0, 1.0, 9.0, 8.0])
    counts = np.array([0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5])
    # expected outputs
    half = np.array([2.0, 6.0, 6.0, 6.0, 9.0, 8.0])
    full = np.array([1.0, 1.0, 3.0, 1.0])
    bin_edges_1 = np.array([0.5, 1.5, 2.5, 3.5])
    bin_edges_2 = np.array([-0.4, 0.6, 1.6, 2.6, 3.6])
    bin_edges_3 = np.array([-1.0, 1.0, 3.0])
    bin_edges_4 = np.array([-0.4, 1.6, 3.6])
    bin_centers_1 = np.array([1.0, 2.0, 3.0])
    bin_centers_2 = np.array([0.1, 1.1, 2.1, 3.1])
    bin_centers_3 = np.array([0.0, 2.0])
    bin_centers_4 = np.array([0.6, 2.6])
    bind_vals_1 = np.array([1.0, 1.0, 3.0, 1.0])
    bind_vals_2 = np.array([1.1, 1.1, 3.1, 1.1])
    bind_vals_3 = np.array([0.0, 0.0, 2.0, 0.0])
    bind_vals_4 = np.array([0.6, 0.6, 2.6, 0.6])

    def setUp(self):
        """
        Set up for some of the tests.
        """
        # load irregular 3-hour time series test rebin and mesh
        self.stress_range_1 = StressRange(
            _values=self.values, _counts=self.counts, bin_width=1
        )
        self.stress_range_2 = StressRange(
            _values=self.values, _counts=self.counts, bin_width=1, _bin_lb=-0.4
        )
        self.stress_range_3 = StressRange(
            _values=self.values, _counts=self.counts, bin_width=2
        )
        self.stress_range_4 = StressRange(
            _values=self.values, _counts=self.counts, bin_width=2, _bin_lb=-0.4
        )

    def test_bin_edges(self):
        """
        Test the bin_edges property.
        """
        assert np.allclose(self.stress_range_1.bin_edges, self.bin_edges_1)
        assert np.allclose(self.stress_range_2.bin_edges, self.bin_edges_2)
        assert np.allclose(self.stress_range_3.bin_edges, self.bin_edges_3)
        assert np.allclose(self.stress_range_4.bin_edges, self.bin_edges_4)

    def test_bin_centers(self):
        """
        Test the bin_centers property.
        """
        assert np.allclose(self.stress_range_1.bin_centers, self.bin_centers_1)
        assert np.allclose(self.stress_range_2.bin_centers, self.bin_centers_2)
        assert np.allclose(self.stress_range_3.bin_centers, self.bin_centers_3)
        assert np.allclose(self.stress_range_4.bin_centers, self.bin_centers_4)

    def test_binned_values(self):
        """
        Test the binned_values property.
        """
        assert np.allclose(self.stress_range_1.binned_values, self.bind_vals_1)
        assert np.allclose(self.stress_range_2.binned_values, self.bind_vals_2)
        assert np.allclose(self.stress_range_3.binned_values, self.bind_vals_3)
        assert np.allclose(self.stress_range_4.binned_values, self.bind_vals_4)

    def test_full(self):
        """
        Test the full property.
        """
        assert np.allclose(self.stress_range_1.full, self.full)
        assert np.allclose(self.stress_range_2.full, self.full)
        assert np.allclose(self.stress_range_3.full, self.full)
        assert np.allclose(self.stress_range_4.full, self.full)

    def test_half(self):
        """
        Test the half property.
        """
        assert np.allclose(self.stress_range_1.half, self.half)
        assert np.allclose(self.stress_range_2.half, self.half)
        assert np.allclose(self.stress_range_3.half, self.half)
        assert np.allclose(self.stress_range_4.half, self.half)


if __name__ == '__main__':
    unittest.main()