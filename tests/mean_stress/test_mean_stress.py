# -*- coding: utf-8 -*-

r"""The following tests are meant to assess the correct behavior of the
MeanStress class.
"""


# from hypothesis import  given, strategies as hy
# Standard imports
import unittest

# Non-standard imports
import numpy as np

from py_fatigue.mean_stress import MeanStress


class TestMeanStress(unittest.TestCase):
    """Test the MeanStress class."""

    # inputs
    values = np.array([0.0, 2.0, 2.0, 2.0, -0.5, 1.5, 2.5, 3.5, 0.5, 0.0])
    counts = np.array([0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5])
    # expected outputs
    half = np.array([0.0, 2.0, 2.0, 2.0, 0.5, 0.0])
    full = np.array([-0.5, 1.5, 2.5, 3.5])
    bin_edges_1 = np.array(
        [-0.75, -0.25, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75]
    )
    bin_edges_2 = np.array(
        [-0.6, -0.1, 0.4, 0.9, 1.4, 1.9, 2.4, 2.9, 3.4, 3.9]
    )
    bin_edges_3 = np.array([-0.5, 0.5, 1.5, 2.5, 3.5])
    bin_edges_4 = np.array([-0.6, 0.4, 1.4, 2.4, 3.4, 4.4])
    bin_centers_1 = np.array([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    bin_centers_2 = np.array(
        [-0.35, 0.15, 0.65, 1.15, 1.65, 2.15, 2.65, 3.15, 3.65]
    )
    bin_centers_3 = np.array([0.0, 1.0, 2.0, 3.0])
    bin_centers_4 = np.array([-0.1, 0.9, 1.9, 2.9, 3.9])
    bind_vals_1 = np.array([-0.5, 1.5, 2.5, 3.5])
    bind_vals_2 = np.array([-0.35, 1.65, 2.65, 3.65])
    bind_vals_3 = np.array([0.0, 1.0, 2.0, 3.0])
    bind_vals_4 = np.array([-0.1, 1.9, 2.9, 3.9])

    def setUp(self):
        """
        Set up for some of the tests.
        """
        # load irregular 3-hour time series test rebin and mesh
        self.mean_stress_1 = MeanStress(
            _values=self.values, _counts=self.counts, bin_width=0.5
        )
        self.mean_stress_2 = MeanStress(
            _values=self.values,
            _counts=self.counts,
            bin_width=0.5,
            _bin_lb=-0.6,
        )
        self.mean_stress_3 = MeanStress(
            _values=self.values, _counts=self.counts, bin_width=1
        )
        self.mean_stress_4 = MeanStress(
            _values=self.values, _counts=self.counts, bin_width=1, _bin_lb=-0.6
        )

    def test_bin_edges(self):
        """
        Test the bin_edges property.
        """
        assert np.allclose(self.mean_stress_1.bin_edges, self.bin_edges_1)
        assert np.allclose(self.mean_stress_2.bin_edges, self.bin_edges_2)
        assert np.allclose(self.mean_stress_3.bin_edges, self.bin_edges_3)
        assert np.allclose(self.mean_stress_4.bin_edges, self.bin_edges_4)

    def test_bin_centers(self):
        """
        Test the bin_centers property.
        """
        assert np.allclose(self.mean_stress_1.bin_centers, self.bin_centers_1)
        assert np.allclose(self.mean_stress_2.bin_centers, self.bin_centers_2)
        assert np.allclose(self.mean_stress_3.bin_centers, self.bin_centers_3)
        assert np.allclose(self.mean_stress_4.bin_centers, self.bin_centers_4)

    def test_binned_values(self):
        """
        Test the binned_values property.
        """
        assert np.allclose(self.mean_stress_1.binned_values, self.bind_vals_1)
        assert np.allclose(self.mean_stress_2.binned_values, self.bind_vals_2)
        assert np.allclose(self.mean_stress_3.binned_values, self.bind_vals_3)
        assert np.allclose(self.mean_stress_4.binned_values, self.bind_vals_4)

    def test_full(self):
        """
        Test the full property.
        """
        assert np.allclose(self.mean_stress_1.full, self.full)
        assert np.allclose(self.mean_stress_2.full, self.full)
        assert np.allclose(self.mean_stress_3.full, self.full)
        assert np.allclose(self.mean_stress_4.full, self.full)

    def test_half(self):
        """
        Test the half property.
        """
        assert np.allclose(self.mean_stress_1.half, self.half)
        assert np.allclose(self.mean_stress_2.half, self.half)
        assert np.allclose(self.mean_stress_3.half, self.half)
        assert np.allclose(self.mean_stress_4.half, self.half)


if __name__ == "__main__":
    unittest.main()
