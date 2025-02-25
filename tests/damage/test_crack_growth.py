# -*- coding: utf-8 -*-

r"""The following tests are meant to assess the correct behavior of the
damage calculation methods in the stress-life approach.
"""


# Standard imports
import os
import sys

# Non-standard imports
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as hy

PROJECT_PATH = os.path.dirname(os.getcwd())
if not PROJECT_PATH in sys.path:
    sys.path.append(PROJECT_PATH)

# Local imports
import py_fatigue.damage as damage
import py_fatigue.geometry as geometry
from py_fatigue import CycleCount, ParisCurve

# PARIS_1A = ParisCurve(slope=8.16, intercept=1.21e-26)
# PARIS_1B = ParisCurve(slope=8.16, intercept=1.21e-26, critical=350)
# PARIS_1C = ParisCurve(slope=8.16, intercept=1.21e-26, threshold=230)

CC_1A = CycleCount(
    count_cycle=np.ones(50000),
    stress_range=np.ones(50000),
    mean_stress=np.zeros(50000),
)
CC_1B = CycleCount(
    count_cycle=np.array([50000.0]),
    stress_range=np.array([1]),
    mean_stress=np.array([0]),
)

os.environ["NUMBA_DISABLE_JIT"] = "0"

class TestCrackGrowth:
    """Test the functions related with the crack growth rules, i.e.:
    - Paris' law
    - Walker's law (TO BE IMPLEMENTED)
    """


    @pytest.mark.parametrize("scf", [112.3, 540.999, 1000.0])
    @pytest.mark.parametrize("initial_depth", [1.0, 5.0, 9.0])
    @pytest.mark.parametrize("cc_clustered, cc_divided", [(CC_1A, CC_1B)])
    @pytest.mark.parametrize(
        "slope, intercept", [(3.1, 1.2e-14), (4.2, 1.1e-17), (5.0, 1.0e-20)]
    )
    # @given(
    #     scf=hy.floats(min_value=100.0, max_value=1000.0),
    #     initial_depth=hy.floats(min_value=1.0, max_value=9.0),
    # )
    # @settings(deadline=None, max_examples=10)
    def test_paris_law_constant_load(
        self,
        cc_clustered: CycleCount,
        cc_divided: CycleCount,
        scf: float,
        slope: float,
        intercept: float,
        initial_depth: float,
    ):
        """Test the Paris' law for a constant load.

        The test is performed by comparing the crack size calculated by the
        Paris' law analytical integration with the crack size calculated by
        the numerical integration.

        The test asserts also that the crack size is calculated correctly
        both for clustered and divided cycle-count data, i.e.:
        - cc_clustered: cycle-count data with a single stress range and
        a single number of cycles
        - cc_divided: cycle-count data with multiple same stress ranges and
        multiple number of cycles
        """
        cc_clustered = cc_clustered * scf
        cc_divided = cc_divided * scf
        paris_pure = ParisCurve(slope=slope, intercept=intercept)

        # fmt: off
        analytical = 2 / (2 - slope) / intercept * np.pi ** (-slope / 2) \
            * scf ** (-slope) * (-(initial_depth ** (1 - slope / 2)))

        geo = geometry.InfiniteSurface(initial_depth=initial_depth)
        # fmt: on

        cg_c_ex = damage.get_crack_growth(cc_clustered, paris_pure, geo, True)
        cg_c_no_ex = damage.get_crack_growth(
            cc_clustered, paris_pure, geo, False
        )
        cg_d_ex = damage.get_crack_growth(cc_divided, paris_pure, geo, True)

        cg_d_no_ex = damage.get_crack_growth(
            cc_divided, paris_pure, geo, False
        )

        # fmt: off
        if cg_c_ex.failure:
            if analytical > 1E4:
                assert np.isclose(cg_c_ex.final_cycles, analytical, rtol=0.2)
                assert np.isclose(cg_d_ex.final_cycles, analytical, rtol=0.2)
            else:
                assert abs(cg_c_ex.final_cycles - analytical) < 1000
                assert abs(cg_d_ex.final_cycles - analytical) < 1000
        else:
            assert all((cg_c_ex.final_cycles, cg_d_ex.final_cycles,
                        cg_c_no_ex.final_cycles, cg_d_no_ex.final_cycles)) \
                        < analytical
            assert all((cg_c_ex.failure, cg_d_ex.failure, cg_c_no_ex.failure,
                        cg_d_no_ex.failure)) is False
        # assert cg_d_ex.final_cycles == cg_d_no_ex.final_cycles \
        #        == cg_c_no_ex.final_cycles
        # fmt: on

    @pytest.mark.parametrize("scf", [112.3, 540.999, 1000.0])
    @pytest.mark.parametrize("initial_depth", [1.0, 5.0, 9.0])
    @pytest.mark.parametrize("cc_clustered", [(CC_1A)])
    @pytest.mark.parametrize(
        "slope, intercept", [(3.1, 1.2e-14), (4.2, 1.1e-17), (5.0, 1.0e-20)]
    )
    # @given(
    #     scf=hy.floats(min_value=100.0, max_value=1000.0),
    #     initial_depth=hy.floats(min_value=1.0, max_value=9.0),
    # )
    # @settings(deadline=None, max_examples=10)
    def test_paris_law_with_limits_constant_load(
        self,
        cc_clustered: CycleCount,
        scf: float,
        slope: float,
        intercept: float,
        initial_depth: float,
    ):
        """Test the Paris' law for a constant load with threshold or critical
        stress itensity factor.
        """
        cc_clustered = cc_clustered * scf
        threshold = scf * np.sqrt(np.pi * initial_depth)
        paris_threshold = ParisCurve(
            slope=slope, intercept=intercept, threshold=1.01 * threshold
        )
        paris_critical = ParisCurve(
            slope=slope, intercept=intercept, critical=1.2 * threshold
        )
        paris_pure = ParisCurve(slope=slope, intercept=intercept)

        # fmt: off
        analytical = 2 / (2 - slope) / intercept * np.pi ** (-slope / 2) \
            * scf ** (-slope) * (-(initial_depth ** (1 - slope / 2)))
        
        geo = geometry.InfiniteSurface(initial_depth=initial_depth)
        # fmt: on

        assert paris_threshold != paris_critical != paris_pure
        cg_th = damage.get_crack_growth(
            cc_clustered, paris_threshold, geo, express_mode=True
        )
        cg_cr = damage.get_crack_growth(
            cc_clustered, paris_critical, geo, express_mode=True
        )
        cg = damage.get_crack_growth(
            cc_clustered, paris_pure, geo, express_mode=True
        )

        with pytest.raises(ValueError):
            cc_clustered.unit = "error_unit"
            damage.get_crack_growth(
                cc_clustered, paris_pure, geo, express_mode=True
            )

        # fmt: off
        if cg.failure:
            if analytical > 1E4:
                assert cg.final_cycles >= cg_cr.final_cycles
            else:
                assert abs(cg.final_cycles - analytical) < 1000

        assert not cg_th.failure

    @pytest.mark.parametrize("cc_clustered", [(CC_1A)])
    def test_paris_law_inconsistent_units(
        self,
        cc_clustered: CycleCount,
    ):
        """Test the Paris' law for a constant load with threshold or critical
        stress itensity factor when units are not consistent.
        """
        initial_depth, slope, intercept = (1.0, 3.1, 1.2e-14)
        threshold = np.sqrt(np.pi * initial_depth)
        paris_curve = ParisCurve(
            slope=slope, intercept=intercept, threshold=1.01 * threshold,
            unit_string="ksi * in^(-1/2)"
        )

        geo = geometry.InfiniteSurface(initial_depth=initial_depth)
        # fmt: on

        with pytest.raises(ValueError):
            cc_clustered.unit = "error_unit"
            damage.get_crack_growth(
                cc_clustered, paris_curve, geo, express_mode=True
            )
