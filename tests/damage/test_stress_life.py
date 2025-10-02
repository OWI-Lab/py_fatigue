# -*- coding: utf-8 -*-

r"""The following tests are meant to assess the correct behavior of the
damage calculation methods in the stress-life approach.
"""


# Standard imports
import datetime as dt
import os
import sys

# Non-standard imports
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as hy

import py_fatigue.damage as damage
# Local imports
from py_fatigue import CycleCount, SNCurve
from tests.cycle_count.test_cycle_count import CC_RF_1

PROJECT_PATH = os.path.dirname(os.getcwd())
if not PROJECT_PATH in sys.path:
    sys.path.append(PROJECT_PATH)

TIMESTAMP_1 = dt.datetime(2019, 1, 1, tzinfo=dt.timezone.utc)
TIMESTAMP_2 = dt.datetime(2019, 1, 2, tzinfo=dt.timezone.utc)
SIGNAL_1 = np.array([-3, 1, -1, 5, -1, 5, -1, 0, -4, 2, 1, 4, 1, 4, 3, 4, 2])
CC_TS_1 = CycleCount.from_timeseries(
    SIGNAL_1,
    timestamp=TIMESTAMP_1,
    range_bin_lower_bound=0.5,
    range_bin_width=1,
    mean_bin_lower_bound=-0.75,
    mean_bin_width=0.5,
    name="Test_CC",
)
CC_TS_2 = CycleCount.from_timeseries(
    SIGNAL_1,
    timestamp=TIMESTAMP_2,
    range_bin_lower_bound=0.5,
    range_bin_width=1,
    mean_bin_lower_bound=-0.75,
    mean_bin_width=0.5,
    name="Test_CC",
)
CC_TS_3 = CycleCount.from_timeseries(
    np.hstack([SIGNAL_1, SIGNAL_1]),
    timestamp=TIMESTAMP_1,
    range_bin_lower_bound=0.5,
    range_bin_width=1,
    mean_bin_lower_bound=-0.75,
    mean_bin_width=0.5,
    name="Test_CC",
)

EXOTIC = SNCurve(
    [3, 5, 7], [10.970, 13.617, 16], endurance=2.134e11, curve="exotic"
)
DNV_B1A = SNCurve(
    [4, 5],
    [15.117, 17.146],
    norm="DNVGL-RP-C203",
    environment="Air",
    curve="B1",
)
DNV_B1A_END = SNCurve(
    [4, 5],
    [15.117, 17.146],
    norm="DNVGL-RP-C203",
    environment="Air",
    curve="B1",
    endurance=1e9,
)
DNV_B1A = SNCurve(
    [4, 5],
    [15.117, 17.146],
    norm="DNVGL-RP-C203",
    environment="Air",
    curve="B1",
)
DNV_B1W = SNCurve(
    [4, 5],
    [14.917, 17.146],
    norm="DNVGL-RP-C203",
    environment="Seawater with cathodic protection",
    curve="B1",
)
DNV_B1C = SNCurve(
    slope=3,  # SNCurve can handle multiple slope and
    intercept=12.436,  # intercept types, as long as their sizes
    norm="DNVGL-RP-C203",  # are compatible. The slope and intercept attrs
    environment="Free corrosion",  # will be stored as numpy arrays.
    curve="B1",
)
DNV_C_C = SNCurve(
    [3],
    12.115,
    norm="DNVGL-RP-C203",
    environment="Free corrosion",
    curve="C",
)
DNV_E_C = SNCurve(
    3,
    intercept=[
        11.533,
    ],
    norm="DNVGL-RP-C203",
    environment="Free corrosion",
    curve="E",
)
DNV_W3C = SNCurve(
    [3],
    intercept=[
        10.493,
    ],
    norm="DNVGL-RP-C203",
    environment="Free corrosion",
    curve="E",
)


class TestPalmgrenMiner:
    """Test the functions related with the Palmgren-Miner rule, i.e.:
    - Linear damage accumulation
    - Damage-equivalent stress range (DES)
    - Damage-equivalent moment (DEM).
    """

    # fmt: off
    @pytest.mark.parametrize(
        "sn_curve", [(DNV_B1A), (DNV_B1A_END), (DNV_B1W), (DNV_B1C)]
    )
    @given(
        peak=hy.integers(min_value=1, max_value=100),
        len_hist=hy.integers(min_value=6, max_value=1e3),
    )
    @settings(deadline=None, max_examples=10)
    # fmt: on
    def test_palmgren_miner_constant_load(
        self, sn_curve: SNCurve, peak: int, len_hist: int
    ):
        """Test the Palmgren-Miner rule.

        Parameters
        ----------
        sn_curve : SNCurve
            The SNCurve object to use.
        peak : int
            The constant stress amplitude to use.
        len_hist : int
            The length of the history to use.
        """
        time_reversals = [peak * (-1) ** _ for _ in range(len_hist)]
        cc_obj = CycleCount.from_timeseries(
            time_reversals,
            timestamp=TIMESTAMP_1,
            range_bin_lower_bound=0.5,
            range_bin_width=1,
            mean_bin_lower_bound=-0.75,
            mean_bin_width=0.5,
            name="Test_CC",
        )
        assert sum(damage.get_pm(cc_obj, sn_curve)) == pytest.approx(
            sum(cc_obj.count_cycle)
            / sn_curve.get_cycles(np.mean(cc_obj.stress_range)),
            1e-12,
        )
        with pytest.raises(ValueError):
            cc_obj.unit = "m"
            damage.get_pm(cc_obj, sn_curve)


    @pytest.mark.parametrize(
        "sn_curve", [(DNV_B1A), (DNV_B1A_END), (DNV_B1W), (DNV_B1C)]
    )
    # fmt: on
    def test_palmgren_miner_variable_load(self, sn_curve: SNCurve):
        """Test the Palmgren-Miner rule.

        Parameters
        ----------
        sn_curve : SNCurve
            The SNCurve object to use.
        peak : int
            The constant stress amplitude to use.
        len_hist : int
            The length of the history to use.
        """
        damage_pm = sum(damage.get_pm(CC_TS_1, sn_curve))
        damage_ref = sum(
            np.divide(
                CC_TS_1.count_cycle, sn_curve.get_cycles(CC_TS_1.stress_range)
            )
        )
        assert damage_pm == pytest.approx(damage_ref, 1e-12)

    @pytest.mark.parametrize(
        "cc", [(CC_TS_1), (CC_TS_2), (CC_TS_3), (CC_RF_1)]
    )
    @pytest.mark.parametrize("exponent", [(3), (4), (5)])
    @pytest.mark.parametrize("eq_cycles", [(1e6), (2e6), (1e7)])
    # fmt: on
    def test_damage_equivalent_stress(
        self, cc: CycleCount, exponent: int, eq_cycles: int
    ):
        """Test the damage-equivalent stress calculation.

        Parameters
        ----------
        cc : CycleCount
            The CycleCount object to use.
        exponent : int
            The exponent to use.
        eq_cycles : int
            The number of equivalent cycles to use.
        """
        des = damage.get_des(cc, exponent, eq_cycles)
        des_ref = np.power(
            np.sum(
                np.multiply(
                    np.power(cc.stress_range, float(exponent)),
                    cc.count_cycle,
                )
                / eq_cycles
            ),
            1 / exponent,
        )
        assert des == pytest.approx(des_ref, 1e-12)

    @pytest.mark.parametrize(
        "r_o, r_i", [(1050, 1000), (1600, 1500), (2000, 1800)]
    )
    @pytest.mark.parametrize("cc", [(CC_TS_1), (CC_TS_3), (CC_RF_1)])
    @pytest.mark.parametrize("exponent", [(3), (4), (5)])
    @pytest.mark.parametrize("eq_cycles", [(1e6), (2e6), (1e7)])
    # fmt: on
    def test_damage_equivalent_moment(
        self,
        r_o: float,
        r_i: float,
        cc: CycleCount,
        exponent: int,
        eq_cycles: int,
    ):
        """Test the damage-equivalent moment calculation.

        Parameters
        ----------
        r_i : float
            The inner radius to use.
        r_o : float
            The outer radius to use.
        cc : CycleCount
            The CycleCount object to use.
        exponent : int
            The exponent to use.
        eq_cycles : int
            The number of equivalent cycles to use.
        """
        with pytest.raises(ValueError) as ve:
            damage.get_dem(r_i, r_o, cc, exponent, eq_cycles)
            assert "outer_radius must be greater" in ve.value.args[0]
        a_i = np.pi / 4 * (r_o**4 - r_i**4)
        dem = damage.get_dem(r_o, r_i, cc, exponent, eq_cycles)
        dem_ref = a_i * 1E6 * damage.get_des(cc, exponent, eq_cycles) / r_o
        assert dem == pytest.approx(dem_ref, 1e-12)

    # fmt: off
    @settings(deadline=None)
    @pytest.mark.parametrize(
        "sn_curve", [(DNV_B1A), (DNV_B1A_END), (DNV_B1W), (DNV_B1C)]
    )
    @given(
        peak=hy.integers(min_value=1, max_value=100),
        len_hist=hy.integers(min_value=6, max_value=1e3),
    )
    # fmt: on
    def test_miner_pandas_accessor_constant_load(
        self, sn_curve: SNCurve, peak: int, len_hist: int
    ):
        """Test the Palmgren-Miner rule.

        Parameters
        ----------
        sn_curve : SNCurve
            The SNCurve object to use.
        peak : int
            The constant stress amplitude to use.
        len_hist : int
            The length of the history to use.
        """
        time_reversals = [peak * (-1) ** _ for _ in range(len_hist)]
        cc_obj = CycleCount.from_timeseries(
            time_reversals,
            timestamp=TIMESTAMP_1,
            range_bin_lower_bound=0.5,
            range_bin_width=1,
            mean_bin_lower_bound=-0.75,
            mean_bin_width=0.5,
            name="Test_CC",
        )
        df = cc_obj.to_df()
        assert isinstance(df, pd.DataFrame)
        df_d = df.miner.damage(sn_curve)
        assert df_d.sn_curve == sn_curve
        assert df_d._metadata["name"] == cc_obj.name
        assert df_d._metadata["timestamp"] == cc_obj.timestamp
        assert df_d._metadata["mean_stress_corrected"] == \
            cc_obj.mean_stress_corrected
        assert df_d._metadata["stress_concentration_factor"] == \
            cc_obj.stress_concentration_factor
        assert df_d._metadata["nr_small_cycles"] == cc_obj.nr_small_cycles
        assert df_d._metadata["lffd_solved"] == cc_obj.lffd_solved
        assert isinstance(df_d, pd.DataFrame)
        assert df_d["pm_damage"].sum() == pytest.approx(
            np.sum(damage.get_pm(cc_obj, sn_curve)), 1e-12
        )
        with pytest.raises(ValueError):
            cc_obj.unit = "m"
            df = cc_obj.to_df()
            df.miner.damage(sn_curve)

    @pytest.mark.parametrize(
        "sn_curve", [DNV_B1A, DNV_B1A_END, DNV_B1W, DNV_B1C]
    )
    @pytest.mark.parametrize(
        "cc", [CC_TS_1, CC_TS_3]
    )
    # fmt: on
    def test_miner_pandas_accessor_variable_load(
        self, cc: CycleCount, sn_curve: SNCurve
    ):
        """Test the Palmgren-Miner rule.

        Parameters
        ----------
        cc : CycleCount
            The CycleCount object to use.
        sn_curve : SNCurve
            The SNCurve object to use.
        peak : int
            The constant stress amplitude to use.
        len_hist : int
            The length of the history to use.
        """
        df = cc.to_df()
        assert isinstance(df, pd.DataFrame)
        df_d = df.miner.damage(sn_curve)
        assert df_d.sn_curve == sn_curve
        assert df_d._metadata["name"] == cc.name
        assert df_d._metadata["timestamp"] == cc.timestamp
        assert df_d._metadata["mean_stress_corrected"] == \
            cc.mean_stress_corrected
        assert df_d._metadata["stress_concentration_factor"] == \
            cc.stress_concentration_factor
        assert df_d._metadata["nr_small_cycles"] == cc.nr_small_cycles
        assert df_d._metadata["lffd_solved"] == cc.lffd_solved
        assert isinstance(df_d, pd.DataFrame)
        assert df_d["pm_damage"].sum() == pytest.approx(
            np.sum(damage.get_pm(cc, sn_curve)), 1e-12
        )

    @pytest.mark.parametrize(
        "cc,", [CC_TS_1, CC_TS_3]
    )
    @given(
        slope=hy.floats(min_value=3, max_value=20),
        n_eq=hy.floats(min_value=1E5, max_value=1e10),
    )
    # fmt: on
    def test_des_pandas_accessor_variable_load(
        self, cc: CycleCount, slope: float, n_eq: float
    ):
        """Test the DES calculation from dataframe.

        Parameters
        ----------
        cc : CycleCount
            The CycleCount object to use.
        slope : float
            The slope for DES calculation.
        n_eq : float
            The number of equivalent cycles to use in the DES calculation.
        peak : int
            The constant stress amplitude to use.
        len_hist : int
            The length of the history to use.
        """
        df = cc.to_df()
        assert isinstance(df, pd.DataFrame)
        assert df._metadata["name"] == cc.name
        assert df._metadata["timestamp"] == cc.timestamp
        assert df._metadata["mean_stress_corrected"] == \
            cc.mean_stress_corrected
        assert df._metadata["stress_concentration_factor"] == \
            cc.stress_concentration_factor
        assert df._metadata["nr_small_cycles"] == cc.nr_small_cycles
        assert df._metadata["lffd_solved"] == cc.lffd_solved
        assert isinstance(df, pd.DataFrame)
        assert df.miner.des(slope=slope, equivalent_cycles=n_eq) == \
            pytest.approx(
                damage.get_des(cc, slope, equivalent_cycles=n_eq), 1e-12
            )

    @pytest.mark.parametrize(
        "cc,", [CC_TS_1, CC_TS_3]
    )
    @given(
        slope=hy.floats(min_value=3, max_value=20),
        n_eq=hy.floats(min_value=1E5, max_value=1e10),
    )
    @pytest.mark.parametrize(
        "r_i, r_o",
        [(3, 3.5), (4000, 4100), pytest.param(2, 1, marks=pytest.mark.xfail)]
    )
    # fmt: on
    def test_dem_pandas_accessor_variable_load(
        self, cc: CycleCount, slope: float, n_eq: float, r_i: float, r_o: float
    ):
        """Test the DEM calculation from dataframe.

        Parameters
        ----------
        cc : CycleCount
            The CycleCount object to use.
        slope : float
            The slope for DES calculation.
        n_eq : float
            The number of equivalent cycles to use in the DEM calculation.
        r_i : float
            The inner radius of the test for DEM calculation.
        r_o : float
            The outer radius of the test for DEM calculation.
        peak : int
            The constant stress amplitude to use.
        len_hist : int
            The length of the history to use.
        """
        df = cc.to_df()
        assert isinstance(df, pd.DataFrame)
        assert df._metadata["name"] == cc.name
        assert df._metadata["timestamp"] == cc.timestamp
        assert df._metadata["mean_stress_corrected"] == \
            cc.mean_stress_corrected
        assert df._metadata["stress_concentration_factor"] == \
            cc.stress_concentration_factor
        assert df._metadata["nr_small_cycles"] == cc.nr_small_cycles
        assert df._metadata["lffd_solved"] == cc.lffd_solved
        assert isinstance(df, pd.DataFrame)
        assert df.miner.dem(
            outer_radius=r_o,
            inner_radius=r_i,
            slope=slope,
            equivalent_cycles=n_eq
        ) == pytest.approx(
            damage.get_dem(
                outer_radius=r_o,
                inner_radius=r_i,
                cycle_count=cc,
                slope=slope,
                equivalent_cycles=n_eq
            ),
            1e-12
        )

class TestGassner:
    """Test the shift factor calculation related with the Gassner curve
    """

        # fmt: off
    @settings(deadline=None)
    @pytest.mark.parametrize(
        "sn_curve", [DNV_B1C, DNV_C_C, DNV_E_C,
                     pytest.param(DNV_B1A, marks=pytest.mark.xfail)
        ]
    )
    @given(
        peak=hy.integers(min_value=1, max_value=100),
        len_hist=hy.integers(min_value=6, max_value=1e3),
    )
    # fmt: on
    def test_g_pandas_accessor_constant_load(
        self, sn_curve: SNCurve, peak: int, len_hist: int
    ):
        """Test the Gassner shift factor that, for constant loading,
        has to fall back to one.

        Parameters
        ----------
        sn_curve : SNCurve
            The SNCurve object to use.
        peak : int
            The constant stress amplitude to use.
        len_hist : int
            The length of the history to use.
        """
        import py_fatigue as pf
        time_reversals = [peak * (-1) ** _ for _ in range(len_hist)]
        cc_obj = pf.CycleCount.from_timeseries(
            time_reversals,
            timestamp=TIMESTAMP_1,
            range_bin_lower_bound=0.5,
            range_bin_width=1,
            mean_bin_lower_bound=-0.75,
            mean_bin_width=0.5,
            name="Test_CC",
        )
        df = cc_obj.to_df()
        assert isinstance(df, pd.DataFrame)
        df_g = df.gassner.g(sn_curve)
        assert df_g.sn_curve == sn_curve
        assert df_g._metadata["name"] == cc_obj.name
        assert df_g._metadata["timestamp"] == cc_obj.timestamp
        assert df_g._metadata["mean_stress_corrected"] == \
            cc_obj.mean_stress_corrected
        assert df_g._metadata["stress_concentration_factor"] == \
            cc_obj.stress_concentration_factor
        assert df_g._metadata["nr_small_cycles"] == cc_obj.nr_small_cycles
        assert df_g._metadata["lffd_solved"] == cc_obj.lffd_solved
        assert isinstance(df_g, pd.DataFrame)
        assert df_g["shift_factor"].sum() == pytest.approx(1, 1e-12)

    @pytest.mark.parametrize(
        "cc", [CC_TS_1, CC_TS_3]
    )
    @pytest.mark.parametrize(
        "sn_curve", [DNV_B1C, DNV_C_C, DNV_E_C,
                     pytest.param(DNV_B1A, marks=pytest.mark.xfail)
        ]
    )
    # fmt: on
    def test_g_pandas_accessor_variable_load(
        self, cc: CycleCount, sn_curve: SNCurve,
    ):
        """Test the Gassner shift factor that has to be less than one
        for variable amplitude stress histories.

        Parameters
        ----------
        cc : CycleCount
            The CycleCount object to use.
        sn_curve : SNCurve
            The SNCurve object to use.
        """
        df = cc.to_df()
        df.gassner.g(sn_curve)
        assert isinstance(df, pd.DataFrame)
        assert df._metadata["name"] == cc.name
        assert df._metadata["timestamp"] == cc.timestamp
        assert df._metadata["mean_stress_corrected"] == \
            cc.mean_stress_corrected
        assert df._metadata["stress_concentration_factor"] == \
            cc.stress_concentration_factor
        assert df._metadata["nr_small_cycles"] == cc.nr_small_cycles
        assert df._metadata["lffd_solved"] == cc.lffd_solved
        assert df["shift_factor"].sum() < 1

@pytest.mark.parametrize("sn_curve", [DNV_B1C, DNV_C_C, DNV_E_C, DNV_B1A])
@pytest.mark.parametrize("load", [
    [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 3, -3, 3, -3, 3, -3, 3, -3],
    [3, -3, 3, -3, 3, -3, 3, -3, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
])
@pytest.mark.parametrize("base_exponent", [0.99, 1, 1.01])
def test_leve_damage_rule(load: list, sn_curve: SNCurve, base_exponent: float):
    """Test the nonlinear damage calculation.

    Parameters
    ----------
    load : list
        The stress history to use.
    sn_curve : SNCurve
        The SNCurve object to use.
    """
    cc = CycleCount.from_timeseries(
        np.asarray(load),
        timestamp=TIMESTAMP_1,
        range_bin_lower_bound=0.1,
        range_bin_width=0.1,
        mean_bin_lower_bound=-4,
        mean_bin_width=0.1,
        name="Test_CC",
    )
    d_nl = damage.get_nonlinear_damage(
        'leve', cc, sn_curve, base_exponent=base_exponent
    )
    d_l = np.sum(damage.get_pm(cc, sn_curve))
    if base_exponent == 1:
        assert d_nl[-1] == pytest.approx(d_l, 1e-14)
    if base_exponent < 1:
        assert d_nl[-1] > d_l
    if base_exponent > 1:
        assert d_nl[-1] < d_l

@pytest.mark.parametrize("sn_curve", [DNV_B1C, DNV_C_C, DNV_E_C, DNV_B1A])
@pytest.mark.parametrize("rule", ["pavlou", "manson", "si jian"])
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_nonlinear_damage_rules(rule: str, sn_curve: SNCurve):
    """Test the nonlinear damage calculation according to Pavlou,
    Manson and Halford, as well as Si Jian et al. damage models.

    ! This test is not exhaustive, but it is a start.

    Parameters
    ----------
    rule : str
        The damage calculation rule to use.
    load : list
        The stress history to use.
    sn_curve : SNCurve
        The SNCurve object to use.
    """
    lh = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 3, -3, 3, -3, 3, -3, 3, -3]

    hl = [3, -3, 3, -3, 3, -3, 3, -3, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
    cc_lh = CycleCount.from_timeseries(
        np.asarray(lh),
        timestamp=TIMESTAMP_1,
        range_bin_lower_bound=0.1,
        range_bin_width=0.1,
        mean_bin_lower_bound=-4,
        mean_bin_width=0.1,
        name="Test_CC",
    )
    cc_hl = CycleCount.from_timeseries(
        np.asarray(hl),
        timestamp=TIMESTAMP_1,
        range_bin_lower_bound=0.1,
        range_bin_width=0.1,
        mean_bin_lower_bound=-4,
        mean_bin_width=0.1,
        name="Test_CC",
    )
    d_nl_lh = damage.get_nonlinear_damage(rule, cc_lh, sn_curve)
    d_l_lh = np.sum(damage.get_pm(cc_lh, sn_curve))
    d_nl_hl = damage.get_nonlinear_damage(rule, cc_hl, sn_curve)
    d_l_hl = np.sum(damage.get_pm(cc_hl, sn_curve))

    assert d_l_hl == pytest.approx(d_l_lh, 1e-14)
    assert isinstance(d_nl_lh[-1], float)
    assert isinstance(d_nl_hl[-1], float)


class TestDamageExponents:
    @pytest.mark.parametrize(
        "damage_rule", ["pavlou", "manson", "leve", "si jian"]
    )
    def test_calc_damage_exponents_no_kwargs(self, damage_rule):
        """Test the _calc_damage_exponents function with no kwargs."""
        stress_range = np.array([100., 200., 300.])
        if "manson" in damage_rule:
            with pytest.raises(ValueError, match="sn_curve must be provided"):
                damage.stress_life._calc_damage_exponents(damage_rule, stress_range)
        else:
            if damage_rule == "pavlou":
                with pytest.warns(UserWarning, match="base_exponent"):
                    with pytest.warns(UserWarning, match="ultimate_stress"):
                        exponents = damage.stress_life._calc_damage_exponents(
                            damage_rule, stress_range
                        )
                        assert isinstance(exponents, np.ndarray)
            if damage_rule == "leve":
                with pytest.warns(UserWarning, match="base_exponent"):
                    exponents = damage.stress_life._calc_damage_exponents(
                        damage_rule, stress_range
                    )
                    assert isinstance(exponents, np.ndarray)
            if damage_rule == "si jian":
                exponents = damage.stress_life._calc_damage_exponents(
                    damage_rule, stress_range
                )
                assert isinstance(exponents, np.ndarray)

    def test_calc_damage_exponents_pavlou(self):
        """Test the _calc_damage_exponents function with pavlou rule."""
        stress_range = np.array([100, 200, 300])
        base_exponent = -0.5
        ultimate_stress = 800
        exponents = damage.stress_life._calc_damage_exponents(
            "pavlou",
            stress_range,
            base_exponent=base_exponent,
            ultimate_stress=ultimate_stress,
        )
        assert isinstance(exponents, np.ndarray)
        assert np.allclose(
            exponents,
            damage.stress_life.calc_pavlou_exponents(
                stress_range, ultimate_stress, base_exponent
            ),
        )

    def test_calc_damage_exponents_manson(self):
        """Test the _calc_damage_exponents function with manson rule."""
        stress_range = np.array([100, 200, 300])
        base_exponent = 0.5
        sn_curve = SNCurve([3, 5, 7], [10.970, 13.617, 16])
        exponents = damage.stress_life._calc_damage_exponents(
            "manson", stress_range, sn_curve=sn_curve, base_exponent=base_exponent
        )
        assert isinstance(exponents, np.ndarray)
        assert np.allclose(
            exponents,
            damage.stress_life.calc_manson_halford_exponents(
                sn_curve.get_cycles(stress_range), base_exponent
            ),
        )

    def test_calc_damage_exponents_leve(self):
        """Test the _calc_damage_exponents function with leve rule."""
        stress_range = np.array([100, 200, 300])
        base_exponent = 3
        exponents = damage.stress_life._calc_damage_exponents(
            "leve", stress_range, base_exponent=base_exponent
        )
        assert isinstance(exponents, np.ndarray)
        assert np.allclose(exponents, base_exponent * np.ones(len(stress_range)))

    def test_calc_damage_exponents_si_jian(self):
        """Test the _calc_damage_exponents function with si jian rule."""
        stress_range = np.array([100, 200, 300])
        exponents = damage.stress_life._calc_damage_exponents("si jian", stress_range)
        assert isinstance(exponents, np.ndarray)
        assert np.allclose(exponents, damage.stress_life.calc_si_jian_exponents(stress_range))

    def test_calc_damage_exponents_unknown_rule(self):
        """Test the _calc_damage_exponents function with unknown rule."""
        stress_range = np.array([100, 200, 300])
        with pytest.raises(ValueError, match="Unknown damage rule: unknown"):
            damage.stress_life._calc_damage_exponents("unknown", stress_range)

@pytest.mark.parametrize("sn_curve", [DNV_B1C, DNV_C_C, DNV_E_C, DNV_B1A])
@pytest.mark.parametrize("load", [
    [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 3, -3, 3, -3, 3, -3, 3, -3],
    [3, -3, 3, -3, 3, -3, 3, -3, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
])
@pytest.mark.parametrize("damage_bands", [
    [0, 0.2, 0.4, 0.6, 0.8, 1],
    [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
])
def test_nonlinear_damage_dca(
    load: list, sn_curve: SNCurve, damage_bands: list
):
    """Test the nonlinear damage calculation with DCA.

    Parameters
    ----------
    load : list
        The stress history to use.
    sn_curve : SNCurve
        The SNCurve object to use.
    damage_bands : list
        The damage bands to use.
    """
    cc = CycleCount.from_timeseries(
        np.asarray(load),
        timestamp=TIMESTAMP_1,
        range_bin_lower_bound=0.1,
        range_bin_width=0.1,
        mean_bin_lower_bound=-4,
        mean_bin_width=0.1,
        name="Test_CC",
    )
    d_nl, _, _, _ = damage.stress_life.get_nonlinear_damage_with_dca(
        'pavlou', cc, sn_curve, np.asarray(damage_bands)
    )
    assert isinstance(d_nl, np.ndarray)

@pytest.mark.parametrize("sn_curve", [DNV_B1C, DNV_C_C, DNV_E_C, DNV_B1A])
@pytest.mark.parametrize("load", [
    [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 3, -3, 3, -3, 3, -3, 3, -3],
    [3, -3, 3, -3, 3, -3, 3, -3, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
])
def test_theil_damage_rule(load: list, sn_curve: SNCurve):
    """Test the nonlinear damage calculation with Theil's method.

    Parameters
    ----------
    load : list
        The stress history to use.
    sn_curve : SNCurve
        The SNCurve object to use.
    """
    cc = CycleCount.from_timeseries(
        np.asarray(load),
        timestamp=TIMESTAMP_1,
        range_bin_lower_bound=0.1,
        range_bin_width=0.1,
        mean_bin_lower_bound=-4,
        mean_bin_width=0.1,
        name="Test_CC",
    )
    d_nl, _, _, _ = damage.stress_life.get_nonlinear_damage_with_dca(
        'theil', cc, sn_curve, damage_bands=np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
    )
    assert isinstance(d_nl, np.ndarray)

