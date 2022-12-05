# -*- coding: utf-8 -*-

r"""The following tests are meant to assess the correct behavior of the
CycleCount class.
"""


# Standard imports
import copy
import datetime as dt
import itertools
import json
import os
import warnings
from hypothesis import given
import pytest
import sys
from typing import Union

PROJECT_PATH = os.path.dirname(os.getcwd())
if not PROJECT_PATH in sys.path:
    sys.path.append(PROJECT_PATH)
# Non-standard imports
import matplotlib 
import numpy as np
from hypothesis import given, strategies as hy

from py_fatigue.cycle_count.cycle_count import (
    _build_input_data_from_json,
    _assess_json_keys,
)
from py_fatigue.cycle_count import CycleCount
from py_fatigue.stress_range import StressRange
from py_fatigue.mean_stress import MeanStress
import py_fatigue.utils as pfu
import py_fatigue.cycle_count.rainflow as rf


# Input testing parameters
TIMESTAMP_1 = dt.datetime(2019, 1, 1, tzinfo=dt.timezone.utc)
TIMESTAMP_2 = dt.datetime(2019, 1, 2, tzinfo=dt.timezone.utc)
SIGNAL_1 = np.array([-3, 1, -1, 5, -1, 5, -1, 0, -4, 2, 1, 4, 1, 4, 3, 4, 2])
SIGNAL_2 = np.array(
    [300, -100, 300, -100, 300, -100, 300, -100, 300, -100, 300, -100]
)
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
CC_TS_4 = CycleCount.from_timeseries(
    np.hstack(SIGNAL_2),
    timestamp=TIMESTAMP_1,
    range_bin_lower_bound=0.5,
    range_bin_width=1,
    name="Test_CC",
)

# fmt: off
AS_DICT_1 = {"nr_small_cycles": 0, "range_bin_lower_bound": 0.5,
             "range_bin_width": 1, "mean_bin_lower_bound": -0.75,
             "mean_bin_width": 0.5,
             "hist": [[1.0], [0.0, 1.0], [], [], [1.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [],
                      [1.0]],
                      "lg_c": [],
                      "res": [[1.0, 8.0], [0.5, 9.0], [0.0, 8.0], [3.0, 2.0]],
                      "res_sig": [-3, 5, -4, 4, 2]}
AS_DICT_1_1D = {"nr_small_cycles": 0, "range_bin_lower_bound": 0.5,
                "range_bin_width": 1, "hist": [3.0, 1.0, 1.0, 0.0, 0.0, 1.0],
                "lg_c": [], "res": [8.0, 9.0, 8.0, 2.0],
                "res_sig": [-3, 5, -4, 4, 2]}
CC_RF_1 = CycleCount.from_rainflow(
    AS_DICT_1,
    timestamp=TIMESTAMP_1,
    name="Test_CC",
)
# fmt: on
CTS_1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5])
S_R_1 = np.array([1.0, 2.0, 1.0, 6.0, 3.0, 1.0, 8.0, 9.0, 8.0, 2.0])
M_S_1 = np.array([-0.5, 0.0, 1.5, 2.0, 2.5, 3.5, 1.0, 0.5, 0.0, 3.0])
S_R_1_COR_DNV8 = np.array([7.4, 8.2, 7.2, 2.0, 1.8, 5.8, 0.8, 1.0, 3.0, 1.0])
M_S_1_COR_DNV8 = np.array([3.7, 4.1, 3.6, 1.0, 0.9, 2.9, 0.4, 0.5, 1.5, 0.5])
S_R_1_COR_DNV6 = np.array([6.8, 7.4, 6.4, 2.0, 1.6, 5.6, 0.6, 1.0, 3.0, 1.0])
M_S_1_COR_DNV6 = np.array([3.4, 3.7, 3.2, 1.0, 0.8, 2.8, 0.3, 0.5, 1.5, 0.5])
STRESS_RANGE_1 = StressRange(CTS_1, S_R_1, _bin_lb=0.5, bin_width=1)
MEAN_STRESS_1 = MeanStress(CTS_1, M_S_1, _bin_lb=-0.75, bin_width=0.5)
STRESS_RANGE_1_COR_DNV8 = StressRange(
    CTS_1, S_R_1_COR_DNV8, _bin_lb=0.5, bin_width=1
)
MEAN_STRESS_1_COR_DNV8 = MeanStress(
    CTS_1, M_S_1_COR_DNV8, _bin_lb=-0.75, bin_width=0.5
)
STRESS_RANGE_1_COR_DNV6 = StressRange(
    CTS_1, S_R_1_COR_DNV6, _bin_lb=0.5, bin_width=1
)
MEAN_STRESS_1_COR_DNV6 = MeanStress(
    CTS_1, M_S_1_COR_DNV6, _bin_lb=-0.75, bin_width=0.5
)
STRESS_RANGE_ADD = StressRange(
    np.hstack([CTS_1, CTS_1]),
    np.hstack([S_R_1, S_R_1]),
    _bin_lb=0.5,
    bin_width=1,
)
MEAN_STRESS_ADD = MeanStress(
    np.hstack([CTS_1, CTS_1]),
    np.hstack([M_S_1, M_S_1]),
    _bin_lb=-0.75,
    bin_width=0.5,
)

RES_SIG_BEFORE_1 = np.hstack([AS_DICT_1["res_sig"], AS_DICT_1["res_sig"]])
_, RES_SIG_AFTER_1, _ = rf.rainflow(RES_SIG_BEFORE_1)

cc_names = "cc, stress_range, mean_stress, res_sig"
cc_data = [
    (CC_TS_1, STRESS_RANGE_1, MEAN_STRESS_1, AS_DICT_1["res_sig"]),
    (CC_RF_1, STRESS_RANGE_1, MEAN_STRESS_1, AS_DICT_1["res_sig"]),
]


def test_build_input_data_from_json():
    """Test the _build_input_data_from_json function for 1D data."""
    dct = _build_input_data_from_json(
        AS_DICT_1, name="Test_CC", timestamp=TIMESTAMP_1
    )
    assert isinstance(dct, dict)
    assert dct["nr_small_cycles"] == 0
    assert dct["range_bin_lower_bound"] == 0.5
    assert dct["range_bin_width"] == 1
    assert np.sum(dct["count_cycle"]) == np.sum(CTS_1)


def test_build_input_data_from_json_legacy():
    """Test the _build_input_data_from_json function for 1D data."""
    dct = _build_input_data_from_json(
        AS_DICT_1_1D, name="Test_CC", timestamp=TIMESTAMP_1
    )
    assert isinstance(dct, dict)
    assert dct["nr_small_cycles"] == 0
    assert dct["range_bin_lower_bound"] == 0.5
    assert dct["range_bin_width"] == 1
    assert np.sum(dct["count_cycle"]) == np.sum(CTS_1)


def test_empty_hist_exception() -> None:
    """Test the _build_input_data_from_json function for 1D data."""
    # fmt: off
    dct = {"nr_small_cycles": 0, "range_bin_lower_bound": 0.5,
                "range_bin_width": 1, "hist": [], "lg_c": [], "res":[],
                "res_sig": [ 1, -1]}
    # fmt: on
    with pytest.raises(ValueError) as ve:
        _build_input_data_from_json(dct, name="Test_CC", timestamp=TIMESTAMP_1)
        assert "empty" in ve.value.args[0]


def test__assess_json_keys() -> None:
    """Test the _assess_json_keys function for 1D data."""
    # fmt: off
    dct_lb_1 = {"nr_small_cycles": 0, "range_bin_lower_bound": 0.5,
                "bin_lb": 0.5, "range_bin_width": 1, "hist": [], "lg_c": [],
                "res":[], "res_sig": [ 1, -1]}
    dct_allow = {"nr_small_cyc": 0, "range_bin_lower_bound": 0.5,
                 "range_bin_width": 1, "hist": [], "lg_c": [],
                 "mean_bin_lower_bound": 0.5, "res":[], "res_sig": [ 1, -1]}
    dct_res_sig = {"nr_small_cycles": 0, "range_bin_lower_bound": 0.5,
                   "range_bin_width": 1, "hist": [], "lg_c": [],
                   "mean_bin_lower_bound": 0.5,
                   "res":[]}
    # fmt: on
    with pytest.raises(KeyError) as ke:
        _assess_json_keys(dct_lb_1)
        assert "both present" in ke.value.args[0]
    with pytest.raises(KeyError) as ke:
        _assess_json_keys(dct_allow)
        assert "not an allowed key" in ke.value.args[0]

    print(_assess_json_keys(dct_res_sig))
    assert "res_sig" in _assess_json_keys(dct_res_sig)


def test_legacy_json_keys() -> None:
    # fmt: off
    dct_1 = {"nr_small_cycles": 0, "range_bin_lower_bound": 0.5,
             "range_bin_width": 1, "hist": [3, 0, 1], "lg_c": [],
             "res": [8.0, 9.0, 8.0, 2.0], "res_sig": [-3, 5, -4, 4, 2]}
    dct_2 = {"no_sm_c": 0, "bin_lb": 0.5,
             "bin_width": 1, "hist": [3, 0, 1], "lg_c": [], 
             "res": [8.0, 9.0, 8.0, 2.0], "res_sig": [-3, 5, -4, 4, 2]}
    dct_3 = {"nr_small_cycles": 0, "bin_lb": 0.5,
             "bin_width": 1, "hist": [3, 0, 1], "lg_c": [],
             "res": [8.0, 9.0, 8.0, 2.0], "res_sig": [-3, 5, -4, 4, 2]}
    # fmt: on
    dct_lst = []
    for dct in [dct_1, dct_2, dct_3]:
        dct_lst.append(
            _build_input_data_from_json(
                dct, name="Test_CC", timestamp=TIMESTAMP_1
            )
        )
        assert pfu.compare(dct_lst[0], dct_lst[-1])


class TestCycleCount:
    """Test the CycleCount class."""

    @pytest.mark.parametrize(cc_names, cc_data)
    def test_properties(
        self,
        cc: CycleCount,
        stress_range: StressRange,
        mean_stress: MeanStress,
        res_sig: Union[list, np.ndarray],
    ) -> None:
        """Testing the instance attributed of the CycleCount class by
        comparing them to the expected values from MeanStress and
        StressRange instances.
        In this test it is also assessed that CycleCount instances
        from timeseries or rainflow are equivalent.
        """

        assert cc.name == "Test_CC"
        assert np.allclose(
            np.sort(cc.stress_range), np.sort(stress_range.values)
        )
        assert np.allclose(
            np.sort(cc.mean_stress), np.sort(mean_stress.values)
        )
        assert np.allclose(
            np.sort(cc.count_cycle), np.sort(stress_range.counts)
        )
        assert np.allclose(
            np.sort(cc.count_cycle), np.sort(mean_stress.counts)
        )
        assert cc.range_bin_lower_bound == stress_range.bin_lb
        assert cc.range_bin_width == stress_range.bin_width
        assert cc.range_bin_upper_bound == stress_range.bin_ub
        assert cc.mean_bin_lower_bound == mean_stress.bin_lb
        assert cc.mean_bin_width == mean_stress.bin_width
        assert cc.mean_bin_upper_bound == mean_stress.bin_ub
        assert cc.timestamp == TIMESTAMP_1
        assert cc.nr_small_cycles == 0
        assert np.allclose(
            np.sort(cc.full_cycles[:, 0]), np.sort(mean_stress.full)
        )
        assert np.allclose(
            np.sort(cc.full_cycles[:, 1]), np.sort(stress_range.full)
        )
        assert np.allclose(
            np.sort(cc.half_cycles[:, 0]), np.sort(mean_stress.half)
        )
        assert np.allclose(
            np.sort(cc.half_cycles[:, 1]), np.sort(stress_range.half)
        )
        assert np.allclose(cc.bin_centers[0], mean_stress.bin_centers)
        assert np.allclose(cc.bin_centers[1], stress_range.bin_centers)
        assert np.allclose(cc.bin_edges[0], mean_stress.bin_edges)
        assert np.allclose(cc.bin_edges[1], stress_range.bin_edges)
        assert np.allclose(cc.residuals_sequence, res_sig)
        assert np.allclose(cc.min_max_sequence, [min(res_sig), max(res_sig)])

    @pytest.mark.parametrize("cc_ts, cc_rf", [(CC_TS_1, CC_RF_1)])
    @given(var=hy.integers(min_value=0, max_value=1e12))
    def test__eq__(
        self,
        cc_ts: CycleCount,
        cc_rf: CycleCount,
        var: int,
    ) -> None:
        """Test the from_timeseries method."""
        cc_ts_rf = CycleCount.from_rainflow(
            cc_ts.as_dict(), timestamp=TIMESTAMP_1, name="Test_CC"
        )
        assert cc_ts_rf == cc_rf
        assert cc_rf != var

    @pytest.mark.parametrize("cc_ts, cc_rf", [(CC_TS_1, CC_RF_1)])
    def test_as_dict(
        self,
        cc_ts: CycleCount,
        cc_rf: CycleCount,
    ) -> None:
        """Test the as_dict method."""
        export_dict = cc_ts.as_dict()
        cc_ts_rf = CycleCount.from_rainflow(
            export_dict, timestamp=TIMESTAMP_1, name="Test_CC"
        )

        export_n = (
            sum(np.asarray(list(itertools.chain(*export_dict["hist"]))))
            + 0.5 * len(export_dict["res"])
            + len(export_dict["lg_c"])
            + export_dict["nr_small_cycles"]
        )
        assert sum(cc_ts.count_cycle) == sum(cc_rf.count_cycle) == export_n
        assert len(export_dict["res_sig"]) == len(export_dict["res"]) + 1
        assert cc_ts_rf == cc_rf

        with pytest.raises(TypeError) as te:
            json.dumps(copy.deepcopy(export_dict))
            json.dumps(cc_rf.as_dict())
            assert not "not JSON serializable" in te.value.args[0]
    
    @pytest.mark.parametrize("cc_ts, cc_rf", [(CC_TS_1, CC_RF_1)])
    def test_as_dict_legacy_export(
        self,
        cc_ts: CycleCount,
        cc_rf: CycleCount,
    ) -> None:
        """Test the as_dict method."""
        export_dict = cc_ts.as_dict(legacy_export=True)
        cc_ts_rf = CycleCount.from_rainflow(
            export_dict, timestamp=TIMESTAMP_1, name="Test_CC"
        )
        cc_rf_rf = CycleCount.from_rainflow(
            cc_rf.as_dict(legacy_export=True),
            timestamp=TIMESTAMP_1,
            name="Test_CC"
        )

        export_n = (
            sum(np.asarray(export_dict["hist"]))
            + 0.5 * len(export_dict["res"])
            + len(export_dict["lg_c"])
            + export_dict["nr_small_cycles"]
        )
        assert sum(cc_ts.count_cycle) == sum(cc_rf.count_cycle) == export_n
        assert len(export_dict["res_sig"]) == len(export_dict["res"]) + 1
        assert cc_ts_rf == cc_rf_rf

        with pytest.raises(TypeError) as te:
            json.dumps(copy.deepcopy(export_dict))
            json.dumps(cc_rf.as_dict(legacy_export=True))
            json.dumps(cc_rf_rf.as_dict(legacy_export=True))
            json.dumps(cc_ts_rf.as_dict(legacy_export=True))
            assert not "not JSON serializable" in te.value.args[0]

    @pytest.mark.parametrize(
        "cc_1, cc_2, stress_range, mean_stress, res_sig",
        [
            (
                CC_TS_1,
                CC_TS_2,
                STRESS_RANGE_ADD,
                MEAN_STRESS_ADD,
                np.hstack([AS_DICT_1["res_sig"], AS_DICT_1["res_sig"]]),
            )
        ],
    )
    def test__add__(
        self,
        cc_1: CycleCount,
        cc_2: CycleCount,
        stress_range: StressRange,
        mean_stress: MeanStress,
        res_sig: Union[list, np.ndarray],
    ) -> None:
        """Test the __add__ method."""
        cc_add = cc_1 + cc_2

        assert cc_add.name == "Test_CC"
        assert len(cc_add.time_sequence) == 2
        assert cc_add.range_bin_lower_bound == stress_range.bin_lb
        assert cc_add.range_bin_width == stress_range.bin_width
        assert cc_add.range_bin_upper_bound == stress_range.bin_ub
        assert cc_add.mean_bin_lower_bound == mean_stress.bin_lb
        assert cc_add.mean_bin_width == mean_stress.bin_width
        assert cc_add.mean_bin_upper_bound == mean_stress.bin_ub
        assert cc_add.timestamp == TIMESTAMP_1
        assert cc_add.nr_small_cycles == 0
        print("------------------")
        print(
            "stress_range.full",
            np.allclose(cc_add.full_cycles[:, 1], stress_range.full),
        )
        print("------------------")
        assert np.allclose(
            np.sort(cc_add.full_cycles[:, 0]), np.sort(mean_stress.full)
        )
        assert np.allclose(
            np.sort(cc_add.full_cycles[:, 1]), np.sort(stress_range.full)
        )
        assert np.allclose(
            np.sort(cc_add.half_cycles[:, 0]), np.sort(mean_stress.half)
        )
        assert np.allclose(
            np.sort(cc_add.half_cycles[:, 1]), np.sort(stress_range.half)
        )
        assert np.allclose(cc_add.bin_centers[0], mean_stress.bin_centers)
        assert np.allclose(cc_add.bin_centers[1], stress_range.bin_centers)
        assert np.allclose(cc_add.bin_edges[0], mean_stress.bin_edges)
        assert np.allclose(cc_add.bin_edges[1], stress_range.bin_edges)
        assert np.allclose(cc_add.residuals_sequence, res_sig)
        assert np.allclose(
            cc_add.min_max_sequence,
            [min(res_sig), max(res_sig), min(res_sig), max(res_sig)],
        )

    @pytest.mark.parametrize("cc", [(CC_TS_1)])
    @given(
        scf_1=hy.integers(min_value=1, max_value=1e3),
        scf_2=hy.integers(min_value=1, max_value=1e3),
    )
    def test__mul__(
        self,
        cc: CycleCount,
        scf_1: int,
        scf_2: int,
    ) -> None:
        """Test the __mul__ method."""

        def assertions_on_scf(
            cc: CycleCount, scf_list: list, scf: Union[float, int]
        ) -> None:
            for cc_scf in scf_list:
                print("cc_scf.mean_stress", cc_scf.mean_stress)
                print("cc_scf.stress_range", cc_scf.stress_range)
                print("cc_scf.residuals_sequence", cc_scf.residuals_sequence)
                print("cc_scf.mean_stress", cc_scf.mean_stress)
                print("cc_scf.mean_stress", cc_scf.count_cycle)
                v = np.ones(len(cc_scf.count_cycle))
                assert np.allclose(
                    np.divide(cc_scf.stress_range, cc.stress_range), scf * v
                )
                assert np.allclose(cc_scf.mean_stress, scf * cc.mean_stress)
                assert np.allclose(cc_scf.count_cycle, cc.count_cycle)
                res_sig = np.divide(
                    cc_scf.residuals_sequence, cc.residuals_sequence
                )
                res_sig[np.where(np.isnan(res_sig))] = 1
                assert np.allclose(
                    res_sig,
                    scf * np.ones(len(res_sig)),
                )

        cc_scf_1 = scf_1 * cc
        cc_scf_2 = cc * scf_1
        assert cc * scf_1 == cc * float(scf_1)
        assert cc * scf_2 == cc * float(scf_2)
        assert cc_scf_1.stress_concentration_factor == scf_1
        assert cc_scf_2.stress_concentration_factor == scf_1
        assert cc_scf_1 == cc_scf_2

        cc_scf_3 = scf_1 * scf_2 * cc
        cc_scf_4 = cc * scf_1 * scf_2
        cc_scf_5 = scf_1 * cc * scf_2
        assert cc_scf_3.stress_concentration_factor == scf_1 * scf_2
        assert cc_scf_4.stress_concentration_factor == scf_1 * scf_2
        assert cc_scf_5.stress_concentration_factor == scf_1 * scf_2
        assert cc_scf_4 == cc_scf_5 != cc_scf_3  # cc_scf_4 has different name

        cc_scf_6 = cc_scf_1 * scf_1
        assert cc_scf_6.stress_concentration_factor == scf_1**2

        assertions_on_scf(cc, [cc_scf_1, cc_scf_2], scf_1)
        assertions_on_scf(cc, [cc_scf_3, cc_scf_4, cc_scf_5], scf_1 * scf_2)
        assertions_on_scf(cc, [cc_scf_6], scf_1**2)

    @pytest.mark.parametrize(
        "cc_1, cc_2, cc_3",
        [(CC_TS_1, CC_TS_2, CC_TS_3)],
    )
    def test_solve_lffd(
        self,
        cc_1: CycleCount,
        cc_2: CycleCount,
        cc_3: CycleCount,
    ) -> None:
        """Test the solve_lffd method.

        Parameters
        ----------
        cc_1 : CycleCount
            The first CycleCount object, coming from single timestamp.
        cc_2 : CycleCount
            The second CycleCount object, coming from single timestamp.
        cc_3 : CycleCount
            The third CycleCount object, coming from concatenation of
            two timestamps.
        """
        cc_add = cc_1 + cc_2
        cc_solv = cc_add.solve_lffd()
        assert cc_solv.lffd_solved
        assert sum(cc_3.count_cycle) == sum(cc_add.count_cycle)
        # assert approx(cc_uni.palmgren_miner(sn_curve=sn))
        # != cc_sum.palmgren_miner(sn_curve=sn)  # Not equal
        assert max(cc_3.stress_range) >= max(cc_add.stress_range)
        assert max(cc_3.stress_range) == max(cc_solv.stress_range)
        assert np.allclose(cc_solv.residuals_sequence, cc_3.residuals_sequence)
        # assert approx(cc_uni.palmgren_miner(sn_curve=sn))
        # ~= cc_sum.palmgren_miner(sn_curve=sn)  # Almost equal

    @pytest.mark.parametrize(
        "cc_1, detail_factor, mean_stress_corr, stress_range_corr",
        [
            (
                CC_TS_1,
                0.8,
                MEAN_STRESS_1_COR_DNV8,
                STRESS_RANGE_1_COR_DNV8,
            ),
            (
                CC_TS_1,
                0.6,
                MEAN_STRESS_1_COR_DNV6,
                STRESS_RANGE_1_COR_DNV6,
            ),
        ],
    )
    def test_dnv_mean_stress_correction(
        self,
        cc_1: CycleCount,
        detail_factor: float,
        mean_stress_corr: MeanStress,
        stress_range_corr: StressRange,
    ) -> None:
        """Test the DNV-GL-RP-C203 mean_stress_correction method.

        Parameters
        ----------
        cc_1 : CycleCount
            The CycleCount object.
        mean_stress : MeanStress
            The mean stress of the CycleCount object.
        mean_stress_corr : MeanStress
            The mean stress of the CycleCount object, corrected.
        stress_range : StressRange
            The stress range of the CycleCount object.
        stress_range_corr : StressRange
            The stress range of the CycleCount object, corrected.
        """
        assert np.allclose(
            stress_range_corr.values / 2, mean_stress_corr.values
        )
        cc_cor = cc_1.mean_stress_correction(detail_factor=detail_factor)

        assert str(detail_factor) in cc_cor.mean_stress_corrected
        assert np.allclose(cc_cor.mean_stress, mean_stress_corr.values)
        assert np.allclose(cc_cor.stress_range, stress_range_corr.values)
        assert "dnv" in cc_cor.mean_stress_corrected.lower()
        assert str(detail_factor) in cc_cor.mean_stress_corrected

    @pytest.mark.parametrize("cc_1", [CC_TS_1, CC_TS_2, CC_TS_3, CC_TS_4])
    @given(detail_factor=hy.floats(min_value=0.81, max_value=0.99))
    def test_dnv_mean_stress_correction_exception(
        self,
        cc_1: CycleCount,
        detail_factor: float,
    ) -> None:
        """Test the DNV-GL-RP-C203 mean_stress_correction method
        ValueError exception when the detail factor is not 0.6 or 0.8.

        Parameters
        ----------
        cc_1 : CycleCount
            The CycleCount object.
        mean_stress : MeanStress
            The mean stress of the CycleCount object.
        mean_stress_corr : MeanStress
            The mean stress of the CycleCount object, corrected.
        stress_range : StressRange
            The stress range of the CycleCount object.
        stress_range_corr : StressRange
            The stress range of the CycleCount object, corrected.
        """
        with pytest.raises(ValueError) as ve:
            _ = cc_1.mean_stress_correction(detail_factor=detail_factor)
            assert "Only 0.6 and 0.8 are permitted." in ve.value.args[0]

    @pytest.mark.parametrize(
        "cc",
        [CC_TS_1, CC_TS_2, CC_TS_3, CC_TS_4],
    )
    @pytest.mark.parametrize(
        "gamma",
        [0.3, 0.4, 0.5, 0.6, 0.7],
    )
    def test_walker_mean_stress_correction(
        self,
        cc: CycleCount,
        gamma: float,
    ) -> None:
        """Test the Walker mean_stress_correction method.

        Parameters
        ----------
        cc : CycleCount
            The CycleCount object.
        gamma: float
            The gamma exponent.
        """
        cc_cor = cc.mean_stress_correction(
            correction_type="walker", gamma=gamma
        )
        range_cor = (cc.mean_stress + cc.stress_amplitude) ** (
            1 - gamma
        ) * cc.stress_amplitude**gamma
        assert np.allclose(np.sort(range_cor), np.sort(cc_cor.stress_range))
        assert "walker" in cc_cor.mean_stress_corrected.lower()
        assert str(gamma) in cc_cor.mean_stress_corrected

    @pytest.mark.parametrize(
        "cc",
        [CC_TS_1, CC_TS_2, CC_TS_3, CC_TS_4],
    )
    def test_swt_mean_stress_correction(
        self,
        cc: CycleCount,
    ) -> None:
        """Test the Smith-Wsatson-Topper mean_stress_correction method.

        Parameters
        ----------
        cc : CycleCount
            The CycleCount object.
        """
        cc_cor = cc.mean_stress_correction(correction_type="swt")
        range_cor = np.sqrt(
            (cc.mean_stress + cc.stress_amplitude) * cc.stress_amplitude
        )
        assert np.allclose(np.sort(range_cor), np.sort(cc_cor.stress_range))
        assert "swt" in cc_cor.mean_stress_corrected.lower()

    # Tests for errors and warnings
    @pytest.mark.parametrize("cc", [(CC_RF_1), (CC_TS_1)])
    @given(
        number=hy.floats(min_value=1),
        list=hy.lists(
            min_size=1, max_size=10, elements=hy.floats(min_value=1)
        ),
    )
    # fmt: on
    def test__add__error_type(self, cc, number, list) -> None:
        """Test __add__ with erroneous types.

        Parameters
        ----------
        cc : CycleCount
            The CycleCount object.
        number : int
            The number to be added to the CycleCount object.
        list : list
            The list to be added to the CycleCount object.
        """
        with pytest.raises(TypeError) as te:
            cc + number
            assert (
                "Trying to add to a non CycleCount object instance"
                in te.value.args[0]
            )
        with pytest.raises(TypeError) as te:
            cc + list
            assert (
                "Trying to add to a non CycleCount object instance"
                in te.value.args[0]
            )

    @pytest.mark.parametrize(
        "cc_1, cc_2",
        [(CC_TS_1, CC_TS_2)],
    )
    def test__add__residuals_sequence_checks(
        self,
        cc_1: CycleCount,
        cc_2: CycleCount,
    ) -> None:
        """Test warnings within the __add__ method when residuals_sequence
        is not defined.
        """
        cc_1_1 = copy.deepcopy(cc_1)
        cc_2_1 = copy.deepcopy(cc_2)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cc_1_1.residuals_sequence = None
            assert cc_1_1.residuals_sequence is None
            cc_1_1 + cc_2
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert "No residuals_sequence found" in str(w[-1].message)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cc_2_1.residuals_sequence = None
            cc_1 + cc_2_1
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert "No residuals_sequence found" in str(w[-1].message)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cc_1_1.residuals_sequence = None
            cc_2_1.residuals_sequence = None
            cc_1_1 + cc_2_1
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert "No residuals_sequence attribute found" in str(
                w[-1].message
            )

    @pytest.mark.parametrize("cc", [(CC_RF_1), (CC_TS_1), (CC_TS_2)])
    # fmt: on
    def test__add__null(self, cc) -> None:
        """Test __add__ zeros, None and or NaN."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cc_0 = cc + 0
            assert cc_0 == cc
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert "non CycleCount object instance" in str(w[-1].message)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cc_none = cc + None
            assert cc_none == cc
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert "non CycleCount object instance" in str(w[-1].message)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cc_nan = cc + np.nan
            assert cc_nan == cc
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert "non CycleCount object instance" in str(w[-1].message)

    @pytest.mark.parametrize(
        "cc_1, cc_2",
        [(CC_TS_1, CC_TS_2)],
    )
    def test__add__timestamp_order(
        self,
        cc_1: CycleCount,
        cc_2: CycleCount,
    ) -> None:
        """Test warnings within the __add__ method when residuals_sequence
        is not defined.
        """
        assert np.max(cc_1.time_sequence) <= np.min(cc_2.time_sequence)
        with pytest.raises(TypeError) as te:
            cc_2 + cc_1
            assert "predates" in te.value.args[0]

        cc_sum = cc_1 + cc_2
        assert len(cc_sum.time_sequence) == 2

    @pytest.mark.parametrize(
        "cc_1, cc_2",
        [(CC_TS_1, CC_TS_2)],
    )
    def test__add__timestamp_order(
        self,
        cc_1: CycleCount,
        cc_2: CycleCount,
    ) -> None:
        """Test that an error is raised when summing timestamps in
        the wrong order.
        """
        with pytest.raises(TypeError) as te:
            cc_2 + cc_1
            assert "predates" in te.value.args[0]
        cc_sum = cc_1 + cc_2
        assert len(cc_sum.time_sequence) == 2

    @pytest.mark.parametrize(
        "cc_1, cc_2",
        [(CC_TS_1, CC_TS_2)],
    )
    def test__add__different_parameters(
        self,
        cc_1: CycleCount,
        cc_2: CycleCount,
    ) -> None:
        """Test warning when summing different parameters."""
        cc_2.name = " ".join([cc_1.name, " modified"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cc_1 + cc_2
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert "Summing different parameters." in str(w[-1].message)

    @pytest.mark.parametrize(
        "cc_1, cc_2",
        [(CC_TS_1, CC_TS_2)],
    )
    @given(
        scf_1=hy.floats(min_value=1, max_value=1e3),
        scf_2=hy.floats(min_value=1, max_value=1e3),
    )
    def test__add__different_scf(
        self,
        cc_1: CycleCount,
        cc_2: CycleCount,
        scf_1: float,
        scf_2: float,
    ) -> None:
        """Test warning when summing different parameters."""
        cc_2.name = " ".join([cc_1.name, " modified"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            cc_1 * scf_1 + cc_2 * scf_2
            if scf_1 != scf_2:
                assert len(w) == 4
                assert issubclass(w[-1].category, UserWarning)
                assert "different SCFs" in str(w[-1].message)


# Add the tests for error messages, specially for the
# - mean_stress_correction,
# - __add__, and
# - solve_lffd methods.
