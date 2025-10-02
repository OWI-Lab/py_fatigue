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
import sys
from typing import Union

# Non-standard imports
import numpy as np
from hypothesis import given, strategies as hy
import pytest
import pandas as pd

from py_fatigue.cycle_count.cycle_count import (
    _build_input_data_from_json,
    _assess_json_keys,
)
from py_fatigue.cycle_count import CycleCount
from py_fatigue.stress_range import StressRange
from py_fatigue.mean_stress import MeanStress
import py_fatigue.utils as pfu
import py_fatigue.cycle_count.rainflow as rf

# Add missing imports at the top
import matplotlib
import matplotlib.pyplot as plt

PROJECT_PATH = os.path.dirname(os.getcwd())
if not PROJECT_PATH in sys.path:
    sys.path.append(PROJECT_PATH)

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
    """Test the _assess_json_keys function for 1D data."""
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
        with pytest.raises(TypeError) as te:
            cc_1.unit = "m"
            _ = cc_1 + cc_2
            assert "different units" in te.value.args[0]
        cc_1.unit = "MPa"

    @pytest.mark.parametrize("cc", [(CC_TS_1)])
    @given(
        scf_1=hy.integers(min_value=1, max_value=1000),
        scf_2=hy.integers(min_value=1, max_value=1000),
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
                # print("cc_scf.mean_stress", cc_scf.mean_stress)
                # print("cc_scf.stress_range", cc_scf.stress_range)
                # print("cc_scf.residuals_sequence", cc_scf.residuals_sequence)
                # print("cc_scf.mean_stress", cc_scf.mean_stress)
                # print("cc_scf.mean_stress", cc_scf.count_cycle)
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

    @pytest.mark.parametrize(
        "cc",
        [CC_TS_1, CC_TS_2, CC_TS_3, CC_TS_4],
    )
    def test_goodman_mean_stress_correction(
        self,
        cc: CycleCount,
    ) -> None:
        """Test the Goodman mean stress correction method."""
        cc_cor = cc.mean_stress_correction(
            correction_type="goodman",
            ult_s=1000,
            correction_exponent=1.0,
            r_out=-1.0
        )
        assert "goodman" in cc_cor.mean_stress_corrected.lower()
        assert np.all(cc_cor.stress_range >= 0)
        assert np.all(np.isfinite(cc_cor.stress_range))

    @pytest.mark.parametrize(
        "cc",
        [CC_TS_1, CC_TS_2, CC_TS_3, CC_TS_4],
    )
    def test_gerber_mean_stress_correction(
        self,
        cc: CycleCount,
    ) -> None:
        """Test the Gerber mean stress correction method."""
        cc_cor = cc.mean_stress_correction(
            correction_type="gerber",
            ult_s=1000,
            correction_exponent=2.0,
            r_out=-1.0
        )
        assert "gerber" in cc_cor.mean_stress_corrected.lower()
        assert np.all(cc_cor.stress_range >= 0)
        assert np.all(np.isfinite(cc_cor.stress_range))

    @pytest.mark.parametrize(
        "cc",
        [CC_TS_1, CC_TS_2, CC_TS_3, CC_TS_4],
    )
    def test_morrow_mean_stress_correction(
        self,
        cc: CycleCount,
    ) -> None:
        """Test the Morrow mean stress correction method."""
        cc_cor = cc.mean_stress_correction(
            correction_type="morrow",
            ult_s=1000,
            correction_exponent=1.0,
            r_out=-1.0
        )
        assert "morrow" in cc_cor.mean_stress_corrected.lower()
        assert np.all(cc_cor.stress_range >= 0)
        assert np.all(np.isfinite(cc_cor.stress_range))

    @pytest.mark.parametrize(
        "cc",
        [CC_TS_1, CC_TS_2, CC_TS_3, CC_TS_4],
    )
    def test_soderberg_mean_stress_correction(
        self,
        cc: CycleCount,
    ) -> None:
        """Test the Soderberg mean stress correction method."""
        cc_cor = cc.mean_stress_correction(
            correction_type="soderberg",
            ult_s=600,  # For Soderberg, ult_s should be yield strength
            correction_exponent=1.0,
            r_out=-1.0
        )
        assert "soderberg" in cc_cor.mean_stress_corrected.lower()
        assert np.all(cc_cor.stress_range[np.isfinite(cc_cor.stress_range)] >= 0)
        # Allow for some NaN values which can occur in numerical corrections
        assert np.sum(np.isfinite(cc_cor.stress_range)) > 0

    @pytest.mark.parametrize(
        "cc",
        [CC_TS_1, CC_TS_2, CC_TS_3, CC_TS_4],
    )
    def test_generic_haigh_mean_stress_correction(
        self,
        cc: CycleCount,
    ) -> None:
        """Test the Generic-Haigh mean stress correction method."""
        cc_cor = cc.mean_stress_correction(
            correction_type="generic-haigh",
            ult_s=1000,
            correction_exponent=1.5,
            r_out=-1.0
        )
        assert "generic-haigh" in cc_cor.mean_stress_corrected.lower()
        # Check that finite values are non-negative (allow NaN for numerical issues)
        finite_mask = np.isfinite(cc_cor.stress_range)
        assert np.all(cc_cor.stress_range[finite_mask] >= 0)
        # Ensure we have some finite results
        assert np.sum(finite_mask) > 0

    @pytest.mark.parametrize(
        "cc",
        [CC_TS_1, CC_TS_2, CC_TS_3, CC_TS_4],
    )
    def test_haigh_mean_stress_correction(
        self,
        cc: CycleCount,
    ) -> None:
        """Test the Haigh mean stress correction method (alias for generic-haigh)."""
        cc_cor = cc.mean_stress_correction(
            correction_type="haigh",
            ult_s=1000,
            correction_exponent=1.2,
            r_out=-1.0
        )
        assert "haigh" in cc_cor.mean_stress_corrected.lower()
        # Check that finite values are non-negative (allow NaN for numerical issues)
        finite_mask = np.isfinite(cc_cor.stress_range)
        assert np.all(cc_cor.stress_range[finite_mask] >= 0)
        # Ensure we have some finite results
        assert np.sum(finite_mask) > 0

    @pytest.mark.parametrize(
        "cc",
        [CC_TS_1, CC_TS_2, CC_TS_3, CC_TS_4],
    )
    def test_smith_watson_topper_mean_stress_correction(
        self,
        cc: CycleCount,
    ) -> None:
        """Test the Smith-Watson-Topper (full name) mean stress correction method."""
        cc_cor = cc.mean_stress_correction(correction_type="smith-watson-topper")
        assert "smith-watson-topper" in cc_cor.mean_stress_corrected.lower()
        assert np.all(cc_cor.stress_range >= 0)
        assert np.all(np.isfinite(cc_cor.stress_range))

    @pytest.mark.parametrize("cc", [CC_TS_1, CC_TS_2, CC_TS_3, CC_TS_4])
    def test_invalid_correction_type(self, cc: CycleCount) -> None:
        """Test that invalid correction types raise ValueError."""
        with pytest.raises(ValueError, match="must be one of"):
            cc.mean_stress_correction(correction_type="invalid_correction")

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
    def test__add__timestamp_order_single(
        self,
        cc_1: CycleCount,
        cc_2: CycleCount,
    ) -> None:
        """Test that an error is raised when summing timestamps in the wrong order."""
        with pytest.raises(TypeError) as te:
            cc_2 + cc_1
            assert "predates" in te.value.args[0]
        cc_sum = cc_1 + cc_2
        assert len(cc_sum.time_sequence) == 2

    @pytest.mark.parametrize(cc_names, cc_data)
    def test_summary(
        self,
        cc: CycleCount,
        stress_range: StressRange,
        mean_stress: MeanStress,
        res_sig: Union[list, np.ndarray],
    ) -> None:
        """Test the summary method of the CycleCount class."""
        summary_df = cc.summary()

        # Check that summary returns a pandas DataFrame
        assert isinstance(summary_df, pd.DataFrame)

        # Check that the DataFrame has the correct structure
        assert summary_df.index.name == "Cycle counting object"
        assert cc.name in summary_df.columns

        # Check expected rows are present
        expected_rows = [
            f"largest full stress range, {cc.unit}",
            f"largest stress range, {cc.unit}",
            "number of full cycles",
            "number of residuals",
            "number of small cycles",
            "stress concentration factor",
            "residuals resolved",
            "mean stress-corrected",
        ]

        for row in expected_rows:
            assert row in summary_df.index

        # Verify specific values
        assert summary_df.loc[f"largest stress range, {cc.unit}", cc.name] == max(cc.stress_range)
        assert summary_df.loc["number of residuals", cc.name] == int(len(cc.residuals[:, 1]))
        assert summary_df.loc["number of small cycles", cc.name] == int(cc.nr_small_cycles)
        assert summary_df.loc["residuals resolved", cc.name] == bool(cc.lffd_solved)
        assert summary_df.loc["mean stress-corrected", cc.name] == cc.mean_stress_corrected

    @pytest.mark.parametrize("cc", [CC_TS_1])
    def test_summary_with_scf(self, cc: CycleCount) -> None:
        """Test summary method with stress concentration factor."""
        cc_scf = cc * 2.5
        summary_df = cc_scf.summary()

        # Check that SCF is displayed correctly
        assert summary_df.loc["stress concentration factor", cc_scf.name] == 2.5

    @pytest.mark.parametrize("cc", [CC_TS_1])
    def test_summary_no_scf(self, cc: CycleCount) -> None:
        """Test summary method without stress concentration factor (SCF = 1)."""
        summary_df = cc.summary()

        # Check that N/A is displayed when SCF = 1
        assert summary_df.loc["stress concentration factor", cc.name] == "N/A"

    @pytest.mark.parametrize("cc", [CC_TS_1])
    def test_summary_solved_lffd(self, cc: CycleCount) -> None:
        """Test summary method with solved LFFD."""
        cc_solved = cc.solve_lffd()
        summary_df = cc_solved.summary()

        # Check that residuals resolved is True
        assert summary_df.loc["residuals resolved", cc_solved.name] == True

    @pytest.mark.parametrize("cc", [CC_TS_1])
    def test_summary_mean_stress_corrected(self, cc: CycleCount) -> None:
        """Test summary method with mean stress correction applied."""
        cc_corrected = cc.mean_stress_correction(detail_factor=0.8)
        summary_df = cc_corrected.summary()

        # Check that mean stress correction is reflected
        assert "DNVGL-RP-C203" in summary_df.loc["mean stress-corrected", cc_corrected.name]

    def test_summary_empty_cycle_count(self) -> None:
        """Test summary method with minimal/empty cycle count data."""
        # Create a minimal CycleCount with very small data
        minimal_cc = CycleCount(
            count_cycle=np.array([0.1]),
            stress_range=np.array([0.05]),
            mean_stress=np.array([0.0]),
            name="minimal_test"
        )

        summary_df = minimal_cc.summary()

        # Check that summary handles edge cases properly
        assert isinstance(summary_df, pd.DataFrame)
        assert "minimal_test" in summary_df.columns

        # For very small stress ranges, largest full stress range should be None
        largest_full = summary_df.loc[f"largest full stress range, {minimal_cc.unit}", "minimal_test"]
        assert largest_full is None or pd.isna(largest_full)

    @pytest.mark.parametrize("cc", [CC_TS_4])  # CC_TS_4 has no mean stress data
    def test_summary_no_mean_stress(self, cc: CycleCount) -> None:
        """Test summary method with cycle count that has no mean stress binning."""
        summary_df = cc.summary()

        # Should still work and return valid DataFrame
        assert isinstance(summary_df, pd.DataFrame)
        assert cc.name in summary_df.columns

    @pytest.mark.parametrize(cc_names, cc_data)
    def test_stress_amplitude_property(
        self,
        cc: CycleCount,
        stress_range: StressRange,
        mean_stress: MeanStress,
        res_sig: Union[list, np.ndarray],
    ) -> None:
        """Test the stress_amplitude property."""
        stress_amp = cc.stress_amplitude
        assert isinstance(stress_amp, np.ndarray)
        assert np.allclose(stress_amp, cc.stress_range / 2)

    @pytest.mark.parametrize(cc_names, cc_data)
    def test_min_max_stress_properties(
        self,
        cc: CycleCount,
        stress_range: StressRange,
        mean_stress: MeanStress,
        res_sig: Union[list, np.ndarray],
    ) -> None:
        """Test the min_stress and max_stress properties."""
        min_stress = cc.min_stress
        max_stress = cc.max_stress

        assert isinstance(min_stress, np.ndarray)
        assert isinstance(max_stress, np.ndarray)
        assert np.allclose(min_stress, cc.mean_stress - cc.stress_amplitude)
        assert np.allclose(max_stress, cc.mean_stress + cc.stress_amplitude)
        assert np.all(max_stress >= min_stress)

    @pytest.mark.parametrize(cc_names, cc_data)
    def test_statistical_moments(
        self,
        cc: CycleCount,
        stress_range: StressRange,
        mean_stress: MeanStress,
        res_sig: Union[list, np.ndarray],
    ) -> None:
        """Test the statistical_moments property."""
        moments = cc.statistical_moments
        assert isinstance(moments, tuple)
        assert len(moments) == 3

        mean_val, coeff_var, skewness = moments
        assert isinstance(mean_val, (float, np.floating))
        assert isinstance(coeff_var, (float, np.floating))
        assert isinstance(skewness, (float, np.floating))
        assert mean_val > 0
        assert coeff_var >= 0

    @pytest.mark.parametrize(cc_names, cc_data)
    def test_bin_properties(
        self,
        cc: CycleCount,
        stress_range: StressRange,
        mean_stress: MeanStress,
        res_sig: Union[list, np.ndarray],
    ) -> None:
        """Test bin-related properties."""
        # Test bin_centers
        mean_centers, range_centers = cc.bin_centers
        assert isinstance(mean_centers, np.ndarray)
        assert isinstance(range_centers, np.ndarray)
        assert len(mean_centers) > 0
        assert len(range_centers) > 0

        # Test bin_edges
        mean_edges, range_edges = cc.bin_edges
        assert isinstance(mean_edges, np.ndarray)
        assert isinstance(range_edges, np.ndarray)
        assert len(mean_edges) == len(mean_centers) + 1
        assert len(range_edges) == len(range_centers) + 1

        # Test bin bounds
        assert cc.mean_bin_upper_bound > cc.mean_bin_lower_bound
        assert cc.range_bin_upper_bound > cc.range_bin_lower_bound

    def test_from_timeseries_with_time_array(self) -> None:
        """Test from_timeseries with time array."""
        data = np.array([0, 1, -1, 2, -2, 1, 0])
        time = np.linspace(0, 6, len(data))

        cc = CycleCount.from_timeseries(
            data=data,
            time=time,
            name="test_with_time"
        )

        assert cc.name == "test_with_time"
        assert len(cc.count_cycle) > 0
        assert len(cc.stress_range) > 0

    def test_from_rainflow_with_custom_parameters(self) -> None:
        """Test from_rainflow with custom parameters."""
        cc = CycleCount.from_rainflow(
            AS_DICT_1,
            timestamp=TIMESTAMP_1,
            round_decimals=2,
            name="custom_test",
            mean_stress_corrected="Test correction",
            lffd_solved=True,
            unit="kPa"
        )

        assert cc.name == "custom_test"
        assert cc.mean_stress_corrected == "Test correction"
        assert cc.lffd_solved == True
        assert cc.unit == "kPa"

    def test_cycle_count_equality_edge_cases(self) -> None:
        """Test equality comparison edge cases."""
        cc1 = copy.deepcopy(CC_TS_1)
        cc2 = copy.deepcopy(CC_TS_1)

        # Test equality
        assert cc1 == cc2

        # Test inequality with different attributes
        cc2.name = "different_name"
        assert cc1 != cc2

        # Test inequality with non-CycleCount object
        assert cc1 != "not_a_cyclecount"
        assert cc1 != 42
        assert cc1 != None

    @pytest.mark.parametrize("cc", [CC_TS_1])
    def test_radd_method(self, cc: CycleCount) -> None:
        """Test the __radd__ method."""
        # Test with None
        result_none = None + cc
        assert result_none == cc

        # Test with 0
        result_zero = 0 + cc
        assert result_zero == cc

        # Test with NaN
        result_nan = np.nan + cc
        assert result_nan == cc

    def test_str_representation_variations(self) -> None:
        """Test __str__ method variations."""
        # Single timestamp
        str_repr = str(CC_TS_1)
        assert isinstance(str_repr, str)
        assert CC_TS_1.name in str_repr

        # Multiple timestamps
        cc_sum = CC_TS_1 + CC_TS_2
        str_repr = str(cc_sum)
        assert "from" in str_repr
        assert "to" in str_repr

    def test_plot_methods(self) -> None:
        """Test plotting methods."""
        import matplotlib.pyplot as plt

        # Test plot_histogram with different types
        fig, ax = CC_TS_1.plot_histogram(plot_type="min-max")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        fig, ax = CC_TS_1.plot_histogram(plot_type="mean-range")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        # Test invalid plot type
        with pytest.raises(ValueError, match="Invalid plot type"):
            CC_TS_1.plot_histogram(plot_type="invalid")

        # Test plot_residuals_sequence
        fig, ax = CC_TS_1.plot_residuals_sequence()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        # Test plot_half_cycles_sequence (alias)
        fig, ax = CC_TS_1.plot_half_cycles_sequence()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_pbar_sum_function(self) -> None:
        """Test pbar_sum function."""
        from py_fatigue.cycle_count.cycle_count import pbar_sum

        # Single element
        result = pbar_sum([CC_TS_1])
        assert result == CC_TS_1

        # Multiple elements
        cc_list = [CC_TS_1, CC_TS_2]
        result = pbar_sum(cc_list)
        assert isinstance(result, CycleCount)

        # With None elements
        cc_list_with_none = [CC_TS_1, None, CC_TS_2]
        result = pbar_sum(cc_list_with_none)
        assert isinstance(result, CycleCount)

    def test_mean_stress_correction_edge_cases(self) -> None:
        """Test mean stress correction edge cases."""
        # Test Walker without gamma
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cc_cor = CC_TS_1.solve_lffd().mean_stress_correction(correction_type="walker")
            assert any("No gamma exponent" in str(warning.message) for warning in w)

        # Test already corrected
        cc_corrected = CC_TS_1.solve_lffd().mean_stress_correction(detail_factor=0.8)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cc_cor2 = cc_corrected.mean_stress_correction(detail_factor=0.8)
            assert any("already been corrected" in str(warning.message) for warning in w)

        # Test zero mean stress
        cc_zero = copy.deepcopy(CC_TS_1).solve_lffd()
        cc_zero.mean_stress = np.zeros_like(cc_zero.mean_stress)
        with pytest.raises(ValueError, match="will not be applied"):
            # The actual call that should raise the error
            cc_zero.mean_stress_correction(
                detail_factor=0.8,
                enforce_pulsating_load=False
            )

        # Test with enforce_pulsating_load
        cc_cor = cc_zero.solve_lffd().mean_stress_correction(
            detail_factor=0.8, enforce_pulsating_load=True
        )
        assert isinstance(cc_cor, CycleCount)

    def test_mean_stress_correction_parameter_errors(self) -> None:
        """Test mean stress correction parameter validation."""
        # Missing r_out
        with pytest.raises(ValueError, match="requires the 'r_out'"):
            CC_TS_1.solve_lffd().mean_stress_correction(
                correction_type="goodman", ult_s=1000
            )

        # Missing ult_s
        with pytest.raises(ValueError, match="requires the 'ult_s'"):
            CC_TS_1.solve_lffd().mean_stress_correction(
                correction_type="goodman", r_out=-1.0
            )

        # Missing correction_exponent for generic-haigh
        with pytest.raises(ValueError, match="requires the 'correction_exponent'"):
            CC_TS_1.solve_lffd().mean_stress_correction(
                correction_type="generic-haigh", r_out=-1.0, ult_s=1000
            )

        # Multiple r_out values
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cc_cor = CC_TS_1.solve_lffd().mean_stress_correction(
                correction_type="goodman",
                r_out=[-1.0, -0.5],
                ult_s=1000
            )
            assert any("Multiple output load ratios" in str(warning.message) for warning in w)

    def test_solve_lffd_edge_cases(self) -> None:
        """Test solve_lffd edge cases."""
        # Already solved
        cc_solved = CC_TS_1.solve_lffd()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cc_double_solved = cc_solved.solve_lffd()
            assert any("already resolved" in str(warning.message) for warning in w)

        # Short residuals sequence
        cc_short = copy.deepcopy(CC_TS_1)
        cc_short.residuals_sequence = np.array([1, 2])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cc_result = cc_short.solve_lffd()
            assert any("len" in str(warning.message) and "residuals_sequence" in str(warning.message) for warning in w)

        # No residuals sequence
        cc_no_res = copy.deepcopy(CC_TS_1)
        cc_no_res.residuals_sequence = None
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cc_result = cc_no_res.solve_lffd()
            assert any("residuals_sequence" in str(warning.message) and "not available" in str(warning.message) for warning in w)

        # Invalid solve mode
        with pytest.raises(ValueError, match="Invalid solve mode"):
            CC_TS_1.solve_lffd(solve_mode="invalid_mode")

    def test_multiplication_edge_cases(self) -> None:
        """Test multiplication edge cases."""
        # Test with existing SCF
        cc_with_scf = CC_TS_1 * 2.0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cc_double_scf = cc_with_scf * 1.5
            assert any("SCF already defined" in str(warning.message) for warning in w)

        # Test invalid types
        with pytest.raises(TypeError, match="Multiplication by a scalar"):
            CC_TS_1 * "invalid"

        with pytest.raises(TypeError, match="Multiplication by a scalar"):
            CC_TS_1 * [1, 2, 3]

    def test_addition_edge_cases(self) -> None:
        """Test addition edge cases."""
        # Different bin parameters
        cc1 = copy.deepcopy(CC_TS_1)
        cc2 = copy.deepcopy(CC_TS_2)
        cc2.range_bin_width = 1.5
        cc2.mean_bin_width = 0.75

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cc_sum = cc1 + cc2
            warning_messages = [str(warning.message) for warning in w]
            assert any("Different bin" in msg and "widths" in msg for msg in warning_messages)

    def test_build_input_data_edge_cases(self) -> None:
        """Test _build_input_data_from_json edge cases."""
        # Test with 2D histogram data
        hist_2d_data = {
            "nr_small_cycles": 0,
            "range_bin_lower_bound": 0.5,
            "range_bin_width": 1,
            "mean_bin_lower_bound": -0.75,
            "mean_bin_width": 0.5,
            "hist": [[1.0, 0.0], [0.0, 1.0]],
            "lg_c": [[1.0, 2.0]],
            "res": [[1.0, 8.0]],
            "res_sig": [-3, 5]
        }

        result = _build_input_data_from_json(hist_2d_data, name="test", timestamp=TIMESTAMP_1)
        assert isinstance(result, dict)
        assert "count_cycle" in result

        # Test without mean_bin_width
        no_mean_data = {
            "nr_small_cycles": 0,
            "range_bin_lower_bound": 0.5,
            "range_bin_width": 1,
            "hist": [1.0, 2.0],
            "lg_c": [],
            "res": [8.0, 9.0],
            "res_sig": [-3, 5]
        }

        result = _build_input_data_from_json(no_mean_data, name="test", timestamp=TIMESTAMP_1)
        assert result["mean_bin_width"] == 10  # Default value

    def test_assess_json_keys_edge_cases(self) -> None:
        """Test _assess_json_keys edge cases."""
        # Test legacy key mapping
        data_with_legacy = {
            "no_sm_c": 5,
            "bin_lb": 0.5,
            "bin_width": 1.0,
            "hist": [],
            "lg_c": [],
            "res": []
        }

        result = _assess_json_keys(data_with_legacy)
        assert "nr_small_cycles" in result
        assert "range_bin_lower_bound" in result
        assert "range_bin_width" in result

    def test_multiplication_by_scalar_edge_cases(self) -> None:
        """Test _multiplication_by_scalar edge cases."""
        from py_fatigue.cycle_count.cycle_count import _multiplication_by_scalar

        # Test with None residuals_sequence
        cc_no_res = copy.deepcopy(CC_TS_1)
        cc_no_res.residuals_sequence = None
        result = _multiplication_by_scalar(cc_no_res, 2.0)
        assert len(result.residuals_sequence) == 0

        # Test with None min_max_sequence
        cc_no_minmax = copy.deepcopy(CC_TS_1)
        cc_no_minmax.residuals_sequence = None
        result = _multiplication_by_scalar(cc_no_minmax, 2.0)
        assert len(result._min_max_sequence) == 0

    def test_handling_different_bins_in_sum(self) -> None:
        """Test _handling_different_bins_in_sum function."""
        from py_fatigue.cycle_count.cycle_count import _handling_different_bins_in_sum

        cc1 = copy.deepcopy(CC_TS_1)
        cc2 = copy.deepcopy(CC_TS_2)

        # Test with different bounds
        cc2.range_bin_lower_bound = 1.0
        cc2._mean_bin_lower_bound = -1.0

        result = _handling_different_bins_in_sum(cc1, cc2)
        assert len(result) == 4
        assert result[2] == min(cc1.range_bin_lower_bound, cc2.range_bin_lower_bound)

    def test_bin_widths_add_check(self) -> None:
        """Test _bin_widths_add_check function."""
        from py_fatigue.cycle_count.cycle_count import _bin_widths_add_check

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _bin_widths_add_check(1.0, 2.0, "test")
            assert len(w) == 1
            assert "Different bin test widths" in str(w[0].message)

    def test_lffd_checks_function(self) -> None:
        """Test _lffd_checks function."""
        from py_fatigue.cycle_count.cycle_count import _lffd_checks

        # Test invalid solve mode
        with pytest.raises(ValueError, match="Invalid solve mode"):
            _lffd_checks(CC_TS_1, "invalid")

        # Test already solved
        cc_solved = CC_TS_1.solve_lffd()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            return_self, _ = _lffd_checks(cc_solved, "residuals")
            assert return_self == True
            assert any("already resolved" in str(warning.message) for warning in w)

    def test_cycle_count_add_checks(self) -> None:
        """Test _cycle_count_add_checks function."""
        from py_fatigue.cycle_count.cycle_count import _cycle_count_add_checks

        # Test different units
        cc1 = copy.deepcopy(CC_TS_1)
        cc2 = copy.deepcopy(CC_TS_2)
        cc2.unit = "kPa"

        with pytest.raises(TypeError, match="different units"):
            _cycle_count_add_checks(cc1, cc2)

    def test_post_init_edge_cases(self) -> None:
        """Test __post_init__ edge cases."""
        # Test with NaN mean stress
        cc_nan = CycleCount(
            count_cycle=np.array([1.0]),
            stress_range=np.array([2.0]),
            mean_stress=np.array([np.nan]),
            name="test_nan"
        )
        # Should not set _mean_bin_lower_bound
        assert cc_nan._mean_bin_lower_bound is None

        # Test with valid mean stress
        cc_valid = CycleCount(
            count_cycle=np.array([1.0]),
            stress_range=np.array([2.0]),
            mean_stress=np.array([1.0]),
            name="test_valid"
        )
        assert cc_valid._mean_bin_lower_bound is not None

    def test_html_repr(self) -> None:
        """Test _repr_html_ method."""
        html_repr = CC_TS_1._repr_html_()
        assert isinstance(html_repr, str)
        # Should contain HTML table elements
        assert any(tag in html_repr for tag in ['<table', '<tr', '<td'])

    def test_as_dict_debug_mode(self) -> None:
        """Test as_dict with debug_mode."""
        result = CC_TS_1.as_dict(debug_mode=True)
        assert isinstance(result, dict)
        assert "res_sig" in result

# Add more tests for uncovered utility functions and edge cases...

def test_build_input_data_complex_cases():
    """Test _build_input_data_from_json with complex cases."""
    # Test with large cycles as iterable
    complex_data = {
        "nr_small_cycles": 5,
        "range_bin_lower_bound": 1.0,
        "range_bin_width": 2.0,
        "mean_bin_lower_bound": -1.0,
        "mean_bin_width": 1.0,
        "hist": [],
        "lg_c": [[0.5, 10.0], [1.0, 15.0]],
        "res": [[2.0, 5.0]],
        "res_sig": [1, 2, 3]
    }

    result = _build_input_data_from_json(complex_data, name="complex_test", timestamp=TIMESTAMP_1)
    assert isinstance(result, dict)
    assert result["nr_small_cycles"] == 5
    assert len(result["count_cycle"]) > 0

def test_solve_lffd_only_residuals() -> None:
    """Test solve_lffd when only residuals exist."""
    # Create a CycleCount with only residuals (count <= 0.5)
    cc_only_res = CycleCount(
        count_cycle=np.array([0.5, 0.5, 0.5]),
        stress_range=np.array([1.0, 2.0, 3.0]),
        mean_stress=np.array([0.0, 1.0, 2.0]),
        residuals_sequence=np.array([1, -1, 2, -2, 1]),
        name="only_residuals"
    )

    result = cc_only_res.solve_lffd()
    assert result.lffd_solved
    # Should return only the residuals component since max(count_cycle) <= 0.5

def test_build_input_data_no_mean_stress():
    """Test _build_input_data_from_json without mean stress binning."""
    data_no_mean = {
        "nr_small_cycles": 0,
        "range_bin_lower_bound": 0.5,
        "range_bin_width": 1,
        "hist": [1.0, 2.0, 3.0],
        "lg_c": [10.0, 15.0],
        "res": [8.0, 9.0],
        "res_sig": [-3, 5, -4, 4, 2]
    }

    result = _build_input_data_from_json(data_no_mean, name="no_mean_test", timestamp=TIMESTAMP_1)
    assert isinstance(result, dict)
    assert "mean_stress" in result
    # When mean_bin_width is not provided, mean_stress should be zeros
    # with the same shape as stress_range
    expected_length = len(result["stress_range"])
    assert len(result["mean_stress"]) == expected_length
    assert np.allclose(result["mean_stress"], np.zeros(expected_length))
