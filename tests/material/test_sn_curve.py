# from hypothesis import  given, strategies as hy
import os
import warnings

import numba as nb
import numpy as np
import pytest

# os.environ["NUMBA_DISABLE_JIT"] = "1"

nb.config.DISABLE_JIT = True

from py_fatigue import SNCurve

# Just to stress out the SN curve class...
# exotic SN curve with 2 knees and endurance values.
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

def test_sncurve_private_attributes():
    """Test the calculation of the stress ranges and number of cycles
    to assure the answer is the same
    """
    # assert type(DNV_W3C.slope) == 'numpy.ndarray'
    with pytest.raises(AttributeError):
        DNV_B1W.linear = False
    with pytest.raises(AttributeError):
        DNV_B1W.slope = -3
    with pytest.raises(AttributeError):
        DNV_B1W.intercept = 45
    with pytest.raises(AttributeError):
        DNV_B1W.endurance = 909
    with pytest.raises(AttributeError):
        DNV_B1W.unit = "pounds"


def test_sncurve_slope_intercept_types():
    """Test the calculation of the stress ranges and number of cycles
    to assure the answer is the same
    """
    # assert type(DNV_W3C.slope) == 'numpy.ndarray'
    assert isinstance(DNV_B1C.slope, np.ndarray)
    assert isinstance(DNV_B1C.intercept, np.ndarray)
    assert isinstance(DNV_C_C.slope, np.ndarray)
    assert isinstance(DNV_C_C.intercept, np.ndarray)
    assert isinstance(DNV_E_C.slope, np.ndarray)
    assert isinstance(DNV_E_C.intercept, np.ndarray)
    assert isinstance(DNV_W3C.slope, np.ndarray)
    assert isinstance(DNV_W3C.intercept, np.ndarray)


def test_linear_sncurve_back_and_forth():
    """Test the calculation of the stress ranges and number of cycles
    to assure the answer is the same
    """
    stress_test = np.logspace(1.7, 3, 20)
    cycles_test = DNV_B1C.get_cycles(stress_test)
    stress_out = DNV_B1C.get_stress(cycles_test)
    np.testing.assert_array_almost_equal(stress_out, stress_test, decimal=4)

    stress_test_var = 180
    cycles_test_var = DNV_B1C.get_cycles(stress_test_var)
    stress_out_var = DNV_B1C.get_stress(cycles_test_var)
    np.testing.assert_almost_equal(stress_out_var, stress_test_var)


def test_bilinear_sncurve_back_and_forth():
    """Test the calculation of the stress ranges and number of cycles
    to assure the answer is the same
    """
    stress_test = np.logspace(1.7, 3, 20)
    cycles_test = DNV_B1A.get_cycles(stress_test)
    stress_out = DNV_B1A.get_stress(cycles_test)
    np.testing.assert_array_almost_equal(stress_out, stress_test, decimal=4)

    stress_test_var = 180
    cycles_test_var = DNV_B1A.get_cycles(stress_test_var)
    stress_out_var = DNV_B1A.get_stress(cycles_test_var)
    np.testing.assert_almost_equal(stress_out_var, stress_test_var)


def test_trilinear_sncurve_back_and_forth():
    """Test the calculation of the stress ranges and number of cycles
    to assure the answer is the same
    """
    stress_test = np.maximum(np.logspace(0.7, 3, 20), 42.5794 * np.ones(20))
    cycles_test = DNV_B1A_END.get_cycles(stress_test)
    stress_out = DNV_B1A_END.get_stress(cycles_test)
    np.testing.assert_array_almost_equal(stress_out, stress_test, decimal=4)
    with pytest.raises(AssertionError):
        _ = DNV_B1A_END.get_cycles(-1)  # negative stresses not allowed
    with pytest.raises(AssertionError):
        _ = DNV_B1A_END.get_stress(-1)  # negative cycles not allowed


def test_backwards_compatibility_n_sigma():
    """Test the calculation of the stress ranges and number of cycles
    to assure the answer is the same
    """
    stress_test = np.maximum(np.logspace(0.7, 3, 20), 42.5794 * np.ones(20))
    cycles_test = DNV_B1A_END.get_cycles(stress_test)
    stress_out = DNV_B1A_END.get_stress(cycles_test)
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        cycles_test_bc = DNV_B1A_END.n(stress_test)
        stress_out_bc = DNV_B1A_END.sigma(cycles_test_bc)
        # Verify some things
        assert len(w) == 2  # 2 warnings
        assert issubclass(w[0].category, FutureWarning)
        assert issubclass(w[1].category, FutureWarning)
        assert "will be removed in a future release" in str(w[0].message)
        assert "will be removed in a future release" in str(w[1].message)
        np.testing.assert_array_almost_equal(
            cycles_test, cycles_test_bc, decimal=4
        )
        np.testing.assert_array_almost_equal(
            stress_out, stress_out_bc, decimal=4
        )


def test_sn_curve_equality():
    """Test multiple SN curves with same data are actually the same (uses the
    `py_fatigue.sn_curve.SNCurve.__eq__()` method)
    """
    from random import Random, seed
    
    seed(2)
    my_random = Random(2)

    sn_1 = SNCurve(slope=[3, 5], intercept=[12.592, 16.320], color="k")
    sn_2 = SNCurve(slope=[3, 5], intercept=[12.592, 16.320], color="k")
    sn_3 = SNCurve(slope=[3], intercept=[12.592], color="k")
    sn_4 = SNCurve(slope=3, intercept=12.592, color="k")
    sn_5 = SNCurve([3, 5, 7], [10.970, 13.617, 16], 2.134e11, "Air", color="r")
    sn_6 = SNCurve([3, 5, 7], [10.970, 13.617, 16], 2.134e11, "Air", color="r")

    assert sn_1 == sn_2  # same input types, same input var => same outputs
    assert sn_3 == sn_4  # same input t. (long), same input var => same outputs
    assert sn_5 == sn_6  # diff input types, same input var => same outputs
    assert sn_1 != sn_3  # same class types, diff input vars => diff outputs
    assert sn_1 != sn_5  # ""
    assert sn_3 != sn_5  # ""
    assert sn_1 != my_random  # diff class tipes => diff outputs
    del sn_1.__dict__["id"]
    # deleting sn_1.id attr to break equality
    assert sn_1 != sn_2  # same input types, diff namespace => same outputs


def test_sn_curve_different_attribute_types():
    """Test that SN curves throw Params lengths must match ValueError if
    slope and intercept have different lengths.
    """
    with pytest.raises(ValueError) as ve:
        _ = SNCurve(slope=[3, 5], intercept=[12.592], endurance=1e9)
    assert "Params lengths must match" in ve.value.args[0]
    with pytest.raises(ValueError) as ve:
        _ = SNCurve(slope=[3, 5], intercept=12.592, endurance=1e9)
    assert "Params lengths must match" in ve.value.args[0]
    with pytest.raises(ValueError) as ve:
        _ = SNCurve([3, 7], [10.970, 13.617, 16], 2.134e11, "Air", color="r")
    assert "Params lengths must match" in ve.value.args[0]
    with pytest.raises(ValueError) as ve:
        _ = SNCurve(3, [10.970, 13.617, 16], 2.134e11, "Air", color="r")
    assert "Params lengths must match" in ve.value.args[0]


def test_knee_point_calculation():
    """Test that SN curves automatically calculate the correct knee point
    based on the length of slopes-intercept instance attributes.
    """
    assert DNV_B1C.linear  # linear SN curve
    assert len(DNV_B1C.get_knee_stress()) == 0  # no knee stress
    assert len(DNV_B1C.get_knee_cycles()) == 0  # and cycles
    assert not DNV_B1A.linear  # bilinear SN curve
    # knee assertions on bilinear SN curve
    np.testing.assert_approx_equal(
        DNV_B1A.get_knee_cycles(), 1e7, significant=2
    )
    np.testing.assert_approx_equal(
        DNV_B1A.get_knee_stress(), 106.91, significant=2
    )
    np.testing.assert_approx_equal(
        DNV_B1A.get_knee_stress(),
        DNV_B1A.get_stress(DNV_B1A.get_knee_cycles()),
        significant=2,
    )
    np.testing.assert_approx_equal(
        DNV_B1A.get_knee_stress(), DNV_B1A.get_stress(1e7), significant=2
    )
    np.testing.assert_approx_equal(
        DNV_B1A.get_knee_cycles(), DNV_B1A.get_cycles(106.91), significant=2
    )
    # knee assertions on trilinear SN curve + endurance
    assert not EXOTIC.linear
    [
        np.testing.assert_approx_equal(c_k, k, significant=2)
        for c_k, k in zip([1e7, 45656000], EXOTIC.get_knee_cycles())
    ]
    [
        np.testing.assert_approx_equal(c_k, k, significant=2)
        for c_k, k in zip([21.06, 15.54], EXOTIC.get_knee_stress())
    ]
    [
        np.testing.assert_approx_equal(c_k, k, significant=2)
        for c_k, k in zip(
            [EXOTIC.get_stress(1e7), EXOTIC.get_stress(45656000)],
            EXOTIC.get_knee_stress(),
        )
    ]
    [
        np.testing.assert_approx_equal(c_k, k, significant=2)
        for c_k, k in zip(
            [EXOTIC.get_cycles(21.06), EXOTIC.get_cycles(15.54)],
            EXOTIC.get_knee_cycles(),
        )
    ]


def test_knee_point_check():
    """Test that when SN curves automatically calculate the correct knee point,
    providing the check_knee attribute actually performs a check on the knee
    point value.
    """
    # size mismatch assertions
    with pytest.raises(ValueError) as ve:
        _ = DNV_B1C.get_knee_cycles(check_knee=[1e7])
    assert "0 knee points expected" in ve.value.args[0]
    with pytest.raises(ValueError) as ve:
        _ = DNV_B1C.get_knee_stress(check_knee=[1e7])
    assert "0 knee points expected" in ve.value.args[0]
    with pytest.raises(ValueError) as ve:
        _ = DNV_B1A.get_knee_cycles(check_knee=[1e7, 1e8])
    assert "1 knee points expected" in ve.value.args[0]
    with pytest.raises(ValueError) as ve:
        _ = DNV_B1A.get_knee_stress(check_knee=[1e7, 1e8])
    assert "1 knee points expected" in ve.value.args[0]
    with pytest.raises(ValueError) as ve:
        _ = EXOTIC.get_knee_cycles(check_knee=[1e7, 1e8, 1e9])
    assert "2 knee points expected" in ve.value.args[0]
    with pytest.raises(ValueError) as ve:
        _ = EXOTIC.get_knee_stress(check_knee=[1e7, 1e8, 1e9])
    assert "2 knee points expected" in ve.value.args[0]
    # check_knee assertions
    DNV_B1A.get_knee_cycles(check_knee=1e7)
    DNV_B1A.get_knee_stress(check_knee=1e7)
    DNV_B1A.get_knee_cycles(check_knee=[1e7])
    DNV_B1A.get_knee_stress(check_knee=[1e7])
    assert DNV_B1A.get_knee_cycles(check_knee=1e7) == DNV_B1A.get_knee_cycles(
        check_knee=[1e7]
    )
    assert DNV_B1A.get_knee_stress(check_knee=1e7) == DNV_B1A.get_knee_stress(
        check_knee=[1e7]
    )
    EXOTIC.get_knee_cycles(check_knee=[1e7, 45656000])
    EXOTIC.get_knee_stress(check_knee=[1e7, 45656000])


def test_endurance_behavior():
    """Test that when SN curves is given an endurance, all the SN values
    meeting the below endurance requirement actually return:
    - cycles = Inf
    - stress = endurance stress
    """
    # endurance assertions on trilinear SN curve + endurance
    for cyc in [2.134e11, 1e12, 2e12, 1e13, 2e13]:
        assert EXOTIC.get_stress(2.134e11) == EXOTIC.get_stress(cyc)
        np.testing.assert_approx_equal(
            EXOTIC.get_stress(cyc), 4.64, significant=2
        )  # above endurance cycles always same stress is returned
    assert (
        EXOTIC.get_cycles(4.64) == np.inf
    )  # just below the endurance returns
    # Inf cycles
    assert EXOTIC.get_cycles(1) == np.inf  # well below the endurance returns
    # Inf cycles
    with pytest.raises(AssertionError):
        _ = EXOTIC.get_cycles(-1)  # too much below the endurance
    with pytest.raises(AssertionError):
        _ = EXOTIC.get_stress(-1)  # too much below the endurance
    with pytest.raises(ValueError) as ve:
        _ = SNCurve([3, 5, 8], [10.970, 13.617, 14.3], endurance=1e9)
    assert "Endurance" in ve.value.args[0]


def test_plotly():
    """Asserting that plotly behaves as expected."""
    # endurance assertions on trilinear SN curve + endurance
    data, layout = EXOTIC.plotly(cycles=200, stress_range=60)
    assert EXOTIC.endurance in data[0].x
    assert np.round(EXOTIC.get_stress(EXOTIC.endurance), 4) == np.round(
        data[0].y[-1], 4
    )
    assert np.round(EXOTIC.get_stress(np.inf), 4) == np.round(data[0].y[-1], 4)
    assert np.any(np.in1d(EXOTIC.get_knee_cycles(), data[0].x))
    assert np.any(
        np.in1d(np.round(EXOTIC.get_knee_stress(), 4), np.round(data[0].y, 4))
    )
    assert 200 in data[1].x
    assert 60 in data[1].y
    assert layout.xaxis.title.text == "Number of cycles"
    assert EXOTIC.unit in layout.yaxis.title.text


def test_format_name():
    """Asserting that curve naming behaves as expected."""
    sn = SNCurve(
        3,
        12.115,
        norm="DNVGL-RP-C203",
        environment="Free corrosion",
        curve="C",
    )
    assert "\n" in sn.name
    sn.format_name(html_format=True)
    assert "<br>" in sn.name
    sn.format_name()
    assert "\n" in sn.name


def test_plot():
    """Asserting that plotly behaves as expected."""
    # endurance assertions on trilinear SN curve + endurance
    _, ax = EXOTIC.plot(cycles=200, stress_range=60)
    sn_curve_cycles = ax.lines[0].get_xdata()
    sn_curve_stress = ax.lines[0].get_ydata()
    other_cycles = ax.lines[1].get_xdata()
    other_stress = ax.lines[1].get_ydata()
    assert EXOTIC.endurance in sn_curve_cycles
    assert np.round(EXOTIC.get_stress(EXOTIC.endurance), 4) == np.round(
        ax.lines[0].get_ydata()[-1], 4
    )
    assert np.round(EXOTIC.get_stress(np.inf), 4) == np.round(
        sn_curve_stress[-1], 4
    )
    assert np.any(np.in1d(EXOTIC.get_knee_cycles(), sn_curve_cycles))
    assert np.any(
        np.in1d(
            np.round(EXOTIC.get_knee_stress(), 4), np.round(sn_curve_stress, 4)
        )
    )
    assert 200 in other_cycles
    assert 60 in other_stress
    assert ax.xaxis.get_label()._text == "Number of cycles"
    assert EXOTIC.unit in ax.yaxis.get_label()._text

nb.config.DISABLE_JIT = False