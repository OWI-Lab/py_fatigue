"""Check the interactive rendering of existing Py-Fatigue figures."""

import json

from matplotlib import pyplot as plt
import numpy as np
from pyecharts.charts import Page
import pytest

from py_fatigue import CycleCount, ParisCurve, SNCurve, WalkerCurve
from py_fatigue.crack_growth import (
    AxialCylinderThroughCrack,
    CircumferentialCylinderThroughCrack,
    SphereThroughCrack,
    CenterThroughCrack,
    EdgeThroughCrack,
    GrowthPoint,
    GrowthResult,
    NasgroCurve,
    RoundBarSurfaceCrack,
    SemiEllipticalSurfaceCrack,
)
from py_fatigue.damage import get_crack_growth
from py_fatigue.geometry import InfiniteSurface
from py_fatigue.mean_stress.corrections import (
    goodman_haigh_mean_stress_correction,
)
from py_fatigue.plotting import to_echarts


def test_rainflow_histogram_keeps_scatter_data_and_color_scale():
    """Use the same stress coordinates and density in both views."""
    cycles = CycleCount.from_timeseries([4, 7, 2, 10, 5, 9, 3, 4, 2, 12])
    fig, ax = cycles.plot_histogram()
    try:
        chart = to_echarts(fig)
        assert "locale: 'EN'" in chart.render_embed()
        data = np.asarray(chart.options["series"][0]["data"])
        scatter = ax.collections[0]
        np.testing.assert_array_equal(data[:, :2], scatter.get_offsets())
        np.testing.assert_array_equal(data[:, 2], scatter.get_array())
        (color_scale,) = chart.options["visualMap"]
        assert (color_scale["min"], color_scale["max"]) == scatter.get_clim()
        assert color_scale["text"][0] == scatter.colorbar.ax.get_ylabel()
    finally:
        plt.close(fig)


def test_goodman_plot_callback_captures_both_panels():
    """Capture the plot without changing the corrected values returned."""
    args = ([100, 200, 300], [0, 0, 0], [-1, 0, 0.5, 0.75, 0.9], 1000, 9)
    figures = []
    expected = goodman_haigh_mean_stress_correction(*args)
    actual = goodman_haigh_mean_stress_correction(*args, plot=figures.append)
    np.testing.assert_array_equal(actual, expected)
    (fig,) = figures
    try:
        axes = fig.axes
        chart = to_echarts(fig)
        assert isinstance(chart, Page)
        html = chart.render_embed()
        assert html.count("locale: 'EN'") == len(axes)
        for ax in axes:
            assert ax.get_title() in html
    finally:
        plt.close(fig)


def test_nasgro_uses_supplied_intensities_and_keeps_zero_rates():
    """Plot the pointwise law, including samples below threshold."""
    curve = NasgroCurve(1e-10, 2.0, 0.5, 1.0, 25.0, 0.0, 500.0, 2.0, 0.3)
    upper = np.array([10.0, 100.0])
    lower = upper * 0.2
    expected = [
        curve.get_growth_rate(4.0, lo, hi) for lo, hi in zip(lower, upper)
    ]
    chart = curve.echarts(4.0, lower, upper, scale="semilogx")
    values = np.asarray(chart.options["series"][0]["data"])
    np.testing.assert_array_equal(values[:, 0], upper)
    np.testing.assert_array_equal(values[:, 1], expected)
    assert values[0, 1] == 0.0


def test_nasgro_plot_sweeps_to_toughness_or_scalar_endpoint():
    """Sample the admissible interval and show where toughness lies."""
    curve = NasgroCurve(1e-10, 2.0, 0.5, 1.0, 25.0, 0.0, 500.0, 2.0, 0.3)
    chart = curve.echarts(length=4.0, k_min=20.0, scale="semilogx")
    growth, limit = chart.options["series"]
    x, y = np.asarray(growth["data"]).T
    assert 20.0 < x[0] < 0.99 * curve.toughness < x[-1] < curve.toughness
    assert y[0] == 0.0 and y[-1] > 0.0
    assert chart.options["xAxis"][0]["max"] == 500.0
    assert np.all(np.asarray(limit["data"])[:, 0] == 500.0)

    subset = curve.echarts(10.0, 1.0, 100.0).options["series"][0]["data"]
    assert 1.0 < subset[0][0] < subset[-1][0] == 100.0
    point = curve.echarts(10.0, 1.0, [100.0]).options["series"][0]["data"]
    assert point == [[100.0, subset[-1][1]]]


def test_nasgro_plots_the_long_crack_reference_by_default() -> None:
    """Plot without extra inputs and recover the long-crack threshold at R=0."""
    curve = NasgroCurve(1e-10, 2.0, 0.5, 1.0, 25.0, 0.0, 500.0, 2.0, 0.3)
    fig, ax = curve.plot()
    try:
        growth, limit = ax.lines
        samples = growth.get_xydata()
        below_threshold = samples[:, 0] <= curve.threshold_zero
        assert ax.get_yscale() == "log"
        assert np.all(samples[below_threshold, 1] == 0.0)
        assert np.all(samples[~below_threshold, 1] > 0.0)
        assert samples[-1, 0] < curve.toughness
        np.testing.assert_array_equal(limit.get_xdata(), [500.0, 500.0])
        assert curve.get_growth_rate(0.0381, 0.0, 20.0) > 0.0

        chart = curve.echarts()
        values = np.asarray(chart.options["series"][0]["data"], dtype=float)
        np.testing.assert_allclose(values[-1], samples[-1])
        assert "long-crack" in chart.options["series"][0]["name"]
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "curve_type,constructor_args,plot_args",
    [
        (SNCurve, (3.0, 12.0), ()),
        (ParisCurve, (3.0, 1e-10), ()),
        (WalkerCurve, (3.0, 1e-10), ()),
        (
            NasgroCurve,
            (1e-10, 2.0, 0.5, 1.0, 25.0, 0.0, 500.0, 2.0, 0.3),
            (4.0, 0.0),
        ),
    ],
)
@pytest.mark.parametrize(
    "scale,xscale,yscale",
    [
        ("loglog", "log", "log"),
        ("semilogx", "log", "linear"),
        ("semilogy", "linear", "log"),
    ],
)
def test_curve_scales_match_between_renderers(
    curve_type, constructor_args, plot_args, scale, xscale, yscale
):
    """Use the selected axis scales for both renderers and existing axes."""
    curve = curve_type(*constructor_args)
    fig, ax = plt.subplots()
    try:
        ax.set_xscale("log")
        ax.set_yscale("log")
        curve.plot(*plot_args, ax=ax, scale=scale)
        assert (ax.get_xscale(), ax.get_yscale()) == (xscale, yscale)
        options = curve.echarts(*plot_args, scale=scale).options
        assert options["xAxis"][0]["type"] == (
            "log" if xscale == "log" else "value"
        )
        assert options["yAxis"][0]["type"] == (
            "log" if yscale == "log" else "value"
        )
    finally:
        plt.close(fig)


def test_ordered_result_plots_recorded_sizes_on_supplied_axes():
    """Keep actual cycle counts and the geometry's dimension labels."""
    result = GrowthResult(
        (
            GrowthPoint(0.0, (4.0, 8.0), (0.0, 0.0), (50.0, 50.0), 0),
            GrowthPoint(5.5, (4.5, 8.25), (0.0, 0.0), (55.0, 55.0), 0),
        ),
        "spectrum_exhausted",
    )
    fig, ax = plt.subplots()
    try:
        plotted_fig, plotted_ax = result.plot(ax=ax)
        assert plotted_fig is fig
        assert plotted_ax is ax
        assert [line.get_label() for line in ax.lines] == [
            "Depth",
            "Half-length",
        ]
        np.testing.assert_array_equal(
            ax.lines[0].get_xydata(), [[0, 4], [5.5, 4.5]]
        )
        assert result.echarts().options["series"][1]["data"] == [
            [0, 8],
            [5.5, 8.25],
        ]
        assert plt.fignum_exists(fig.number)
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "geometry, initial, final, start_x, end_x",
    [
        (CenterThroughCrack(50.0, 4.0), (4.0,), (6.0,), [-4, 4], [-6, 6]),
        (EdgeThroughCrack(100.0, 4.0), (4.0,), (6.0,), [0, 4], [0, 6]),
        (
            SemiEllipticalSurfaceCrack(50.0, 15.0, 4.0, 8.0),
            (4.0, 8.0),
            (5.0, 9.0),
            [8, -8],
            [9, -9],
        ),
    ],
)
def test_crack_profile_overlay_uses_the_accepted_endpoints(
    geometry, initial, final, start_x, end_x
) -> None:
    """Draw both endpoints in their physical coordinates in both plot modes."""
    result = GrowthResult(
        (
            GrowthPoint(0.0, initial, (), (), 0),
            GrowthPoint(25.0, final, (), (), 0),
        ),
        "spectrum_exhausted",
    )
    fig, axes = geometry.plot(result=result)
    try:
        _, start, end = axes.lines
        np.testing.assert_array_equal(start.get_xdata()[[0, -1]], start_x)
        np.testing.assert_array_equal(end.get_xdata()[[0, -1]], end_x)
        if len(initial) == 2:
            assert axes.yaxis_inverted()
            assert start.get_ydata().max() == initial[0]
            assert end.get_ydata().max() == final[0]
        chart = geometry.echarts(result=result)
        data = chart.options["series"]
        assert data[-2]["lineStyle"]["type"] == "dashed"
        assert data[-1]["name"] == "Final accepted crack"
        np.testing.assert_array_equal(data[-1]["data"], end.get_xydata())
    finally:
        plt.close(fig)


def test_crack_profile_with_no_accepted_state_uses_the_initial_geometry() -> (
    None
):
    """Do not label an invented final profile when a run stops at its start."""
    crack = EdgeThroughCrack(100.0, 4.0)
    result = GrowthResult((), "toughness")
    chart = crack.echarts(result=result)
    assert [series["name"] for series in chart.options["series"]] == [
        "Plate width",
        "Crack",
    ]


def test_legacy_plot_uses_cycle_counts_at_start_of_each_step():
    """The last recorded depth precedes completion of the last step."""
    cycles = CycleCount(
        stress_range=np.array([50.0, 70.0]),
        mean_stress=np.zeros(2),
        count_cycle=np.array([3.0, 2.0]),
    )
    curve = ParisCurve(slope=3.0, intercept=1e-10)
    crack = InfiniteSurface(initial_depth=1.0)
    result = get_crack_growth(cycles, curve, crack)
    plotted = np.asarray(result.echarts().options["series"][0]["data"])
    np.testing.assert_array_equal(
        plotted[:, 0], np.r_[0.0, np.cumsum(result.count_cycle[:-1])]
    )
    np.testing.assert_array_equal(plotted[:, 1], result.crack_depth)


def test_surface_crack_sketch_uses_the_specified_dimensions():
    """Show the crack tips and draw depth increasing downwards."""
    chart = SemiEllipticalSurfaceCrack(50.0, 15.0, 4.0, 8.0).echarts()
    outline, front = chart.options["series"]
    assert outline["data"][2] == [50.0, 15.0]
    np.testing.assert_allclose(
        front["data"][::50], [[8, 0], [0, 4], [-8, 0]], atol=1e-12
    )
    assert json.loads(chart.dump_options_with_quotes())["yAxis"][0]["inverse"]


def test_sc07_profiles_meet_the_bar_surface_orthogonally() -> None:
    crack = RoundBarSurfaceCrack(20.0, 2.0)
    result = GrowthResult(
        (
            GrowthPoint(0.0, (2.0,), (), (), 0),
            GrowthPoint(1.0, (4.0,), (), (), 0),
        ),
        "spectrum_exhausted",
    )
    chart = crack.echarts(result=result)
    for depth, series in zip((2.0, 4.0), chart.options["series"][1:]):
        front = np.asarray(series["data"], dtype=float)
        contacts = front[[0, -1]]
        np.testing.assert_allclose(np.linalg.norm(contacts, axis=1), 10.0)
        tip = 10.0 - depth
        np.testing.assert_allclose(
            front[len(front) // 2], [tip, 0.0], atol=1e-12
        )
        center = [(100.0 + tip**2) / (2 * tip), 0.0]
        np.testing.assert_allclose(
            np.sum(contacts * (contacts - center), axis=1), 0.0, atol=1e-12
        )


@pytest.mark.parametrize(
    "geometry",
    [
        AxialCylinderThroughCrack(100.0, 1.0, 4.0),
        SphereThroughCrack(100.0, 1.0, 4.0),
        CircumferentialCylinderThroughCrack(100.0, 1.0, 4.0, 2500.0, 0.3),
    ],
)
def test_through_shell_plot_uses_crack_length_in_mm(geometry) -> None:
    result = GrowthResult(
        (
            GrowthPoint(0.0, (4.0,), (), (), 0),
            GrowthPoint(1.0, (8.0,), (), (), 0),
        ),
        "spectrum_exhausted",
    )
    series = geometry.echarts(result=result).options["series"]
    assert series[-2]["name"] == "Initial crack"
    assert series[-1]["name"] == "Final accepted crack"
    for half_length, line in zip((4.0, 8.0), series[-2:]):
        data = np.asarray(line["data"])
        if isinstance(geometry, AxialCylinderThroughCrack):
            np.testing.assert_allclose(
                data, [[-half_length, 0], [half_length, 0]]
            )
        else:
            np.testing.assert_allclose(np.linalg.norm(data, axis=1), 100)
            np.testing.assert_allclose(
                100 * np.arctan2(data[[0, -1], 1], data[[0, -1], 0]),
                [-half_length, half_length],
            )
