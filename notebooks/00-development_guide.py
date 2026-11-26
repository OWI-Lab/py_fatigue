"""Interactive development examples for Py-Fatigue."""

import marimo

__generated_with = "0.26.0"
app = marimo.App()


@app.cell
def imports():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from py_fatigue import SNCurve, __version__
    from py_fatigue.cycle_count.rainflow import rainflow
    from py_fatigue.plotting import to_echarts

    return SNCurve, __version__, mo, np, plt, rainflow, to_echarts


@app.cell
def introduction(__version__, mo):
    mo.md(f"""
        # Py-Fatigue development guide

        Running Py-Fatigue {__version__}.

        The first digit groups the lessons. **1x** covers cycle counting and
        residual recovery, **2x** covers stress-life analysis, and **3x** covers
        crack growth. Each lesson runs with its own example inputs.

        Start here to compare rainflow methods and see how several S–N
        curves can share one figure. Run each lesson in marimo to modify inputs
        and inspect how the curves change.
        """)


@app.cell
def compare_rainflow(mo, np, plt, rainflow, to_echarts):
    signal = np.array(
        [
            4,
            7,
            2,
            10,
            5,
            9,
            3,
            4,
            2,
            12,
            5,
            11,
            1,
            4,
            3,
            10,
            6,
            12,
            4,
            8,
            1,
            9,
            4,
            6,
        ],
        dtype=float,
    )
    fig, ax = plt.subplots()
    ax.plot(signal, color="#666666", label="Stress history")
    for label, method, marker in [
        ("ASTM", "astm", "o"),
        ("Four-point", "fourpoint", "D"),
    ]:
        _, residuals, indices = rainflow(
            signal, method=method, extended_output=True
        )
        ax.plot(
            indices, residuals, linestyle="none", marker=marker, label=label
        )
    ax.set(xlabel="Sample", ylabel="Stress")
    ax.legend()

    mo.vstack(
        [
            mo.md(
                "## Compare rainflow residuals\n"
                "The markers identify which turning points each algorithm "
                "leaves in the residual sequence."
            ),
            mo.ui.tabs(
                {
                    "Static": fig,
                    "ECharts": mo.iframe(
                        to_echarts(fig).render_embed(), height="420px"
                    ),
                }
            ),
        ]
    )


@app.cell
def sn_curves(SNCurve, mo, plt, to_echarts):
    curves = [
        SNCurve(
            slope=slope,
            intercept=intercept,
            norm="DNVGL-RP-C203",
            environment="Free corrosion",
            curve=label,
        )
        for slope, intercept, label in [
            (3, 12.436, "B1"),
            ([3], 12.115, "C"),
            (3, [11.533], "E"),
            ([3], [10.493], "W3-C"),
        ]
    ]
    sn_fig, sn_ax = plt.subplots()
    for curve in curves:
        curve.plot(fig=sn_fig, ax=sn_ax)
        sn_ax.lines[-1].set_label(curve.curve)
    sn_ax.legend()
    mo.vstack(
        [
            mo.md("## S–N curves in free corrosion"),
            mo.ui.tabs(
                {
                    "Static": sn_fig,
                    "ECharts": mo.iframe(
                        to_echarts(sn_fig).render_embed(), height="420px"
                    ),
                }
            ),
        ]
    )
    return (curves,)


@app.cell
def long_case(curves, np, plt):
    from _examples.synthetic import (
        long_case_view,
        synthetic_peaks,
        tensile_waveform,
    )
    from py_fatigue import CycleCount
    from py_fatigue.damage import get_pm

    peaks = synthetic_peaks()
    counted = CycleCount.from_timeseries(tensile_waveform(peaks))
    edges = np.histogram_bin_edges(counted.stress_range, bins=60)
    centres = (edges[:-1] + edges[1:]) / 2
    long_fig, long_ax = plt.subplots()
    for sn in curves:
        damage_by_range, _ = np.histogram(
            counted.stress_range, bins=edges, weights=get_pm(counted, sn)
        )
        long_ax.plot(centres, damage_by_range, label=sn.curve)
    long_ax.set(
        xlabel="Stress range (MPa)", ylabel="Miner damage per range bin"
    )
    long_ax.legend()
    long_case_view(
        long_fig,
        f"Rainflow-count **{peaks.size:,} generated cycles**. Apply the four "
        "S–N curves above to the same spectrum and compare which stress "
        "ranges contribute the most Miner damage.",
    )
    return


if __name__ == "__main__":
    app.run()
