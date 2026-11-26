"""Compare ordered crack propagation in two plate geometries."""

import marimo

__generated_with = "0.26.0"
app = marimo.App()


@app.cell
def introduction():
    import marimo as mo

    mo.md("""
        # Ordered plate crack growth

        Apply the same ordered tensile membrane load blocks to a center crack
        and a semi-elliptical surface crack. The material and loads are
        synthetic teaching inputs. Use calibrated input data for engineering
        assessments.

        Compare the applied minimum and maximum stresses, the accepted crack
        dimensions, and the peak stress intensity at each crack-front point.
        At a block boundary, the peak intensity can jump even though the
        crack dimensions are continuous. The dashed line marks toughness.
        """)
    return (mo,)


@app.cell
def propagate():
    from math import ceil

    from py_fatigue.crack_growth import (
        CenterThroughCrack,
        LoadBlock,
        NasgroCurve,
        SemiEllipticalSurfaceCrack,
        simulate_crack_growth,
    )

    curve = NasgroCurve(
        coefficient=1e-10,
        exponent=2.0,
        p=0.5,
        q=1.0,
        threshold_zero=25.0,
        threshold_fanning=0.0,
        toughness=500.0,
        constraint=2.0,
        peak_to_flow=0.3,
    )
    loads = [
        LoadBlock(10.0, 70.0, 50_000),
        LoadBlock(20.0, 90.0, 25_005),
        LoadBlock(10.0, 60.0, 25_000),
        LoadBlock(40.0, 100.0, 10_000),
    ]
    max_cycle_step = 10.0
    minimum_steps = sum(ceil(load.cycles / max_cycle_step) for load in loads)
    # Leave room for the solver to reject trials and retry with smaller steps.
    max_trials = max(10_000, 2 * minimum_steps)
    cases = {
        "Center crack": (CenterThroughCrack(50.0, 4.0), ("Half-length",)),
        "Surface crack": (
            SemiEllipticalSurfaceCrack(50.0, 15.0, 4.0, 8.0),
            ("Depth", "Half-length"),
        ),
    }
    results = {
        name: (
            simulate_crack_growth(
                geometry,
                curve,
                loads,
                max_cycle_step=max_cycle_step,
                length_tolerance=1e-6,
                cycle_tolerance=1e-4,
                max_trials=max_trials,
            ),
            dimensions,
        )
        for name, (geometry, dimensions) in cases.items()
    }
    return cases, curve, loads, results


@app.cell
def show_results(curve, loads, mo, results):
    import matplotlib.pyplot as plt
    import numpy as np
    from py_fatigue.plotting import to_echarts

    fig, (loading, shape, intensity) = plt.subplots(
        3, 1, figsize=(7, 9), sharex=True, layout="constrained"
    )
    boundaries = np.r_[0.0, np.cumsum([block.cycles for block in loads])]
    intervals = np.repeat(boundaries, 2)[1:-1]
    loading.plot(
        intervals,
        np.repeat([block.minimum for block in loads], 2),
        label="Minimum",
    )
    loading.plot(
        intervals,
        np.repeat([block.maximum for block in loads], 2),
        label="Maximum",
    )
    loading.set(ylabel="Applied stress (MPa)")
    loading.legend()

    summaries = []
    for name, (result, dimensions) in results.items():
        result.plot(ax=shape, label=name)
        tip_names = (
            ("Tip",)
            if len(dimensions) == 1
            else ("Deepest point", "Free surface")
        )
        for index, tip in enumerate(tip_names):
            intensity.plot(
                [point.cycles for point in result.history],
                [point.k_max[index] for point in result.history],
                label=f"{name} · {tip}",
            )
        summaries.append(
            f"- **{name}** · {result.final_cycles:g} cycles, "
            f"{result.termination.replace('_', ' ')}"
        )
    intensity.axhline(
        curve.toughness, linestyle="--", color="0.4", label="Toughness"
    )
    intensity.set(
        xlabel="Cumulative cycles",
        ylabel="Peak stress intensity (MPa √mm)",
        xlim=(0.0, boundaries[-1]),
    )
    intensity.legend()
    mo.vstack(
        [
            mo.ui.tabs(
                {
                    "Static": fig,
                    "ECharts": mo.iframe(
                        to_echarts(fig).render_embed(), height="1240px"
                    ),
                }
            ),
            mo.md("\n".join(summaries)),
        ]
    )
    return


@app.cell
def _(cases):
    cases["Center crack"][0].half_length

    return


@app.cell
def long_case(cases, curve):
    import matplotlib.pyplot as _plt
    from _examples.synthetic import (
        MINIMUM_STRESS,
        display_indices,
        long_case_view,
        synthetic_peaks,
    )
    from py_fatigue.crack_growth import (
        GrowthResult,
        LoadBlock as _LoadBlock,
        simulate_crack_growth as _simulate,
    )

    peaks = synthetic_peaks()
    schedule = [_LoadBlock(MINIMUM_STRESS, float(peak), 1.0) for peak in peaks]
    long_fig, ax = _plt.subplots()
    outcomes = []
    for label, (long_geometry, _) in cases.items():
        long_result = _simulate(
            long_geometry,
            curve,
            schedule,
            max_cycle_step=1.0,
            length_tolerance=1e-6,
            cycle_tolerance=1e-4,
            max_trials=len(schedule),
        )
        indices = display_indices(len(long_result.history))
        display = GrowthResult(
            tuple(long_result.history[long_index] for long_index in indices),
            long_result.termination,
        )
        display.plot(ax=ax, label=label)
        stop = long_result.termination.replace("_", " ")
        outcomes.append(
            f"- **{label}** · {stop} at {long_result.final_cycles:,.2f} cycles."
        )
        del long_result
    long_case_view(
        long_fig,
        f"Apply **{peaks.size:,} cycles** in order to both geometries. "
        "The curves show the accepted crack dimensions with fewer points "
        "for display.\n\n" + "\n".join(outcomes),
    )
    return


if __name__ == "__main__":
    app.run()
