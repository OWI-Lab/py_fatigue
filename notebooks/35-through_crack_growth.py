"""Propagate through cracks in plates and shells."""

import marimo

__generated_with = "0.26.0"
app = marimo.App(width="medium")


@app.cell
def introduction():
    import marimo as mo

    mo.md("""
        # Through-crack growth in plates and shells

        Compare a center crack (TC01) and a single edge crack (TC02) under the
        same ordered tensile loads. The center crack grows symmetrically, so
        the solver records the **half-length**. The edge crack grows from one
        side, so the solver records the **length from that edge**.

        Each profile shows the initial crack as a dashed line and the last
        accepted crack in red. The material and loads are synthetic.
        """)
    return (mo,)


@app.cell
def propagate():
    from math import ceil

    from py_fatigue.crack_growth import (
        CenterThroughCrack,
        EdgeThroughCrack,
        LoadBlock,
        NasgroCurve,
        simulate_crack_growth,
    )

    curve = NasgroCurve(1e-10, 2.0, 0.5, 1.0, 25.0, 0.0, 500.0, 2.0, 0.3)
    geometries = {
        "TC01": CenterThroughCrack(half_width=50.0, half_length=4.0),
        "TC02": EdgeThroughCrack(width=100.0, length=4.0),
    }
    loads = [
        LoadBlock(10.0, 70.0, 50_000),
        LoadBlock(20.0, 90.0, 25_005),
        LoadBlock(10.0, 60.0, 25_000),
    ]
    step = 10.0
    # Start with the steps needed to cover each block and allow for retries.
    trial_budget = 2 * sum(ceil(block.cycles / step) for block in loads)
    results = {
        name: simulate_crack_growth(
            geometry,
            curve,
            loads,
            max_cycle_step=step,
            length_tolerance=1e-6,
            cycle_tolerance=1e-4,
            max_trials=trial_budget,
        )
        for name, geometry in geometries.items()
    }
    return LoadBlock, curve, geometries, loads, results, simulate_crack_growth


@app.cell
def ordered_profiles(geometries, loads, mo, results):
    import matplotlib.pyplot as plt
    from py_fatigue.plotting import to_echarts

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), layout="constrained")
    summaries = []
    for (name, geometry), profile in zip(geometries.items(), axes[:2]):
        result = results[name]
        geometry.plot(ax=profile, result=result)
        result.plot(ax=axes[2], label=name)
        net_stress = geometry.net_section_stress(loads[0].maximum)
        summaries.append(
            f"- **{name}** · {result.final_cycles:,.0f} accepted cycles, "
            f"{result.termination.replace('_', ' ')}. "
            f"Initial net-section stress at the first peak is {net_stress:.1f} MPa."
        )
    mo.vstack(
        [
            mo.md("\n".join(summaries)),
            mo.ui.tabs(
                {
                    "Static": fig,
                    "ECharts": mo.iframe(
                        to_echarts(fig).render_embed(), height="1240px"
                    ),
                }
            ),
        ]
    )


@app.cell
def shell_cases(curve, loads, mo, simulate_crack_growth):
    from math import ceil as _ceil
    import matplotlib.pyplot as _shell_plt
    from py_fatigue.crack_growth import (
        AxialCylinderThroughCrack,
        CircumferentialCylinderThroughCrack,
        SphereThroughCrack,
    )
    from py_fatigue.plotting import to_echarts as _to_echarts

    shells = (
        SphereThroughCrack(mean_radius=100.0, thickness=1.0, half_length=4.0),
        AxialCylinderThroughCrack(
            mean_radius=100.0, thickness=1.0, half_length=4.0
        ),
        CircumferentialCylinderThroughCrack(
            mean_radius=100.0,
            thickness=1.0,
            half_length=4.0,
            length=2500.0,
            poisson=0.3,
        ),
    )
    shell_step = 10.0
    shell_fig, shell_axes = _shell_plt.subplots(
        2, 2, figsize=(9, 7), layout="constrained"
    )
    shell_summaries = []
    for (_case, _shell), _profile in zip(
        zip(("TC06", "TC07", "TC08"), shells), shell_axes.flat
    ):
        _run = simulate_crack_growth(
            _shell,
            curve,
            loads,
            max_cycle_step=shell_step,
            length_tolerance=1e-6,
            cycle_tolerance=1e-4,
            max_trials=2
            * sum(_ceil(block.cycles / shell_step) for block in loads),
        )
        _shell.plot(ax=_profile, result=_run)
        _run.plot(ax=shell_axes[1, 1], label=_case)
        shell_summaries.append(
            f"- **{_case}** · {_run.final_cycles:,.0f} cycles, "
            f"{_run.termination.replace('_', ' ')}."
        )
    mo.vstack(
        [
            mo.md(
                "## Through cracks in shells\n\n"
                "Here the same synthetic membrane-stress history is applied "
                "in the opening direction for each crack. That means hoop "
                "stress for TC07 and axial stress for TC08.\n\n"
                + "\n".join(shell_summaries)
            ),
            mo.ui.tabs(
                {
                    "Static": shell_fig,
                    "ECharts": mo.iframe(
                        _to_echarts(shell_fig).render_embed(), height="1640px"
                    ),
                }
            ),
        ]
    )


@app.cell
def long_case(LoadBlock, curve, geometries, simulate_crack_growth):
    import matplotlib.pyplot as _plt
    from _examples.synthetic import (
        MINIMUM_STRESS,
        long_case_view,
        synthetic_peaks,
    )

    peaks = synthetic_peaks()
    schedule = [LoadBlock(MINIMUM_STRESS, float(peak), 1.0) for peak in peaks]
    long_fig, long_axes = _plt.subplots(
        2, 1, figsize=(7, 5), layout="constrained"
    )
    outcomes = []
    for (case, crack), long_profile in zip(geometries.items(), long_axes):
        long_result = simulate_crack_growth(
            crack,
            curve,
            schedule,
            max_cycle_step=1.0,
            length_tolerance=1e-6,
            cycle_tolerance=1e-4,
            max_trials=2 * len(schedule),
        )
        crack.plot(ax=long_profile, result=long_result)
        outcomes.append(
            f"- **{case}** · {long_result.termination.replace('_', ' ')} after "
            f"{long_result.final_cycles:,.2f} cycles."
        )
        del long_result
    long_case_view(
        long_fig,
        f"Apply **{peaks.size:,} tensile cycles** in their generated order.\n\n"
        + "\n".join(outcomes),
        height="840px",
    )


if __name__ == "__main__":
    app.run()
