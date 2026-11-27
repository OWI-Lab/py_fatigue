"""Evaluate fatigue and crack-growth limits under a long synthetic signal."""

import marimo

__generated_with = "0.26.0"
app = marimo.App(css_file="../docs/stylesheets/custom.css")


@app.cell
def introduction():
    import marimo as mo

    mo.md("""
        # From a synthetic signal to failure

        Use Py-Fatigue's signal generator to draw a peak stress for each
        cycle. Return to the same tensile minimum between peaks so the
        cycle order is known. This gives one history for two separate
        predictions with teaching inputs.

        **Stress life** uses an S–N curve and marks the first cycle where
        cumulative Miner damage reaches 1. Rainflow counting the complete
        waveform provides a check on the final damage sum.

        **Crack growth** applies the generated peaks in order to a plate
        crack. The final plotted markers show the last accepted crack state.
        Inspect the termination reason to see which limit the solver reached.
        """)
    return (mo,)


@app.cell
def generate_signal():
    import numpy as np
    from _examples.synthetic import (
        MINIMUM_STRESS,
        synthetic_peaks,
        tensile_waveform,
    )

    minimum = MINIMUM_STRESS
    peaks = synthetic_peaks()
    stress = tensile_waveform(peaks)
    return minimum, np, peaks, stress


@app.cell
def count_damage(minimum, np, peaks, stress):
    from py_fatigue import CycleCount, SNCurve
    from py_fatigue.damage import calc_pm, get_pm

    # The W3-C curve from lesson 00.
    sn = SNCurve(slope=3.0, intercept=10.493)
    counted = CycleCount.from_timeseries(stress, unit="MPa")
    rainflow_damage = get_pm(counted, sn).sum()
    increments = calc_pm(peaks - minimum, np.ones_like(peaks), sn)
    damage = np.r_[0.0, np.cumsum(increments)]
    crossings = np.flatnonzero(damage >= 1.0)
    fatigue_failure = int(crossings[0]) if crossings.size else None
    return damage, fatigue_failure, rainflow_damage


@app.cell
def propagate(minimum, peaks):
    from py_fatigue.crack_growth import (
        SemiEllipticalSurfaceCrack,
        CenterThroughCrack,
        LoadBlock,
        NasgroCurve,
        simulate_crack_growth,
    )

    geometry = CenterThroughCrack(half_width=50.0, half_length=5.0)
    geometry = SemiEllipticalSurfaceCrack(
        half_length=10, half_width=100, thickness=10, depth=2
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
    loads = [LoadBlock(minimum, float(peak), 1.0) for peak in peaks]
    result = simulate_crack_growth(
        geometry,
        curve,
        loads,
        max_cycle_step=1.0,
        length_tolerance=1e-6,
        cycle_tolerance=1e-4,
        max_trials=len(loads),
    )
    return curve, result


@app.cell
def show_results(
    damage,
    fatigue_failure,
    mo,
    np,
    peaks,
    rainflow_damage,
    result,
    stress,
):
    import matplotlib.pyplot as plt
    from _examples.synthetic import display_indices
    from py_fatigue.crack_growth import GrowthResult
    from _examples.charts import plot_views

    fig, (signal, miner, growth) = plt.subplots(
        3, 1, figsize=(7, 9), layout="constrained"
    )
    preview = stress[:51]
    signal.plot(np.arange(preview.size) / 2, preview, "o-")
    signal.set(
        xlabel="Cycle position",
        ylabel="Stress (MPa)",
        title=f"First {(preview.size - 1) // 2} cycles",
    )

    # Thin the histories for display and retain the failure markers.
    samples = display_indices(damage.size)
    if fatigue_failure is not None:
        samples = np.union1d(samples, fatigue_failure)
    miner.plot(samples, damage[samples], label="Cumulative Miner damage")
    miner.axhline(1.0, linestyle="--", color="0.4", label="Damage limit")
    if fatigue_failure is not None:
        miner.scatter(
            fatigue_failure,
            damage[fatigue_failure],
            color="firebrick",
            label="First cycle at the limit",
        )
    miner.set(xlabel="Cycles", ylabel="Miner damage", xlim=(0, peaks.size))
    miner.legend()

    indices = display_indices(len(result.history))
    display_growth = GrowthResult(
        tuple(result.history[index] for index in indices), result.termination
    )
    display_growth.plot(ax=growth)
    if result.history:
        last = result.history[-1]
        growth.scatter(
            [last.cycles] * len(last.sizes),
            last.sizes,
            color="firebrick",
            label="Last accepted state",
        )
        growth.legend()
    growth.set_xlim(0, peaks.size)

    fatigue_summary = (
        f"Miner reached 1 at cycle **{fatigue_failure:,}**."
        if fatigue_failure is not None
        else "Miner damage stayed below 1 over the supplied history."
    )
    termination = result.termination.replace("_", " ")
    mo.vstack(
        [
            mo.md(f"""
            The waveform has **{stress.size:,} samples** over
            **{peaks.size:,} cycles**.
            {fatigue_summary}
            Full-record damage is **{damage[-1]:.5f}** by cycle and
            **{rainflow_damage:.5f}** by rainflow counting.

            **Crack growth** · {termination}. The last accepted state was
            at **{result.final_cycles:,.2f} cycles**.
            """),
            plot_views(fig, height="1240px"),
        ]
    )


@app.cell
def _(curve):
    from _examples.charts import plot_views as _plot_views

    _plot_views(curve.plot()[0])


if __name__ == "__main__":
    app.run()
