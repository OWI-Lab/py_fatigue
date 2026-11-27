"""Interpolate an S–N curve and calculate damage by stress range."""

import marimo

__generated_with = "0.26.0"
app = marimo.App(css_file="../docs/stylesheets/custom.css")


@app.cell
def introduction():
    import marimo as mo

    mo.md("""
        # S–N and Miner damage

        Build a piecewise S–N curve from three points in the user guide and
        give it a finite endurance limit. The first plot marks the interior
        knee, the endurance limit, and the applied ranges with finite lives.

        The second plot shows the Palmgren–Miner contribution of each range.
        The 50 MPa range falls below the endurance stress, so its predicted
        life is infinite and its Miner contribution is zero.
        """)
    return (mo,)


@app.cell
def calculate(mo):
    import matplotlib.pyplot as plt
    import numpy as np
    from py_fatigue import CycleCount, SNCurve
    from _examples.charts import plot_views

    cycles = CycleCount(
        stress_range=np.array([50.0, 80.0, 120.0]),
        mean_stress=np.zeros(3),
        count_cycle=np.array([1000.0, 1000.0, 100.0]),
    )
    curve = SNCurve.from_knee_points(
        knee_stress=[150.0, 86.0, 56.0],
        knee_cycles=[207_880, 963_397, 7_525_202],
        endurance=7_525_202,
    )
    damage = cycles.to_df().miner.damage(curve)
    fig, (ax, contributions) = plt.subplots(2, 1, layout="constrained")
    curve.plot(ax=ax)
    ax.lines[-1].set_label("Interpolated S–N curve")
    knees = curve.get_knee_cycles()
    ax.scatter(knees, curve.get_stress(knees), label="Interior knee", zorder=4)
    ax.axhline(curve.endurance_stress, linestyle="--", label="Endurance stress")
    finite = np.isfinite(damage.cycles_to_failure)
    ax.scatter(
        damage.loc[finite, "cycles_to_failure"],
        damage.loc[finite, "stress_range"],
        label="Applied ranges with finite life",
        zorder=3,
    )
    ax.legend()
    damage.miner.plot(ax=contributions)

    mo.vstack(
        [
            plot_views(fig, height="830px"),
            mo.md(f"**Total Miner damage** `{damage.pm_damage.sum():.4g}`"),
            mo.ui.table(
                damage[
                    [
                        "stress_range",
                        "count_cycle",
                        "cycles_to_failure",
                        "pm_damage",
                    ]
                ]
            ),
        ]
    )
    return (curve,)


@app.cell
def long_case(curve):
    import matplotlib.pyplot as _plt
    import numpy as _np
    from _examples.synthetic import (
        MINIMUM_STRESS,
        display_indices,
        long_case_view,
        synthetic_peaks,
    )
    from py_fatigue import SNCurve as _SNCurve
    from py_fatigue.damage import calc_pm

    peaks = synthetic_peaks()
    ranges = peaks - MINIMUM_STRESS
    unit_counts = _np.ones_like(ranges)
    long_fig, long_ax = _plt.subplots()
    for sn, label in (
        (curve, "Piecewise curve"),
        (_SNCurve(3, 10.493), "W3-C"),
    ):
        long_damage = _np.r_[0, _np.cumsum(calc_pm(ranges, unit_counts, sn))]
        indices = display_indices(long_damage.size)
        long_ax.plot(indices, long_damage[indices], label=label)
    long_ax.axhline(1.0, color="0.4", linestyle="--", label="Miner limit")
    long_ax.set(xlabel="Cumulative cycles", ylabel="Miner damage")
    long_ax.legend()
    below_endurance = _np.count_nonzero(ranges <= curve.endurance_stress)
    long_case_view(
        long_fig,
        f"Apply **{peaks.size:,} cycles** in order to both S–N curves. "
        f"**{below_endurance:,} cycles** are at or below the endurance stress "
        "of the piecewise curve. Compare its damage history with W3-C.",
    )
    return


if __name__ == "__main__":
    app.run()
