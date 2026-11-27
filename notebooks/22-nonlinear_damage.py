"""Compare Miner, Pavlou, and Theil under ordered stress blocks."""

import marimo

__generated_with = "0.26.0"
app = marimo.App(css_file="../docs/stylesheets/custom.css")


@app.cell
def introduction():
    import marimo as mo

    mo.md("""
        # Nonlinear damage under ordered loading

        Apply three blocks in increasing stress order, then reverse their
        order. Miner sums the same contributions in both cases. Compare that
        with the Pavlou damage curve approach, which weights increments using
        the accumulated damage.

        The second panel shows a separate calculation, the Theil effective
        stress history, against the S–N curve. The blocks are teaching inputs
        supplied in order.
        """)
    return (mo,)


@app.cell
def compare(mo):
    import matplotlib.pyplot as plt
    import numpy as np
    from py_fatigue import SNCurve
    from py_fatigue.damage.stress_life import (
        calc_nonlinear_damage_with_dca,
        calc_pm,
        calc_theil_sn_damage,
    )
    from _examples.charts import plot_views

    ranges = np.array([50.0, 100.0, 150.0])
    counts = np.array([1000.0, 2000.0, 3000.0])
    curve = SNCurve(slope=[3, 5], intercept=[11.764, 15.606])
    damage_bands = np.array(
        [0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1]
    )
    fig, (damage, history) = plt.subplots(2, 1, layout="constrained")
    end_damage = []
    for label, stress, cycles in (
        ("Low to high", ranges, counts),
        ("High to low", ranges[::-1], counts[::-1]),
    ):
        cumulative_cycles = np.r_[0.0, np.cumsum(cycles)]
        miner = np.cumsum(calc_pm(stress, cycles, curve))
        _, pavlou, _, _ = calc_nonlinear_damage_with_dca(
            "pavlou",
            stress,
            cycles,
            curve,
            damage_bands,
            ultimate_stress=900.0,
            base_exponent=-0.75,
        )
        damage.plot(
            cumulative_cycles,
            np.r_[0.0, miner],
            "o--",
            label=f"Miner · {label}",
        )
        damage.plot(
            cumulative_cycles,
            np.r_[0.0, pavlou],
            "o-",
            label=f"Pavlou · {label}",
        )
        end_damage.append(
            {
                "Order": label,
                "Miner": float(miner[-1]),
                "Pavlou": float(pavlou[-1]),
            }
        )
    damage.set(xlabel="Cumulative cycles", ylabel="Accumulated damage")
    damage.legend()

    curve.plot(ax=history, scale="semilogx")
    history.lines[-1].set_label("S–N curve")
    for label, cycles, stress in calc_theil_sn_damage(ranges, counts, curve):
        history.plot(cycles, stress, label=label)
    history.set(xlabel="Cumulative cycles", ylabel="Stress range (MPa)")
    history.legend()

    mo.vstack(
        [
            plot_views(fig, height="830px"),
            mo.ui.table(end_damage),
        ]
    )
    return curve, damage_bands


@app.cell
def long_case(curve, damage_bands):
    import matplotlib.pyplot as _plt
    import numpy as _np
    from _examples.synthetic import (
        MINIMUM_STRESS,
        display_indices,
        long_case_view,
        synthetic_peaks,
    )
    from py_fatigue.damage.stress_life import (
        calc_nonlinear_damage_with_dca as _calc_dca,
        calc_pm as _calc_pm,
    )

    peaks = synthetic_peaks()
    long_ranges = peaks - MINIMUM_STRESS
    long_counts = _np.ones_like(long_ranges)
    long_fig, ax = _plt.subplots()
    totals = []
    for long_label, ordered_ranges in (
        ("Generated order", long_ranges),
        ("Reverse order", long_ranges[::-1]),
    ):
        long_miner = _np.r_[
            0, _np.cumsum(_calc_pm(ordered_ranges, long_counts, curve))
        ]
        _, cumulative_damage, _, _ = _calc_dca(
            "pavlou",
            ordered_ranges,
            long_counts,
            curve,
            damage_bands,
            ultimate_stress=900.0,
            base_exponent=-0.75,
        )
        long_pavlou = _np.r_[0, cumulative_damage]
        indices = display_indices(long_miner.size - 1) + 1
        ax.plot(
            indices,
            long_miner[indices],
            linestyle="--",
            label=f"Miner · {long_label}",
        )
        ax.plot(indices, long_pavlou[indices], label=f"Pavlou · {long_label}")
        totals.append(
            f"- **{long_label}** · Miner {long_miner[-1]:.8g}, "
            f"Pavlou {long_pavlou[-1]:.8g}."
        )
    ax.set(
        xlabel="Cumulative cycles", ylabel="Accumulated damage", yscale="log"
    )
    ax.legend()
    long_case_view(
        long_fig,
        f"Run both damage rules over **{peaks.size:,} cycles**, first as "
        "generated and then in reverse. The end totals show the effect of "
        "order. The logarithmic axis separates the two damage histories.\n\n"
        + "\n".join(totals),
    )
    return


if __name__ == "__main__":
    app.run()
