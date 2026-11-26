"""Example of rainflow counting."""

import marimo

__generated_with = "0.26.0"
app = marimo.App()


@app.cell
def introduction():
    import marimo as mo

    mo.md("""
        # Rainflow counting

        Count a short stress history and inspect the full and half cycles.
        Export the spectrum as a sparse dictionary with `as_dict()`, then
        rebuild it with `CycleCount.from_rainflow()`. Compare the summed
        counts by range before and after binning. Each half cycle has
        weight 0.5.
        """)
    return (mo,)


@app.cell
def calculate(mo):
    import matplotlib.pyplot as plt
    import numpy as np
    from py_fatigue import CycleCount
    from py_fatigue.plotting import to_echarts

    stress = np.array([4, 7, 2, 10, 5, 9, 3, 4, 2, 12], dtype=float)
    cycles = CycleCount.from_timeseries(
        stress, unit="MPa", range_bin_width=1.0, mean_bin_width=1.0
    )
    exported = cycles.as_dict()
    restored = CycleCount.from_rainflow(exported, unit=cycles.unit)

    fig, (signal, counted) = plt.subplots(2, 1, layout="constrained")
    signal.plot(np.arange(stress.size), stress, "o-")
    signal.set(xlabel="Sample", ylabel="Stress (MPa)", title="Input history")

    for spectrum, label, marker in (
        (cycles, "From signal", "o"),
        (restored, "From sparse dictionary", "D"),
    ):
        ranges, groups = np.unique(spectrum.stress_range, return_inverse=True)
        counted.plot(
            ranges,
            np.bincount(groups, weights=spectrum.count_cycle),
            linestyle="none",
            marker=marker,
            label=label,
        )
    counted.set(xlabel="Stress range (MPa)", ylabel="Total cycle count")
    counted.legend()

    mo.vstack(
        [
            mo.ui.tabs(
                {
                    "Static": fig,
                    "ECharts": mo.iframe(
                        to_echarts(fig).render_embed(), height="830px"
                    ),
                }
            ),
            mo.md(
                f"**{cycles.count_cycle.sum():g}** cycles from the signal, "
                f"**{restored.count_cycle.sum():g}** after reloading. "
                "Binned cycles are represented by their bin centres."
            ),
            mo.ui.table(
                cycles.to_df()[["stress_range", "mean_stress", "count_cycle"]]
            ),
        ]
    )


@app.cell
def long_case():
    import matplotlib.pyplot as _plt
    import numpy as _np
    from _examples.synthetic import (
        long_case_view,
        synthetic_peaks,
        tensile_waveform,
    )
    from py_fatigue import CycleCount as _CycleCount

    peaks = synthetic_peaks()
    waveform = tensile_waveform(peaks)
    original = _CycleCount.from_timeseries(waveform)
    long_restored = _CycleCount.from_rainflow(original.as_dict())
    edges = _np.histogram_bin_edges(
        _np.r_[original.stress_range, long_restored.stress_range], bins=60
    )
    centres = (edges[:-1] + edges[1:]) / 2
    long_fig, ax = _plt.subplots()
    for long_label, long_spectrum in (
        ("Rainflow", original),
        ("After sparse reload", long_restored),
    ):
        counts, _ = _np.histogram(
            long_spectrum.stress_range,
            bins=edges,
            weights=long_spectrum.count_cycle,
        )
        ax.plot(centres, counts, label=long_label)
    ax.set(xlabel="Stress range (MPa)", ylabel="Counted cycles per bin")
    ax.legend()
    original_count = original.count_cycle.sum()
    restored_count = long_restored.count_cycle.sum()
    long_case_view(
        long_fig,
        f"Count **{waveform.size:,} samples** and repeat the round-trip. "
        f"The original contains **{original_count:,g} cycles** "
        f"and the reloaded spectrum contains **{restored_count:,g}**. "
        "Compare the two distributions on the same display bins.",
    )
    return


if __name__ == "__main__":
    app.run()
