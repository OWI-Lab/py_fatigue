"""Compare whole-signal counting with segmented cycle count aggregation."""

import marimo

__generated_with = "0.26.0"
app = marimo.App()


@app.cell
def introduction():
    import marimo as mo

    mo.md("""
        # Add cycle counts and recover residuals

        Count a sampled history in two windows, add their `CycleCount` objects,
        then call `solve_lffd()` to recount the turning points left over at
        the window boundaries. Compare with counting the whole signal at once.

        The example uses a short synthetic signal. Check how both the spectrum
        and Miner damage change when the residuals are resolved.
        """)
    return (mo,)


@app.cell
def compare(mo):
    from datetime import datetime, timedelta, timezone

    import matplotlib.pyplot as plt
    import numpy as np
    from py_fatigue import CycleCount, SNCurve
    from py_fatigue.damage import get_pm
    from py_fatigue.plotting import to_echarts

    stress = np.array(
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
    split = len(stress) // 2
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    first = CycleCount.from_timeseries(stress[:split], timestamp=start)
    second = CycleCount.from_timeseries(
        stress[split:], timestamp=start + timedelta(seconds=split)
    )
    combined = first + second
    recovered = combined.solve_lffd()
    whole = CycleCount.from_timeseries(stress, timestamp=start)
    curve = SNCurve(slope=3, intercept=12)

    fig, (signal, spectrum) = plt.subplots(2, 1, layout="constrained")
    signal.plot(stress, marker="o")
    signal.axvline(
        split - 0.5, color="0.4", linestyle="--", label="Window split"
    )
    signal.set(xlabel="Sample at 1 Hz", ylabel="Stress (MPa)")
    signal.legend()

    summary = []
    for label, cc in (
        ("Added windows", combined),
        ("Residuals resolved", recovered),
        ("Whole signal", whole),
    ):
        ranges, bins = np.unique(cc.stress_range, return_inverse=True)
        counts = np.bincount(bins, weights=cc.count_cycle)
        spectrum.plot(ranges, np.cumsum(counts[::-1])[::-1], "o-", label=label)
        summary.append(
            {
                "Spectrum": label,
                "Remaining turning points": len(cc.residuals_sequence),
                "Miner damage": float(get_pm(cc, curve).sum()),
            }
        )
    spectrum.set(
        xlabel="Stress range (MPa)", ylabel="Cycles at this range or higher"
    )
    spectrum.legend()

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
            mo.ui.table(summary),
        ]
    )


@app.cell
def long_case():
    import matplotlib.pyplot as _plt
    import numpy as _np
    from _examples.synthetic import (
        count_windows,
        long_case_view,
        synthetic_peaks,
        tensile_waveform,
    )
    from py_fatigue import CycleCount as _CycleCount, SNCurve as _SNCurve
    from py_fatigue.damage import get_pm as _get_pm

    peaks = synthetic_peaks()
    waveform = tensile_waveform(peaks)
    long_combined = count_windows(waveform)
    resolved = long_combined.solve_lffd()
    reference = _CycleCount.from_timeseries(waveform)
    edges = _np.histogram_bin_edges(reference.stress_range, bins=60)
    centres = (edges[:-1] + edges[1:]) / 2
    sn = _SNCurve(slope=3, intercept=10.493)
    reference_damage, _ = _np.histogram(
        reference.stress_range, bins=edges, weights=_get_pm(reference, sn)
    )
    long_fig, ax = _plt.subplots()
    for long_label, long_spectrum in (
        ("Added windows", long_combined),
        ("Residuals resolved", resolved),
    ):
        damage_by_range, _ = _np.histogram(
            long_spectrum.stress_range,
            bins=edges,
            weights=_get_pm(long_spectrum, sn),
        )
        ax.plot(centres, damage_by_range - reference_damage, label=long_label)
    ax.set(
        xlabel="Stress range (MPa)",
        ylabel="Miner damage difference from whole-signal count",
    )
    ax.legend()
    long_case_view(
        long_fig,
        f"Count two consecutive pieces of a **{peaks.size:,}-cycle** waveform. "
        "Compare the damage in each range bin against the whole-signal result. "
        f"The whole-signal damage is **{reference_damage.sum():.5f}**.",
    )
    return


if __name__ == "__main__":
    app.run()
