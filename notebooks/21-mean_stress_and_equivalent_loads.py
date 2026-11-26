"""Compare mean-stress corrections and their effect on equivalent loads."""

import marimo

__generated_with = "0.26.0"
app = marimo.App()


@app.cell
def introduction():
    import marimo as mo

    mo.md("""
        # Mean stress and equivalent loads

        Add two windows of a synthetic stress history and resolve the
        residuals. Apply two mean-stress corrections independently to that
        combined spectrum. Use one S–N curve to compute Miner damage for the
        original and both corrected spectra.

        The equivalent stress ranges and moments use the same S–N slope and
        reference cycle count. The moment example uses a hollow circular
        section with radii specified in metres.
        """)
    return (mo,)


@app.cell
def compare(mo):
    from datetime import datetime, timedelta, timezone

    import matplotlib.pyplot as plt
    import numpy as np
    from py_fatigue import CycleCount, SNCurve
    from py_fatigue.damage import get_dem, get_des, get_pm
    from py_fatigue.plotting import to_echarts

    stress = (
        10
        * np.array(
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
        - 30
    )
    split = len(stress) // 2
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    first = CycleCount.from_timeseries(stress[:split], timestamp=start)
    second = CycleCount.from_timeseries(
        stress[split:], timestamp=start + timedelta(seconds=split)
    )
    spectrum = (first + second).solve_lffd()
    corrections = {
        "Original": spectrum,
        "DNVGL-RP-C203": spectrum.mean_stress_correction(
            "DNVGL-RP-C203", detail_factor=0.8
        ),
        "Smith–Watson–Topper": spectrum.mean_stress_correction("SWT"),
    }
    sn = SNCurve(slope=3, intercept=12)
    reference_cycles = spectrum.count_cycle.sum()
    outer_radius, inner_radius = 0.5, 0.4

    fig, (ranges, damage) = plt.subplots(2, 1, layout="constrained")
    summary = []
    for label, cc in corrections.items():
        contributions = get_pm(cc, sn)
        ranges.plot(spectrum.stress_range, cc.stress_range, "o", label=label)
        damage.plot(spectrum.stress_range, contributions, "o", label=label)
        des = get_des(cc, sn.slope[0], reference_cycles)
        dem = get_dem(
            outer_radius, inner_radius, cc, sn.slope[0], reference_cycles
        )
        summary.append(
            {
                "Correction": label,
                "Miner damage": float(contributions.sum()),
                "DES (MPa)": float(des),
                "DEM (kNm)": float(dem / 1000),
            }
        )

    ranges.set(
        xlabel="Original range (MPa)", ylabel="Range after correction (MPa)"
    )
    damage.set(
        xlabel="Original range (MPa)", ylabel="Miner damage contribution"
    )
    ranges.legend()
    damage.legend()
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
                f"DES and DEM use **{reference_cycles:g} reference cycles**."
            ),
            mo.ui.table(summary),
        ]
    )
    return sn, outer_radius, inner_radius


@app.cell
def long_case(sn, outer_radius, inner_radius):
    import matplotlib.pyplot as _plt
    import numpy as _np
    from _examples.synthetic import (
        count_windows,
        long_case_view,
        synthetic_peaks,
        tensile_waveform,
    )
    from py_fatigue.damage import (
        get_dem as _get_dem,
        get_des as _get_des,
        get_pm as _get_pm,
    )

    peaks = synthetic_peaks()
    combined = count_windows(tensile_waveform(peaks)).solve_lffd()
    corrected = combined.mean_stress_correction("SWT")
    edges = _np.histogram_bin_edges(
        _np.r_[combined.stress_range, corrected.stress_range], bins=60
    )
    centres = (edges[:-1] + edges[1:]) / 2
    long_fig, ax = _plt.subplots()
    totals = []
    long_reference_cycles = combined.count_cycle.sum()
    for long_label, long_spectrum in (
        ("Original", combined),
        ("SWT", corrected),
    ):
        long_damage = _get_pm(long_spectrum, sn)
        by_range, _ = _np.histogram(
            long_spectrum.stress_range, bins=edges, weights=long_damage
        )
        ax.plot(centres, by_range, label=long_label)
        long_des = _get_des(long_spectrum, sn.slope[0], long_reference_cycles)
        long_dem = _get_dem(
            outer_radius,
            inner_radius,
            long_spectrum,
            sn.slope[0],
            long_reference_cycles,
        )
        totals.append(
            f"- **{long_label}** · damage {long_damage.sum():.4g}, "
            f"DES {long_des:.2f} MPa, DEM {long_dem / 1000:.2f} kNm."
        )
    ax.set(xlabel="Stress range (MPa)", ylabel="Miner damage per range bin")
    ax.legend()
    long_case_view(
        long_fig,
        f"Combine and resolve the **{peaks.size:,}-cycle** waveform, then "
        "apply SWT to the resulting spectrum. Compare damage and equivalent "
        "loads with the same S–N curve and reference cycle count.\n\n"
        + "\n".join(totals),
    )
    return


if __name__ == "__main__":
    app.run()
