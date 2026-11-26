"""Calculate stress-life damage from a tube's bending history."""

import marimo

__generated_with = "0.26.0"
app = marimo.App()


@app.cell
def introduction():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from notebooks._examples.synthetic import (
        count_windows,
        display_indices,
        plot_views,
        synthetic_peaks,
        tensile_waveform,
    )
    from py_fatigue import CycleCount, SNCurve
    from py_fatigue.damage import calc_pm, get_dem, get_des, get_pm

    mo.md("""
        # Stress-life assessment of a tube in bending

        Define an S–N curve for the detail, convert a bending-moment history
        to nominal stress, and rainflow count the signal in two windows.
        Compare Miner damage and equivalent loads with and without a
        Smith–Watson–Topper (SWT) mean-stress correction.

        The section and moment history are illustrative. The S–N inputs
        follow the W3-C demonstration in lesson 00. Use MPa for stress
        and metres for the tube radii.
        """)
    return (
        CycleCount,
        SNCurve,
        calc_pm,
        count_windows,
        display_indices,
        get_dem,
        get_des,
        get_pm,
        mo,
        np,
        plot_views,
        plt,
        synthetic_peaks,
        tensile_waveform,
    )


@app.cell
def material_geometry_and_load(
    SNCurve, mo, np, plot_views, plt, synthetic_peaks, tensile_waveform
):
    curve = SNCurve(slope=3.0, intercept=10.493)
    outer_radius, inner_radius = 0.15, 0.13
    section_modulus = (
        np.pi * (outer_radius**4 - inner_radius**4) / (4 * outer_radius)
    )
    # Use the seeded generator's values as moment peaks in kN m.
    moment = tensile_waveform(synthetic_peaks())
    stress = moment * 1e3 / section_modulus / 1e6

    _fig, (section, history, law) = plt.subplots(
        3, 1, figsize=(8, 9), layout="constrained"
    )
    angle = np.linspace(0, 2 * np.pi, 101)
    for radius, label in (
        (outer_radius, "Outer wall"),
        (inner_radius, "Inner wall"),
    ):
        section.plot(
            radius * np.cos(angle), radius * np.sin(angle), label=label
        )
    section.set(xlabel="Section x (m)", ylabel="Section y (m)", aspect="equal")
    section.legend()
    history.plot(stress[:41])
    history.set(
        xlabel="Sample · first 20 cycles", ylabel="Nominal stress (MPa)"
    )
    curve.plot(ax=law)
    mo.vstack(
        [
            mo.md(
                f"## Section, material and load\n\nSection modulus is **{section_modulus:.5f} m³**. "
                f"Count all **{stress.size:,} samples** using the elastic relation σ = M/W."
            ),
            plot_views(_fig),
        ]
    )
    return curve, inner_radius, outer_radius, stress


@app.cell
def damage_analysis(
    CycleCount,
    calc_pm,
    count_windows,
    curve,
    display_indices,
    get_dem,
    get_des,
    get_pm,
    inner_radius,
    mo,
    np,
    outer_radius,
    plot_views,
    plt,
    stress,
):
    combined = count_windows(stress).solve_lffd()
    whole = CycleCount.from_timeseries(stress)
    corrected = combined.mean_stress_correction("SWT")
    reference_cycles = float(whole.count_cycle.sum())
    slope = float(curve.slope[0])
    rows = []
    _fig, (spectrum, cumulative) = plt.subplots(
        2, 1, figsize=(8, 7), layout="constrained"
    )
    edges = np.histogram_bin_edges(
        np.r_[combined.stress_range, corrected.stress_range], bins=50
    )
    for case, counted in (("Raw", combined), ("SWT", corrected)):
        damage = get_pm(counted, curve)
        bins, _ = np.histogram(counted.stress_range, bins=edges, weights=damage)
        spectrum.plot((edges[:-1] + edges[1:]) / 2, bins, label=case)
        rows.append(
            {
                "Spectrum": case,
                "Miner damage": float(damage.sum()),
                "DES (MPa)": float(get_des(counted, slope, reference_cycles)),
                "DEM (kN m)": float(
                    get_dem(
                        outer_radius,
                        inner_radius,
                        counted,
                        slope,
                        reference_cycles,
                    )
                    / 1000
                ),
            }
        )
    # This waveform returns to the same minimum between all peaks.
    # Its ordered peak-to-minimum ranges give a separate Miner history.
    ranges = stress[1::2] - stress[0]
    damage_history = np.r_[
        0.0, np.cumsum(calc_pm(ranges, np.ones_like(ranges), curve))
    ]
    indices = display_indices(damage_history.size)
    cumulative.plot(
        indices, damage_history[indices], label="Raw ordered cycles"
    )
    cumulative.axhline(
        1.0, color="firebrick", linestyle="--", label="Miner D = 1"
    )
    spectrum.set(xlabel="Stress range (MPa)", ylabel="Miner damage per bin")
    cumulative.set(xlabel="Applied cycles", ylabel="Cumulative Miner damage")
    spectrum.legend()
    cumulative.legend()
    first_limit = np.searchsorted(damage_history, 1.0)
    limit = (
        f"Raw Miner damage reaches one at cycle **{first_limit:,}**."
        if first_limit < damage_history.size
        else "Raw Miner damage stays below one."
    )
    mo.vstack(
        [
            mo.md(
                f"## Count, correct and assess\n\nThe full-signal count gives "
                f"raw damage **{get_pm(whole, curve).sum():.4f}**. Compare it with "
                "the combined windows below. DES and DEM use the full-signal "
                f"count of **{reference_cycles:,.0f} reference cycles**. {limit}\n\n"
                "For the SWT row, Py-Fatigue uses √(maximum stress × stress "
                "amplitude) as the corrected range. On this spectrum that "
                "reduces the range used in the damage calculation."
            ),
            mo.ui.table(
                rows,
                format_mapping={
                    "Miner damage": "{:.4f}",
                    "DES (MPa)": "{:.2f}",
                    "DEM (kN m)": "{:.2f}",
                },
            ),
            plot_views(_fig),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
