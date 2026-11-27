"""Apply the published FAC-1 stress intensities to a complete Paris analysis."""

import marimo

__generated_with = "0.26.0"
app = marimo.App(css_file="../../docs/stylesheets/custom.css")


@app.cell
def model():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from notebooks._examples.afgrow import (
        FAC1_CRACK_RATIOS,
        FAC1_SIF,
        MERC3_COEFFICIENT,
        MERC3_EXPONENT,
        crack_history,
        fac1_totals,
        native_paris_growth,
        paris_reference_cycles,
    )
    from notebooks._examples.synthetic import (
        display_indices,
        plot_views,
        synthetic_peaks,
    )
    from py_fatigue import ParisCurve
    from py_fatigue.geometry import TabulatedGeometryFactor

    mo.md("""
        # AFGROW FAC-1 · From a loaded-hole K table to crack growth

        Use the 10 in wide, 1 in thick plate and centered 1 in hole from
        FAC-1 Example 1. At the reference load, the bypass stress is
        10 ksi and the pin force is 10 kip. A single through crack grows
        radially from the hole toward the plate edge.

        FAC-1 supplies the stress intensities. This tutorial adds an
        illustrative life analysis using the independently published
        7075-T6 Paris fit from MERC-3. The reported lives compare the
        three tabular K inputs under this tutorial's load history.
        Compare the published K values with the Py-Fatigue geometry
        model, then compare two independently integrated growth histories.
        [FAC-1 geometry and K tables](https://www.afgrow.net/applications/DTDHandbook/Examples/page1_0.aspx)
        · [MERC-3 material fit](https://www.afgrow.net/applications/DTDHandbook/Examples/page2_2.aspx)
        """)
    width, diameter = 10.0, 1.0  # in
    bypass_reference = 10.0  # ksi
    bearing_reference = 10.0  # kip applied through pin
    curve = ParisCurve(
        slope=MERC3_EXPONENT, intercept=MERC3_COEFFICIENT, unit_string="ksi √in"
    )
    # Cycle peaks scale both reference loads. Minima are zero.
    load_fraction = synthetic_peaks() / 100.0
    table_length = FAC1_CRACK_RATIOS * diameter
    length = np.union1d(
        np.linspace(table_length[0], table_length[-1], 1001), table_length
    )
    return (
        FAC1_SIF,
        MERC3_COEFFICIENT,
        MERC3_EXPONENT,
        TabulatedGeometryFactor,
        bearing_reference,
        bypass_reference,
        crack_history,
        curve,
        diameter,
        display_indices,
        fac1_totals,
        length,
        load_fraction,
        mo,
        native_paris_growth,
        np,
        paris_reference_cycles,
        plot_views,
        plt,
        table_length,
        width,
    )


@app.cell
def analysis(
    FAC1_SIF,
    MERC3_COEFFICIENT,
    MERC3_EXPONENT,
    TabulatedGeometryFactor,
    bearing_reference,
    bypass_reference,
    crack_history,
    curve,
    diameter,
    display_indices,
    fac1_totals,
    length,
    load_fraction,
    mo,
    native_paris_growth,
    np,
    paris_reference_cycles,
    plot_views,
    plt,
    table_length,
    width,
):
    _fig, axes = plt.subplots(3, 2, figsize=(11, 11), layout="constrained")
    curve.plot(ax=axes[0, 0], scale="loglog")
    axes[0, 0].set(ylabel="Growth rate (in/cycle)")
    samples = display_indices(load_fraction.size)
    axes[0, 1].plot(samples + 1, load_fraction[samples])
    axes[0, 1].set(
        xlabel="Applied cycles", ylabel="Fraction of reference loads"
    )
    components, totals, growth, profile = (
        axes[1, 0],
        axes[1, 1],
        axes[2, 0],
        axes[2, 1],
    )
    radius = diameter / 2
    angle = np.linspace(0, 2 * np.pi, 101)
    profile.plot(
        radius * np.cos(angle), radius * np.sin(angle), "k-", label="Hole"
    )
    profile.axvline(width / 2, color="grey", label="Plate edge")
    profile.plot(
        [radius, radius + length[0]],
        [0, 0],
        "k--",
        label="Initial crack",
        zorder=5,
    )

    rows = []
    for case, total in fac1_totals().items():
        published_total = FAC1_SIF[case][2]
        delta_k = np.interp(length, table_length, total)
        reference = paris_reference_cycles(
            length, delta_k, MERC3_COEFFICIENT, MERC3_EXPONENT
        )
        factors = delta_k / (bypass_reference * np.sqrt(np.pi * length))
        geometry = TabulatedGeometryFactor(length[0], length, factors)
        result = native_paris_growth(
            geometry, curve, bypass_reference * load_fraction
        )
        history, reached = crack_history(
            length, reference, load_fraction, MERC3_EXPONENT
        )
        saved = display_indices(history.size)
        sampled = display_indices(result.crack_depth.size)
        native_k = result.sif / load_fraction[: result.sif.size]
        (k_line,) = totals.plot(
            result.crack_depth[sampled], native_k[sampled], label=case
        )
        color = k_line.get_color()
        totals.plot(length, delta_k, "--", color=color)
        totals.plot(
            table_length, published_total, "o", color=color, markersize=3
        )
        growth.plot(saved, history[saved], "--", color=color)
        depths = result.crack_depth
        if result.termination == "geometry_limit":
            depths = np.r_[depths, length[-1]]
        native_samples = display_indices(depths.size)
        growth.plot(
            native_samples, depths[native_samples], color=color, label=case
        )
        profile.plot(
            [radius, radius + depths[-1]],
            [0, 0],
            label=f"{case} final",
            alpha=0.65,
            color=color,
        )
        rows.append(
            {
                "K source": case,
                "Max rounding difference (ksi √in)": round(
                    float(np.max(np.abs(total - published_total))), 2
                ),
                "Max Py-Fatigue K difference (ksi √in)": round(
                    float(
                        np.max(
                            np.abs(
                                native_k
                                - np.interp(
                                    result.crack_depth, table_length, total
                                )
                            )
                        )
                    ),
                    6,
                ),
                "Reference cycles": history.size - 1,
                "Py-Fatigue cycles": int(result.final_cycles),
                "Py-Fatigue − reference (cycles)": int(result.final_cycles)
                - (history.size - 1),
                "Py-Fatigue last length (in)": round(depths[-1], 3),
                "Reference outcome": (
                    "End of K table" if reached else "Spectrum ended"
                ),
                "Py-Fatigue stop": result.termination.replace("_", " "),
            }
        )
    bypass, bearing, _ = FAC1_SIF["NASGRO TC03"]
    components.plot(table_length, bypass, "o-", label="Bypass")
    components.plot(table_length, bearing, "o-", label="Bearing")
    components.set(xlabel="Crack length (in)", ylabel="Component K (ksi √in)")
    totals.set(xlabel="Crack length (in)", ylabel="Combined K (ksi √in)")
    growth.set(xlabel="Applied cycles", ylabel="Crack length (in)")
    profile.set(
        xlabel="Position across plate (in)",
        ylabel="Position along plate (in)",
        aspect="equal",
        title="Hole and crack path · plan view",
    )
    for axis in (components, totals, growth, profile):
        axis.legend(fontsize="small")
    mo.vstack(
        [
            mo.md(f"""
            ## Superpose the loads and integrate growth

            Each of the **{load_fraction.size:,} cycles** scales the
            {bypass_reference:g} ksi bypass stress and {bearing_reference:g}
            kip pin force by the same fraction. The minima are zero, so
            the scaled sum of the two K components is ΔK. The two
            FRANC2D meshes and NASGRO TC03 give three reference curves.
            The dots show the handbook's separately rounded totals.
            Divide each Py-Fatigue K value by its applied load fraction
            to compare the history at the same reference load.

            Integrate cycles versus crack length at full reference load.
            For each applied cycle, advance the reference history by
            the load fraction raised to the Paris exponent. For
            Py-Fatigue, express the combined K table as a dimensionless
            geometry factor against the reference bypass stress. Run
            `get_crack_growth` with the resulting factor table and the
            applied bypass ranges. The tabulated factor incorporates
            pin bearing because both loads scale together.

            Dashed lines show the independent integration. Solid lines
            show Py-Fatigue's cycle-by-cycle integration. Both stop at
            c/D = 1.35 or at the end of the spectrum. The K table
            controls the range of this analysis. The end of the table
            is a model limit, with no inferred failure criterion.
            """),
            mo.ui.table(rows),
            plot_views(_fig),
        ]
    )


if __name__ == "__main__":
    app.run()
