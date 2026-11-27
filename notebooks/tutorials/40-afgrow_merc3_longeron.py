"""Reproduce the MERC-3 longeron life and apply a variable load history."""

import marimo

__generated_with = "0.26.0"
app = marimo.App(css_file="../../docs/stylesheets/custom.css")


@app.cell
def model():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from notebooks._examples.afgrow import (
        MERC3_COEFFICIENT,
        MERC3_DIGITIZED,
        MERC3_EXPONENT,
        MERC3_FINAL_LENGTH,
        MERC3_INITIAL_LENGTH,
        MERC3_PUBLISHED_CYCLES,
        MERC3_STRESS_RANGE,
        crack_history,
        merc3_geometry_factors,
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
        # AFGROW MERC-3 · Life of an aircraft longeron

        Follow the through-crack path from a fastener hole toward the
        longeron edge. Define the geometry by its published β(a) data
        and use the 7075-T6 aluminum Paris fit at R = 0 from Figure
        MERC-3.1. Use inches and ksi throughout the analysis.

        Reconstruct the handbook's constant-range lives independently
        and run the same cases with Py-Fatigue's crack-growth solver.
        Compare both methods again under a seeded variable-amplitude
        history, using 1.25 in as the end of each geometry table.
        [USAF DTD Handbook, MERC-3](https://www.afgrow.net/applications/DTDHandbook/Examples/page2_2.aspx)
        """)
    curve = ParisCurve(
        slope=MERC3_EXPONENT, intercept=MERC3_COEFFICIENT, unit_string="ksi √in"
    )
    length = np.linspace(MERC3_INITIAL_LENGTH, MERC3_FINAL_LENGTH, 1201)
    factors = merc3_geometry_factors(length)
    # The generated peaks are in MPa. Use zero as the minimum for R = 0.
    stress_range = synthetic_peaks() / 6.894757293168
    return (
        MERC3_COEFFICIENT,
        MERC3_DIGITIZED,
        MERC3_EXPONENT,
        MERC3_PUBLISHED_CYCLES,
        MERC3_STRESS_RANGE,
        TabulatedGeometryFactor,
        crack_history,
        curve,
        display_indices,
        factors,
        length,
        mo,
        native_paris_growth,
        np,
        paris_reference_cycles,
        plot_views,
        plt,
        stress_range,
    )


@app.cell
def results(
    MERC3_COEFFICIENT,
    MERC3_DIGITIZED,
    MERC3_EXPONENT,
    MERC3_PUBLISHED_CYCLES,
    MERC3_STRESS_RANGE,
    TabulatedGeometryFactor,
    crack_history,
    curve,
    display_indices,
    factors,
    length,
    mo,
    native_paris_growth,
    np,
    paris_reference_cycles,
    plot_views,
    plt,
    stress_range,
):
    _fig, axes = plt.subplots(3, 2, figsize=(11, 11), layout="constrained")
    curve.plot(ax=axes[0, 0], scale="loglog")
    axes[0, 0].set(ylabel="Growth rate (in/cycle)")
    indices = display_indices(stress_range.size)
    axes[0, 1].plot(indices + 1, stress_range[indices])
    axes[0, 1].set(xlabel="Applied cycles", ylabel="Stress range (ksi)")
    geometry, constant, variable, profile = (
        axes[1, 0],
        axes[1, 1],
        axes[2, 0],
        axes[2, 1],
    )
    geometry.plot(*MERC3_DIGITIZED.T, "k.", label="Digitized samples")
    reference_stress = np.full_like(stress_range, MERC3_STRESS_RANGE)
    rows = []
    for i, (case, geometry_factor) in enumerate(factors.items()):
        reference_k = (
            MERC3_STRESS_RANGE * geometry_factor * np.sqrt(np.pi * length)
        )
        reference = paris_reference_cycles(
            length, reference_k, MERC3_COEFFICIENT, MERC3_EXPONENT
        )
        factor_table = TabulatedGeometryFactor(
            length[0], length, geometry_factor
        )
        native_constant = native_paris_growth(
            factor_table, curve, reference_stress
        )
        native_variable = native_paris_growth(factor_table, curve, stress_range)
        history, reached = crack_history(
            length, reference, stress_range / MERC3_STRESS_RANGE, MERC3_EXPONENT
        )
        saved = display_indices(history.size)
        (beta_line,) = geometry.plot(length, geometry_factor, label=case)
        color = beta_line.get_color()
        constant.plot(reference, length, "--", color=color)
        variable.plot(saved, history[saved], "--", color=color)
        for axis, result in (
            (constant, native_constant),
            (variable, native_variable),
        ):
            depths = result.crack_depth
            if result.termination == "geometry_limit":
                depths = np.r_[depths, length[-1]]
            native_samples = display_indices(depths.size)
            axis.plot(
                native_samples, depths[native_samples], color=color, label=case
            )
        final_length = (
            length[-1]
            if native_variable.termination == "geometry_limit"
            else native_variable.crack_depth[-1]
        )
        profile.plot(
            [0, final_length], [i, i], color=color, label=f"{case} final"
        )
        profile.plot(
            [0, length[0]],
            [i, i],
            "k--",
            label="Initial" if i == 0 else "_initial",
        )
        published = MERC3_PUBLISHED_CYCLES[case]
        rows.append(
            {
                "Case": case,
                "Handbook cycles": published,
                "Reconstructed cycles": round(reference[-1]),
                "Py-Fatigue cycles": int(native_constant.final_cycles),
                "Py-Fatigue vs handbook (%)": round(
                    100 * (native_constant.final_cycles / published - 1), 2
                ),
                "Variable reference cycles": history.size - 1,
                "Variable Py-Fatigue cycles": int(native_variable.final_cycles),
                "Variable reference outcome": (
                    "Target length" if reached else "Spectrum ended"
                ),
                "Variable Py-Fatigue stop": native_variable.termination.replace(
                    "_", " "
                ),
            }
        )
    geometry.set(xlabel="Crack length (in)", ylabel="Geometry factor β")
    constant.set(xlabel="Constant-range cycles", ylabel="Crack length (in)")
    variable.set(xlabel="Applied cycles", ylabel="Crack length (in)")
    profile.set(
        xlabel="Distance from crack origin (in)", title="Crack path schematic"
    )
    profile.yaxis.set_visible(False)
    for axis in (geometry, constant, variable, profile):
        axis.legend(fontsize="small")
    mo.vstack(
        [
            mo.md("""
            ## Compare the handbook, numerical reconstruction and Py-Fatigue

            Interpolate the digitized β curve and hold the last sample
            from 1.235 to 1.25 in. Apply the handbook's 10% perturbations
            to the small and large β values. Compute ΔK at the 10 ksi
            reference range and integrate the inverse growth rate in
            crack length. Py-Fatigue instead advances the crack one cycle
            at a time with `get_crack_growth` and a
            `TabulatedGeometryFactor`. Dashed lines show the independent
            reconstruction. Solid lines show Py-Fatigue.

            The variable-amplitude run advances that reference history
            by (Δσ / 10 ksi)ⁿ equivalent cycles per applied cycle. This
            follows from the single-slope Paris law at R = 0. The two
            inputs are separable because ΔK is proportional to stress
            range for a given crack length. The full 300,000-cycle
            spectrum is available to each case.

            MERC-1’s full center-wing prediction uses an unpublished load
            spectrum and material curve, plus Willenborg retardation.
            MERC-3 publishes the inputs needed for this life comparison.
            [MERC-1 model description](https://www.afgrow.net/applications/DTDHandbook/Examples/page2_0.aspx)
            """),
            mo.ui.table(rows),
            plot_views(_fig),
        ]
    )


if __name__ == "__main__":
    app.run()
