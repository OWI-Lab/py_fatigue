"""Check the handbook's blend-out repair life using Py-Fatigue."""

import marimo

__generated_with = "0.26.0"
app = marimo.App(css_file="../../docs/stylesheets/custom.css")


@app.cell
def inputs():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from notebooks._examples.afgrow import crack_history, native_paris_growth
    from notebooks._examples.synthetic import (
        display_indices,
        plot_views,
        synthetic_peaks,
    )
    from py_fatigue import ParisCurve
    from py_fatigue.geometry import TabulatedGeometryFactor

    mo.md("""
        # AFGROW handbook · Edge crack after a blend-out repair

        Example 9.6.1 compares an as-manufactured edge with a shallow
        machined repair in a 7079-T6 aluminum angle. Both start with
        a 0.050 in crack measured from the local surface. The published
        load is 0–27 ksi. The repair removes 0.080 in of material and
        has a short-crack stress concentration factor of 1.16.

        Use the handbook's Paris fit and its reported terminal crack
        sizes of 0.325 in and 0.245 in. Compare the reported lives,
        the analytic integral and Py-Fatigue's cycle-by-cycle result.
        [USAF DTD Handbook, Example 9.6.1](https://www.afgrow.net/applications/DTDHandbook/pdfs/sec9_6.pdf)
        """)
    coefficient, exponent = 5.84e-10, 4.09
    curve = ParisCurve(exponent, coefficient, unit_string="ksi √in")
    stress = 27.0
    initial = 0.050
    blueprint_limit = 0.325
    repair_depth = 0.080
    concentration = 1.16
    peaks = stress * synthetic_peaks() / 100
    return (
        TabulatedGeometryFactor,
        blueprint_limit,
        coefficient,
        concentration,
        crack_history,
        curve,
        display_indices,
        exponent,
        initial,
        mo,
        native_paris_growth,
        np,
        peaks,
        plot_views,
        plt,
        repair_depth,
        stress,
    )


@app.cell
def analysis(
    TabulatedGeometryFactor,
    blueprint_limit,
    coefficient,
    concentration,
    crack_history,
    curve,
    display_indices,
    exponent,
    initial,
    mo,
    native_paris_growth,
    np,
    peaks,
    plot_views,
    plt,
    repair_depth,
    stress,
):
    transition = repair_depth / (concentration**2 - 1)
    power = 1 - exponent / 2
    edge_factor = 1.12

    def life_between(start, end):
        return (end**power - start**power) / (
            power
            * coefficient
            * (edge_factor * stress * np.sqrt(np.pi)) ** exponent
        )

    _fig, axes = plt.subplots(3, 2, figsize=(11, 11), layout="constrained")
    law, load, beta_plot, constant, variable, profile = axes.ravel()
    curve.plot(ax=law)
    law.set(ylabel="Growth rate (in/cycle)")
    samples = display_indices(peaks.size)
    load.plot(samples + 1, peaks[samples])
    load.axhline(stress, linestyle="--", label="Handbook range")
    load.set(xlabel="Applied cycles", ylabel="Stress range (ksi)")
    load.legend()
    rows = []
    for index, (case, offset, published) in enumerate(
        [("As manufactured", 0.0, 2910), ("Blend-out", repair_depth, 2033)]
    ):
        limit = blueprint_limit - offset
        length = np.linspace(initial, limit, 1001)
        factor = np.full_like(length, edge_factor)
        reference = life_between(initial, length)
        printed_life = reference[-1]
        if offset:
            length = np.union1d(length, [transition])
            factor = edge_factor * np.minimum(
                concentration, np.sqrt((length + offset) / length)
            )
            short_life = life_between(initial, np.minimum(length, transition))
            long_life = life_between(
                transition + offset, np.maximum(length, transition) + offset
            )
            reference = short_life / concentration**exponent + long_life
            printed_life = short_life[-1] / concentration**2 + long_life[-1]

        geometry = TabulatedGeometryFactor(initial, length, factor)
        (line,) = beta_plot.plot(length, factor, label=case)
        color = line.get_color()
        native_constant = native_paris_growth(
            geometry, curve, np.full_like(peaks, stress)
        )
        native_variable = native_paris_growth(geometry, curve, peaks)
        history, _ = crack_history(length, reference, peaks / stress, exponent)
        saved = display_indices(history.size)
        constant.plot(reference, length, "--", color=color)
        variable.plot(saved, history[saved], "--", color=color)
        for axis, result in (
            (constant, native_constant),
            (variable, native_variable),
        ):
            depth_history = result.crack_depth
            if result.termination == "geometry_limit":
                depth_history = np.r_[depth_history, limit]
            saved = display_indices(depth_history.size)
            axis.plot(saved, depth_history[saved], color=color, label=case)

        final = (
            limit
            if native_variable.termination == "geometry_limit"
            else native_variable.crack_depth[-1]
        )
        profile.plot(
            [offset, offset + final],
            [index, index],
            "-",
            color=color,
            label=case,
        )
        profile.plot(
            [offset, offset + initial],
            [index, index],
            "k--",
            label="Initial" if index == 0 else "_initial",
        )
        profile.plot([offset], [index], "|", color="black", markersize=12)
        rows.append(
            {
                "Geometry": case,
                "Handbook reported cycles": published,
                "Handbook printed integral": round(printed_life),
                "Paris equation integral": round(reference[-1]),
                "Py-Fatigue cycles": int(native_constant.final_cycles),
                "Variable Py-Fatigue cycles": int(native_variable.final_cycles),
            }
        )
    beta_plot.set(xlabel="Crack length (in)", ylabel="Geometry factor")
    constant.set(xlabel="Constant-amplitude cycles", ylabel="Crack length (in)")
    variable.set(xlabel="Variable-amplitude cycles", ylabel="Crack length (in)")
    profile.set(
        xlabel="Distance from original edge (in)",
        title="Crack profiles · bars mark local edges",
        yticks=[0, 1],
        yticklabels=["Original", "Repair"],
    )
    for axis in (beta_plot, constant, variable, profile):
        axis.legend(fontsize="small")
    mo.vstack(
        [
            mo.md(f"""
                ## Check the published life calculation

                The handbook uses a Paris fit to data at R = 0.05
                for this R = 0 example. Its short-crack factor is
                1.12 kₜ. Its long-crack factor is 1.12 √((a + d) / a).
                They meet at **{transition:.3f} in**.

                Substitution in the Paris law makes the short-crack
                life scale by 1/kₜⁿ. The printed repair integral uses
                1/kₜ². The table evaluates both expressions, using the
                handbook’s reported terminal sizes. The printed
                integrals also differ from the reported cycle counts.

                Dashed lines show the analytic histories using 1/kₜⁿ.
                Solid lines show Py-Fatigue. The variable-load run
                makes **{peaks.size:,} synthetic cycles** available and
                uses the same crack-length endpoints for comparison.
                """),
            mo.ui.table(rows),
            plot_views(_fig),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
