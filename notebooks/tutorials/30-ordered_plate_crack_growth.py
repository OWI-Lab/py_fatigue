"""Assess a surface crack and a through crack under ordered tensile loads."""

import marimo

__generated_with = "0.26.0"
app = marimo.App(css_file="../../docs/stylesheets/custom.css")


@app.cell
def inputs():
    import marimo as mo
    import matplotlib.pyplot as plt
    from notebooks._examples.synthetic import (
        MINIMUM_STRESS,
        display_indices,
        plot_views,
        synthetic_peaks,
    )
    from py_fatigue.crack_growth import (
        CenterThroughCrack,
        LoadBlock,
        NasgroCurve,
        SemiEllipticalSurfaceCrack,
        simulate_crack_growth,
    )

    mo.md("""
        # Ordered fatigue crack growth in a plate

        Assess two initial defects in a 100 mm wide, 10 mm thick plate.
        The center through crack advances one half-length. The
        semi-elliptical surface crack advances in depth and along the
        surface. Both see the same nominal tensile stress history.

        The material fit and loading are synthetic. Compare the
        last admissible profiles and the reasons each run stopped.
        """)
    geometries = {
        "TC01": CenterThroughCrack(half_width=50.0, half_length=5.0),
        "SC01": SemiEllipticalSurfaceCrack(
            half_width=50.0, thickness=10.0, depth=2.0, half_length=10.0
        ),
    }
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
    peaks = synthetic_peaks()
    loads = [LoadBlock(MINIMUM_STRESS, float(peak), 1.0) for peak in peaks]
    return (
        MINIMUM_STRESS,
        curve,
        display_indices,
        geometries,
        loads,
        mo,
        peaks,
        plot_views,
        plt,
        simulate_crack_growth,
    )


@app.cell
def material_and_load(
    MINIMUM_STRESS, curve, display_indices, mo, peaks, plot_views, plt
):
    _fig, (_law, _load) = plt.subplots(
        2, 1, figsize=(8, 7), layout="constrained"
    )
    curve.plot(ax=_law, scale="loglog")
    _indices = display_indices(peaks.size)
    _load.plot(_indices + 1, peaks[_indices], label="Peak")
    _load.axhline(MINIMUM_STRESS, linestyle="--", label="Minimum")
    _load.set(xlabel="Applied cycles", ylabel="Tensile stress (MPa)")
    _load.legend()
    mo.vstack(
        [
            mo.md(
                f"## Material and load history\n\nApply **{peaks.size:,} cycles** in "
                "generator order. The material plot shows the long-crack reference "
                "at R = 0. The solver uses each tip’s actual minimum and maximum K."
            ),
            plot_views(_fig),
        ]
    )
    return


@app.cell
def analysis(
    curve,
    display_indices,
    geometries,
    loads,
    mo,
    plot_views,
    plt,
    simulate_crack_growth,
):
    _fig, _axes = plt.subplots(2, 3, figsize=(15, 7), layout="constrained")
    rows = []
    for (case, geometry), (profile, evolution, intensity) in zip(
        geometries.items(), _axes
    ):
        result = simulate_crack_growth(
            geometry,
            curve,
            loads,
            max_cycle_step=1.0,
            length_tolerance=1e-6,
            cycle_tolerance=1e-4,
            max_trials=2 * len(loads),
        )
        geometry.plot(ax=profile, result=result)
        points = [
            result.history[i] for i in display_indices(len(result.history))
        ]
        cycles = [point.cycles for point in points]
        for index, label in enumerate(result.dimension_names):
            evolution.plot(
                cycles,
                [point.sizes[index] for point in points],
                label=label,
            )
            intensity.plot(
                cycles, [point.k_max[index] for point in points], label=label
            )
        intensity.axhline(
            curve.toughness,
            color="firebrick",
            linestyle="--",
            label="Toughness",
        )
        evolution.set(
            title=f"{case} crack dimensions",
            xlabel="Applied cycles",
            ylabel="Crack dimension (mm)",
        )
        intensity.set(
            title=f"{case} peak stress intensity",
            xlabel="Applied cycles",
            ylabel="Kₘₐₓ (MPa √mm)",
        )
        evolution.legend()
        intensity.legend()
        rows.append(
            {
                "Case": case,
                "Cycles accepted": round(result.final_cycles, 2),
                "Stop": result.termination.replace("_", " "),
                "Final dimensions (mm)": ", ".join(
                    f"{size:.3f}" for size in result.history[-1].sizes
                ),
            }
        )
    mo.vstack(
        [
            mo.md("""
            ## Propagate and interpret the stop

            The solver checks both toughness and geometry limits as the
            cracks grow. The trial budget scales with the supplied load
            count and allows step refinement. The profiles show the
            last accepted step, which is on the admissible side of a
            terminal boundary.
            """),
            mo.ui.table(rows),
            plot_views(_fig),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
