"""Propagate an external pipe crack under repeated axial loads."""

import marimo

__generated_with = "0.26.0"
app = marimo.App(css_file="../../docs/stylesheets/custom.css")


@app.cell
def inputs():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from notebooks._examples.synthetic import (
        display_indices,
        plot_views,
        synthetic_peaks,
    )
    from py_fatigue import CycleCount, ParisCurve
    from py_fatigue.damage import get_crack_growth
    from py_fatigue.geometry import HollowCylinder

    mo.md("""
        # Paris growth of an external pipe crack

        Define the pipe dimensions and the initial flaw.
        Convert a synthetic axial force history to nominal stress using
        the gross annulus area. Feed one row per known tensile cycle to
        the legacy Paris solver and inspect the recorded crack surfaces.

        The dimensions, load history and Paris coefficients are
        illustrative. The legacy pipe model keeps its crack width-to-depth
        ratio fixed as depth increases.
        """)
    geometry = HollowCylinder(
        initial_depth=1.0,
        outer_diameter=100.0,
        thickness=10.0,
        height=100.0,
        crack_position="external",
    )
    curve = ParisCurve(
        slope=3.0, intercept=1e-10, threshold=20.0, critical=2000.0
    )
    outer_radius = geometry.outer_diameter / 2
    area = np.pi * (outer_radius**2 - (outer_radius - geometry.thickness) ** 2)
    force_max = synthetic_peaks()  # kN
    force_min = 10.0  # kN
    minimum = force_min * 1e3 / area
    maximum = force_max * 1e3 / area
    cycles = CycleCount(
        stress_range=maximum - minimum,
        mean_stress=(maximum + minimum) / 2,
        count_cycle=np.ones_like(maximum),
        unit="MPa",
    )
    return (
        curve,
        cycles,
        display_indices,
        geometry,
        get_crack_growth,
        maximum,
        minimum,
        mo,
        plot_views,
        plt,
    )


@app.cell
def analysis(
    curve,
    cycles,
    display_indices,
    geometry,
    get_crack_growth,
    maximum,
    minimum,
    mo,
    plot_views,
    plt,
):
    result = get_crack_growth(cycles, curve, geometry)
    final_depth = float(result.crack_depth[-1])
    _fig, axes = plt.subplots(3, 2, figsize=(11, 11), layout="constrained")
    curve.plot(ax=axes[0, 0])
    indices = display_indices(maximum.size)
    axes[0, 1].plot(indices + 1, maximum[indices], label="Peak")
    axes[0, 1].axhline(minimum, linestyle="--", label="Minimum")
    axes[0, 1].set(xlabel="Cycle", ylabel="Axial stress (MPa)")
    axes[0, 1].legend()

    saved = display_indices(result.crack_depth.size)
    axes[1, 0].plot(saved, result.crack_depth[saved])
    axes[1, 1].plot(saved, result.sif[saved])
    axes[1, 0].set(xlabel="Cycle start", ylabel="Crack depth (mm)")
    axes[1, 1].set(xlabel="Cycle start", ylabel="ΔK (MPa √mm)")
    geometry.plot(ax=axes[2, 0])
    geometry.plot(ax=axes[2, 1], depth=final_depth)
    for profile, title in zip(
        axes[2], ("Initial crack", "Last recorded crack")
    ):
        profile.set(
            title=title, xlabel="Section x (mm)", ylabel="Section y (mm)"
        )

    outcome = (
        "the solver reported failure"
        if result.failure
        else "the spectrum ended"
    )
    mo.vstack(
        [
            mo.md(
                f"## Growth and stop\n\nOf **{maximum.size:,} prescribed cycles**, "
                f"the solver processed **{result.final_cycles:,.0f}** before {outcome}. "
                f"The last recorded start-of-step depth was **{final_depth:.3f} mm**. "
                "Compare the Paris curve, ΔK history, and the two cross sections."
            ),
            plot_views(_fig),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
