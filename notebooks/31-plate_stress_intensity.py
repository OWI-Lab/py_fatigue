"""Example of plate stress intensity."""

import marimo

__generated_with = "0.26.0"
app = marimo.App(css_file="../docs/stylesheets/custom.css")


@app.cell
def introduction():
    import marimo as mo

    mo.md("""
        # Plate stress intensity

        Compare the tip of a center crack with the deepest and free-surface
        points of a surface crack under tension up to 60 MPa. The slope of
        each line shows how much the intensity changes per MPa of stress.
        The sketches show where each intensity is evaluated. Then hold the
        stress and half-length fixed and increase the surface crack depth to
        see how the two front points respond.
        """)
    return (mo,)


@app.cell
def calculate(mo):
    from dataclasses import replace

    import matplotlib.pyplot as plt
    import numpy as np
    from py_fatigue.crack_growth import (
        CenterThroughCrack,
        SemiEllipticalSurfaceCrack,
    )
    from _examples.charts import plot_views

    center = CenterThroughCrack(half_width=50.0, half_length=4.0)
    surface = SemiEllipticalSurfaceCrack(
        half_width=50.0, thickness=15.0, depth=4.0, half_length=8.0
    )
    stress = [0.0, 60.0]
    center_tip = [center.stress_intensity(value) for value in stress]
    deepest, surface_tip = zip(
        *(surface.stress_intensity(value) for value in stress)
    )

    fig, (ax, sweep) = plt.subplots(2, 1, layout="constrained")
    ax.plot(stress, center_tip, "o-", label="Center crack tip")
    ax.plot(stress, deepest, "o-", label="Surface crack, deepest point")
    ax.plot(
        stress, surface_tip, "o-", label="Surface crack, free-surface point"
    )
    ax.set(xlabel="Tensile stress (MPa)", ylabel="Stress intensity (MPa √mm)")
    ax.legend()

    depths = np.linspace(surface.depth, surface.half_length, 25)
    deepest_by_depth, surface_by_depth = zip(
        *(
            replace(surface, depth=float(depth)).stress_intensity(stress[-1])
            for depth in depths
        )
    )
    sweep.plot(
        depths / surface.thickness, deepest_by_depth, label="Deepest point"
    )
    sweep.plot(
        depths / surface.thickness, surface_by_depth, label="Free surface"
    )
    sweep.set(
        xlabel="Crack depth / plate thickness",
        ylabel="Stress intensity (MPa √mm)",
        title=f"Surface crack at {stress[-1]:g} MPa",
    )
    sweep.legend()

    plot_views(fig, height="830px")
    return center, surface


@app.cell
def geometry_views(center, surface):
    import matplotlib.pyplot as _plt
    from _examples.charts import plot_views as _plot_views

    profiles, (center_ax, surface_ax) = _plt.subplots(
        2, 1, figsize=(7, 7), layout="constrained"
    )
    center.plot(fig=profiles, ax=center_ax)
    surface.plot(fig=profiles, ax=surface_ax)
    _plot_views(profiles, height="830px")
    return


@app.cell
def long_case(center, surface):
    import matplotlib.pyplot as _plt
    import numpy as _np
    from _examples.synthetic import long_case_view, synthetic_peaks

    peaks = synthetic_peaks()
    depth_factor, surface_factor = surface.stress_intensity(1.0)
    intensities = {
        "Center tip": peaks * center.stress_intensity(1.0),
        "Surface crack · deepest point": peaks * depth_factor,
        "Surface crack · free surface": peaks * surface_factor,
    }
    edges = _np.histogram_bin_edges(
        _np.concatenate(list(intensities.values())), bins=60
    )
    centres = (edges[:-1] + edges[1:]) / 2
    long_fig, long_ax = _plt.subplots()
    for label, intensity in intensities.items():
        counts, _ = _np.histogram(intensity, bins=edges)
        long_ax.plot(centres, counts, label=label)
    long_ax.set(
        xlabel="Peak stress intensity (MPa √mm)", ylabel="Cycles per bin"
    )
    long_ax.legend()
    long_case_view(
        long_fig,
        f"Keep both crack geometries fixed and evaluate **{peaks.size:,} "
        "generated stress peaks** at their tips. The distribution shows "
        "how the same tensile loading maps to different peak intensities.",
    )
    return


if __name__ == "__main__":
    app.run()
