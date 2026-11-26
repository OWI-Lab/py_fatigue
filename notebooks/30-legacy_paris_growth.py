"""Use Paris and Walker curves and compare legacy propagation geometries."""

import marimo

__generated_with = "0.26.0"
app = marimo.App()


@app.cell
def introduction():
    import marimo as mo

    mo.md("""
        # Paris and Walker curves and legacy growth

        Compare Paris and Walker rates using the same synthetic material
        coefficients. The Walker example specifies a stress ratio and
        changes the rate as a function of stress intensity range.

        Then use the Paris curve with two geometries under the same two
        stress ranges. The depth and stress intensity in the lower panels
        are the values recorded at the start of each step.
        """)
    return (mo,)


@app.cell
def calculate(mo):
    import matplotlib.pyplot as plt
    import numpy as np
    from py_fatigue import CycleCount, ParisCurve, WalkerCurve
    from py_fatigue.damage import get_crack_growth
    from py_fatigue.geometry import HollowCylinder, InfiniteSurface
    from py_fatigue.plotting import to_echarts

    material = dict(slope=3.0, intercept=1e-10, threshold=20.0, critical=2000.0)
    paris = ParisCurve(**material)
    walker = WalkerCurve(**material, load_ratio=0.1, walker_exponent=0.5)
    cycles = CycleCount(
        stress_range=np.array([50.0, 70.0]),
        mean_stress=np.zeros(2),
        count_cycle=np.array([3.0, 2.0]),
    )
    geometries = {
        "Infinite surface": InfiniteSurface(initial_depth=1.0),
        "External cylinder crack": HollowCylinder(
            initial_depth=1.0,
            outer_diameter=100.0,
            thickness=10.0,
            height=100.0,
            crack_position="external",
        ),
    }

    fig, (law, depth, intensity) = plt.subplots(
        3, 1, figsize=(7, 9), layout="constrained"
    )
    paris.plot(ax=law)
    law.lines[-1].set_label("Paris")
    walker.plot(ax=law)
    law.lines[-1].set_label(f"Walker · R = {walker.load_ratio:g}")
    law.set(ylabel="Growth rate (mm/cycle)")
    law.legend()

    summary = []
    for label, crack in geometries.items():
        result = get_crack_growth(cycles, paris, crack)
        result.plot(ax=depth)
        depth.lines[-1].set_label(label)
        saved_steps = len(result.crack_depth)
        step_start = np.r_[
            0.0, np.cumsum(result.count_cycle[: saved_steps - 1])
        ]
        intensity.plot(step_start, result.sif, "o-", label=label)
        summary.append(
            {
                "Geometry": label,
                "Cycles processed": result.final_cycles,
                "Last recorded depth (mm)": float(result.crack_depth[-1]),
                "Last geometry factor": float(result.geometry_factor[-1]),
            }
        )
    depth.set_ylabel("Crack depth (mm)")
    depth.legend()
    intensity.set(xlabel="Cycles at start of step", ylabel="ΔK (MPa √mm)")
    intensity.legend()

    mo.vstack(
        [
            mo.ui.tabs(
                {
                    "Static": fig,
                    "ECharts": mo.iframe(
                        to_echarts(fig).render_embed(), height="1240px"
                    ),
                }
            ),
            mo.ui.table(summary),
        ]
    )
    return paris, geometries


@app.cell
def long_case(paris, geometries):
    import matplotlib.pyplot as _plt
    import numpy as _np
    from _examples.synthetic import (
        MINIMUM_STRESS,
        display_indices,
        long_case_view,
        synthetic_peaks,
    )
    from py_fatigue import CycleCount as _CycleCount
    from py_fatigue.damage import get_crack_growth as _get_crack_growth

    peaks = synthetic_peaks()
    spectrum = _CycleCount(
        stress_range=peaks - MINIMUM_STRESS,
        mean_stress=(peaks + MINIMUM_STRESS) / 2,
        count_cycle=_np.ones_like(peaks),
    )
    long_fig, ax = _plt.subplots()
    outcomes = []
    for long_label, geometry in geometries.items():
        growth = _get_crack_growth(spectrum, paris, geometry)
        indices = display_indices(growth.crack_depth.size)
        ax.plot(indices, growth.crack_depth[indices], label=long_label)
        outcome = "failure" if growth.failure else "spectrum completed"
        outcomes.append(
            f"- **{long_label}** · {outcome} after "
            f"{growth.final_cycles:,.0f} cycles."
        )
    ax.set(
        xlabel="Cycles at start of step",
        ylabel="Crack depth (mm)",
        yscale="log",
    )
    ax.legend()
    long_case_view(
        long_fig,
        f"Apply the Paris curve to **{peaks.size:,} generated cycles** in "
        "order. Each input row represents one cycle. Compare how the two "
        "geometries grow and where the legacy analysis stops.\n\n"
        + "\n".join(outcomes),
    )
    return


if __name__ == "__main__":
    app.run()
