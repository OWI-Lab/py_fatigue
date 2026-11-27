"""Propagate surface cracks in plates, pipes, bars and bolt threads."""

import marimo

__generated_with = "0.26.0"
app = marimo.App(width="medium", css_file="../docs/stylesheets/custom.css")


@app.cell
def introduction():
    from math import ceil

    import marimo as mo
    import matplotlib.pyplot as plt
    from _examples.charts import plot_views

    mo.md("""
        # Surface-crack growth

        Start with a finite-width plate and a semi-elliptical surface
        crack (SC01). Under ordered tensile loads, the deepest point and
        the free-surface point grow at different rates. The solver records
        the **depth** and **surface half-length** in that order.

        Overlay the initial crack front and the last accepted front on the
        plate cross section. Depth is measured downward from the free
        surface. The material and loads are synthetic.

        In the round-bar case (SC07), only the deepest point advances.
        The crack front stays circular and meets the bar surface at right angles.

        The additional cases show an external pipe crack (SC05), a
        rolled bolt thread (SC08), and symmetric cracks at a plate hole
        (SC11). In each case, use the stress specified by its geometry.
        """)
    return ceil, mo, plt, plot_views


@app.cell
def propagate(ceil):
    from py_fatigue.crack_growth import (
        LoadBlock,
        NasgroCurve,
        SemiEllipticalSurfaceCrack,
        simulate_crack_growth,
    )

    geometry = SemiEllipticalSurfaceCrack(
        half_width=50.0, thickness=15.0, depth=4.0, half_length=8.0
    )
    curve = NasgroCurve(1e-10, 2.0, 0.5, 1.0, 25.0, 0.0, 500.0, 2.0, 0.3)
    loads = [
        LoadBlock(10.0, 70.0, 50_000),
        LoadBlock(20.0, 90.0, 25_005),
        LoadBlock(10.0, 60.0, 25_000),
    ]
    step = 10.0
    trial_budget = 2 * sum(ceil(block.cycles / step) for block in loads)
    result = simulate_crack_growth(
        geometry,
        curve,
        loads,
        max_cycle_step=step,
        length_tolerance=1e-6,
        cycle_tolerance=1e-4,
        max_trials=trial_budget,
    )
    return (
        LoadBlock,
        curve,
        geometry,
        loads,
        result,
        simulate_crack_growth,
        step,
    )


@app.cell
def ordered_profile(geometry, loads, mo, plt, result, plot_views):
    fig, (profile, history) = plt.subplots(
        2, 1, figsize=(7, 6), layout="constrained"
    )
    geometry.plot(ax=profile, result=result)
    result.plot(ax=history)
    net_stress = geometry.net_section_stress(loads[0].maximum)
    mo.vstack(
        [
            mo.md(
                f"**{result.final_cycles:,.0f} accepted cycles**. "
                f"Stop: {result.termination.replace('_', ' ')}. "
                f"Initial net-section stress at the first peak is {net_stress:.1f} MPa."
            ),
            plot_views(fig, height="840px"),
        ]
    )


@app.cell
def round_bar_case(
    ceil,
    curve,
    loads,
    mo,
    plt,
    simulate_crack_growth,
    plot_views,
):
    from py_fatigue.crack_growth import RoundBarSurfaceCrack

    bar = RoundBarSurfaceCrack(diameter=100.0, depth=4.0)
    bar_step = 10.0
    bar_result = simulate_crack_growth(
        bar,
        curve,
        loads,
        max_cycle_step=bar_step,
        length_tolerance=1e-6,
        cycle_tolerance=1e-4,
        max_trials=2 * sum(ceil(block.cycles / bar_step) for block in loads),
    )
    bar_fig, (bar_profile, bar_history) = plt.subplots(
        2, 1, figsize=(7, 8), layout="constrained"
    )
    bar.plot(ax=bar_profile, result=bar_result)
    bar_result.plot(ax=bar_history)
    mo.vstack(
        [
            mo.md(
                f"## SC07 under the same ordered loads\n\n"
                f"Last accepted profile at **{bar_result.final_cycles:,.0f} "
                f"cycles**. Stop: {bar_result.termination.replace('_', ' ')}."
            ),
            plot_views(bar_fig, height="840px"),
        ]
    )


@app.cell
def additional_geometries():
    from py_fatigue.crack_growth import (
        CircumferentialCylinderSurfaceCrack,
        HoleBoreSurfaceCrack,
        RolledBoltThreadSurfaceCrack,
    )

    surface_cases = {
        "SC05 · Pipe": CircumferentialCylinderSurfaceCrack(
            inner_radius=20.0, thickness=10.0, depth=4.0, half_length=5.0
        ),
        "SC08 · Rolled thread": RolledBoltThreadSurfaceCrack(
            major_diameter=12.0, minor_diameter=10.0, depth=1.5
        ),
        "SC11 · Hole bore": HoleBoreSurfaceCrack(
            width=80.0,
            thickness=12.0,
            hole_diameter=12.0,
            depth=2.2,
            half_length=2.5,
            crack_count=2,
        ),
    }
    return (surface_cases,)


@app.cell
def additional_growth(
    ceil,
    curve,
    loads,
    mo,
    plt,
    simulate_crack_growth,
    step,
    surface_cases,
    plot_views,
):
    _cases = {}
    for _name, _geometry in surface_cases.items():
        _result = simulate_crack_growth(
            _geometry,
            curve,
            loads,
            max_cycle_step=step,
            length_tolerance=1e-6,
            cycle_tolerance=1e-4,
            max_trials=2 * sum(ceil(block.cycles / step) for block in loads),
        )
        _fig, (_profile, _history) = plt.subplots(
            2, 1, figsize=(7, 8), layout="constrained"
        )
        _geometry.plot(ax=_profile, result=_result)
        _result.plot(ax=_history)
        _cases[_name] = mo.vstack(
            [
                mo.md(
                    f"Accepted **{_result.final_cycles:,.0f} cycles** · "
                    f"{_result.termination.replace('_', ' ')}."
                ),
                plot_views(_fig, height="840px"),
            ]
        )
    mo.vstack(
        [
            mo.md("""
                ## Pipe, thread and plate-hole cases

                - **SC05** uses axial stress on the uncracked pipe section.
                  It advances the radial depth and outer-surface arc half-length.
                - **SC08** uses nominal axial stress on the thread-root
                  section. The arc half-length equals the crack depth.
                - **SC11** uses remote gross-section plate stress. It
                  advances radial depth and half-length along the bore.
                  The two cracks grow symmetrically.

                The same numbers are used for the three stress histories
                to illustrate the geometry effects. Each solver stops at
                toughness, at its geometry limit, or when the loads end.
                """),
            mo.ui.tabs(_cases),
        ]
    )


@app.cell
def long_case(
    LoadBlock,
    curve,
    geometry,
    plt,
    simulate_crack_growth,
    surface_cases,
):
    from _examples.synthetic import (
        MINIMUM_STRESS,
        long_case_view,
        synthetic_peaks,
    )

    peaks = synthetic_peaks()
    schedule = [LoadBlock(MINIMUM_STRESS, float(peak), 1.0) for peak in peaks]
    long_fig, _axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
    _summary = []
    for (_name, _geometry), _ax in zip(
        {"SC01 · Plate": geometry, **surface_cases}.items(), _axes.flat
    ):
        _result = simulate_crack_growth(
            _geometry,
            curve,
            schedule,
            max_cycle_step=1.0,
            length_tolerance=1e-6,
            cycle_tolerance=1e-4,
            max_trials=2 * len(schedule),
        )
        _geometry.plot(ax=_ax, result=_result)
        _summary.append(
            f"- **{_name}** · {_result.final_cycles:,.2f} cycles, "
            f"{_result.termination.replace('_', ' ')}."
        )
    long_case_view(
        long_fig,
        f"Apply up to **{peaks.size:,} tensile cycles** to each geometry in "
        "their generated order, and compare the last accepted profiles.\n\n"
        + "\n".join(_summary),
        height="1640px",
    )


if __name__ == "__main__":
    app.run()
