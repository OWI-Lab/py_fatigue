"""Compare ordered crack propagation in two plate geometries."""

import marimo

__generated_with = "0.26.0"
app = marimo.App()


@app.cell
def introduction():
    import marimo as mo

    mo.md(
        """
        # Ordered plate crack growth

        Apply the same two tensile membrane load blocks to a center crack
        and a semi-elliptical surface crack. The material and loads are
        synthetic teaching inputs. Use calibrated input data for engineering
        assessments.

        Stress is in MPa, crack dimensions are in mm, and stress intensity
        is in MPa √mm.
        """
    )
    return (mo,)


@app.cell
def propagate():
    from py_fatigue.crack_growth import (
        CenterThroughCrack,
        LoadBlock,
        NasgroCurve,
        SemiEllipticalSurfaceCrack,
        simulate_crack_growth,
    )

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
    loads = [LoadBlock(10.0, 70.0, 500.0), LoadBlock(20.0, 90.0, 250.5)]
    cases = {
        "Center crack": (CenterThroughCrack(50.0, 4.0), ("Half-length",)),
        "Surface crack": (
            SemiEllipticalSurfaceCrack(50.0, 15.0, 4.0, 8.0),
            ("Depth", "Half-length"),
        ),
    }
    results = {
        name: (
            simulate_crack_growth(
                geometry,
                curve,
                loads,
                max_cycle_step=10.0,
                length_tolerance=1e-6,
                cycle_tolerance=1e-4,
                max_trials=10_000,
            ),
            dimensions,
        )
        for name, (geometry, dimensions) in cases.items()
    }
    return (results,)


@app.cell
def show_results(mo, results):
    import plotly.graph_objects as go

    figure = go.Figure()
    summaries = []
    for name, (result, dimensions) in results.items():
        for index, dimension in enumerate(dimensions):
            figure.add_scatter(
                x=[point.cycles for point in result.history],
                y=[point.sizes[index] for point in result.history],
                mode="lines",
                name=f"{name} · {dimension}",
            )
        summaries.append(
            f"- **{name}** · {result.final_cycles:g} cycles, "
            f"{result.termination.replace('_', ' ')}"
        )
    figure.update_layout(xaxis_title="Cycles", yaxis_title="Crack dimension (mm)")
    mo.vstack([mo.md("\n".join(summaries)), figure])
    return


if __name__ == "__main__":
    app.run()
