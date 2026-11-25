"""Interactive development examples for Py-Fatigue."""

import marimo

__generated_with = "0.26.0"
app = marimo.App()


@app.cell
def imports():
    import marimo as mo
    import numpy as np
    from py_fatigue import SNCurve, __version__
    from py_fatigue.cycle_count.rainflow import rainflow

    return SNCurve, __version__, mo, np, rainflow


@app.cell
def introduction(__version__, mo):
    mo.md(
        f"""
        # Developing with Py-Fatigue

        Running Py-Fatigue {__version__}. Prototype in a notebook, move reusable
        code into the package with docstrings, and add tests for its behavior.
        """
    )
    return


@app.cell
def compare_rainflow(mo, np, rainflow):
    signal = np.array(
        [4, 7, 2, 10, 5, 9, 3, 4, 2, 12, 5, 11, 1, 4, 3, 10, 6, 12, 4, 8, 1, 9, 4, 6],
        dtype=float,
    )
    mo.vstack(
        [
            mo.md("## Compare ASTM and four-point rainflow"),
            mo.ui.tabs(
                {
                    label: mo.vstack(list(rainflow(signal, method=method, extended_output=True)))
                    for label, method in [("ASTM", "astm"), ("Four-point", "fourpoint")]
                }
            ),
        ]
    )
    return


@app.cell
def sn_curves(SNCurve, mo):
    curves = [
        SNCurve(
            slope=slope,
            intercept=intercept,
            norm="DNVGL-RP-C203",
            environment="Free corrosion",
            curve=label,
        )
        for slope, intercept, label in [
            (3, 12.436, "B1"),
            ([3], 12.115, "C"),
            (3, [11.533], "E"),
            ([3], [10.493], "W3-C"),
        ]
    ]
    mo.vstack([mo.md("## S–N curves"), *curves])
    return


if __name__ == "__main__":
    app.run()
