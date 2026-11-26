"""Reconstruct the handbook FAC-1 Mode I stress-intensity example."""

import marimo

__generated_with = "0.26.0"
app = marimo.App(width="medium")


@app.cell
def introduction():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from _examples.afgrow import FAC1_CRACK_RATIOS, FAC1_SIF, fac1_totals
    from py_fatigue.plotting import to_echarts

    mo.md("""
        # FAC-1 · Stress intensity at a loaded hole

        A through crack starts at a pin-loaded hole in a plate. Figure
        FAC-1.1 specifies a 10 in plate width, a centered 1 in hole,
        a 1 in thickness, a 10 ksi bypass stress, and a 10 kip pin force.
        The pin pressure follows a cosine distribution over half the hole.
        The nominal bearing stress P/(Dt) is 10 ksi.

        Reconstruct Example 1 from its Mode I results for NASGRO TC03 and
        two FRANC2D meshes. The inputs here are the published K components
        for each crack length. They form a reference for a future py-fatigue
        TC03 geometry that supports both load components.
        [Handbook problem FAC-1](https://www.afgrow.net/applications/DTDHandbook/Examples/page1_0.aspx)
        """)
    return FAC1_CRACK_RATIOS, FAC1_SIF, fac1_totals, mo, np, plt, to_echarts


@app.cell
def compare(FAC1_CRACK_RATIOS, FAC1_SIF, fac1_totals, mo, np, plt, to_echarts):
    _fig, (_components, _totals, _errors) = plt.subplots(
        3, 1, figsize=(7, 9), layout="constrained"
    )
    _reconstructed = fac1_totals()
    _baseline = FAC1_SIF["NASGRO TC03"][2]
    _rows = []
    for _label, (_bypass, _bearing, _published) in FAC1_SIF.items():
        (_line,) = _components.plot(
            FAC1_CRACK_RATIOS, _bypass, "o-", label=f"{_label} · bypass"
        )
        _components.plot(
            FAC1_CRACK_RATIOS,
            _bearing,
            "o--",
            color=_line.get_color(),
            label=f"{_label} · bearing",
        )
        _totals.plot(
            FAC1_CRACK_RATIOS, _reconstructed[_label], "o-", label=_label
        )
        _deviation = 100 * (_published / _baseline - 1)
        if _label != "NASGRO TC03":
            _errors.plot(FAC1_CRACK_RATIOS, _deviation, "o-", label=_label)
        _rows.append(
            {
                "Source": _label,
                "Max |sum − printed total| (ksi √in)": round(
                    np.max(np.abs(_reconstructed[_label] - _published)), 2
                ),
                "Mean signed difference from NASGRO (%)": round(
                    np.mean(_deviation), 2
                ),
                "Mean absolute difference from NASGRO (%)": round(
                    np.mean(np.abs(_deviation)), 2
                ),
            }
        )
    _components.set(ylabel="Mode I components (ksi √in)")
    _totals.set(ylabel="Reconstructed Kᵢ (ksi √in)")
    _errors.set(ylabel="Difference from NASGRO (%)")
    for _ax in (_components, _totals, _errors):
        _ax.set_xlabel("Crack length / hole diameter, c/D")
        _ax.legend(fontsize="small")
    mo.vstack(
        [
            mo.md("""
                ## Superpose bypass and pin loading

                Add the K values from bypass tension and pin bearing at
                each crack length. Compare the sum with the published total.
                Each component and total is rounded separately in the
                handbook. For the mesh comparison, use the printed totals
                and divide each difference by the NASGRO total at the same
                crack length. The table reports both the signed mean and
                the mean magnitude, which measures discrepancies even when
                errors have opposite signs.
                [Coarse-mesh table](https://www.afgrow.net/applications/DTDHandbook/Examples/images/page1_1/image014.gif)
                · [Refined-mesh table](https://www.afgrow.net/applications/DTDHandbook/Examples/images/page1_1/image016.gif)
                """),
            mo.ui.table(_rows),
            mo.ui.tabs(
                {
                    "Static": _fig,
                    "ECharts": mo.iframe(
                        to_echarts(_fig).render_embed(), height="1150px"
                    ),
                }
            ),
        ]
    )


@app.cell
def long_case(FAC1_CRACK_RATIOS, FAC1_SIF, np, plt):
    from _examples.synthetic import (
        display_indices,
        long_case_view,
        synthetic_peaks,
    )

    _peaks = synthetic_peaks()
    _scale = _peaks / _peaks.max()
    _row = np.searchsorted(FAC1_CRACK_RATIOS, 0.45)
    _bypass, _bearing, _ = FAC1_SIF["NASGRO TC03"][:, _row]
    _bypass_peaks = _bypass * _scale
    _bearing_peaks = _bearing * _scale
    _total_peaks = _bypass_peaks + _bearing_peaks
    _indices = display_indices(_peaks.size)

    _fig, _ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for _label, _values in (
        ("Bypass", _bypass_peaks),
        ("Bearing", _bearing_peaks),
        ("Total", _total_peaks),
    ):
        _ax.plot(_indices + 1, _values[_indices], label=_label)
    _ax.set(xlabel="Applied cycles", ylabel="Peak Kᵢ (ksi √in)")
    _ax.legend()
    long_case_view(
        _fig,
        "Hold c/D at 0.45 and scale both reference loads together using "
        "each of the package's 300,000 synthetic peaks divided by the "
        "maximum peak. Calculate the combined intensity at every peak "
        "before sampling for display. The crack length is fixed.",
        height="460px",
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
