"""Example of nasgro growth rate."""

import marimo

__generated_with = "0.26.0"
app = marimo.App(css_file="../docs/stylesheets/custom.css")


@app.cell
def introduction():
    import marimo as mo

    mo.md("""
        # NASGRO growth rate

        `curve.plot()` and `curve.echarts()` show the long-crack reference at
        stress ratio R = 0. Both sweep maximum stress intensity up to the
        dashed toughness limit. The semilog X panel shows zero growth below
        threshold. The log–log panel shows the positive rates.

        The last panel applies the short-crack threshold correction at a
        crack length of 4 mm. Set `k_min = R * k_max` for each sample to
        compare two stress ratios at the same maximum intensity.
        """)
    return (mo,)


@app.cell
def calculate(mo):
    import matplotlib.pyplot as plt
    import numpy as np
    from py_fatigue.crack_growth import NasgroCurve
    from _examples.charts import plot_views

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
    length = 4.0
    fig, (semilog, loglog, ratios) = plt.subplots(
        3, 1, figsize=(7, 9), layout="constrained"
    )
    curve.plot(ax=semilog, scale="semilogx")
    curve.plot(ax=loglog, scale="loglog")
    semilog.set_title("Semilog X · includes zero growth")
    loglog.set_title("Log–log · positive growth rates")
    semilog.legend()

    k_max = np.linspace(0.0, curve.toughness, 201)[1:-1]
    for ratio in (0.0, 0.4):
        curve.plot(length, ratio * k_max, k_max, ax=ratios, scale="loglog")
        ratios.lines[-1].set_label(f"R = {ratio:g}")
    ratios.axvline(
        curve.toughness, color="0.4", linestyle="--", label="Toughness"
    )
    ratios.set_title("Compare fixed stress ratios")
    ratios.legend()

    plot_views(fig, height="1240px")
    return curve, length


@app.cell
def long_case(curve, length):
    import matplotlib.pyplot as _plt
    import numpy as _np
    from _examples.synthetic import (
        MINIMUM_STRESS,
        long_case_view,
        synthetic_peaks,
    )
    from py_fatigue.crack_growth import CenterThroughCrack

    peaks = synthetic_peaks()
    geometry = CenterThroughCrack(half_width=50.0, half_length=length)
    factor = geometry.stress_intensity(1.0)
    k_min = MINIMUM_STRESS * factor
    long_k_max = peaks * factor
    rates = _np.fromiter(
        (curve.get_growth_rate(length, k_min, value) for value in long_k_max),
        dtype=float,
        count=peaks.size,
    )
    positive = rates[rates > 0]
    zero_count = peaks.size - positive.size
    long_fig, ax = _plt.subplots()
    if positive.size:
        edges = _np.geomspace(positive.min(), positive.max(), 61)
        counts, _ = _np.histogram(positive, bins=edges)
        ax.semilogx(_np.sqrt(edges[:-1] * edges[1:]), counts)
    else:
        ax.set_title("Zero growth across this history")
    ax.set(xlabel="Growth rate (mm/cycle)", ylabel="Cycles per rate bin")
    long_case_view(
        long_fig,
        f"Evaluate the NASGRO law for **{peaks.size:,} generated cycles** "
        f"with the center crack held at a **{length:g} mm** half-length. "
        "The histogram shows positive rates. "
        f"**{zero_count:,} cycles** give zero growth at this crack length.",
    )
    return


if __name__ == "__main__":
    app.run()
