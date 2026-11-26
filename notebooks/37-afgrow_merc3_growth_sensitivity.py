"""Reproduce the MERC-3 worked crack-growth example."""

import marimo

__generated_with = "0.26.0"
app = marimo.App(width="medium")


@app.cell
def introduction():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from py_fatigue.plotting import to_echarts
    from _examples.synthetic import display_indices

    mo.md("""
        # MERC-3 · Crack-growth sensitivity

        How much does an error in the geometry factor β change predicted
        life? Follow the handbook's constant-β derivation, then reproduce
        the aircraft longeron example with a variable β.

        Both calculations use the handbook's Paris fit for 7075-T6 Al
        at R = 0, with exponent 3.6 and coefficient 2.5 × 10⁻⁹ in the
        inch-cycle and ksi √in unit system. The stress range is 10 ksi.
        The crack grows from 0.05 to 1.25 in.
        [Handbook problem MERC-3](https://www.afgrow.net/applications/DTDHandbook/Examples/page2_2.aspx)
        """)
    return display_indices, mo, np, plt, to_echarts


@app.cell
def constant_factor(display_indices, mo, plt, to_echarts):
    from _examples.afgrow import merc3_constant_beta

    _fig, _ax = plt.subplots(figsize=(7, 4), layout="constrained")
    _rows = []
    for _beta in (1.0, 1.1):
        _exact, _result = merc3_constant_beta(_beta)
        _indices = display_indices(_result.crack_depth.size)
        _ax.plot(
            _indices, _result.crack_depth[_indices], label=f"β = {_beta:g}"
        )
        _rows.append(
            {
                "β": _beta,
                "Eq. (11), cycles": round(_exact, 1),
                "py-fatigue, cycles": int(_result.final_cycles),
                "Difference, cycles": round(_result.final_cycles - _exact, 1),
            }
        )
    _ax.set(xlabel="Cycles", ylabel="Crack length (in)")
    _ax.legend()
    _drop = 1 - _rows[1]["Eq. (11), cycles"] / _rows[0]["Eq. (11), cycles"]
    mo.vstack(
        [
            mo.md(
                "## Constant geometry factor\n\n"
                "With constant β, ΔK = β Δσ √(πa). The exact life in "
                "Eq. (11) comes from integrating the inverse Paris rate. "
                "In py-fatigue, the unit-factor `InfiniteSurface` takes the "
                "effective range β Δσ. Set `ParisCurve.critical` to the "
                "intensity at 1.25 in so the native solver reaches the same "
                "end criterion. Give it a stress history 1% longer than the "
                "analytical life to let the stop occur with one-cycle steps.\n\n"
                f"Increasing β by 10% shortens the exact life by "
                f"{_drop:.1%}. The handbook's −3.6 sensitivity gives "
                "the first-order estimate of a 36% reduction."
            ),
            mo.ui.table(_rows),
            mo.ui.tabs(
                {
                    "Static": _fig,
                    "ECharts": mo.iframe(
                        to_echarts(_fig).render_embed(), height="460px"
                    ),
                }
            ),
        ]
    )


@app.cell
def longeron(mo, np, plt, to_echarts):
    from _examples.afgrow import (
        MERC3_DIGITIZED,
        MERC3_FINAL_LENGTH,
        MERC3_INITIAL_LENGTH,
        MERC3_PUBLISHED_CYCLES,
        merc3_beta_histories,
    )

    length = np.linspace(MERC3_INITIAL_LENGTH, MERC3_FINAL_LENGTH, 1201)
    histories = merc3_beta_histories(length)
    _fig, (_beta_ax, _life_ax) = plt.subplots(
        2, 1, figsize=(7, 7), layout="constrained"
    )
    _rows = []
    for _case, (_beta, _cycles) in histories.items():
        _beta_ax.plot(length, _beta, label=_case)
        _life_ax.plot(_cycles, length, label=_case)
        _published = MERC3_PUBLISHED_CYCLES[_case]
        _rows.append(
            {
                "Case": _case,
                "Handbook cycles": _published,
                "Calculated cycles": round(_cycles[-1]),
                "Difference from handbook (%)": round(
                    100 * (_cycles[-1] / _published - 1), 2
                ),
            }
        )
    _beta_ax.plot(*MERC3_DIGITIZED.T, "k.", label="Digitized samples")
    _beta_ax.set(xlabel="Crack length (in)", ylabel="Geometry factor β")
    _life_ax.set(xlabel="Cycles", ylabel="Crack length (in)")
    _beta_ax.legend()
    _life_ax.legend()
    mo.vstack(
        [
            mo.md("""
                ## Variable β in the longeron

                Interpolate the samples digitized from Figure MERC-3.4.
                The final sample is at 1.235 in. Hold its β through 1.25 in.
                Follow the handbook's 10% increases at β ≥ 3 and β ≤ 1,
                with linear transitions between those values. Calculate
                each local rate with `ParisCurve.get_growth_rate` and
                integrate its reciprocal over crack length by the
                trapezoidal rule.

                Compare the resulting lives with Figure MERC-3.5. The
                small-β correction has a larger effect because the crack
                spends more cycles in that part of the history.
                [Digitized β source](https://www.afgrow.net/applications/DTDHandbook/Examples/images/page2_3/image055.gif)
                · [Published crack-growth lives](https://www.afgrow.net/applications/DTDHandbook/Examples/images/page2_3/image057.gif)
                """),
            mo.ui.table(_rows),
            mo.ui.tabs(
                {
                    "Static": _fig,
                    "ECharts": mo.iframe(
                        to_echarts(_fig).render_embed(), height="860px"
                    ),
                }
            ),
        ]
    )
    return histories, length


@app.cell
def long_case(display_indices, histories, length, np, plt):
    from _examples.afgrow import MERC3_EXPONENT, MERC3_STRESS_RANGE
    from _examples.synthetic import (
        MINIMUM_STRESS,
        long_case_view,
        synthetic_peaks,
    )

    _stress_range = (synthetic_peaks() - MINIMUM_STRESS) / 6.894757293168
    # A variable-amplitude cycle advances the constant-range history by
    # (Δσ / 10 ksi)^n equivalent cycles under this Paris model.
    _equivalent = np.r_[
        0.0, np.cumsum((_stress_range / MERC3_STRESS_RANGE) ** MERC3_EXPONENT)
    ]
    _fig, (_stress_ax, _life_ax) = plt.subplots(
        2, 1, figsize=(7, 7), layout="constrained"
    )
    _stress_indices = display_indices(_stress_range.size)
    _stress_ax.plot(_stress_indices + 1, _stress_range[_stress_indices])
    _stress_ax.set(xlabel="Applied cycles", ylabel="Stress range (ksi)")
    _summary = []
    for _case, (_, _reference_cycles) in histories.items():
        _reached = np.searchsorted(_equivalent, _reference_cycles[-1])
        _end = min(_reached, _stress_range.size)
        _indices = display_indices(_end + 1)
        _depth = np.interp(_equivalent[_indices], _reference_cycles, length)
        _life_ax.plot(_indices, _depth, label=_case)
        if _reached < _equivalent.size:
            _summary.append(f"**{_case}** reaches 1.25 in by cycle {_end:,}.")
    _life_ax.set(xlabel="Applied cycles", ylabel="Crack length (in)")
    _life_ax.legend()
    long_case_view(
        _fig,
        "Use the same β curves and R = 0 Paris fit with the package's "
        "300,000 synthetic stress ranges converted to ksi. Each range "
        "advances the constant-range model by (Δσ / 10 ksi)³·⁶ equivalent "
        "cycles. This follows from the separable Paris law. It treats "
        "growth as a function of ΔK alone. All cycles contribute to the "
        "calculation before sampling for display.\n\n" + " ".join(_summary),
        height="860px",
    )


if __name__ == "__main__":
    app.run()
