# 7. Simulate ordered crack growth { #ordered-plate-crack-propagation }

[Jump to the worked notebook](#worked-notebook)

The [plate crack tutorial](../../tutorials/30-ordered_plate_crack_growth.html)
follows a through crack and a surface crack from their initial profiles to
their final dimensions, using a NASGRO material fit and the same tensile
load history for both cases.

[Open lesson 33](../../33-ordered_crack_growth.html) to compare
both plate geometries using the same ordered load blocks. The plots align
applied stress, crack dimensions, and peak stress intensity on cumulative
cycles. To edit and rerun it locally, use
`uv run marimo edit notebooks/33-ordered_crack_growth.py`.

[Open lesson 34](../../34-synthetic_signal_to_failure.html) to generate a
long tensile waveform and follow it to a Miner damage limit and a
crack-growth toughness stop.

Use the family notebooks to run ordered crack propagation and overlay the
initial and last accepted profiles. [Through cracks · lesson 35](../../35-through_crack_growth.html)
· [Surface cracks · lesson 36](../../36-surface_crack_growth.html)

For an SC07 circular surface crack in a round bar, open lesson 36 or
follow the [same three tensile load blocks through MCP](../../mcp/07-propagate-sc07.md).

## Choose a geometry and tensile load { #what-each-new-name-does }

Match the crack shape and loading direction to a supported case. Check
the dimensions and validity limits in the
[geometry reference](../../api/damage/ordered_crack_growth.md#what-each-new-name-does).
Supply the tensile load blocks in their original order.

<span id="plate-solutions-and-loads"></span>
<span id="through-cracks-in-thin-shells"></span>
<span id="surface-cracks-in-pipes-bars-threads-and-plate-holes"></span>
<span id="net-section-stress"></span>

## Material inputs

Use a calibrated NASGRO curve with parameters in MPa, mm, and MPa·√mm.
See the [material parameter reference](../../api/damage/ordered_crack_growth.md#material-inputs).

## Solver controls and result

The following center-cracked plate example uses two tensile load blocks
and a synthetic NASGRO curve.

```python
from py_fatigue.crack_growth import (
    CenterThroughCrack,
    LoadBlock,
    NasgroCurve,
    simulate_crack_growth,
)

geometry = CenterThroughCrack(half_width=50, half_length=5)
curve = NasgroCurve(
    coefficient=1e-10, exponent=2, p=0.5, q=1,
    threshold_zero=25, threshold_fanning=0, toughness=500,
    constraint=2, peak_to_flow=0.3,
)
loads = [LoadBlock(0, 40, 8.5), LoadBlock(10, 50, 10)]
result = simulate_crack_growth(
    geometry, curve, loads,
    max_cycle_step=1, length_tolerance=1e-6,
    cycle_tolerance=1e-4, max_trials=40,
)
print(result.termination, result.final_cycles)
```

Expect `spectrum_exhausted` at 18.5 cycles. Compare the last accepted
geometry with the initial flaw. The
[solver reference](../../api/damage/ordered_crack_growth.md#solver-controls-and-result)
describes the controls, output conventions, and validation cases.

## Plot the geometry, material law and result

Call `geometry.plot()` or `geometry.echarts()` to draw the initial plate
crack. Pass `result=result` to overlay the last accepted profile.
Surface cracks are drawn in the crack plane of the plate, pipe, bar, or
thread. If a run stops before accepting a state, the plot shows the
starting geometry.

```python
fig, ax = geometry.plot(result=result)
geometry.echarts(result=result)
```

Call `curve.plot()` or `curve.echarts()` to show the long-crack reference
curve at stress ratio `R = 0` (`k_min = 0`). Both sweep maximum stress
intensity up to toughness and mark that limit with a dashed line. Supply
`length` to apply the finite-crack correction to the growth threshold and
`k_min` to change the minimum stress intensity.

```python
fig, ax = curve.plot()
curve.echarts()

fig, ax = curve.plot(length=4.0, k_min=20.0, scale="loglog")

fig, ax = result.plot()
result.echarts()
```

A scalar `k_max` sets a shorter sweep endpoint. Supply an array to evaluate
exact sample points, or a one-element array to show one point. With explicit
samples, `k_min` can be one value or an array paired with `k_max`.
Use `scale="semilogx"` to show zero growth below threshold, or
`scale="semilogy"` for the default logarithmic growth-rate axis.

To compare results on shared axes, use `other_result.plot(ax=ax,
label="Surface crack")`. The plot uses the accepted points in
`result.history`.

## Worked notebook

[Open notebook](../../33-ordered_crack_growth.html) ·
[Edit the notebook locally](08-example-notebooks.md)

<iframe class="docs-notebook" src="../../33-ordered_crack_growth.html" title="Ordered crack-growth analysis" loading="lazy"></iframe>
