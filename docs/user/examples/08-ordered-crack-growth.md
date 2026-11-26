# Ordered plate crack propagation

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

## What each new name does

The following API is experimental in this release candidate. Each operation can be learned
independently before running a propagation analysis.

| Name | Inputs and output |
| --- | --- |
| `CenterThroughCrack(half_width, half_length)` | TC01 symmetric center crack. Its `stress_intensity(stress)` method returns one opening-mode intensity. Both dimensions are half-dimensions. |
| `EdgeThroughCrack(width, length)` | TC02 single edge crack. Give the full plate width and the distance from the edge to the tip. Its `stress_intensity(stress)` method returns one opening-mode intensity. |
| `SphereThroughCrack(mean_radius, thickness, half_length)` | TC06 through-wall arc on a great circle. The half-length follows the mid-wall arc. |
| `AxialCylinderThroughCrack(mean_radius, thickness, half_length)` | TC07 longitudinal through-wall crack. Use hoop membrane stress to open the crack. |
| `CircumferentialCylinderThroughCrack(mean_radius, thickness, half_length, length, poisson)` | TC08 through-wall arc in a thin cylinder. The half-length follows the mid-wall arc and `length` is the full axial cylinder length. Use axial membrane stress. |
| `SemiEllipticalSurfaceCrack(half_width, thickness, depth, half_length)` | SC01 surface crack. Its `stress_intensity(stress)` method returns a pair ordered as deepest point, free surface. |
| `CircumferentialCylinderSurfaceCrack(inner_radius, thickness, depth, half_length)` | SC05 external pipe crack under axial tension. `half_length` follows the outer-surface arc. Intensities are ordered as deepest point, free surface. |
| `RoundBarSurfaceCrack(diameter, depth)` | SC07 circular surface crack in a solid bar. Its `stress_intensity(stress)` method returns one deepest-point intensity for remote tension. |
| `RolledBoltThreadSurfaceCrack(major_diameter, minor_diameter, depth)` | SC08 thumbnail in a rolled thread. The single intensity is at the deepest point. |
| `HoleBoreSurfaceCrack(width, thickness, hole_diameter, depth, half_length, crack_count)` | SC11 one or two symmetric cracks at a plate hole. Give the full plate dimensions. Intensities are ordered as radial tip, bore tip. |
| `NasgroCurve` | Stores calibrated material inputs. `get_growth_rate(length, k_min, k_max)` evaluates one growth rate in mm/cycle at a crack-front point. This is a material calculation without integration. |
| `LoadBlock(minimum, maximum, cycles)` | Represents repeated loading between two stress extrema. A sequence of blocks preserves the applied order. |
| `simulate_crack_growth(geometry, curve, loads, ...)` | Combines the geometry, NASGRO material law, and ordered loads to advance the crack over time. |
| `GrowthResult` and `GrowthPoint` | The result contains accepted points and a termination reason. A point records cycles, crack dimensions, per-tip intensities, and the zero-based block index. |

[Open the plate intensity example](../../31-plate_stress_intensity.html) or
[open the NASGRO rate example](../../32-nasgro_growth_rate.html) to see these
operations on their own.

## Plate solutions and loads

The ordered solver uses tensile membrane loads. TC01 uses the finite-width
secant solution, TC02 uses Tada's single-edge tension equation for lengths
through 0.8 of the plate width, and SC01 uses the Newman–Raju finite-width
surface-crack tension solution. TC02 was checked against a published
uniform-tension strip benchmark. [Tada equation and loading conventions](https://ntrs.nasa.gov/citations/20240000858)
· [NASA strip benchmark, Table 3](https://ntrs.nasa.gov/citations/19860020138)
· [Newman–Raju surface crack](https://ntrs.nasa.gov/citations/19800005257)

## Through cracks in thin shells

TC06 and TC07 use the FLAGRO 2.0 expressions reproduced in NASA CR-4706,
with `half_length / sqrt(mean_radius * thickness)` limited to 3 and 10,
respectively. The inputs to `stress_intensity()` and `LoadBlock` are
opening membrane stresses in MPa. If starting from pressure `p`, use
`p * mean_radius / (2 * thickness)` for TC06 and
`p * mean_radius / thickness` for TC07. [NASA CR-4706, TC06](https://ntrs.nasa.gov/api/citations/19960016651/downloads/19960016651.pdf#page=150)
· [NASA CR-4706, TC07](https://ntrs.nasa.gov/api/citations/19960016651/downloads/19960016651.pdf#page=141)

TC08 uses the Forman–Sanders solution for axial membrane tension. The
supported domain requires a total arc angle below 270°, an axial length
of at least `2 * mean_radius * sqrt(mean_radius / thickness)`, and a
mean radius at least ten times the wall thickness. The last restriction
keeps the implementation in the thin-shell range used for comparison.
The bending component and net-section stress for the shell models
remain outside this implementation. [NASFLA geometry, Fig. 2.2.6](https://www.swri.org/sites/default/files/industries/k-solutions-nasgro-manual.pdf#page=10)
· [Rege, Table 8.2](https://ebooks.uis.no/index.php/USPS/catalog/view/31/29/126)

## Surface cracks in pipes, bars, threads and plate holes

**SC05** uses the external-crack, axial-tension factors from Raju and
Newman's Table 1, with linear interpolation along each table axis.
Supply gross-section axial stress in MPa. With pipe inner radius `R`,
thickness `t`, radial depth `a` and outer arc half-length `c`, the
table spans `1 ≤ R/t ≤ 10`, `0.2 ≤ a/t ≤ 0.8`, and
`0.6 ≤ a/c ≤ 1`. [NASA TM-87594, Table 1](https://ntrs.nasa.gov/api/citations/19850027111/downloads/19850027111.pdf#page=18)

**SC07** uses the Forman–Shivakumar round-bar tension factor for
`0 < depth < diameter / 2`. The solver advances the deepest point and
the plot constructs the circular front from depth. This case supports
remote tension. Bending and the Appendix B net-section calculation
remain outside this implementation. [Handbook, Table 11.3.13](https://www.afgrow.net/applications/DTDHandbook/Sections/page11_3.aspx)
· [NASFLA Fig. 2.2.52](https://www.swri.org/sites/default/files/industries/k-solutions-nasgro-manual.pdf#page=41)

**SC08** covers a thumbnail at first contact in a rolled aerospace bolt
thread. Use `S₀ = 4F/(πd²)` in MPa, where `F` is the applied axial
force in N and `d` is the minor diameter in mm. Supply the actual tensile
extrema, including any known preload. The deepest tip uses
`K = S₀ F₀ √(πa)`, with the `a/c = 1` column of Table 2 interpolated
linearly over `0 < a/d ≤ 0.5`. Fig. 2 identifies `d` as the reference
diameter for this branch. The thread cross section is a schematic that
preserves the radial depth and surface arc length.
[Mettu et al., NLR-TP-99313, §5, Table 2 and Fig. 2](https://reports.nlr.nl/server/api/core/bitstreams/e8e307d8-250e-4aa5-a89b-baa9ba94e62d/content#page=9)

**SC11** uses Newman and Raju's finite-width tension equations for a
centered hole and either one crack or two identical cracks, both at the
plate mid-plane. Supply gross-section remote tensile stress in MPa.
The public `depth` is the radial distance into the ligament and
`half_length` is the half-length along the bore. The source calls these
`c` and `a`, respectively. The fit requires
`0.2 ≤ half_length/depth ≤ 2`, `half_length < thickness/2`,
`0.5 ≤ hole_diameter/thickness ≤ 2` and
`(hole_diameter + 2·depth)/width < 0.5`.
[NASA TM-85793, Eqs. 53–64 and Table 1](https://ntrs.nasa.gov/api/citations/19840015857/downloads/19840015857.pdf#page=18)

These additions cover axial or remote tension. Internal SC05 cracks,
machined SC08 threads, SC11 pin bearing, and bending need their own load
and geometry treatments. Growth stops when the crack leaves the
supported domain. The last accepted profile is available without
a through-crack transition.

### Net-section stress

For TC01, TC02 and SC01, `geometry.net_section_stress(stress)` returns
the Appendix B net-section stress in MPa under the supplied nonnegative
membrane stress. TC02 and SC01 include bending caused by the shift in the
net-section centroid. This method reports a diagnostic. The propagation
solver evaluates stress intensity from the applied gross-section stress
and the geometry solution. The Appendix B yield and flow checks require
separate material inputs.

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

## Material inputs

`coefficient` and `exponent` set the scale and power of the growth law.
The coefficient depends on the units and the exponent. `threshold_zero`
is the long-crack threshold at stress ratio zero. `threshold_fanning`
controls how the threshold changes with stress ratio. `p` controls the
threshold term, which is disabled at `p = 0`. `toughness` is the critical
maximum stress intensity. `q` controls the near-toughness term. The solver's
toughness stop still applies at `q = 0`. `constraint` and `peak_to_flow`
parameterize the crack-opening law. Both stay fixed for the current run.

The pointwise rate method requires a positive crack length and
`0 ≤ k_min < k_max < toughness`. The propagation solver handles a
zero-range block and checks toughness before evaluating the rate.

## Solver controls and result

The current solver requires four controls. `max_cycle_step` caps a single
integration step in cycles. `length_tolerance` controls the absolute
difference in mm between one full step and two half steps.
`cycle_tolerance` controls the terminal bracket width in cycles.
`max_trials` is an implementation limit that counts accepted and rejected
trials. Exhausting it raises an error. Its design is under review.

`termination` identifies spectrum exhaustion, a toughness stop, or a geometry
limit. `final_cycles` reports the last accepted cycle count, or zero if no state
was accepted. At a stop inside a step, it reports the admissible side of the
bracket. Treat the output accordingly when comparing against a reference life.

Use `simulate_crack_growth` for an ordered sequence of tensile membrane load
blocks. Each block carries a minimum stress, a maximum stress, and a fractional
cycle count. Preserve the order in which the blocks occur. A binned cycle count
does not establish that order.

The plate geometries use crack dimensions and plate dimensions in mm. Use MPa
for applied stresses and MPa √mm for stress intensity. A center crack takes
crack and plate half-widths. A surface crack takes depth and half-length and
returns intensities at the deepest point and the free surface in that order.

The implemented center-crack range is $0 < a/b \le 0.8$, and the surface fit
uses $0 < a/c \le 1$, $0 < a/t \le 0.8$, and $0 < c/b < 0.5$.
The solution uses linear-elastic stress intensity in uniform membrane tension.
The solver stops at a toughness or geometry limit. Its final point is the
last accepted state on the admissible side of the terminal cycle bracket.

`NasgroCurve` uses the NASGRO 3.0 growth and threshold equations with a fixed,
user-supplied toughness and the Newman opening law at nonnegative stress ratios.
Supply calibrated material inputs in the stated unit system. Bending,
retardation, geometry transitions, and the NASGRO material database are outside
this implementation.

The geometry equations and growth law are based on published sources. Numerical
checks against those sources cover the supported subset. Reproducing the full
NASFLA application requires further case-specific verification.

- [Center crack finite-width solution](https://ntrs.nasa.gov/citations/19750019342)
- [Newman–Raju surface crack fit](https://ntrs.nasa.gov/citations/19800005257)
- [NASGRO 3.0 rate and threshold equations](https://ntrs.nasa.gov/citations/19990028759)
- [Newman opening law](https://ntrs.nasa.gov/citations/19840050000)

::: py_fatigue.crack_growth
