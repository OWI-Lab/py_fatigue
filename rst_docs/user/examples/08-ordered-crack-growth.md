# Ordered plate crack propagation

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
