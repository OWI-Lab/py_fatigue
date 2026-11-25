# Architecture

Py-Fatigue ships as one Python distribution. Applications use the existing
`CycleCount`, `SNCurve`, `ParisCurve`, and `WalkerCurve` classes and their
existing import paths. The `py_fatigue.stress_life` and
`py_fatigue.crack_growth` modules provide domain-specific entry points.

| Layer | Responsibilities |
| --- | --- |
| Python API | Public classes, cycle-count arithmetic and residual recovery, input preparation, pandas accessors, plotting, and result formatting |
| Private native bindings | `py_fatigue._rust` converts prepared inputs into Rust values and maps results and errors back to Python |
| Rust numerical core | Bounded kernels for crossing detection, curve evaluation, root solving, damage accumulation, and crack propagation |

Maturin builds the native extension into each platform wheel. Cargo can test
the numerical core without Python. The binding feature adds PyO3 and NumPy
support for installed-package tests.

## Choose a crack-growth workflow

Existing analyses use `get_crack_growth` or the pandas `.cg` accessor with their
current curve and geometry types. Keep this route for analyses driven by a
`CycleCount` histogram.

The ordered plate workflow takes a sequence of `LoadBlock` objects, a plate
geometry, a `NasgroCurve`, and explicit solver controls. Each block supplies
minimum and maximum stress and a cycle count. Geometry determines the stress
intensities at each crack-front point. The material law determines the growth
rate from those intensities. The integrator advances the crack dimensions and
returns the accepted history and termination reason. A center through-crack
tracks half-length. A semi-elliptical surface crack tracks depth and half-length.

A binned histogram cannot reconstruct the load sequence. Keep the original
ordered load blocks for this workflow. See the
[ordered plate crack-growth example](user/examples/08-ordered-crack-growth.md)
for units, supported loads, and validation limits.
