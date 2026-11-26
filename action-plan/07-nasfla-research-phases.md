# Qualify and implement public NASFLA geometry solutions

This is the execution checklist for [APPB-05](05-appendix-b-crack-families.md).
Use the [source map](06-nasfla-source-evidence.md) to open each reference and
record progress by case and load component.

## SIF-01 · Establish the numerical source

**Inputs.** Appendix B, NASFLA §§2.2.3–2.2.4, the public source map, and the
existing TC01, TC02, and SC01 implementations.

- [X] **SIF-01.1** — Screen the 22 case IDs and record source access and
  version caveats. 2026-11-26. [Public source map](06-nasfla-source-evidence.md)
- [ ] **SIF-01.2** — Complete the SC07 source sheet first. Work from AFGROW
  Table 11.3.13 and catalog Fig. 2.2.52. Confirm the K evaluation point,
  how depth determines the surface arc, and the range supported by each
  source. Extract the usable inputs from NASA TM-102165 Table 12 and identify
  which quantities come from measurements and which come from FLAGRO.
- [ ] **SIF-01.3** — Prepare TC06 and TC07 source sheets from NASA CR-4706
  pp. A-36 and A-27. Read the scanned equations directly. Compare at least
  one independently available K reference per candidate and document the
  crack length, radius, shell parameter and pressure conversion.
- [ ] **SIF-01.4** — Prepare the SC08 source sheet from NLR-TP-99313 §5,
  Table 2 and Fig. 2. Confirm the normalized factors, the tabular gaps
  and any documented interpolation. Trace preload separately through the
  versioned manual and identify comparable cases in NASA TM-102165 Table 13.
- [ ] **SIF-01.5** — Work the other rows of the source map by geometry and
  load component. Confirm the specific hole, lug, shell, thread, and
  weight-function variants. Record inaccessible sources and unresolved
  conventions before assigning an implementation. Check TC04 and SC12
  legacy sources before deciding whether to add either superseded model.

**Artifacts and gate.** Proposed source sheets under `references/nasfla/`
with the document revision, pages, geometry sketch or dimensions, K
normalization and evaluation point, load meaning and sign, applicable range,
crack-state variables and benchmark candidates. Advance a load component when
these are unambiguous and at least one usable benchmark exists. If terms are
missing, narrow the supported model to the verified component or leave it open.

## SIF-02 · Implement a scoped numerical case

**Prerequisite.** A source sheet through SIF-01 for the specific load and
geometry being added. The [Python entry point](../py_fatigue/crack_growth.py)
and the [Rust solver](../rust/crack_growth.rs) currently use ordered tensile
membrane extrema and one or two crack dimensions.

- [X] **SIF-02.1** — Implement the SC07 tensile factor and circular profile.
  2026-11-26. `RoundBarSurfaceCrack` advances one depth and draws the
  orthogonal circular front. Bending remains part of SIF-02.4.
  [Python case](../py_fatigue/crack_growth.py) · [Rust kernel](../rust/crack_growth.rs)
- [X] **SIF-02.2** — Integrate TC06 and TC07 with the version-labeled
  published equations and TC08 with the Forman–Sanders tension solution.
  2026-11-26. All three accept only opening membrane stress in the
  solver. The guide documents pressure-to-stress conversion for TC06
  and TC07. Domain and profile tests cover the three cases.
- [ ] **SIF-02.3** — Implement SC08's supported factor table once its
  interpolation and test points are settled. Document the admissible data
  region without filling table gaps from an assumed extrapolation.
- [ ] **SIF-02.4** — For later cases, choose a load/state representation
  supported by their sources. Preserve existing `LoadBlock` behavior.
  Specify how bending, bearing, biaxial load, crack-face pressure or stress
  distributions map to tip K before adding them to ordered growth.

**Artifacts and gate.** Rust kernels and Python bindings for each qualified
case, plus geometry plotting entry points. Run formatting, Rust tests and
Clippy for core and bindings. Require explicit invalid-domain behavior,
tested pure-load limits, and compatibility with the existing public entry
points. Appendix B net-section stress can be exposed independently. A
yield or flow stop requires separate material inputs and event tests.

## SIF-03 · Separate numerical checks from validation

**Prerequisite.** The implementation and source sheet for the case.

- [ ] **SIF-03.1** — Compare each K or correction factor at documented
  points with an independently evaluated source table or calculation.
  Record conventions, rounding, and acceptance tolerances from the
  benchmark before comparing held-out values.
- [ ] **SIF-03.2** — Verify domain boundaries, vanishing and nonzero
  driving loads, and ordered solver convergence. Compare a single-case
  history and stopping event to an independent integration where possible.
  Test breakthrough only when the transition and successor have qualified
  models and both sides of the transition have coverage.
- [ ] **SIF-03.3** — Keep experimental growth and versioned NASFLA output
  comparisons in separate fixtures. For SC07 and SC08, document material
  curves, load ratio, geometry and stop criteria before comparing life.
  Disagreement with a legacy run may indicate a difference in those inputs
  or the growth law. It does not by itself identify an error in the K factor.

**Artifacts and gate.** Sourced regression tests in
[`tests/test_crack_growth.py`](../tests/test_crack_growth.py) or a dedicated
case module, and a comparison record linked from the source sheet. Label
the most specific claim the evidence supports. Published-model agreement
and NASGRO 3.0 parity have separate acceptance evidence.

## SIF-04 · Explain the cases and update the family notebooks

**Prerequisite.** Qualified propagation and plotting for a supported case.

- [ ] **SIF-04.1** — Document the input units, applicable loads,
  geometry limits, stop handling and source version in the
  [ordered-growth guide](../docs/user/examples/08-ordered-crack-growth.md).
- [ ] **SIF-04.2** — Add each new case to the corresponding existing
  [through-crack](../notebooks/35-through_crack_growth.py) or
  [surface-crack](../notebooks/36-surface_crack_growth.py) lesson. Use
  ordered blocks and the seeded long-signal example where their loads are
  within the supported domain. Plot the initial and last accepted profiles
  with matched units and axes in Matplotlib and ECharts.
- [ ] **SIF-04.3** — Run the full Python suite, build Zensical and export
  the notebooks. Verify the static and interactive profiles in a browser.

**Artifacts and gate.** Published-solution attribution, updated docs and
executable marimo lessons. Check both plot modes and record numerical
verification separately from the illustrative notebook results.

2026-11-26. SC07 remote tension passed the RD1 Table 5 comparison, ordered
growth and midpoint-domain tests, and the orthogonal-profile check. Rust
format, tests and Clippy passed. The full Python run reported 697 passed
and 5 expected failures. Zensical built and Chromium verified the SC07
static and ECharts profile views in the exported surface-crack lesson.
SC07 life validation, bending propagation, and
the next geometry source qualifications remain open.

2026-11-26. TC06 and TC07 passed direct CR-4706 equation regression checks.
TC08 passed two comparisons to Rege Table 8.2. All three passed growth
and profile tests. Full Python verification reported 707 passed and
5 expected failures. Rust tests, formatting and both Clippy
configurations passed. Zensical built and Chromium verified the shell
profiles in lesson 35. The Pyodide 0.29.3 wheel passed 70 targeted tests
in Node.js and the installation smoke test in Chromium.
Independent TC06/TC07 K comparisons, version
parity, and shell bending remain open.

### SIF-03.4 · Worked handbook examples

- [X] Compare native constant-β Paris growth with MERC-3 Eq. (11), then
  compare the digitized longeron reconstruction with Figure MERC-3.5.
- [X] Reconstruct FAC-1 Example 1 by superposing its published bypass and
  bearing intensities. Keep its NASGRO TC03 values for a future native
  TC03 check. The coarse and refined finite-element rows quantify the
  source comparison.
- [X] Run the regressions and verify both plot modes for the two marimo
  notebooks in the documentation build.

2026-11-26. All five [regressions](../tests/test_afgrow_examples.py) passed.
Native Paris life was within four cycles of Eq. (11). The three digitized
longeron lives differed from Figure MERC-3.5 by at most 1.31%. Doubling
the grid resolution changed each life by less than one cycle. The FAC-1
component sums agreed with the published totals within 0.01 ksi √in.
The full Python run reported 712 passed and 5 expected failures.
Zensical built, both notebooks passed strict marimo checks, and Chromium
rendered five static figures and nine ECharts panels.
[MERC-3 notebook](../notebooks/37-afgrow_merc3_growth_sensitivity.py) ·
[FAC-1 notebook](../notebooks/38-afgrow_fac1_sif_history.py)

FAC-2 uses an elastic–plastic shell model, rivet load transfer, and a
critical crack-tip opening angle for stable tearing. Its residual-strength
results do not provide a like-for-like benchmark for the current elastic
single-crack shell solutions.
[MERC-3](https://www.afgrow.net/applications/DTDHandbook/Examples/page2_2.aspx) ·
[FAC-1](https://www.afgrow.net/applications/DTDHandbook/Examples/page1_0.aspx) ·
[FAC-2](https://www.afgrow.net/applications/DTDHandbook/Examples/page1_1.aspx)

### SIF-02.5 · Additional surface cracks in tension

- [X] Implement SC05 external axial tension from NASA TM-87594 Table 1,
  SC08 rolled-thread axial tension from NLR-TP-99313, and SC11 remote
  tension from NASA TM-85793. Record the exact domain and stress for
  each case in the [source register](06-nasfla-source-evidence.md).
- [X] Verify source factors, interpolation, out-of-domain behavior,
  ordered growth, and initial and last accepted profiles.
- [X] Update the ordered-growth guide and surface-crack notebook. Run
  the Rust checks, Python suite, docs build, and both plot modes.

2026-11-26. SC05 and SC08 matched printed table values and interpolation
checks. SC11 matched the plotted wide-plate factors within 0.11 and
passed the single-crack conversion check. The ordered SC11 regression
stopped on the admissible side of its aspect-ratio boundary. The full
suite reported 729 passed and 5 expected failures. Rust tests,
formatting, and Clippy passed in the core and Python binding
configurations. Zensical built, strict marimo checks passed, and
Chromium rendered the new static plots and ECharts views. In the
synthetic long case SC05 and SC08 completed 300,000 cycles. SC11
reached the toughness stop at about 75,346 cycles.
[Tests](../tests/test_crack_growth.py) ·
[Notebook](../notebooks/36-surface_crack_growth.py)

Power of Ten review. Rules 1–3 and 6–9 were reviewed in
[`surface.rs`](../rust/crack_growth/surface.rs). The evaluations are
nonrecursive, use fixed tables and bounded searches, allocate no
buffers, and return optional factors after checking their domains.
New numerical functions fit within one page for rule 4. The existing
shared geometry dispatcher is longer. Domain and stop regressions
exercise rule 5's runtime checks, and the Rust checks verify rule 10.
