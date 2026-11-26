# Appendix B through and surface crack families

Implement sourced TC and SC cases for ordered crack growth, with static and
ECharts plots of the starting and last accepted crack profiles. Give each
family its own numbered marimo lesson, including a long synthetic load case.

## Work packages

- [X] **APPB-01** — Inventory Appendix B and trace the existing solver and plotting entry points. 2026-11-26.
- [X] **APPB-02** — Add TC02 to the ordered solver and Appendix B membrane net-section stress to TC01, TC02, and SC01. 2026-11-26. The [crack growth tests](../tests/test_crack_growth.py) check TC02 against NASA Table 3, both through-crack net sections against direct calculations, and the SC01 net section against independent cross-section integration.
- [X] **APPB-03** — Add initial and final profiles to the geometry plotters. 2026-11-26. The [plot tests](../tests/test_plotting.py) check the endpoint coordinates, surface depth, ECharts traces, and the case with no accepted state.
- [X] **APPB-04** — Add [35 Through-crack growth](../notebooks/35-through_crack_growth.py) and [36 Surface-crack growth](../notebooks/36-surface_crack_growth.py) and link both from the [ordered-growth guide](../docs/user/examples/08-ordered-crack-growth.md). 2026-11-26. Ran short ordered blocks and 300,000-cycle cases and verified both HTML exports in Chromium.
- [ ] **APPB-05** — Extend coverage case by case as geometry definitions, stress-intensity equations and verification references become available. For each case record the source version, allowable loads and dimensions, crack state, limiting events, and benchmark evidence.

## APPB-05 research checklist

The [public source map](06-nasfla-source-evidence.md) screens all 22 cases.
SC07 remote tension and the membrane branches of TC06, TC07 and TC08 are
implemented. Continue by qualifying SC04's pressure model and SC08's
coefficient table. SC08 has version-specific coefficient data and
also needs a separate check of the table domain and preload rules.

- [ ] [**SIF-01**](07-nasfla-research-phases.md#sif-01--establish-the-numerical-source) — Qualify source, loads, limits and benchmark per case. The first source screen is complete.
- [ ] [**SIF-02**](07-nasfla-research-phases.md#sif-02--implement-a-scoped-numerical-case) — Implement each qualified load case and plotter.
- [ ] [**SIF-03**](07-nasfla-research-phases.md#sif-03--separate-numerical-checks-from-validation) — Verify the K factors, ordered growth, and stop conditions. Record version parity evidence separately.
- [ ] [**SIF-04**](07-nasfla-research-phases.md#sif-04--explain-the-cases-and-update-the-family-notebooks) — Update the two family notebooks, build docs, and verify the profiles.

Update the phase checklist with dated, linked evidence per case. Check off
a work package when its documented acceptance checks pass. Record a
published solution and NASGRO version parity as separate claims.

## Case inventory

| Case | Appendix B geometry | Propagation source or gap |
| --- | --- | --- |
| TC01 | Center of a plate | Existing center-crack secant solution under membrane tension |
| TC02 | Edge of a plate | Implemented for remote tension using Tada's equation through a length/width of 0.8. Compared to the NASA elastic strip benchmark at 0.1, 0.4 and 0.8. |
| TC03 | Offset hole in a plate | Offset-hole magnification factors |
| TC04 | Hole in a lug | Lug pin-load solution |
| TC05 | Hole in a plate with a row of holes | Hole-spacing and crack-count factors |
| TC06 | Sphere | Implemented with the NASA CR-4706 FLAGRO 2.0 sphere expression and opening membrane stress. |
| TC07 | Longitudinal cylinder crack | Implemented with the Newman fit, using hoop membrane stress. |
| TC08 | Circumferential cylinder crack | Implemented for axial membrane stress with the Forman–Sanders thin-shell fit. Bending remains open. |
| TC09 | Hole in a plate under combined loading | Combined-load factors and definition of gamma |
| TC10 | Cracks from a hole in a cylinder | Cylinder-hole and multiple-crack factors |
| SC01 | Finite-width plate in tension and bending | Existing Newman–Raju tension solution. The current ordered solver accepts membrane stresses. |
| SC02 | Finite-width plate with nonlinear stress | Through-thickness stress and weight function |
| SC03 | Sphere | Internal and external spherical surface solutions and pressure term |
| SC04 | Longitudinal hollow cylinder crack | NASA TM-80073 gives a candidate for the internally pressurized branch. External loading and the NASFLA through-wall stress inputs need separate sources. |
| SC05 | Circumferential hollow cylinder thumbnail | Circumferential thumbnail factors and crack-front definition |
| SC06 | Circumferential hollow cylinder crack | Distinct internal and external circumferential factors |
| SC07 | Solid cylinder thumbnail | Implemented for remote tension, with a circular front derived from depth. Bending and net-section checks remain open. |
| SC08 | Bolt thread thumbnail | Thread notch and preload solution |
| SC09 | Circumferential bolt thread crack | Circumferential thread solution |
| SC10 | Threaded pipe | Pipe-thread factors for uniform and nonuniform radial stress |
| SC11 | One or two surface cracks from a plate hole | NASA TM-85793 gives a candidate for tension and bending. Pin loading and crack-count applicability still need confirmation. |
| SC12 | One or two surface cracks from a lug hole | Lug pin-loading and crack-count factors |

## Sources and progress

- 2026-11-26. The supplied `data/nasgro-docs/appb.pdf` is the net-section stress appendix. Its introduction uses net-section stress for yielding and flow checks. Crack propagation requires Mode I stress-intensity factors separately. Appendix D describes the stress-intensity routines. The main manual references Appendix C for the equations and tables, and the local manual set has no copy of those pages.
- 2026-11-26. The existing `Plate` Rust enum and `simulate_crack_growth` use one center-crack length or two surface-crack dimensions. Preserve their public constructors, histories and units. Use the existing ordered tensile membrane load workflow for the initial implementations.
- 2026-11-26. Record Appendix B net-section stresses as diagnostics for supported plate loads. Net-section material failure criteria require yield and ultimate stresses and a separate solver integration design.
- 2026-11-26. TC02 uses [Tada's equation (1)](https://ntrs.nasa.gov/citations/20240000858) and passed comparison against [NASA strip benchmark Table 3](https://ntrs.nasa.gov/citations/19860020138). The supported upper ratio is the largest benchmark ratio checked here. TC02 net-section eccentricity uses the crack length from Figure 16 of the local manual and the 0.13 ligament inset specified in Appendix B.
- 2026-11-26. Verification included Rust format, tests and Clippy with both core and Python features. The new Rust calculations use fixed-size inputs, typed validation and enum dispatch. The solver's trial bounds and buffer allocation are unchanged. Pytest reported 694 passed and 5 expected failures. A local Zensical build produced all 13 notebook exports. Pyodide 0.29.3 installed the rebuilt wheel and passed 57 targeted tests in Node.js.
- 2026-11-26. The long examples supplied 300,000 synthetic cycles per geometry. TC01 and SC01 exhausted the spectrum. TC02 reached the toughness stop with its last accepted profile at 211,703.17 cycles.
- The remaining 19 cases are open. Appendix C or a published solution with matching geometry, dimensions, load components, and comparison data can advance a case. Candidate references include the [NASA cylinder through-crack comparison](https://ntrs.nasa.gov/citations/19770007577), [NASA TM-80073 on internal cylindrical surface cracks](https://ntrs.nasa.gov/citations/19790019361), and [NASA TM-85793 on finite plates with holes](https://ntrs.nasa.gov/citations/19840015857).
- 2026-11-26. The August 2022 NASFLA catalog in `data/nasgro-docs/k-solutions-nasgro-manual.pdf` adds case diagrams, load definitions, dimensional conventions, some geometry limits, and transition tables. Use §2.2.3 for plotters and load conventions and §2.2.4 for transition rules. Table 2.2.1 lists TC04 and SC12 as superseded. The catalog refers to Appendix C for the stress-intensity formulations and weight-function data, so case-specific numerical sources are still needed.

[Existing propagation references](03-crack-growth.md) · [SwRI geometry catalog](https://www.swri.org/sites/default/files/NASFLA.pdf)
