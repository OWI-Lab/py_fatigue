# Public sources for Appendix B crack cases

This register covers TC01–TC10 and SC01–SC12 from the supplied Appendix B.
Read it with the [implementation tracker](05-appendix-b-crack-families.md)
and the [research and validation phases](07-nasfla-research-phases.md).

## Evidence and version boundaries

| Source | Inspected material | Role |
| --- | --- | --- |
| Local `data/nasgro-docs/appb.pdf` and `appd.pdf` | All TC and SC headings in B, and D.1 | Appendix B defines net-section stresses. Appendix D describes the interfaces for stress-intensity models. |
| August 2022 NASFLA manual | §§2.2.3–2.2.4, Table 2.2.1, SC07 diagram | Case dimensions, loads, some limits and transition rules. Modern names and capabilities need to be checked against the older case. [SwRI catalog](https://www.swri.org/sites/default/files/industries/k-solutions-nasgro-manual.pdf) |
| USAF Damage Tolerant Design Handbook | §11.3, Tables 11.3.4, 11.3.10 and 11.3.13, and SC07 equation images | Public engineering compilation with traceable equations. A matching diagram and equation identify a candidate. Version parity calls for a versioned source or a comparison against NASFLA outputs. [AFGROW handbook](https://www.afgrow.net/applications/DTDHandbook/Sections/page11_3.aspx) |
| MechaniCalc | Stress-intensity reference page inspected | Secondary reference for reading basic factors and notation. It adds no evidence of which formulation NASGRO 3.0 used. [MechaniCalc reference](https://mechanicalc.com/reference/stress-intensity-factor-solutions) |
| NASA CR-4706 | Appendix A, printed pages A-27 and A-36 | Printed numerical formulas explicitly identified as FLAGRO 2.0 TC07 and TC06. That identification is evidence for the predecessor version. [TC07 page](https://ntrs.nasa.gov/api/citations/19960016651/downloads/19960016651.pdf#page=141) · [TC06 page](https://ntrs.nasa.gov/api/citations/19960016651/downloads/19960016651.pdf#page=150) |
| NASA/CR-20250011200 | §4.4, Tables 4.4-3 through 4.4-5 | Modern correspondence with API 579. Useful for distinguishing families and loads. Recover the formulation and allowable ranges separately. [NASA/SwRI report](https://ntrs.nasa.gov/citations/20250011200) |

The original 1986 Forman–Shivakumar paper was inspected through its publisher
and NASA abstracts. The public handbook supplies the numerical SC07
equations. A versioned NASGRO 3.0 Appendix C, the legacy lug formulations
as shipped, and reference program output decks were not obtained in this pass.

## Case-by-case source map

| Case | Source and evidence level | Open check before propagation |
| --- | --- | --- |
| TC01 | Existing membrane secant implementation and NASA reference. [Baseline evidence](03-crack-growth.md) | Validate the bending term separately if adding it. |
| TC02 | Existing Tada remote-tension implementation with NASA strip comparisons. [Baseline evidence](05-appendix-b-crack-families.md#sources-and-progress) | Confirm the applicable fixture assumptions and source the in-plane and through-thickness bending terms if adding them. |
| TC03 | AFGROW Table 11.3.10 includes radial through cracks from a hole with tension and bearing terms. Newman's NASA TN D-6376 provides numerical results for cracks at holes. [AFGROW handbook](https://www.afgrow.net/applications/DTDHandbook/Sections/page11_3.aspx) · [Newman](https://ntrs.nasa.gov/citations/19710022830) | Check offset and ligament conventions, crack count, pin-load normalization, and Appendix C tables C1–C3 referenced by the local manual. Grade the remote-tension branch separately. |
| TC04 | AFGROW Table 11.3.10 includes a through crack in a bearing-loaded lug. The NASA NASCRAC/FLAGRO comparison includes a pin-loaded lug example. [AFGROW handbook](https://www.afgrow.net/applications/DTDHandbook/Sections/page11_3.aspx) · [NASA code comparison](https://ntrs.nasa.gov/citations/19900016044) | Check finite lug contour, hole/width ratio and reference load. The 2022 catalog lists TC04 as superseded. Verify any proposed successor as its own model. |
| TC05 | The 2022 catalog, Fig. 2.2.5, shows the three crack configurations. Local manual table index identifies C4–C10 for tension, pin loading and coefficients. [SwRI catalog](https://www.swri.org/sites/default/files/industries/k-solutions-nasgro-manual.pdf) | Locate the interaction factors for a row of holes. Later TC44 covers its own curved-crack configurations and is a distinct case. |
| TC06 | NASA CR-4706 p. A-36 prints the Erdogan–Kibler FLAGRO 2.0 sphere solution with pressure-to-membrane conversion and a shell-parameter limit. [NASA CR-4706](https://ntrs.nasa.gov/api/citations/19960016651/downloads/19960016651.pdf#page=150) · [Erdogan and Kibler](https://ntrs.nasa.gov/citations/19690063052) | Confirm arc convention, radius convention and 3.0 version. Verify independent K values and separate pressure-derived membrane loading from any bending solution. |
| TC07 | NASA CR-4706 p. A-27 prints the Newman FLAGRO 2.0 axial through-crack equation under pressure. NASA TN D-8325 includes pressurized cylinder fracture tests. [NASA CR-4706](https://ntrs.nasa.gov/api/citations/19960016651/downloads/19960016651.pdf#page=141) · [NASA TN D-8325](https://ntrs.nasa.gov/citations/19770007577) | Check shell parameter, mean radius, hoop stress and fit range. Use direct K evidence to verify the kernel and keep fracture tests as separate evidence. |
| TC08 | Forman et al. (1985) address circumferential through-cracks in complete shells under tension and bending. AFGROW Table 11.3.13 lists factors for this family. [Original paper record](https://www.sciencedirect.com/science/article/abs/pii/S0013794485800497) · [AFGROW handbook](https://www.afgrow.net/applications/DTDHandbook/Sections/page11_3.aspx) | Read the complete source and verify the arc, radius, bending sign and geometry ranges. Axial through-crack bulging fits do not cover TC08. |
| TC09 | 2022 catalog Fig. 2.2.6 documents hole geometry and biaxial loading. Published infinite-sheet hole solutions can serve as limiting checks. [SwRI catalog](https://www.swri.org/sites/default/files/industries/k-solutions-nasgro-manual.pdf) · [Infinite sheet study](https://deepblue.lib.umich.edu/bitstream/handle/2027.42/42767/10704_2004_Article_BF00033700.pdf?sequence=1) | Obtain the specific combined-load SIFs, verify what gamma means in Appendix B, and check finite-width and boundary assumptions. |
| TC10 | 2022 catalog Fig. 2.2.7 defines the circumferential cylinder-hole case. [SwRI catalog](https://www.swri.org/sites/default/files/industries/k-solutions-nasgro-manual.pdf) | Find curved-shell hole and pin-load factors and the supported crack counts. NASA CR-4706 pp. A-29–30 discuss a longitudinal crack and a depressurized area, which do not establish the TC10 factors. |
| SC01 | Existing Newman–Raju finite-width tension implementation. NASA TP-1578 is the local basis. [Baseline evidence](03-crack-growth.md) · [NASA surface crack research](https://ntrs.nasa.gov/citations/19800005257) | The current solver handles tension. Qualify bending and breakthrough as separate additions. |
| SC02 | Shen and Glinka (1991) report closed-form weight functions at the deepest and surface points. The publisher abstract was inspected. [Original paper](https://doi.org/10.1016/0167-8442(91)90023-D) | Obtain the complete formulas, check finite-width corrections and the stress input convention, then compare the two-tip results under nonlinear stress. |
| SC03 | 2022 catalog Fig. 2.2.49 distinguishes internal and external spherical surface cracks. NASA/CR-20250011200 Table 4.4-4 gives a modern API/NASGRO cross-reference. [SwRI catalog](https://www.swri.org/sites/default/files/industries/k-solutions-nasgro-manual.pdf) · [NASA/SwRI report](https://ntrs.nasa.gov/citations/20250011200) | Find the corresponding internal/external factors and crack-face pressure convention. TC06 is a separate through-wall solution. |
| SC04 | NASA TM-80073 provides an internal semi-elliptical crack solution for pressurized cylinders. The modern SC04/SC34 cross-reference distinguishes additional load variants. [NASA TM-80073](https://ntrs.nasa.gov/citations/19790019361) · [NASA/SwRI report](https://ntrs.nasa.gov/citations/20250011200) | Start with an internal pressure candidate. Check the source shape and wall ratios, pressure term and both tip factors. Establish external and arbitrary-stress formulations separately. |
| SC05 | Raju–Newman NASA TM-87594, Tables 1–2, gives finite-element values for pipes under tension and bending. AFGROW Table 11.3.13 prints hollow-cylinder thumbnail factors. [NASA TM-87594](https://ntrs.nasa.gov/citations/19850027111) · [AFGROW handbook](https://www.afgrow.net/applications/DTDHandbook/Sections/page11_3.aspx) | Compare the transformed crack front, inner/outer location and crack-face load. Check the source behind the legacy hollow-cylinder function before claiming correspondence. |
| SC06 | SwRI v9.2 release notes describe verification for internal and external constant-depth cracks. NASA/CR-20250011200 Table 4.4-3 maps modern SC06 to API 579 9B.5.8–9. [SwRI v9.2 notes](https://member-nasgro.swri.org/system/tdf/nasgro_v9.2f_release_notes.pdf?file=1&force=0&id=41401&type=node) · [NASA/SwRI report](https://ntrs.nasa.gov/citations/20250011200) | Obtain weight-function coefficients and distinguish this 360° front from a thumbnail crack. Verify which features existed in the target version. |
| SC07 | AFGROW Table 11.3.13 provides closed-form tension and bending factors. Forman–Shivakumar is the underlying circular-front study. NASA TM-102165 Table 12 has named SC07 predictions and round-bar experiments. [AFGROW handbook](https://www.afgrow.net/applications/DTDHandbook/Sections/page11_3.aspx) · [Original study](https://doi.org/10.1520/STP17388S) · [NASA life comparison](https://ntrs.nasa.gov/api/citations/19910009960/downloads/19910009960.pdf) | Confirm the K evaluation point and arc-length relation. NASA TM-87594 Table 3 can cross-check only overlapping rod geometry. Treat the life predictions as versioned program comparisons with their own material law and termination inputs. |
| SC08 | Mettu et al., NLR-TP-99313 §5, Fig. 2 and Table 2 specify SC08 as implemented in **NASGRO 3.0** with tabulated tension and bending factors. NASA TM-102165 Table 13 supplies earlier SC08 life data. [Original implementation paper](https://reports.nlr.nl/server/api/core/bitstreams/e8e307d8-250e-4aa5-a89b-baa9ba94e62d/content#page=9) · [NASA life comparison](https://ntrs.nasa.gov/api/citations/19910009960/downloads/19910009960.pdf) | Check the normalization against the typeset PDF, the missing shallow-depth entries, interpolation and preload rules. Confirm trial comparisons across versions before using them as parity fixtures. |
| SC09 | 2022 catalog Fig. 2.2.54 and Appendix B define a constant-depth circumferential bolt-thread crack. [SwRI catalog](https://www.swri.org/sites/default/files/industries/k-solutions-nasgro-manual.pdf) | Obtain thread-root tension and bending factors specific to the annular crack. SC08's localized thumbnail tables address a different front. |
| SC10 | 2022 catalog and Appendix B define the constant-depth threaded-pipe case. [SwRI catalog](https://www.swri.org/sites/default/files/industries/k-solutions-nasgro-manual.pdf) | Find the thread-root SIF for uniform and radial stress inputs. Confirm inner and outer radius conventions, allowable depths and treatment of threads and pressure. |
| SC11 | Newman–Raju NASA TM-85793 gives a semi-elliptical crack along a circular-hole bore under tension and bending. [NASA TM-85793](https://ntrs.nasa.gov/citations/19840015857) | Check hole radius and crack placement, one/two-crack geometry and bearing factors against the case definition. |
| SC12 | Appendix B defines the legacy lug net-section calculation. The 2022 catalog lists SC12 as superseded and describes SC32 as a weight-function lug model. [SwRI catalog](https://www.swri.org/sites/default/files/industries/k-solutions-nasgro-manual.pdf) | Source legacy pin-load factors and crack-count conditions, or explicitly qualify SC32 as a separate case. |

## How to use this register

A first implementation can claim a named published solution over the
verified geometry and load domain. Claiming NASGRO 3.0 parity also requires
version-specific evidence, including convention checks, reference factors,
and a growth comparison with equivalent material and stop conditions.
Use the primary paper for a published model, the handbook to recover
equations and trace references, and comparison data for independent checks.

SC07 remote tension now uses the handbook factor and the NASFLA circular
front for `0 < a/D < 0.5`. NASA TM-102165 Table 5, RD1, gives a deepest-tip
FLAGRO comparison of 73.9 ksi √in. The implementation gives 73.917 ksi √in
using the printed inputs. Table 12 is still a separate life benchmark.

TC06 uses the version-labeled CR-4706 expression. AFGROW Table 11.3.13
prints a different sphere factor, so it cannot serve as a pointwise parity
fixture. TC07's AFGROW expression agrees with CR-4706.

For TC08, the long-crack beta term in AFGROW's equation image puts `b`
under the inner square root. The printed Sanders expression places it
outside that root. The latter matches Rege's independent shell
finite-element factors within 0.02 at the test points. The former gives
1.682 where Table 8.2 reports 2.16 for radius/thickness 40 and half-angle
40°. [AFGROW table](https://www.afgrow.net/applications/DTDHandbook/Sections/page11_3.aspx)
· [Sanders expression](https://www-it.jwes.or.jp/ae/free/ae-1204/ae-1204-00.pdf)
· [Rege, Table 8.2](https://ebooks.uis.no/index.php/USPS/catalog/view/31/29/126)

SC04 and SC05 also have case-specific material in NASA TM-111707,
including checked influence coefficients and separate internal and
external cases. The report is a candidate for the next qualification
pass. Its interpolation and stress-profile conventions still need to
be mapped to the solver. [NASA TM-111707](https://ntrs.nasa.gov/citations/19960038141)

### Surface-crack tension variants

- **SC05 external.** NASA TM-87594 Table 1 tabulates the deepest and
  free-surface factors for axial tension, normalized by
  `S₀√(πa/Q)`. The axes are radial depth over arc half-length, inner
  radius over wall thickness, and radial depth over wall thickness.
  The implementation uses trilinear interpolation inside the table,
  with no extrapolation. The plot uses the Appendix's conformal map.
  [Raju and Newman](https://ntrs.nasa.gov/citations/19850027111)
- **SC08 rolled.** Mettu et al. define `F₀ = K/(S₀√(πa))`. Use the
  `a/c = 1` tension column and Fig. 2's minor-diameter convention for
  both the crack-depth ratio and remote axial stress. The interpolation is
  piecewise linear. Machined threads have a gap in the shallow-crack data
  and use a different nominal-stress section.
  [NLR-TP-99313 §5, Table 2 and Fig. 2](https://reports.nlr.nl/server/api/core/bitstreams/e8e307d8-250e-4aa5-a89b-baa9ba94e62d/content#page=9)
- **SC11 remote tension.** NASA TM-85793 Eq. (62) supplies the
  finite-width factor for one and two cracks. Eq. (64) estimates the
  single crack from the two-crack result before applying width
  correction. Figure 2(d) establishes the bore-span and radial-depth
  conventions. Relative crack growth at the two tips follows the K
  pair from Eq. (53).
  [Newman and Raju](https://ntrs.nasa.gov/citations/19840015857)
- **SC04 pressure fit.** In NASA TM-80073, the printed Eqs. (6)–(9)
  materially overpredict the Fig. 3 factors for `a/c = 1` and
  `a/t = 0.8`. The influence-coefficient route in NASA TM-111707
  remains a candidate for implementing SC04 without this fit.
  [NASA TM-80073](https://ntrs.nasa.gov/citations/19790019361)
