# Relevant guidelines

## Choose the v3 analysis workflow

| Available inputs and analysis goal | Entry point |
| --- | --- |
| Stress history and an S–N curve for fatigue usage or equivalent loads | Use `CycleCount.from_timeseries()` or a stress-history connector, then the [stress–life API](../../api/damage/stress_life.md). |
| Existing cycle counts, a Paris or Walker material curve, and a known initial flaw | Use [`get_crack_growth()`](../../api/damage/crack_growth.md). Use `TabulatedGeometryFactor` for a supplied $Y(a)$ table on one growing dimension. |
| Ordered tensile loading, a supported geometry, and material inputs for a NASGRO model | Use [`simulate_crack_growth()`](../../api/damage/ordered_crack_growth.md) with `NasgroCurve` and a sequence of `LoadBlock` instances. |

## Set up and check ordered crack growth

1. **Match the crack and load to the published solution.** Confirm what the
   dimensions mean, which stress opens the crack, which front points advance,
   and the allowed ratios throughout the analysis. The
   [geometry guide](../examples/08-ordered-crack-growth.md) records the source
   and domain for the plate, shell, round-bar, thread, and hole cases.
2. **Preserve loading order and units.** Supply the observed or prescribed
   minimum and maximum tensile stress for each block, along with its cycle
   count. Use mm for geometry, MPa for these loads, and MPa √mm for
   intensity, threshold, and toughness. Treat fracture toughness and yield
   or flow strength as different material inputs.
3. **Document the fit.** Record the source, stress ratio, test environment,
   units, and any calibration used for the material inputs. NASGRO closure,
   threshold dependence, and the near-toughness rate term affect different
   portions of the growth curve.
4. **Check numerical convergence and the stop reason.** Repeat with smaller
   cycle steps and tighter tolerances, compare terminal cycles and
   dimensions, and plot the initial and final profiles. A geometry limit
   marks the end of the supported solution. Spectrum exhaustion marks the
   end of the supplied loading. A toughness stop marks the intensity limit.
5. **Compare against a source case.** Check stress intensity at the initial
   flaw and along the growth path before comparing life. The
   [AFGROW tutorials](../examples/01-notebook_setup.md) separate the
   published geometry, loads, and benchmark results from illustrative
   extensions. The ordered solver supports the documented tension cases.
   Overload retardation, bending, through-wall transitions, and full
   fracture assessment procedures require additional models.

The [AFGROW Damage Tolerant Design Handbook](https://www.afgrow.net/applications/DTDHandbook/Sections/page11_3.aspx)
catalogues case-specific stress-intensity solutions. The
[NASGRO 3.0 report](https://ntrs.nasa.gov/citations/19990028759) documents the
growth and threshold equations. Use the project-specified assessment
standard for acceptance criteria and the applicability of an engineering
assessment.

## Crack-growth and stress-intensity references

The sources below define the equations and tables used in the documented
models. The [geometry and load guide](../examples/08-ordered-crack-growth.md)
gives the implementation’s dimensions, stress components and validity
limits. A NASFLA case code identifies the geometry and loading family.

| Model or case | Source and how it is used |
| --- | --- |
| NASGRO growth and threshold | Mettu et al., *NASGRO 3.0 — A Software for Analyzing Aging Aircraft*, §2.1.1. The growth law combines the Newman opening function, a size- and ratio-dependent threshold, and a near-toughness term. [NASA report](https://ntrs.nasa.gov/citations/19990028759) |
| TC01, central through crack | Finite-width secant correction for uniform tension. [AFGROW Handbook, §11.3, through cracks in plates](https://www.afgrow.net/applications/DTDHandbook/Sections/page11_3.aspx) |
| TC02, edge through crack | Tada's remote-tension equation and independent elastic strip comparisons. [Tada equation and conventions](https://ntrs.nasa.gov/citations/20240000858) · [NASA strip benchmark, Table 3](https://ntrs.nasa.gov/citations/19860020138) |
| TC06 and TC07, spherical and axial cylindrical through cracks | Erdogan–Kibler sphere and Newman axial cylinder bulging fits, as reproduced in FLAGRO 2.0. [NASA CR-4706, TC06](https://ntrs.nasa.gov/api/citations/19960016651/downloads/19960016651.pdf#page=150) · [NASA CR-4706, TC07](https://ntrs.nasa.gov/api/citations/19960016651/downloads/19960016651.pdf#page=141) |
| TC08, circumferential cylindrical through crack | Forman–Sanders axial tension fit for thin shells. [Forman et al., cylinder solutions](https://www.sciencedirect.com/science/article/abs/pii/S0013794485800497) · [Rege, Table 8.2](https://ebooks.uis.no/index.php/USPS/catalog/view/31/29/126) |
| SC01, semi-elliptical surface crack in a plate | Newman and Raju's finite-width tension solution, with separate factors at the deepest and surface points. [NASA TP-1578](https://ntrs.nasa.gov/citations/19800005257) · [1981 surface-crack equation](https://ntrs.nasa.gov/citations/19820030778) |
| SC05, external circumferential pipe crack | Raju and Newman's gross-section tension factors, interpolated on the three geometry ratios in Table 1. [NASA TM-87594](https://ntrs.nasa.gov/api/citations/19850027111/downloads/19850027111.pdf#page=18) |
| SC07, round-bar surface crack | Forman and Shivakumar's tension factor for a circular crack front. [Original study](https://doi.org/10.1520/STP17388S) · [AFGROW Handbook, Table 11.3.13](https://www.afgrow.net/applications/DTDHandbook/Sections/page11_3.aspx) |
| SC08, rolled-bolt-thread crack | Mettu et al., §5, Fig. 2 and Table 2. Use the minor diameter for the reference stress and interpolate the circular-front tension column. [NLR-TP-99313](https://reports.nlr.nl/server/api/core/bitstreams/e8e307d8-250e-4aa5-a89b-baa9ba94e62d/content#page=9) |
| SC11, cracks at a centered hole | Newman and Raju's tension factors at the radial and bore tips. [NASA TM-85793, Eqs. 53–64 and Table 1](https://ntrs.nasa.gov/api/citations/19840015857/downloads/19840015857.pdf#page=18) |
| Geometry definitions and net-section stress | The SwRI case catalogue defines the crack shapes and stresses. Appendix B of the NASGRO 3.0 manual supplies the net-section formulation for TC01, TC02 and SC01. [SwRI NASFLA catalogue](https://www.swri.org/sites/default/files/industries/k-solutions-nasgro-manual.pdf) |

### Comparison cases

Use the [notebook catalogue](../examples/01-notebook_setup.md) to run the
Py-Fatigue comparisons. MERC-3 includes a crack-growth calculation.
FAC-1 compares stress-intensity factors for bypass and pin loading. The
blend-out example checks the stress-intensity and growth procedure used
for a repaired surface. [MERC-3](https://www.afgrow.net/applications/DTDHandbook/Examples/page2_2.aspx)
· [FAC-1](https://www.afgrow.net/applications/DTDHandbook/Examples/page1_0.aspx)
· [Blend-out repair, Example 9.6.1](https://www.afgrow.net/applications/DTDHandbook/pdfs/sec9_6.pdf)

## Standards and background

**DNV**

DNV (formerly DNV GL) is an international accredited registrar and classification society headquartered in Høvik, Norway. The company provides services for several industries including maritime, oil & gas, renewable energy, electrification, food & beverage and healthcare. DNV GL was created in 2013 as a result of a merger between two leading organizations in the field --- Det Norske Veritas (Norway) and Germanischer Lloyd (Germany).

In 2021, DNV GL changed its name to DNV, while retaining its post-merger structure. DNV is also the leading independent advisor and certifier to the renewable industry, notably within wind energy. Within the sector of renewables DNV could be involved in every step of the project lifecycle. Its services include a.o: feasibility studies, energy yield assessments, wind turbine type certification, design certification advisory roles, measurement and testing services, marine warranty of offshore operations, inspections and many more. Based on the available knowledge, technology and information there is a series of rules and standards provided by DNV helping engineers in the design process of an offshore wind turbine structure. These rules and standards are published as Service Specifications (SE), Standards (ST) or Recommended Practices (RP). The frequency to update the rules and standards up to DNV's disccretion. The use of the rules and standards by others than DNV is at the user's sole risk. Today's (mid 2022) most relevant rules and standards used in the assessment of offshore wind foundations are listed hereafter. Given the recent rebranding, all rules and standards remain valid and will be gradually rebranded.

- *DNVGL-ST-0126. Support Structures for wind turbines. 2018.*

  :   New DNV GL standard for wind turbine support structures offers pragmatic design approaches and can for example be used for design of steel and concrete towers, gravity-based concrete foundations, together with steel founda tions such as monopile foundations, jacket structure foundations and suction buckets foundations. The standard contains as well requirements to materials, execution, operations and maintenance (O&M) related to the design. This is done by inclusion of guidance text, or for example by reference to material and execution standards. This document number supersedes DNV-OS-J1011

- *DNVGL-ST-0437. Loads and site conditions for wind turbines. 2016.*

  :   Principles, technical requirements and guidance for loads and site conditions of wind turbines. For the first time, the requirements for onshore and bottom mounted offshore wind turbines are defined consistently and concisely in a single document. The definition of offshore turbulence classes has been revised significantly to include turbulence distributions adapted to the offshore environment. In addition marine parameters have been included to define offshore type classes which may be used for the type certification of an offshore rotor-nacelle assembly and substructure.

- *DNVGL-RP-C203. Fatigue Design of Offshore Steel Structures. 2016.*

  :   Recommendations in relation to fatigue analyses based on fatigue tests and fracture mechanics. Formerly known as DNVGL-RP-0005.

- *DNVGL-RP-0416 Corrosion protection of Wind Turbines - DNVGL Guidelines.*

  :   Principles, technical recommendations and guidance for design and construction and inservice inspection of corrosion protection systems for wind turbines. The emphasis is put on the protection of support Structures of offshore wind farms. The recommended practice can also be used for design of corrosion protection for other structures in an offshore wind farm, such as offshore substations or meteorological masts.

- *2016.DNVGL-ST-0262. Lifetime Extension of Wind Turbines - DNVGL Guidelines. 2016*

  :   Principles, technical requirements and guidance for extending the lifetime of wind turbines onshore and offshore. When designing wind turbines, a design lifetime of 20 years is generally assumed as a basis for dimensioning. Extension of lifetime for existing wind turbines can be justifiable from both a technical and an economic point of view.

- *DNVGL-SE-0263. Certification of Lifetime Extension of Wind Turbines. 2016.*

  :   Description of DNV GL's services related to the certification of wind turbine lifetime extension and it is referred to as a contractual document in the certification contract between the customer and DNV GL. The document specifies the mutual obligations of DNV GL and the customer when a wind turbine or wind farm is certified according to this service specification. The document also specifies the relevant tasks and requirements to be fulfilled for the certification.

- *DNVGL-ST-0145. Offshore substations, 2020.*

  :   DNV GL standard for the design of offshore substations. The standard is applicable for typical AC substation associated with offshore renewable energy projects, collectively denoted as "offshore substations".

- *DNVGL-RP-0497. Data quality assessment framework, 2017.*

  :   The framework outlined in this recommended practice consists of three main parts: (i) data quality assessment, (ii) organisational maturity assessment, and (iii) data quality risk assessment.

- *DNVGL-ST-0510. Framework for assurance of data driven algorithms and models, 2020*

**BS fatigue guidelines**

For the FLS assessment of structures, we also make use of a document provided by British Standards (BS). "British Standards" means formal consensus standards as set out in BS 0-1 paragraph 3.2 and based upon the principles of standardisation recognised inter alia in European standardisation policy.

!!! important "Important"

    *- Memorandum of Understanding Between the United Kingdom Government and the British Standards Institu-tion in Respect of its Activities as the United Kingdom's National Standards Body, United Kingdom Depart-ment for Business, Innovation, and Skills -*

Products and services which BSI certifies as having met the requirements of specific standards within designated schemes are awarded the Kitemark.

- *BS7910. Guide to Methods for Assessing the Acceptability of Flaws in Metallic Structures. 2013.*

  :   This British Standard gives guidance and recommendations for assessing the acceptability of flaws in all types of structures and components. Although emphasis is placed on welded fabrications in ferritic and austenitic steels and aluminium alloys, the procedures may be used for analysing flaws in structures made from other metallic materials and in non-welded components or structures. The methods described are applicable at the design, fabrication and operational phases of a structure's life.
