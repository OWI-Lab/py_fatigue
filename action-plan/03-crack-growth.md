# NASFLA geometry and crack growth

- [ ] **CG-01** — Trace TC01 and SC01 to NASFLA geometry documentation and published independent references. Record conventions, load and geometry domains, equations, fixture licensing and tolerances. The local NASGRO 9.0 PDF ends before its referenced Appendix C. Keep the ignored manuals, software and materials database out of the distribution.
- [X] **CG-02** — Add explicit ordered tensile membrane load blocks, finite plate geometries and typed one- and two-dimensional state. Evaluate per-tip stress intensities together. Preserve legacy CycleCount and pandas workflows.
- [ ] **CG-03** — Implement the documented NASGRO growth law with Newman closure, threshold and user-supplied toughness. Use bounded step doubling and terminal event bracketing. Verify both geometries, per-tip outputs, units, accuracy, load order, and the separately benchmarked growth law before release.

[1999 NASGRO 3.0 overview](https://ntrs.nasa.gov/citations/19990028759) · [Center crack reference](https://ntrs.nasa.gov/citations/19750019342) · [Newman–Raju surface crack](https://ntrs.nasa.gov/citations/19800005257) · [Newman closure](https://ntrs.nasa.gov/citations/19840050000)

## Evidence and release gate

- 2026-11-25. The [plate regression tests](../tests/test_crack_growth.py) exercise the NASA center-crack secant correction, TP-1578 Table I tension intensities at both surface-crack points, the NASGRO 3.0 Figure 1 A357 fit, and threshold and toughness boundaries. A separated integral checks the 1D life. An independently evaluated RK4 calculation checks the 2D depth and half-length trajectory endpoint. These are tests of the published-physics implementation.
- The local manual set has no usable NASFLA 3.0 Appendix C case tables or authorized program runs for equivalent input decks. The NASGRO 9.0 manual explicitly documents a later threshold equation and case-specific near-surface corrections. Obtain versioned case equations and output decks for TC01 and SC01 before claiming full NASFLA equivalence or cutting 3.0.
