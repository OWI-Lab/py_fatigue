# Py-Fatigue 3.0

Use the existing repository and one distribution. Preserve public fatigue behavior and replace Numba with Rust. Ship independently validated TC01 and SC01 geometry cases and the NASGRO equation. Move notebooks to marimo and documentation to Zensical.

## Work packages

- [ ] [BASE-01 and P10-01](01-baseline.md) — Freeze the v2.1.1 contract and apply Power of Ten.
- [X] [RUST-01 through RUST-03](02-rust.md) — Port kernels and remove Numba.
- [ ] [CG-01 through CG-03](03-crack-growth.md) — Validate and implement ordered crack propagation.
- [ ] [NB-01, DOC-01 and REL-01](04-release.md) — Migrate examples, docs and release automation.

Baseline at `7a9cc4c` on CPython 3.14.5 with pytest 8.4.2 and setuptools was 671 passed and 5 xfailed. The Sphinx build succeeded with three warnings and all 18 local Numba benchmark cases ran.

Start with contract tests, the Power of Ten skill and a source inventory. Rust parity, engineering validation, artifact verification and release handoff are separate gates.
