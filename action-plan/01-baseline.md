# Baseline and Power of Ten

- [ ] **BASE-01** — Fix the distutils fixture and pytest 9 parametrization. Run Python 3.10–3.14. Record v2.1.1 comparisons for old imports, CycleCount addition, sparse round-trips, ASTM and four-point rainflow, curve knees and inverses, mean stress and damage. Cover fractional counts, clustered DataFrame results, legacy threshold and failure boundaries, and Walker parameters. Compare discrete results exactly and document floating-point tolerances.
- [X] **P10-01** — Install and validate `~/.agents/skills/power-of-ten/SKILL.md`. Review the first native kernel under each rule. Keep the [repository review criteria](../POWER_OF_TEN.md) usable without the local skill.

## Evidence

- 2026-11-25. The Python 3.14.5 v2.1.1 baseline ran 671 tests with 5 expected failures. The two test harness fixes also passed with pytest 9 before the backend change. Dispatcher-only assertions were removed with the helpers.
- 2026-11-25. The skill was installed under `~/.agents/skills/power-of-ten` and passed the skill-creator validator. Rust formatting, unit tests and Clippy passed locally. The runtime audit checks numerical code separately from Python bindings.
