# Rust kernel migration

- [X] **RUST-01** — Build a mixed Python/Rust distribution with Maturin, PyO3 and NumPy bindings. Start with rainflow zero crossings behind the existing wrapper. Verify exact parity, native tests, editable installs and wheel smoke tests.
- [X] **RUST-02** — Port S–N and Paris/Walker kernels, Goodman and S–N root solvers, Theil accumulation, hollow-cylinder geometry and legacy propagation. Retain Python callbacks in the existing Python root helpers. Preserve legacy classes and import paths. Add stress-life and crack-growth entry points within the same distribution.
- [X] **RUST-03** — Retire Numba helpers and configuration after parity. Compare v2.1.1 and 3.0 cold and warm timings and peak memory. Verify clean wheel metadata and installs have no Numba or llvmlite.

## Evidence

- 2026-11-25. PyO3 and Maturin builds work on Linux x86-64. An installed manylinux wheel passed the smoke test. The full Python test suite ran on CPython 3.10 through 3.14 locally.
- 2026-11-25. The [backend benchmark](../scripts/benchmark_backends.py) used CPython 3.14 and NumPy 2.4.6 in both environments. In one local run, import dropped from 5.40 s to 0.81 s and peak RSS from 411,556 KiB to 141,600 KiB. Warm legacy propagation dropped from 0.362 ms to 0.048 ms. Scalar S–N increased from 0.004 ms to 0.020 ms. Small array calls incur Python parameter preparation and NumPy binding overhead. The long rainflow ranges, means, counts and residuals matched exactly. S–N, Paris, Miner, Theil, Goodman and legacy growth matched within the stated benchmark comparison tolerances of 5e-12 relative and 5e-13 absolute.
- Build the local extension with `uv sync`. After changing Rust, run `uv run --with maturin maturin develop --uv` from the project root and rerun the affected Python and Cargo tests.
