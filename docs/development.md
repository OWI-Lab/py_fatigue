# Development guidelines

Read the [architecture](architecture.md) before changing an analysis
boundary. Keep Python and MCP callers on the public fatigue API.
Numerical changes should have reference or regression evidence for the
affected inputs, units, stopping criteria, and failure paths.

## Check a change locally

From the repository root, with uv and Rust installed, run the core checks
for the code you changed.

```sh
uv sync --no-default-groups --group dev
uv run --no-sync pytest
cargo +stable fmt --all --check
cargo +stable test --locked
cargo +stable clippy --locked --all-targets -- -D warnings
```

Run the MCP subprocess tests from the companion project when changing
its input, output, or launch contract.

```sh
cd extensions/py-fatigue-mcp
uv run pytest
```

## Review native kernels

Use Holzmann’s Power of Ten as a review guide for the Rust kernels.

- Keep control flow simple and avoid recursion. Give each loop a finite
  bound and validate dimensions and solver budgets before execution.
- Allocate scratch and output buffers before entering the calculation.
  Keep functions short enough to review on one page and use narrow,
  immutable bindings where possible.
- Aim for an average of two meaningful invariant checks per function
  and return typed errors on failure,
  including in release builds. Handle fallible results at the boundary.
- Use safe borrows and exhaustive dispatch. Keep authored unsafe code,
  foreign callbacks, and binding macros out of the numerical kernels.
  Justify conditional compilation and keep the numerical logic explicit.
- Run formatting, warning-free compilation, Clippy, and tests in CI
  and scheduled checks.
- Review loop bounds, allocations, call graphs, and error paths manually
  where the automated checks cannot establish them.

## Maintain the documentation

Use the existing Py-Fatigue theme in
`docs/stylesheets/documentation.css`. The homepage uses ruled rectangular
cards and short examples. Guides use a reading column with a sidebar index,
an on-page outline, and copyable code blocks. Use the shared theme tokens
and controls for new pages so navigation, focus, and colors stay consistent
in light and dark mode.

Add each new guide to the navigation in `zensical.toml`, check its links,
and run the build from the repository root.

```sh
uv sync --group dev --group docs
uv run --no-sync invoke docs.build
```

Check the rendered page at desktop and phone widths. Verify that long
code blocks scroll within their frames and that copied examples run as
written.
