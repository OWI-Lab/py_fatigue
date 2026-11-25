# Notebooks, documentation and release

- [X] **NB-01** — Convert the tracked development guide to marimo and verify headless execution. Update Invoke and setup instructions. Remove the Jupyter dev dependency.
- [X] **DOC-01** — Migrate Sphinx to Zensical, including docstring reference pages, examples, math, assets, glossary and changelog. Check redirects for published routes. Use static marimo export for embedded examples. Browser interactivity is deferred.
- [ ] **REL-01** — Verify the CPython 3.10–3.14 wheel matrix across Linux x86-64 and ARM64, Intel and ARM macOS, and Windows x86-64. Rebuild from the sdist. Release through protected PRs to develop, validate TestPyPI and Pages, and promote the tested artifacts.

## Evidence and release gate

- 2026-11-25. The development guide passed marimo lint and exported from an installed package. Zensical preserved the prior published HTML paths and produced the API reference and notebook export. Griffe reports warnings from legacy docstrings.
- 2026-11-25. Local manylinux x86-64 wheel and source distribution checks passed. The source archive explicitly excludes local reference PDFs. The release workflow builds platform wheels and verifies the staged workflow run before reusing its files for PyPI promotion.
- Release after the TC01 and SC01 reference gate is cleared and the platform matrix, TestPyPI install and Pages preview pass. Restore the previous documentation artifact and pin version 2.1.1 if rollback is required.
