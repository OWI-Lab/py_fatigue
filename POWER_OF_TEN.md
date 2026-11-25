# Power of Ten for native kernels

Apply [Holzmann's Power of Ten](https://spinroot.com/gerard/pdf/P10.pdf) when implementing or reviewing Rust.

1. Use straight-line or simple branching and no recursion.
2. Show a finite bound for every loop. Check input sizes and solver budgets before use.
3. Allocate scratch and output capacity before entering the numerical calculation.
4. Keep functions small enough to review on one page.
5. Review for an average of two meaningful invariant checks per function. Propagate failed checks as typed errors in release builds.
6. Use narrow variable scope and immutable bindings where possible.
7. Validate boundary inputs and handle fallible results.
8. Isolate binding macros and justify conditional compilation. Keep numerical logic explicit.
9. Use safe borrows and exhaustive dispatch. Keep authored unsafe code and callbacks out of kernels.
10. Run formatting, warning-free compilation, Clippy and tests in CI and scheduled checks.

Review loop bounds, allocations, call graphs, and error paths manually where the tools cannot verify them.
