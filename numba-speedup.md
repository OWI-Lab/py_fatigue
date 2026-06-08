# Numba speedup investigation

This is an explanation/reference report for maintainers of `py_fatigue`.
The goal is to identify where numba compilation is paid repeatedly and where
small refactors could improve performance without changing fatigue logic.

## Package overview

`py_fatigue` provides tools for fatigue analysis:

- `cycle_count`: rainflow counting, turning points, histograms, and cycle-count
  objects.
- `material`: S-N curves and Paris/Walker crack-growth curves.
- `damage`: Palmgren-Miner, non-linear stress-life damage, Theil stress-life
  damage, and crack-growth integration.
- `mean_stress`: DNVGL, Walker/SWT, and Goodman-Haigh mean-stress correction.
- `geometry`: crack geometry models and stress-intensity geometry factors.
- `stress_range`, `testing`, and `utils`: supporting data structures,
  numerical helpers, plotting helpers, and root finders.

The numba usage is concentrated in material curve evaluation, crack-growth
geometry/SIF calculations, Theil damage, rainflow crossing detection, and the
generic bisection/Newton wrappers in `py_fatigue.utils`.

## Benchmark setup

I added:

- `scripts/benchmark_numba_speedups.py`: rich CLI benchmark and numba inventory.
- `scripts/numba_benchmark_results.json`: raw benchmark output from the main run.

The script uses `rich` and `rich-argparse`. The environment was created and
managed with `uv`:

```bash
uv sync --no-default-groups --group dev
uv run python scripts/benchmark_numba_speedups.py \
  --calls 8 \
  --repeats 3 \
  --json-output scripts/numba_benchmark_results.json
```

The timings below are median wall-clock timings on this branch with
Python 3.13 + Numba 0.65.1. They are intended to compare call patterns, not
to be absolute performance claims. Compatibility validation was also run on
Numba 0.61.2, 0.62.1, 0.64.0, and 0.65.1 on Python 3.13, plus Python 3.14
with Numba 0.65.1.

## Numba inventory

| Area | Symbol | Mechanism | Assessment |
| --- | --- | --- | --- |
| `cycle_count.rainflow` | `_findcross` | `@njit` with explicit signature and `cache=True` | Compiles once per process/signature and can reuse numba's disk cache. |
| `damage.crack_growth` | `CalcCrackGrowth` | `@jitclass(spec)` | Compiles on first matching instantiation. Warm calls reuse compiled methods. |
| `damage.crack_growth` | `get_geometry_factor`, `get_sif` | `@nb.njit(..., cache=True)` | Already cached and signature-pinned. |
| `damage.stress_life` | `_calc_theil_sn_damage` | `@nb.njit(cache=True)` | Already cached. |
| `geometry.cylinder` | `f_hol_cyl_01` | `@nb.njit(..., cache=True)` | Already cached and signature-pinned. |
| `material.crack_growth_curve` | `_calc_growth_rate`, `_calc_sif` | `@nb.njit(cache=True)` | Warm calls reuse compilation and can reuse numba's disk cache. |
| `material.sn_curve` | `_calc_cycles`, `_calc_cycles_2`, `_calc_stress` | `@nb.njit(cache=True)` | Warm calls reuse compilation and can reuse numba's disk cache. `_calc_cycles_2` is used by `SNCurve.get_cycles`; `_calc_cycles` and `_calc_stress` appear to be legacy/unused alternatives. |
| `material.sn_curve` | `_calc_stress_2` | `@nb.njit(cache=True)` | Already cached and used by `SNCurve.get_stress`. |
| `material.sn_curve` | `__jit_sn_curve_residuals` | cached `compile_specialized_bisect(...)` | Fixed: the specialized bisection function is a numba dispatcher with a nopython bisection loop. |
| `mean_stress.corrections` | `__jit_goodman_equation` | `compile_specialized_newton(...)` | OK: returns a numba dispatcher, so `numba_newton(...)` does not recompile on each call. |
| `utils` | `numba_bisect`, `numba_newton` | generic wrappers with cached specialization | Fixed: repeated calls with the same Python residual reuse the same specialized root finder in-process. Bisection now executes its loop in nopython mode when JIT is enabled. |
| `utils` / package API | `warmup_numba` | opt-in warmup function | New: compiles common numba call paths for the current process. |

## Benchmark results

| Case | Current pattern | Hoisted/cached pattern | Observed impact |
| --- | ---: | ---: | ---: |
| `numba_bisect(py_func)` | 0.002 ms/call | 0.001 ms/call | Repeated-call overhead removed; loop runs in nopython mode. |
| `numba_newton(py_func)` | 0.002 ms/call | 0.001 ms/call | Repeated-call overhead removed. |
| `SNCurve.get_cycles` | 0.414 ms/call | N/A | Warm dispatcher is reused. |
| `SNCurve.get_stress` | 0.448 ms/call | N/A | Warm dispatcher is reused. |
| `ParisCurve.get_growth_rate` | 1.273 ms/call | N/A | Warm dispatcher is reused; compatibility guards for older Numba do not affect the 0.65 path. |
| `ParisCurve.get_sif` | 2.177 ms/call | N/A | Warm dispatcher is reused; compatibility guards for older Numba do not affect the 0.65 path. |
| `rainflow.findcross` | 0.126 ms/call | N/A | Warm dispatcher is reused via cached signature. |
| `rainflow.findtp` | 0.168 ms/call | N/A | Warm dispatcher is reused via `findcross`. |
| `f_hol_cyl_01` | 0.003 ms/call | N/A | Warm dispatcher is reused and cached. |
| `crack_growth.get_sif` | 0.006 ms/call | N/A | Warm dispatcher is reused and cached. |
| `CalcCrackGrowth` construction | 0.717 ms/call | N/A | Jitclass is warm after first instantiation. |
| `calc_theil_sn_damage` | 0.024 ms/call | N/A | Warm dispatcher is reused and cached. |
| `goodman_haigh_mean_stress_correction` | 0.823 ms/call | N/A | The existing module-level Newton specialization avoids repeated compilation. |
| `find_sn_curve_intersection` | 0.066 ms/call | N/A | Fixed; no longer fails with the bisection argument limit. |

The original root-finder numbers showed the issue clearly: when a Python
residual was sent through `numba_bisect` or `numba_newton` on every solve, the
wrapper created a new numba dispatcher every time. The implemented cache removes
that repeated compilation for repeated calls with the same residual function.

## Findings

1. The direct `@njit` functions are not recompiling on every warm call. The
   retained direct dispatchers now use `cache=True` where practical, so they can
   also reuse numba's disk cache across processes.
2. `numba_newton` is safe in the Goodman-Haigh public path because
   `__jit_goodman_equation` is a numba dispatcher, not a plain Python function.
3. The S-N intersection path is fixed. It now calls the already-specialized
   bisection function directly, and the bisection wrapper no longer imposes the
   previous "three extra arguments" limit.
4. The generic wrappers now cache specialized root finders by residual function
   identity. Any loop that calls `numba_bisect(py_func, ...)` or
   `numba_newton(py_func, ...)` repeatedly with the same function avoids repeated
   compilation.
5. The bisection algorithm itself now runs in nopython mode when JIT is enabled,
   while preserving scalar-like residual return compatibility for scalar,
   one-item tuple, one-item array, one-item list, and numpy scalar returns.
6. Python 3.13 compatibility is now validated on Numba 0.61.2, 0.62.1, 0.64.0,
   and 0.65.1. Python 3.14 is validated on Numba 0.65.1. The oldest supported
   line (0.61) needs a small compatibility fallback for the low-level
   Paris/Walker kernel symbols, but the public curve APIs and the targeted test
   surface stay green.

## Notebook and docs validation

All tutorial notebooks in `notebooks/tutorials-repository/notebooks` now
execute on Python 3.13 and 3.14.

Key fixes:

1. The `0.5.x` use-case notebooks no longer hard-fail when the private CSV data
   is absent; they fall back to a small synthetic dataset and use Pandas 3.x
   compatible aggregation aliases.
2. The `0.6.x` nonlinear-damage notebooks were aligned with the current
   `py_fatigue` API and had stale scratch appendix cells trimmed or updated so
   automated execution is reliable.
3. `1.0.0 - AM.General.Usage.ipynb` now uses a valid `python3` kernelspec and
   Pandas 3.x compatible positional indexing examples.

## Recommended refactors

These are performance refactors only; they do not require changing the fatigue
equations or output logic.

1. Use `py_fatigue.warmup_numba()` at application startup when predictable
   runtime latency is more important than startup time.
2. Keep the root-finder residual functions stable. The cache is keyed by the
   Python function object, so repeatedly creating new lambdas/functions will
   still create new specializations.
3. For maximum reuse of compiled root finders, define residual functions once at
   module scope instead of creating new lambdas/functions inside tight loops.

## Priority

The highest-value root-finder fixes have been applied: repeated calls now reuse
specializations, bisection executes in nopython mode when JIT is enabled, and
`find_sn_curve_intersection` no longer fails under JIT. `warmup_numba()` provides
opt-in in-process warmup. Full "always precompiled" behavior is only partially
possible: signature-pinned `cache=True` dispatchers can reuse numba's disk cache
across processes, but jitclasses and dynamic root-finder closures still need
in-process warmup after each process starts. The remaining compatibility work is
around old-Numpy/old-Numba constructor quirks rather than repeated-compilation
overhead.
