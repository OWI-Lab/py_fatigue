# Migrate from v2 to v3

Keep your existing analysis code for the common v2 workflows. The
`CycleCount`, `SNCurve`, `ParisCurve`, and `WalkerCurve` imports remain
available at package level.

## Install in a separate environment

Py-Fatigue 3.0.0 supports 64-bit CPython 3.10–3.15. A compatible wheel
includes the Rust extension. Install Rust if pip needs to build from source.

```bash
python -m pip install "py-fatigue==3.0.0"
```

## Run an existing v2 analysis

This script uses the same calls in v2.1.1 and v3. It prints the sum of the
Palmgren–Miner damage contributions for a short stress history.

```python
from py_fatigue import CycleCount, SNCurve
from py_fatigue.damage.stress_life import calc_pm

cycles = CycleCount.from_timeseries([0, 75, -15, 50, 0])
curve = SNCurve(slope=3, intercept=12)
damage = calc_pm(cycles.stress_range, cycles.count_cycle, curve)
print(float(damage.sum()))
```

| Existing workflow | v3 entry point |
| --- | --- |
| Count or combine stress histories | `CycleCount.from_timeseries()`, `count_a + count_b`, and `solve_lffd()` |
| Export and reload a cycle histogram | `CycleCount.as_dict()` and `CycleCount.from_rainflow()` |
| Correct for mean stress or calculate stress–life damage | `CycleCount.mean_stress_correction()`, `calc_pm()`, or the `DataFrame.miner` accessor |
| Calculate Paris or Walker crack growth | The existing `py_fatigue.damage.crack_growth.get_crack_growth()` with a legacy geometry, or the `DataFrame.cg` accessor |

## Update code that used Numba helpers

Remove calls to `warmup_numba()`. The Rust kernels ship already compiled.
Use ordinary Python or NumPy inputs in place of `to_numba_dict()`.
Replace imports of `jit_sn_curve_residuals` and other removed
Numba-specific helpers where your own code relied on them. If you call
low-level helpers directly, check their signatures in the [API index](../api/01-index.md).

Compare the outputs of a representative workload in both environments
before switching a production analysis. Include the counted cycles and
residuals, plus the damage or final crack size you use downstream.

## Adopt new features when needed

- Use `CycleCount.from_sep005()` or `CycleCount.from_pyexsi()` to pass
  a supported measured or generated stress history directly.
- Continue to use Matplotlib through `.plot()`. Where available,
  `.echarts()` provides an interactive view of the same analysis.
- Use [ordered crack growth](examples/08-ordered-crack-growth.md) when
  the inputs are an explicit sequence of load blocks, a NASGRO material
  fit, and a supported geometry.
- Open the [marimo notebook catalogue](examples/01-notebook_setup.md) for
  current examples and tutorials.
