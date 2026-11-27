# Getting started

Count the cycles in a 48-sample stress history and calculate its
Palmgren–Miner damage. You will use the same synthetic data and
illustrative S–N curve as the [first MCP example](../mcp/04-analyze-stress-history.md).

## Install Py-Fatigue

In your Python environment, install the package.

```sh
python -m pip install py-fatigue
```

## Count the cycles

The stress samples are in MPa and in acquisition order. Run the following
code in a Python session.

```python
from py_fatigue import CycleCount, SNCurve
from py_fatigue.damage.stress_life import calc_pm

stress = [
    27, 43, 54, 54, 27, 15, 30, 46, 70, 83, 57, 27, 15, 9, 28, 53,
    45, 23, 2, -23, -21, 3, 10, 12, 8, -13, -19, -5, 4, 20, 35, 25,
    19, 25, 25, 39, 61, 60, 53, 47, 28, 29, 48, 54, 55, 46, 11, -10,
]
cycles = CycleCount.from_timeseries(stress)
print(len(cycles.count_cycle), cycles.count_cycle.sum())
```

Expect **10 cycle rows** and a **total weight of 7.0 cycles**. Each
closed cycle contributes 1.0 and each half-cycle contributes 0.5.

## Calculate Miner damage

Continue in the same Python session. The curve uses a slope of 3 and a
log10 intercept of 12 for stress ranges in MPa. These values are
illustrative.

```python
curve = SNCurve(slope=3, intercept=12)
damage = calc_pm(cycles.stress_range, cycles.count_cycle, curve)
print(f"{damage.sum():.8g}")
```

Expect a dimensionless total of about **`1.4062865e-6`**. Each row
contributes its cycle weight divided by the number of cycles to failure
at that stress range.

## Continue with the notebook

The existing notebook covers plots, histogram export, DataFrame
accessors, and a legacy Paris-law crack-growth example. The
[Python guide](02-user-guide.md) lists the individual analysis tasks.

[Open notebook](getting-started-notebook.html) ·
[Download Python source](getting-started-notebook.py){ download }

<iframe class="docs-notebook" id="runnable-notebook" src="getting-started-notebook.html" title="Getting started with Py-Fatigue" loading="lazy"></iframe>
