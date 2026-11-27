<section class="home-intro" markdown="1">

# Py-Fatigue

Count stress cycles, estimate fatigue damage, and predict crack growth with Python.

[Install Py-Fatigue](https://pypi.org/project/py-fatigue/){ .home-action }
[Upgrade from v2](user/migrate-v3.md)

</section>

<div class="home-divider" aria-hidden="true"></div>

<section class="home-section home-destinations" aria-labelledby="choose-a-workflow" markdown="1">

## Choose a workflow

<div class="home-links" markdown="1">

- **[Cycle counting](user/examples/04-cycle_counting.md)**

    Cycle-count time series and combine counting windows.

- **[Stress–life analysis](user/examples/07-damage.md)**

    Perform (non)linear damage accumulation analysis.

- **[Crack growth](user/examples/08-ordered-crack-growth.md)**

    Perform complex crack propagation analysis.

- **[Interactive notebooks](user/examples/01-notebook_setup.md)**

    Run complete analyses and compare results with AFGROW examples.

</div>

</section>

<div class="home-divider" aria-hidden="true"></div>

<section class="home-section home-example" aria-labelledby="run-a-cycle-count" markdown="1">

<div markdown="1">

## Run a cycle count

Pass a stress history to `CycleCount` and inspect the stress range and count of each cycle.

[Follow the beginner’s guide](user/01-absolute-noob.md)

</div>

```python
from py_fatigue import CycleCount

stress = [0, 50, -20, 45, -10, 0]
cycles = CycleCount.from_timeseries(stress)
print(cycles.stress_range)
print(cycles.count_cycle)
```

</section>

<div class="home-divider home-divider--line" aria-hidden="true"></div>

<section class="home-section home-example home-example--reversed" aria-labelledby="run-a-stress-life-analysis" markdown="1">

<div markdown="1">

## Run a stress-life analysis

Count cycles in a stress history and use an illustrative S–N curve to sum the Palmgren–Miner damage.

[Explore stress-life analysis](user/examples/07-damage.md)

</div>

```python
from py_fatigue import CycleCount, SNCurve
from py_fatigue.damage.stress_life import calc_pm

cycles = CycleCount.from_timeseries([0, 75, -15, 50, 0])
curve = SNCurve(slope=3, intercept=12)
damage = calc_pm(
    cycles.stress_range, cycles.count_cycle, curve
)
print(f"Total Miner damage: {damage.sum():.3g}")
```

</section>

<div class="home-divider home-divider--line home-divider--grid" aria-hidden="true"></div>

<section class="home-section home-example" aria-labelledby="run-a-crack-growth-analysis" markdown="1">

<div markdown="1">

## Run a crack growth analysis

Grow a through crack in a plate under repeated tensile loading. This example uses an illustrative NASGRO fit and returns the final crack half-length in mm.

[Explore ordered crack growth](user/examples/08-ordered-crack-growth.md)

</div>

```python
from math import ceil
from py_fatigue.crack_growth import (
    CenterThroughCrack, LoadBlock, NasgroCurve,
    simulate_crack_growth,
)

plate = CenterThroughCrack(half_width=50, half_length=4)
curve = NasgroCurve(
    coefficient=1e-10, exponent=2.0, p=0.5, q=1.0,
    threshold_zero=25.0, threshold_fanning=0.0,
    toughness=500.0, constraint=2.0, peak_to_flow=0.3,
)
load = LoadBlock(minimum=10, maximum=70, cycles=50_000)
step = 10.0  # cycles

result = simulate_crack_growth(
    plate, curve, [load],
    max_cycle_step=step,
    length_tolerance=1e-6,
    cycle_tolerance=1e-4,
    max_trials=2 * ceil(load.cycles / step),
)
print(f"Final crack half-length: {result.history[-1].sizes[0]:.3f} mm")
```

</section>

<div class="home-divider" aria-hidden="true"></div>

<section class="home-section home-resources" aria-labelledby="reference-and-background" markdown="1">

## Reference and background

- [API reference](api/01-index.md)
- [Glossary](user/additional/glossary.md)
- [Engineering guidelines](user/additional/relevant_guidelines.md)
- [Architecture](architecture.md)

</section>
