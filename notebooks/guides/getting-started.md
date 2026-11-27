# Getting started

Welcome to the absolute beginner's guide to py-fatigue! If you have comments or suggestions, please don't hesitate to file an issue or make a pull request on our GitHub repo: **https://github.com/OWI-Lab/py_fatigue/**.

## Welcome to py-fatigue!

py-fatigue was originally part of the DYNAwind suite of tools for wind turbine structural health monitoring and analysis. The tool has been detached from the project and now relies on the following PyPi packages only:

- NumPy
- pyecharts
- pandas
- Matplotlib
- Pydantic

The package bundles the main functionality for performing cyclic stress (fatigue and crack growth) analysis and cycle-counting.

## Installing py-fatigue

py-fatigue is freely available on GitHub and gets maintained through CI/CD phylosophy. Py-Fatigue 3 targets 64-bit CPython 3.10–3.15. A source build requires Rust.

Choose the command that matches your environment.

=== "pip"

    ```bash
    python -m pip install py-fatigue
    ```

=== "uv"

    ```bash
    uv pip install py-fatigue
    ```

## How to import py-fatigue

To access py-fatigue and its functions import it in your Python code like this:

``` python
import matplotlib.pyplot as plt
import numpy as np
import py_fatigue as pf
```

## Static and interactive plots

Use `.plot()` for a Matplotlib figure and axes, or `.echarts()` for an
interactive chart. Both methods take the same domain inputs, such as the
intensity samples to evaluate a NASGRO curve.

| Object | What its plot shows |
| --- | --- |
| `CycleCount` | Rainflow histogram. Use `plot_type` to select the axes. |
| `SNCurve`, `ParisCurve`, `WalkerCurve`, `NasgroCurve` | The material curve, including any supplied measurement overlay for S–N, Paris, or Walker curves. |
| Plate, cylinder and infinite-surface geometries | A sketch using the dimensions supplied to the geometry. |
| `CalcCrackGrowth` from `get_crack_growth` | Crack depth recorded at the start of each legacy step. |
| `GrowthResult` from `simulate_crack_growth` | Accepted crack dimensions against cumulative cycles. |
| `df.miner` and `df.gassner` | Contributions by stress range after calculating Miner damage or Gassner shift factors. |

```python
curve = pf.SNCurve(slope=3, intercept=12.0)
sn_figure, sn_axis = curve.plot()
plot_views(sn_figure)
```

Use `sn_figure.savefig("sn-curve.png")` to save the static plot or
`curve.echarts().render("sn-curve.html")` to write the interactive view.

For material curves, pass `scale="loglog"`, `scale="semilogx"`, or
`scale="semilogy"` to either method. The semilog options put the x or y axis
on a logarithmic scale, respectively. S–N, Paris, and Walker curves default
to log–log. NASGRO defaults to a logarithmic growth-rate axis.

Use `to_echarts(fig)` from `py_fatigue.plotting` when you have composed
several plots into an existing figure. Pass `to_echarts(ax)` for a single
panel. The HTML includes hover, zoom, reset and PNG export, and loads
ECharts from the pyecharts CDN.

For calculations that display a mean stress correction figure, pass a
callback to `plot` to receive the figure and render it separately.

```py
from py_fatigue.plotting import to_echarts

def save_interactive(fig):
    to_echarts(fig).render("mean-stress.html")

corrected = cycle_count.mean_stress_correction(plot=save_interactive)
```

## How to use py-fatigue

1. Use [`CycleCount`](../api/cycle_count/cycle_count_.md) to extract cycles
   from a stress history.
2. Choose the material curve for the calculation.
    - An [S–N curve](additional/glossary.md#sn-curve) gives the life at a
      given stress range.
    - A [Paris, Walker, or NASGRO curve](examples/02-crack-growth-curves.md) gives the
      growth rate of an existing crack.
3. Choose an analysis.
    - For [stress–life damage](examples/06-stress-life-damage.md), combine cycle
      counts with the S–N curve.
    - For [crack growth](examples/07-ordered-crack-growth.md), choose a
      geometry and combine its stress-intensity solution with a growth curve.
4. Use [`get_sampled_time` and `get_random_data`](../api/testing.md) to
   generate a sample history for the example below.

[Upgrading an existing installation?](migrate-v3.md)

## Example

In the following example, starting from a randomly generated signal, we will:

- calculate the rainflow
- export the cycle-count matrix
- reconstruct the CycleCount object from the exported matrix
- define the SN curve
- calculate the Palmgren-Miner damage
- define the Paris' law
- define a crack geometry
- run a crack growth analysis.

The damage and crack growth analyses will be also executed by means of the pandas DataFrame accessors that Py-Fatigue implements.

### Random signal simulation

``` python
import py_fatigue.testing as test

# Simulate a random signal
t = test.get_sampled_time(duration=10000, fs=10)
s = test.get_random_data(
    t=t, min_=-30, range_=180, random_type="weibull", a=2., seed=42
)
# Plot the signal
signal_figure, signal_axis = plt.subplots()
signal_axis.plot(t[:1001], s[:1001], lw=0.5)
signal_axis.set(
    xlabel="Time, s", ylabel="Signal, MPa", title="First 100 seconds"
)
plot_views(signal_figure)
```

### Cycle-count definition

``` python
# CycleCount definition
cycle_count = pf.CycleCount.from_timeseries(
    time=t, data=s, mean_bin_width=3., range_bin_width=3.,
)
cycle_count.summary()
```

### Cycle-count matrix

``` python
# Exporting the cycle-count matrix in the legacy format, i.e. not
# accounting for mean stresses. This function has been kept for
# backwards compatibility.
exp_dict_legacy = cycle_count.as_dict(
    max_consecutive_zeros=20,
    damage_tolerance_for_binning=0.2,
    legacy_export=True,
)
print(exp_dict_legacy)
```

``` python
# Exporting the cycle-count matrix
exp_dict = cycle_count.as_dict(
    max_consecutive_zeros=20, damage_tolerance_for_binning=1
)
print(exp_dict)
```

``` python
# Reconstructing the CycleCount instance from the exported matrix
cycle_count_d = pf.CycleCount.from_rainflow(exp_dict, name="Random Signal")
cycle_count_d.summary()
```

Compare the two spectra on a common mean-stress and stress-range grid.
Each marker represents an occupied bin. Its color gives the cycle count,
including half cycles.

``` python
mean_edges = np.histogram_bin_edges(
    np.r_[cycle_count.mean_stress, cycle_count_d.mean_stress], bins=50
)
range_edges = np.histogram_bin_edges(
    np.r_[cycle_count.stress_range, cycle_count_d.stress_range], bins=50
)
histograms = [
    np.histogram2d(
        spectrum.mean_stress,
        spectrum.stress_range,
        bins=(mean_edges, range_edges),
        weights=spectrum.count_cycle,
    )[0]
    for spectrum in (cycle_count, cycle_count_d)
]
mean_centers = (mean_edges[:-1] + mean_edges[1:]) / 2
range_centers = (range_edges[:-1] + range_edges[1:]) / 2
max_count = max(histogram.max() for histogram in histograms)
histogram_figure, axs = plt.subplots(
    1, 2, figsize=(12, 4.5), layout="constrained"
)
for axis, histogram, title in zip(
    axs, histograms, ("Cycle count from signal", "Cycle count from matrix")
):
    mean_bin, range_bin = histogram.nonzero()
    markers = axis.scatter(
        mean_centers[mean_bin],
        range_centers[range_bin],
        c=histogram[mean_bin, range_bin],
        vmin=0,
        vmax=max_count,
        cmap="viridis",
    )
    histogram_figure.colorbar(markers, ax=axis, label="Cycles per bin")
    axis.set(
        title=title,
        xlabel="Mean stress (MPa)",
        ylabel="Stress range (MPa)",
        xlim=(mean_edges[0], mean_edges[-1]),
        ylim=(range_edges[0], range_edges[-1]),
    )
plot_views(histogram_figure)
```

### Stress-Life

#### SN Curve definition

``` python
c_air = pf.SNCurve(
    [3, 5],
    intercept=[12.592, 16.320],
    norm="DNVGL-RP-C203/2016",
    environment='Air',
    curve='C'
)
sn_code_figure, sn_code_axis = c_air.plot()
plot_views(sn_code_figure)
```

#### Damage calculation analysis

``` python
# Calculate damage for the cycle-count objects
damage = pf.damage.stress_life.get_pm(cycle_count=cycle_count, sn_curve=c_air)
damage_d = pf.damage.stress_life.get_pm(cycle_count=cycle_count_d, sn_curve=c_air)
print(f"damage from signal: {sum(damage)}")
print(f"damage from matrix: {sum(damage_d)}")
```

### Crack growth

A crack growth simulation necessitates of three ingredients (objects):

- Cycle-counted stress history
- Crack growth curve (e.g. Paris' law)
- Crack geometry

#### Paris' curve definition

``` python
SLOPE = np.array([2.88, 5.1, 8.16, 5.1, 2.88])
INTER = np.array([1E-16, 1E-20, 1E-27, 1E-19, 1E-13])
THRESHOLD = 20
CRITICAL = 2000
pc = pf.ParisCurve(slope=SLOPE, intercept=INTER, threshold=THRESHOLD,
                   critical=CRITICAL, norm="The norm",
                   environment="Environment", curve="nr.")
paris_figure, paris_axis = pc.plot()
plot_views(paris_figure)
```

#### Crack geometry definition

``` python
geo = pf.geometry.HollowCylinder(
  initial_depth=5.,
  thickness=10.,
  height=30.,
  outer_diameter=30.,
  crack_position="external"
)
geo
```

#### Crack growth analysis

``` python
cg = pf.damage.crack_growth.get_crack_growth(
  cycle_count, pc, geo, express_mode=True
)
print(f"Cycles to end: {int(cg.final_cycles)}")
```

``` python
cg_d = pf.damage.crack_growth.get_crack_growth(
  cycle_count_d, pc, geo, express_mode=True
)
print(f"Cycles to end: {int(cg_d.final_cycles)}")
```

``` python
growth_figure, growth_axis = plt.subplots()
growth_rows = display_indices(cg_d.crack_depth.size)
growth_axis.plot(np.cumsum(cg_d.count_cycle)[growth_rows], cg_d.crack_depth[growth_rows])
growth_axis.set(xlabel="Cumulative cycles", ylabel="Crack size, mm")
plot_views(growth_figure)
```

### Working with pandas DataFrames

It's possible to translate the CycleCount object to a pandas DataFrame and use the implemented accessors `py_fatigue.damage.stress_life.PalmgrenMiner` (`miner`) and `py_fatigue.damage.crack_growth.CrackGrowth` (`cg`) to perform the stress-life and propagation analyses as shown above.

``` python
df = cycle_count.to_df()

# Stress-life
df.miner.damage(sn_curve=c_air)
# Crack-growth
df.cg.calc_growth(cg_curve=pc, crack_geometry=geo)
df.head()
```

The plots show a sample of the calculated rows. The damage plot includes
the largest stress range and largest damage contribution. The crack
history includes the first and last rows.

``` python
damage_rows = np.union1d(
    display_indices(len(df)),
    [df.stress_range.argmax(), df.pm_damage.argmax()],
)
damage_figure, damage_axis = df.iloc[damage_rows].miner.plot_histogram()
plot_views(damage_figure)
```

``` python
accessor_axis = df.iloc[display_indices(len(df))].plot(
    kind="scatter", x="cumul_cycle", y="crack_depth", s=2
)
plot_views(accessor_axis.figure)
```
