# 8. MCP tool reference

<span id="6-mcp-tool-reference"></span>

Look up the inputs, units, and output fields for the fatigue tools.

## Count cycles

Call `rainflow_count` with stress samples in MPa, in sample order.

```json
{"stress": [0, 75, -15, 50, 0]}
```

The result gives the number of cycle rows, the sum of their weights
(`total_cycles`), and a preview with stress range, mean stress, and cycle
weight. A half cycle contributes 0.5 to the sum. The preview shows unbinned
cycles, including any counted half cycles.
If the core counter returns no rows for a varying history, the tool reports
an error. A constant history gives zero counted cycles.

## Calculate Miner damage

Call `miner_damage` with a stress history and S–N parameters. Intercepts are
log10 intercepts for stress ranges in MPa. You can use a scalar slope and
intercept or paired lists for a piecewise curve. Specify `endurance` in
cycles when your curve has an endurance limit.

```json
{
  "source": {"kind": "stress_history", "stress": [0, 75, -15, 50, 0]},
  "curve": {"slope": 3, "intercept": 12}
}
```

To use cycles that have already been counted, pass
`{"kind":"counted_cycles","cycles":[{"stress_range":75,"count_cycle":0.5}]}`
as the source. The result sums the damage from every cycle row and previews
individual contributions. Pass the original stress history or full counted
data when you need total damage for a large dataset.

## Simulate ordered crack growth

Call `ordered_crack_growth` with a geometry, calibrated curve, load blocks
in applied order, and solver controls. Use MPa for tensile membrane stresses,
mm for lengths, and MPa·√mm for stress intensity.

| Argument | Contents |
| --- | --- |
| `geometry` | A `case` code and that geometry's constructor inputs. See the [case list and validity limits](../api/damage/ordered_crack_growth.md#what-each-new-name-does). |
| `curve` | The calibrated [NASGRO parameters](../api/damage/ordered_crack_growth.md#material-inputs). |
| `loads` | Blocks in application order, each with `minimum`, `maximum`, and `cycles`. |
| `controls` | `max_cycle_step`, `length_tolerance`, `cycle_tolerance`, and `max_trials`. See the [solver controls](../api/damage/ordered_crack_growth.md#solver-controls-and-result). |

See the [SC07 worked example](07-propagate-sc07.md) for a complete request.

`termination` reports `spectrum_exhausted`, `toughness`, or
`geometry_limit`. The final state is the last accepted point. It may be
null if the solver stops at the initial condition. The history uses
zero-based load-block indices and geometry-specific crack dimensions and
tip order. At a geometry or toughness limit, the final state is on the
admissible side of the terminal cycle bracket.

## Recover cycles across windows

`lffd_recovery` takes `windows`, an inline array of 2 to 64 consecutive,
non-overlapping sample arrays in acquisition order. Use MPa and at most
100,000 samples across all windows. Each window must contain enough
reversals for ASTM rainflow counting to return cycle rows.

| Result field | Meaning |
| --- | --- |
| `window_count` | Number of input windows. |
| `windowed_total_cycles`, `windowed_max_stress_range` | Cycle weight and largest range in MPa before recounting residuals. |
| `total_cycles`, `max_stress_range` | Cycle weight and largest range in MPa after recovery. |
| `remaining_turning_points` | Number of points left in the recounted residual sequence. |
| `row_count`, `preview` | Recovered row count and up to five rows with range, mean stress, and weight. |

See the [four-window example](06-recover-lffd.md). Use MCP 0.2.0
or a server that lists this tool.

## Use local CSV files

CSV inputs and exports require the [local stdio setup](02-run-locally.md).

Replace an inline list with a relative file reference such as
`{"csv_path":"measurements/run-01.csv"}`. Files use UTF-8, commas, and a
header row. Extra columns are allowed.

| Data | Required columns | Where to pass the file reference |
| --- | --- | --- |
| Stress history | `stress` in MPa | `stress` in rainflow or in a `stress_history` source for Miner |
| Counted cycles | `stress_range` in MPa, `count_cycle` | `cycles` in a `counted_cycles` source for Miner |
| Ordered tensile loads | `minimum`, `maximum` in MPa, `cycles` | `loads` in ordered crack growth |

Each response previews at most five data rows. Larger results show the
first three and last two. For the tools listed in the table, add
`"export_csv": true` to save every result row. The server returns the generated
filename relative to the configured output directory and the data row
count. Crack-growth exports number the dimension and intensity columns
starting at 1, following the dimension and tip order in the tool result.

| Limit | Local stdio | Hosted HTTPS |
| --- | --- | --- |
| Stress samples or counted rows per call | 1,000,000 | 100,000 |
| CSV input size | 64 MiB per file | Inline data |
| HTTP request size | — | 1 MiB |
| Ordered load blocks per call | 10,000 | 10,000 |
| Crack-growth trials per call | 100,000 | 100,000 |

Use a smaller dataset when a call exceeds an input limit.

## Build, add, and scale native CycleCount

`cycle_count(stress, method)` accepts an ordered MPa sample array,
`synthetic_history`, a local CSV reference, or a
`cycle_count_windows` specification. The count reports unbinned rows,
full-cycle weight, half-cycle rows, maximum range, and residuals. Set
`method` to `astm` or `fourpoint`.

| Window specification field | Meaning |
| --- | --- |
| `windows` | One to 64 windows in acquisition order, with at most 100,000 samples in total. Each window provides `stress` and an optional positive `stress_concentration_factor`. |
| `resolve_residuals` | Recount the joined residuals after adding the windows. Defaults to false. |
| `stress_concentration_factor` | Positive multiplier applied to the resulting count after optional recovery. Defaults to 1. |

Per-window factors apply before addition. Each factor scales ranges,
means, and residuals while keeping cycle weights. Use consecutive,
non-overlapping windows for residual recovery. Inspect `lffd_solved`
to confirm recovery ran. For Miner damage from these exact cycles,
supply a source of `kind: "cycle_count"` with the same `stress` and
`method`. See [the worked addition](09-count-and-combine-cycles.md).

## Evaluate an S–N curve

`sn_curve` takes a `curve` with `slope`, log10 `intercept`, and
optional `endurance`. Supply up to 1,000 positive `stress_ranges` in
MPa, positive `cycles`, or both. Results come back in input order.
A null element in `cycles_to_failure` represents infinite life.

## Generate reusable data

`generate_synthetic_data` takes a `specification` and returns it with
a bounded preview. This is the same input used to regenerate the full
data in another call.

| Specification kind | Bounds and inputs | Analysis input |
| --- | --- | --- |
| `synthetic_history` | 2 to 100,000 time samples. Supply `fs` in Hz, `duration` in seconds, optionally `start` in seconds, and `minimum` and `maximum` in MPa. | `stress` of `rainflow_count` or `cycle_count`, a window's `stress`, or `source.stress` with `kind: "stress_history"` for `miner_damage`. |
| `synthetic_loads` | 2 to 10,000 ordered blocks. Supply `block_count`, `minimum`, `lowest_peak`, and `highest_peak` in MPa as applicable, and `cycles_per_block`. Peaks must exceed the minimum. | `loads` of `ordered_crack_growth` or a corresponding native plot. |

Both take a `seed` in the unsigned 32-bit range, a `random_type`, and
optional numeric `parameters`. The default distribution is normal
and the default seed is 2.

| Distribution | Accepted parameters |
| --- | --- |
| `normal` | `loc`, `scale` |
| `uniform` | `low`, `high` |
| `weibull` | `a` (required) |
| `rayleigh` | `scale` |
| `lognormal` | `mean`, `sigma` |
| `exponential` | `scale` |

Local stdio supports `export_csv: true`. Stress-history files contain
`time,stress`, and load-block files contain `minimum,maximum,cycles`.

## Evaluate geometry stress intensity

`geometry_stress_intensity` takes any of the ten ordered cases and
up to 1,000 nonnegative `stress_mpa` values. Output has one row per
applied stress, with the geometry's stress intensity factors in native
crack-tip order and in MPa·√mm.

## Plot native results

Call `plot_py_fatigue` with a typed `request`. Its `kind` selects
a native Py-Fatigue method. PNG returns an MCP image. ECharts returns
an embedded HTML resource, up to 2 MiB. The content is displayed or
saved by the client.

| Kind | Native result |
| --- | --- |
| `cycle_count` | `histogram`, `residuals`, or `half_cycles`. Histogram inputs include min–max, mean–range, counts–range, cumulative counts–range, or counts–range 2D axes. Set bins, bin range, density mode, and normalization for the two binned views. Accepts the same histories and windows as `cycle_count`. |
| `sn_curve` | Curve with scale and optional paired cycle counts and measured stress ranges. |
| `miner_damage` | Contribution scatter or histogram from the same `source` and `curve` accepted by `miner_damage`. |
| `geometry` | Initial sketch for any ordered case, plus `infinite_surface`, `cylinder`, `hollow_cylinder`, and `tabulated`. |
| `crack_growth` | Native growth `history` or `profile`. Takes the same geometry, curve, loads, and controls as `ordered_crack_growth`. |
| `nasgro_curve` | Growth rate versus maximum intensity, with optional length, minimum intensity, and maximum-intensity sweep or query samples. |
| `paris_curve`, `walker_curve` | Growth curves with optional paired observations. Use coefficients for their intercepts. |

See the [usage prompts](01-overview.md#usage-prompts) for examples.
