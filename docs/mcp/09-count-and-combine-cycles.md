# 9. Count, combine, and scale cycles

Use `cycle_count` to inspect full cycles, half-cycles, and residual
turning points in ordered stress data. It builds the native
`CycleCount` object. Use `plot_py_fatigue` whenever you want the native
histogram or residual plot.

Start with [MCP 0.2.0](02-run-locally.md). For a CSV input or export,
configure the local directories first.

## Count and inspect

Use these prompts independently. In a client that accepts direct calls,
pass the sample list as `stress` and select `method`.

- “Count these MPa samples with native CycleCount and ASTM rainflow:
  `[0, 75, -15, 50, 0, 60, -20, 90, 0, -30, 70, 0]`. Tell me the total
  weight, full cycles, half-cycle rows, and maximum range.”
- “Repeat with four-point rainflow. Compare the residual turning-point
  counts from the two methods.”
- “Use the local input `stress-history.csv` from the CSV example, count
  with ASTM, and export every cycle row. Report the generated filename.”
- “Plot the native min–max histogram for that history.”
- “For the same history, show mean stress versus range with 30 bins
  and the native valley density mode.”
- “Show a normalized CycleCount min–max histogram for the same history.”
- “Use the native counts–range view for that history, then show the
  cumulative counts–range view.”
- “Show the native counts–range 2D view of the same data.”
- “Draw a mean–range histogram with 20 mean-stress bins and 30
  stress-range bins. Limit the mean axis to −30 to 80 MPa and
  the range axis to 0 to 150 MPa.”
- “Plot the residual turning-point sequence using the native method.”
- “Use the native half-cycle sequence view and return the interactive
  ECharts version.”

## Add two counts and apply a stress concentration factor

Each window represents consecutive samples from one location. Keep them
in acquisition order. The tool creates a count per window and adds
them with native `CycleCount` addition.

Paste the arguments below into `cycle_count`. In a prompt-based client,
ask it to use the JSON as given and compare the result with a factor of 1.

```json
{
  "stress": {
    "kind": "cycle_count_windows",
    "windows": [
      {"stress": [0, 75, -15, 50, 0, 60, -20, 90, 0, -30, 70, 0]},
      {"stress": [0, 55, -25, 40, 0, 90, -5, 75, -15, 0]}
    ],
    "resolve_residuals": true,
    "stress_concentration_factor": 1.25
  }
}
```

This applies `(first + second).solve_lffd() * 1.25`. The factor multiplies
stress ranges, mean stresses, and residuals. Cycle weights stay the same.
To scale an individual window before adding, put
`stress_concentration_factor` inside that window. A request with one
window can scale a single count.

Follow up with one of these prompts.

- “Repeat the addition with residual recovery off. Compare the maximum
  range, total weight, and residual count in the two cases.”
- “Apply a factor of 1.5 to the first window and leave the second at 1.
  Show the resulting native histogram with the same bins as before.”
- “Apply a factor of 1.25 to the first window alone. Confirm that the
  cycle weights stayed the same and the stress ranges changed.”
- “Use the same two-window specification to plot the recovered histogram
  after applying the overall factor. Return PNG.”
- “Calculate Miner damage from the recovered and scaled count with
  S–N slope 3 and log10 intercept 12. Compare it with the unscaled result.”
- “With the same curve, plot Miner contributions from the scaled count
  using the native contribution scatter plot.”

For a direct plot, set `request.kind` to `cycle_count` and reuse the
`stress` object. For Miner analysis, set `source.kind` to `cycle_count`
and reuse the `source.stress` object with the same counting `method`.
