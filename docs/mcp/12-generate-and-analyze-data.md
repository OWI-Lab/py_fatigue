# 12. Generate and analyze synthetic data

Use `generate_synthetic_data` to create reproducible inputs with
`py_fatigue.testing`. The response previews up to five rows and
returns a `specification` you can reuse to regenerate every row
in another tool call. Local stdio can also export all rows to CSV.

## Generate a sampled stress record

In a prompt-based client, ask it to call `generate_synthetic_data`
with the arguments below. The example returns 512 samples at 16 Hz.

```json
{
  "specification": {
    "kind": "synthetic_history",
    "fs": 16,
    "duration": 32,
    "seed": 42,
    "minimum": -40,
    "maximum": 100,
    "random_type": "weibull",
    "parameters": {"a": 2}
  }
}
```

The time axis is in seconds, and stress values are scaled to the
requested MPa bounds. Try these prompts.

- “Generate this seeded Weibull record. Tell me the sample count and
  preview the first and last rows with units.”
- “Generate a normal record at 8 Hz for 30 seconds, starting at 10
  seconds. Scale it between −35 and 85 MPa and use seed 17.”
- “Repeat with a uniform distribution and the same specified stress
  bounds. Compare only the generated shape and extrema.”
- “Generate a Rayleigh record at 20 Hz for 10 seconds, with seed 3 and
  stress bounds from 0 to 110 MPa.”
- “Generate an exponential record with scale 2 and the same sample
  rate, duration, seed, and stress bounds.”
- “For the Weibull example, export a CSV with all time and stress
  samples using the configured local output directory.”
- “Analyze the full Weibull specification with both cycle-counting
  methods. Compare the counts and the residual lengths.”
- “Plot the native mean–range histogram and residual sequence for that
  complete synthetic record.”
- “Compute Miner damage for the same record with S–N slope 3 and
  log10 intercept 12. Plot the native damage contributions.”
- “Regenerate the same record with seed 43. Compare the results
  using the same S–N parameters and counting method.”
- “Generate two 16-second windows using seeds 42 and 43 and the same
  sampling settings. Add their CycleCount objects in order, recover
  residuals, and apply an overall stress concentration factor of 1.25.”

For direct calls, reuse the returned specification as `stress` in
`cycle_count`, or nest it under `source.stress` with
`source.kind: "stress_history"` in `miner_damage`.

## Generate ordered tensile blocks

A crack-growth simulation needs load extrema in applied order. Use a
`synthetic_loads` specification rather than inferring load order from
the rainflow histogram. Each block uses the same minimum stress and
cycle count, and the generator varies the maximum in draw order.

- “Generate six ordered tensile blocks with seed 17, a minimum of
  10 MPa, peaks from 40 to 65 MPa, and 10 cycles per block.”
- “Apply those synthetic blocks directly to the SC07 geometry and
  NASGRO curve from the [round-bar example](07-propagate-sc07.md).
  Use its solver controls and report the final depth and termination.”
- “Plot the native crack-depth history for the same synthetic blocks.”
- “Draw the last accepted crack front over the initial SC07 profile
  for the same run.”
- “Generate a second six-block spectrum with seed 18 and otherwise
  identical inputs. Compare the final depths under both ordered spectra.”
- “Export every generated block with its minimum, maximum, and cycles
  so I can repeat the same run from a local CSV.”

To make a direct call, use the following object as `specification`
for generation or as `loads` for `ordered_crack_growth`.

```json
{
  "kind": "synthetic_loads",
  "block_count": 6,
  "minimum": 10,
  "lowest_peak": 40,
  "highest_peak": 65,
  "cycles_per_block": 10,
  "seed": 17
}
```
