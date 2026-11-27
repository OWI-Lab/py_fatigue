# 10. Evaluate and plot material curves

Evaluate S–N curve points with `sn_curve`. Request native figures
with `plot_py_fatigue`. The examples use illustrative material
parameters. Supply your calibrated parameters and units in real analyses.

Request `format: "png"` for an image or `format: "echarts"` for an
embedded HTML chart. Your client determines how it displays these results.

## S–N curves

- “With S–N slope 3 and log10 intercept 12, calculate cycles to failure
  at stress ranges of 80, 100, and 200 MPa.”
- “For the same curve, calculate the stress ranges for one and ten
  million cycles.”
- “Set the endurance limit to one million cycles. Evaluate life at 20
  and 200 MPa and explain the infinite-life result.”
- “Evaluate the piecewise curve with slopes 3 and 5, log10 intercepts
  10.970 and 13.617, and endurance at 100 million cycles. Give ranges
  for 10,000 and 1,000,000 cycles.”
- “Plot the native S–N curve with slope 3 and log10 intercept 12 on
  log-log axes.”
- “Add measured pairs at 100,000 cycles and 220 MPa, and 1,000,000 cycles
  and 105 MPa. Keep the native S–N curve visible.”
- “Use the same data and generate an interactive S–N chart.”
- “Show the piecewise curve on semilog-x axes with its endurance limit.”

A direct plot call uses paired arrays for an overlay.

```json
{
  "request": {
    "kind": "sn_curve",
    "curve": {"slope": 3, "intercept": 12},
    "cycles": [100000, 1000000],
    "stress_ranges": [220, 105],
    "scale": "loglog"
  },
  "format": "echarts"
}
```

## Miner damage

- “Using the [inline stress-history example](04-analyze-stress-history.md),
  calculate Miner damage and show the native contributions by stress range.”
- “Show the native Miner damage histogram for the same history and curve.”
- “Use the local `stress-history.csv` input and an S–N slope of 3 with
  log10 intercept 12. Export every contribution and return the plot as PNG.”
- “Use three counted stress ranges of 50, 80, and 120 MPa with weights
  10, 4, and 0.5, respectively. Calculate and plot their Miner contributions.”
- “Use the [combined CycleCount example](09-count-and-combine-cycles.md)
  and compare Miner damage for stress concentration factors of 1 and 1.25.”

## Paris, Walker, and NASGRO curves

Use MPa·√mm for stress intensity and mm/cycle for growth rate. Paris
and Walker `intercept` values are coefficients in the selected unit
system. S–N intercepts use the log10 convention.

- “Use the native Paris curve with slope 3, coefficient `1e-12`,
  threshold 25 and critical intensity 500. Return a log-log plot.”
- “Plot the same fit using the Walker law with load ratio 0.3 and
  Walker exponent 0.5.”
- “Overlay observations of `1e-6` at 100 and `8e-6` at 200 on the
  native Paris curve. Show the original fit and both observations.”
- “Plot the NASGRO law from the [SC07 example](07-propagate-sc07.md)
  in its long-crack limit.”
- “For that NASGRO curve, set the crack length to 4 mm and minimum
  intensity to 20 MPa·√mm. Sweep up to 400 MPa·√mm.”
- “For the same curve and length, sample exactly 75, 150, and
  300 MPa·√mm. Return the interactive plot on semilog-y axes.”
