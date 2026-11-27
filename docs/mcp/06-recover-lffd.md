# 6. Recover cycles across windows

A load excursion can start in one recording window and end in another.
Use `lffd_recovery` to recount the residual turning points in order and
compare the recovered spectrum with the independent window counts. LFFD
stands for low-frequency fatigue dynamics.

Use a server that lists `lffd_recovery`. You can
[run the tool from a checkout](02-run-locally.md#run-from-a-checkout).
The example uses four consecutive windows from one synthetic stress
record. Each window contains 24 samples in MPa. A slow load variation
spans the windows while smaller oscillations repeat within them.

## Recover the cross-window cycles

In a prompt-based client, paste the JSON below and ask it to call
`lffd_recovery` with those exact arguments. Ask for the largest stress
range and total cycle weight before and after recovery. You can also call
the tool directly.

```json
{
  "windows": [
    [54, 56, 44, 41, 55, 65, 66, 68, 66, 58, 63, 81,
     86, 78, 74, 75, 76, 84, 96, 91, 77, 77, 86, 90],
    [93, 94, 84, 73, 81, 93, 90, 83, 79, 72, 69, 80,
     86, 74, 61, 61, 61, 62, 68, 64, 47, 38, 46, 51],
    [47, 44, 36, 23, 24, 36, 37, 24, 17, 14, 11, 16,
     25, 17, 1, -1, 6, 9, 11, 12, 1, -10, -1, 11],
    [10, 5, 3, -2, -3, 11, 21, 12, 4, 7, 12, 17,
     28, 30, 17, 14, 27, 37, 39, 41, 39, 31, 37, 55]
  ]
}
```

| Result | Independent windows | After recovery |
| --- | ---: | ---: |
| Largest stress range | 58 MPa | 106 MPa |
| Total cycle weight | 17.0 | 16.5 |

The recovered spectrum has 19 rows and six remaining turning points.
Here the recovered totals also match `rainflow_count` on all 96 samples
joined in order. The cycle weight decreases while the largest stress
range grows because recovery counts the long excursion across the
window boundaries.

For your own record, keep the windows contiguous and non-overlapping.
Merge windows that contain too few reversals to count. See
[how residuals are counted](../user/additional/02-rainflow-and-residuals.md)
and the [cycle-count aggregation notebook](../11-cycle_count_aggregation.html)
for the Python workflow.
