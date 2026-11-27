# 5. Analyze and export a CSV

Use a local MCP client to analyze a 240-sample synthetic stress record
and save the damage contribution from every counted row.

## Prepare the input

[Configure both CSV directories](02-run-locally.md#configure-csv-directories).
Download [stress-history.csv](samples/stress-history.csv) and save it under
that name in your input directory. The file has one column, `stress`,
with values in MPa in acquisition order.

## Count the cycles

Send the prompt, or call `rainflow_count` with the JSON arguments below.

```text
Use rainflow_count with the configured input file stress-history.csv.
Report the number of cycle rows and total cycle weight.
```

```json
{"stress": {"csv_path": "stress-history.csv"}}
```

Expect **39 cycle rows** with a total weight of **35.5 cycles**. The
preview shows the first three and last two rows.

## Calculate and export the damage

Ask the client to run `miner_damage` on the same file with an illustrative
S–N slope of 3 and log10 intercept of 12. Ask it to export the full
contribution table and report the returned relative path.

```json
{
  "source": {
    "kind": "stress_history",
    "stress": {"csv_path": "stress-history.csv"}
  },
  "curve": {"slope": 3, "intercept": 12},
  "export_csv": true
}
```

The total damage should be approximately **`8.0198315e-6`**. Open the file
in your configured output directory using the relative path in the result.
It contains 39 data rows with `stress_range`, `count_cycle`, and
`damage` columns. The sum of `damage` is the reported total.
