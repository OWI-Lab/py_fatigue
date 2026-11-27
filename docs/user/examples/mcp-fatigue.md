# Count cycles and calculate Miner damage with MCP

In this example, use a short stress history to check your MCP connection,
then save the individual damage contributions as a CSV. The example S–N
curve is illustrative.

## Prepare the input

Follow the [MCP setup guide](../mcp.md#connect-codex) and configure both
CSV directories. Save this file as `stress.csv` inside the input directory.
Its values are stress samples in MPa, in the order they were measured.

```csv
stress
0
75
-15
50
0
```

## Count the cycles

In Codex or another connected client, enter this prompt.

```text
Use the py-fatigue rainflow_count tool with the configured input file
stress.csv. Report the number of cycle rows, total cycle weight, and
stress range of each row.
```

A client that accepts direct tool arguments can send this to
`rainflow_count`.

```json
{"stress": {"csv_path": "stress.csv"}}
```

You should get four rows with ranges of 75, 90, 65, and 50 MPa, in that
order. Each is a half cycle, so the total cycle weight is 2.

## Calculate and export the damage

Ask the client to run `miner_damage` on the same file using slope 3
and a log10 intercept of 12, and to export the full contribution table.
You can also pass these arguments directly.

```json
{
  "source": {
    "kind": "stress_history",
    "stress": {"csv_path": "stress.csv"}
  },
  "curve": {"slope": 3, "intercept": 12},
  "export_csv": true
}
```

The total damage is approximately `7.7525e-7`. Open the generated file
under the configured output directory using the relative path in the
tool result. The file contains four data rows and columns for the
stress range, cycle weight, and damage contribution.

For larger inputs, the preview shows up to five rows and the total uses
the entire history. See the
[ordered crack-growth example](../mcp.md#simulate-ordered-crack-growth) for
a separate workflow based on load blocks in their applied order.
