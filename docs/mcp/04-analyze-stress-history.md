# 4. Analyze a stress history

Analyze 48 ordered stress samples from a synthetic operating record. The
stress fluctuates around a changing mean, with larger excursions during
parts of the record. You will count the cycles, then estimate their
Palmgren–Miner damage with an illustrative S–N curve.

Start with a connected [local](02-run-locally.md) or [hosted](03-connect-hosted.md)
MCP client.

## Count rainflow cycles

Send the prompt below. If your client accepts direct tool calls, select
`rainflow_count` and use the JSON arguments.

```text
Use the Py-Fatigue MCP tool rainflow_count on these ordered stress samples
in MPa: [27, 43, 54, 54, 27, 15, 30, 46, 70, 83, 57, 27, 15, 9, 28, 53,
45, 23, 2, -23, -21, 3, 10, 12, 8, -13, -19, -5, 4, 20, 35, 25, 19, 25,
25, 39, 61, 60, 53, 47, 28, 29, 48, 54, 55, 46, 11, -10].
Report the number of rows and the total cycle weight. Explain why they differ.
```

```json
{
  "stress": [
    27, 43, 54, 54, 27, 15, 30, 46, 70, 83, 57, 27, 15, 9, 28, 53,
    45, 23, 2, -23, -21, 3, 10, 12, 8, -13, -19, -5, 4, 20, 35, 25,
    19, 25, 25, 39, 61, 60, 53, 47, 28, 29, 48, 54, 55, 46, 11, -10
  ]
}
```

Approve the call if prompted. Expect **10 cycle rows** with a total
weight of **7.0 cycles**. Six rows are half-cycles, which each add 0.5.
The preview contains five of the ten rows.

## Ask a follow-up about Miner damage

```text
Use miner_damage on the original 48 stress samples in their original order.
Use an illustrative S–N curve with slope 3 and log10 intercept 12.
Report the total dimensionless Miner damage and the number of contributions.
```

For a direct call, pass the same 48 samples under `source.stress`.

```json
{
  "source": {
    "kind": "stress_history",
    "stress": [
      27, 43, 54, 54, 27, 15, 30, 46, 70, 83, 57, 27, 15, 9, 28, 53,
      45, 23, 2, -23, -21, 3, 10, 12, 8, -13, -19, -5, 4, 20, 35, 25,
      19, 25, 25, 39, 61, 60, 53, 47, 28, 29, 48, 54, 55, 46, 11, -10
    ]
  },
  "curve": {"slope": 3, "intercept": 12}
}
```

Expect total damage of about **`1.4062865e-6`** from all ten rows. For a
longer record and the full contribution table, try
[analyzing and exporting a CSV](05-analyze-local-csv.md). To reproduce the
same 48-sample calculation in Python, follow [Getting started](../user/01-absolute-noob.md).
