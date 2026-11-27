# 4. Analyze a stress history

After you [run the MCP locally](02-run-locally.md) or
[connect to hosted MCP](03-connect-hosted.md), use the same ordered
stress samples for rainflow counting and Miner damage. The example uses
an illustrative S–N curve with slope 3 and log10 intercept 12.

## Count rainflow cycles

Send this prompt in a client that accepts natural-language requests.

```text
Use the Py-Fatigue MCP tool rainflow_count on these ordered stress
samples in MPa: [0, 75, -15, 50, 0].
Report each stress range and cycle weight, and the total cycle weight.
```

In a client that accepts direct tool calls, select `rainflow_count` and
pass these arguments.

```json
{"stress": [0, 75, -15, 50, 0]}
```

Approve the call if your client prompts you. The tool returns stress
ranges **75, 90, 65, and 50 MPa**. Each row has weight **0.5**, for a total
cycle weight of **2**.

## Ask a follow-up about Miner damage

```text
Use the Py-Fatigue MCP tool miner_damage on the original stress
samples. Use an illustrative S–N curve with slope 3 and log10 intercept
12. Report the total dimensionless Miner damage.
```

For a direct call to `miner_damage`, pass the original samples.

```json
{
  "source": {"kind": "stress_history", "stress": [0, 75, -15, 50, 0]},
  "curve": {"slope": 3, "intercept": 12}
}
```

Approve the call if prompted. The total damage is approximately
**`7.7525e-7`**. The tool uses every counted row when calculating damage,
even when a longer history produces a shortened preview.

For input from a file and a full contribution table, follow the
[local CSV example](05-analyze-local-csv.md).
