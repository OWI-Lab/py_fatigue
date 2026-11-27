# Run Py-Fatigue from an MCP client

The Py-Fatigue MCP gives Codex and other MCP clients access to ASTM
rainflow counting, Palmgren–Miner damage, and ordered NASGRO crack growth.

- [Try the hosted MCP in Codex or Claude Code](examples/mcp-hosted.md)
  with a bearer token and inline data.
- [Run the local CSV example](examples/mcp-fatigue.md) to analyze a file
  and export the damage contributions from a subprocess on your machine.

## Connect Codex

Install the companion package from PyPI with uv. If uv asks you to add its
tool directory to `PATH`, run `uv tool update-shell` and open a new shell.

```sh
uv tool install py-fatigue-mcp==0.1.0
```

Register the installed executable with Codex.

```sh
codex mcp add py-fatigue -- "$(command -v py-fatigue-mcp)"
```

Start a new Codex session and run `/mcp` to check the connection. In other
stdio clients, use the absolute path to the installed `py-fatigue-mcp`
command as the launch command.

To enable CSV access when registering the server, create the input and output
directories and use this command. Choose an input directory containing files
you want the client to be able to read.

```sh
codex mcp add py-fatigue \
  --env "PY_FATIGUE_MCP_INPUT_DIR=/absolute/path/to/input" \
  --env "PY_FATIGUE_MCP_OUTPUT_DIR=/absolute/path/to/output" \
  -- "$(command -v py-fatigue-mcp)"
```

Other MCP clients can set the same environment variables when launching the
server. Inline analyses work with both settings omitted.

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
mm for lengths, and MPa·√mm for stress intensity. This example uses the
synthetic parameters from the repository tests.

```json
{
  "geometry": {"case": "TC01", "half_width": 50, "half_length": 5},
  "curve": {
    "coefficient": 1e-10, "exponent": 2, "p": 0.5, "q": 1,
    "threshold_zero": 25, "threshold_fanning": 0, "toughness": 500,
    "constraint": 2, "peak_to_flow": 0.3
  },
  "loads": [
    {"minimum": 0, "maximum": 40, "cycles": 8.5},
    {"minimum": 10, "maximum": 50, "cycles": 10}
  ],
  "controls": {
    "max_cycle_step": 1, "length_tolerance": 1e-6,
    "cycle_tolerance": 1e-4, "max_trials": 40
  }
}
```

The ten geometry case codes and their constructor inputs are listed in the
[ordered crack-growth guide](examples/08-ordered-crack-growth.md). Use those
field names inside `geometry`. The guide also describes the supported load
directions and geometry limits. The applied order must come from the load
history. Rainflow counts alone cannot establish that order.

`termination` reports `spectrum_exhausted`, `toughness`, or
`geometry_limit`. The final state is the last accepted point. It may be
null if the solver stops at the initial condition. The history uses
zero-based load-block indices and geometry-specific crack dimensions and
tip order. At a geometry or toughness limit, the final state is on the
admissible side of the terminal cycle bracket.

## Use local CSV files

Replace an inline list with a relative file reference such as
`{"csv_path":"measurements/run-01.csv"}`. Files use UTF-8, commas, and a
header row. Extra columns are allowed.

| Data | Required columns | Where to pass the file reference |
| --- | --- | --- |
| Stress history | `stress` in MPa | `stress` in rainflow or in a `stress_history` source for Miner |
| Counted cycles | `stress_range` in MPa, `count_cycle` | `cycles` in a `counted_cycles` source for Miner |
| Ordered tensile loads | `minimum`, `maximum` in MPa, `cycles` | `loads` in ordered crack growth |

Each response previews at most five data rows. Larger results show the
first three and last two. Add `"export_csv": true` at the top level of a
tool call to save every result row. The server returns the generated
filename relative to the configured output directory and the data row
count. Crack-growth exports number the dimension and intensity columns
starting at 1, following the dimension and tip order in the tool result.

CSV inputs are limited to 64 MiB per file. The server accepts up to one
million stress samples or counted rows and 10,000 load blocks per call,
with up to 100,000 crack-growth trials. Use a smaller dataset when a call
exceeds an input limit.
