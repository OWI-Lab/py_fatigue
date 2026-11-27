# 2. Install and run locally

Use a local MCP client that can launch a stdio subprocess. This setup
can read CSV inputs and export full results on your machine.

## Install from PyPI

Install the companion package from PyPI with uv. If uv asks you to add its
tool directory to `PATH`, run `uv tool update-shell` and open a new shell.

```sh
uv tool install py-fatigue-mcp==0.2.0
```

## Connect a stdio client

Set the transport to stdio and use the absolute path returned by
`command -v py-fatigue-mcp` as the launch command. In Codex, register
the executable like this.

```sh
codex mcp add py-fatigue -- "$(command -v py-fatigue-mcp)"
```

Check tool discovery in your client. In a new Codex session, use `/mcp`.
Version 0.2.0 adds `cycle_count`, `sn_curve`, `geometry_stress_intensity`,
`generate_synthetic_data`, and `plot_py_fatigue`. Check the
[tool reference](08-tool-reference.md) for the full list.

## Run from a checkout

Use this setup to try tools from a development branch. Clone the
repository and use its absolute path in the
command below. Building the Python library from the checkout requires Rust.

```sh
uv run --project /absolute/path/to/py_fatigue/extensions/py-fatigue-mcp py-fatigue-mcp
```

For a stdio client, set the executable to `uv` and pass the command after
`uv` as its arguments. Use the absolute path to `uv` if the client uses
a different `PATH` than your shell.

## Configure CSV directories

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

Try the [inline stress history](04-analyze-stress-history.md) or the
[CSV analysis and export](05-analyze-local-csv.md).
