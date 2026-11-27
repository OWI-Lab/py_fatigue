# 2. Install and run locally

Use a local MCP client that can launch a stdio subprocess. This setup
can read CSV inputs and export full results on your machine.

## Install from PyPI

Install the companion package from PyPI with uv. If uv asks you to add its
tool directory to `PATH`, run `uv tool update-shell` and open a new shell.

```sh
uv tool install py-fatigue-mcp==0.1.0
```

## Connect a stdio client

Set the transport to stdio and use the absolute path returned by
`command -v py-fatigue-mcp` as the launch command. In Codex, register
the executable like this.

```sh
codex mcp add py-fatigue -- "$(command -v py-fatigue-mcp)"
```

Check tool discovery in your client. In a new Codex session, use `/mcp`.
The server exposes `rainflow_count`, `miner_damage`, and
`ordered_crack_growth`.

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
