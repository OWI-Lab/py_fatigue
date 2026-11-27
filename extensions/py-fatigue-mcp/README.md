# Py-Fatigue MCP

Run ASTM rainflow counting, Palmgren–Miner damage, and ordered NASGRO crack
growth through a local stdio MCP client. The `py-fatigue-mcp` command starts
the server. See the [quickstart](https://github.com/OWI-Lab/py_fatigue/blob/develop/docs/mcp/01-overview.md)
for setup, tool inputs, and CSV examples. A development checkout also
provides `lffd_recovery` for recounting residual cycles across ordered
stress windows.

## HTTP pilot

From a development checkout, supply a strong secret as
`PY_FATIGUE_MCP_BEARER_TOKEN` and run the HTTP adapter on loopback.

```sh
cd extensions/py-fatigue-mcp
uv run uvicorn py_fatigue_mcp.http_server:create_app --factory --host 127.0.0.1 --port 18732
```

The `/mcp` endpoint accepts inline inputs and requires an
`Authorization: Bearer` header. Use HTTPS at the reverse proxy when the
endpoint is reachable outside the host.

## Release

The Py-Fatigue MCP workflow uploads through the repository's
`PYPI_API_TOKEN` secret in the `pypi-mcp` environment. The token needs
permission to publish `py-fatigue-mcp`.

Tag the merged revision with `mcp-v` followed by the package version,
then dispatch the Py-Fatigue MCP workflow at that tag.
