# Py-Fatigue MCP

Run Py-Fatigue analysis and native plots through an MCP client. The
`py-fatigue-mcp` command starts the local stdio server.

- Build, add, or apply a stress concentration factor to native `CycleCount`
  objects. Work with ASTM or four-point counting and residual recovery.
- Evaluate S–N curves and Miner damage. Run ordered NASGRO across ten
  supported crack geometries.
- Generate reproducible stress histories and ordered tensile blocks with
  `py_fatigue.testing`, then pass the specification to another tool.
- Request native plots as PNG or ECharts HTML.

See the [guide](https://py-fatigue.dev/mcp/01-overview.html) for setup,
examples, and reference.

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
