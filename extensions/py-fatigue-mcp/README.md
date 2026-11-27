# Py-Fatigue MCP

Py-Fatigue MCP is the companion server for the Py-Fatigue 3 Python
library. It calls the same numerical routines and plotting methods
you can use directly in Python. Installing `py-fatigue-mcp` also
installs the core library. [Py-Fatigue on PyPI](https://pypi.org/project/py-fatigue/)
· [Py-Fatigue documentation](https://py-fatigue.dev/)
· [MCP documentation](https://py-fatigue.dev/mcp/01-overview.html)

- Build, add, or apply a stress concentration factor to native `CycleCount`
  objects. Work with ASTM or four-point counting and residual recovery.
- Evaluate S–N curves and Miner damage. Run ordered NASGRO across ten
  supported crack geometries.
- Generate reproducible stress histories and ordered tensile blocks with
  `py_fatigue.testing`, then pass the specification to another tool.
- Request native plots as PNG or ECharts HTML.

## Quickstart

```sh
uv tool install py-fatigue-mcp
command -v py-fatigue-mcp
```

Use the executable path from the second command as the launch command
in a local stdio MCP client. The local server requires no token.
[Local client setup](https://py-fatigue.dev/mcp/02-run-locally.html)
· [Hosted setup](https://py-fatigue.dev/mcp/03-connect-hosted.html)

## Example prompts

### Count cycles and plot the result

```text
Use the Py-Fatigue MCP to count ASTM rainflow cycles for these stress
samples in MPa: [0, 75, -15, 50, 0]. Report the cycle weights and show
the native min–max histogram.
```

Expect four half-cycles with a total weight of 2. For a direct tool call,
select `cycle_count` and pass `{"stress": [0, 75, -15, 50, 0]}`.

### Calculate Miner damage

```text
Use miner_damage on the same stress samples, with an illustrative
S–N slope of 3 and log10 intercept of 12. Report the total damage.
```

Expect a dimensionless total of about `7.7525e-7`.

### Combine and scale cycle counts

```text
Count these two consecutive MPa windows separately:
[0, 75, -15, 50, 0, 60, -20, 90, 0, -30, 70, 0]
[0, 55, -25, 40, 0, 90, -5, 75, -15, 0]
Add the CycleCount objects in order, recover the joined residuals,
and apply a stress concentration factor of 1.25. Plot the resulting
mean–range histogram using Py-Fatigue.
```

### Generate a synthetic stress history

```text
Generate 32 seconds of Weibull stress samples at 16 Hz with seed 42,
shape a = 2, and stress bounds of -40 to 100 MPa. Use the full
generated history for cycle counting and show the native residual plot.
```

The generator returns a preview and a reusable specification for all
512 samples. [MCP walkthroughs](https://py-fatigue.dev/mcp/01-overview.html)
· [Tool reference](https://py-fatigue.dev/mcp/08-tool-reference.html)

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
