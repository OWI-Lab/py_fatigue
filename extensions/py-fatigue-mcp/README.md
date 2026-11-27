# Py-Fatigue MCP

Run ASTM rainflow counting, Palmgren–Miner damage, and ordered NASGRO crack
growth through a local stdio MCP client. The `py-fatigue-mcp` command starts
the server. See the [quickstart](https://github.com/OWI-Lab/py_fatigue/blob/develop/docs/user/mcp.md)
for setup, tool inputs, and CSV examples.

## Release

The Py-Fatigue MCP workflow uploads through the repository's
`PYPI_API_TOKEN` secret in the `pypi-mcp` environment. The token needs
permission to publish `py-fatigue-mcp`.

Tag the merged revision with `mcp-v` followed by the package version,
then dispatch the Py-Fatigue MCP workflow at that tag.
