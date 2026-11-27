# Py-Fatigue MCP

Run ASTM rainflow counting, Palmgren–Miner damage, and ordered NASGRO crack
growth through a local stdio MCP client. The `py-fatigue-mcp` command starts
the server. See the [quickstart](https://github.com/OWI-Lab/py_fatigue/blob/develop/docs/user/mcp.md)
for setup, tool inputs, and CSV examples.

## Release

Configure the PyPI Trusted Publisher with owner `OWI-Lab`, repository
`py_fatigue`, workflow `mcp.yml`, and environment `pypi-mcp`.
Tag the merged revision with `mcp-v` followed by the package version,
then dispatch the Py-Fatigue MCP workflow at that tag.
