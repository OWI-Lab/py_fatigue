# Py-Fatigue v3 MCP

Deliver a stdio MCP server in the independently installable
`py-fatigue-mcp` subproject. Local clients launch `py-fatigue-mcp`.
The pilot targets Codex and other stdio clients on the user's machine.

- [X] **MCP-01** Expose ASTM rainflow counting, Palmgren–Miner damage,
  and ordered NASGRO crack growth for all ten v3 geometries. Reuse the
  existing fatigue API and require caller-supplied material parameters.
- [X] **MCP-02** Accept inline data and CSV inputs under a configured
  directory. Return totals and at most five preview rows. Attempt a full
  result export only when requested and return its relative path and row
  count.
- [X] **MCP-03** Verify the stdio protocol, library parity, CSV boundaries,
  and full exports against a newly built wheel. Verify the companion package in CI.
  The Python 3.13 wheel test run on 2026-11-27 passed 769 tests with 5
  expected failures.
- [X] **MCP-04** Document local setup in the
  [MCP quickstart](../docs/user/mcp.md). The Zensical build rendered the
  quickstart on 2026-11-27.

The starting revision was `60085a9`. The focused numerical tests on the
published 3.0.0 wheel reported 285 passed and 5 expected failures. The
checkout's pre-existing compiled extension failed four SC01 tests, so use
a fresh build for verification.

Use a feature pull request to hand the change off to `develop`.

## Publish the companion package

Create a pending Trusted Publisher on PyPI for `py-fatigue-mcp`, using
owner `OWI-Lab`, repository `py_fatigue`, workflow `mcp.yml`, and
environment `pypi-mcp`. Tag the merged revision `mcp-v0.1.0` and
dispatch the Py-Fatigue MCP workflow at that tag. The workflow builds and
publishes only the companion wheel and source distribution.
