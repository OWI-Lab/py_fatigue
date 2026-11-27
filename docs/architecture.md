# Architecture

## Current boundaries

The Py-Fatigue 3 numerical core ships as one Python distribution. Its
public classes live at their original import paths. Python owns the interface, binning,
serialization and plots. Maturin builds the private `py_fatigue._rust`
extension. PyO3 translates arrays and errors at the boundary. Cargo can test
the numerical core independently of Python.

Curves, geometries and analysis results define what data their plots
show. Their Python plot methods draw the same data for a static
Matplotlib figure and an interactive ECharts view.

Moving a kernel to Rust does not change its responsibilities. The current
`CycleCount` still owns several kinds of operations, and the ordered
crack-growth solver uses NASGRO with supported plate, shell, bar, thread,
and plate-hole geometries.

## MCP companion

The independently installable `py-fatigue-mcp` package translates tool
arguments and local CSV data into calls to the public Python API. The
core library owns the numerical algorithms. The MCP package owns tool
schemas, file handling for stdio, and the bearer-authenticated HTTP
adapter for inline requests. See the [MCP guide](mcp/01-overview.md) for
setup and examples.

## Proposed architecture

See the [design proposal](architecture-proposal.md) for possible
changes to the numerical and Python boundaries.

<span id="migration-order"></span>

See the [notebook index](user/examples/08-example-notebooks.md) for short examples
of the current API and the
[ordered crack-growth guide](user/examples/07-ordered-crack-growth.md)
for supported geometries and load cases in Py-Fatigue 3.
