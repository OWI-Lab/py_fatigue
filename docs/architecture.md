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
The design below is a proposal for splitting those responsibilities.

## MCP companion

The independently installable `py-fatigue-mcp` package translates tool
arguments and local CSV data into calls to the public Python API. The
core library owns the numerical algorithms. The MCP package owns tool
schemas, local file handling, and the stdio entry point. See the
[MCP guide](mcp/01-overview.md) for local setup and examples.

## Proposed architecture

Organize the domain around the data each calculation actually needs.

```text
Stress samples ── rainflow ──► Counted spectrum ──► S–N and equivalent loads
                        └───► Ordered turning-point residue

Ordered load history ──► Crack propagation ──► Crack history and stop status
                            │
                            ├── Geometry: state + load → per-tip intensities
                            ├── Growth law: state + intensities → per-tip rates
                            └── Integrator: rates + step → next state
```

| Boundary | Responsibility |
| --- | --- |
| Counted spectrum | Own the range, mean and count arrays. Keep binning and serialization as operations on these arrays. |
| Rainflow state | Own the ordered residual turning points needed to continue a history. State the order requirements for combining histories and recovering residual cycles. |
| Ordered load history | Carry actual load extrema and cycle counts in sequence. Populate it from known input histories. An aggregated histogram has lost this ordering. |
| Crack geometry | Define admissible dimensions and evaluate intensities at the supported crack-front points. |
| Material law | Evaluate growth from the crack state and intensities. Its required inputs depend on the law. NASGRO needs maximum intensity and crack size as well as the intensity range. |
| Integrator | Advance a crack state, estimate step error, and locate a terminal event. Keep engineering criteria such as toughness distinct from execution limits such as a user-set resource budget. |
| Adapters | Convert native results into Python objects, DataFrames, exports or plots. The numerical core takes typed values and has no dependency on pandas or the presentation layer. |

Keep `CycleCount` as the facade for existing users. Its `+` operator already
handles bin alignment, residual recovery, and metadata. Capture those behaviors
in compatibility tests before extracting the operations. Existing
`SNCurve`, `ParisCurve`, `WalkerCurve`, `get_crack_growth`, and pandas accessors
remain the entry points for existing workflows.

Keep the Rust or NumPy choice private to each algorithm. Plain array reductions
can stay in NumPy. Stateful kernels can use Rust where the port improves
maintenance or measured performance. Start with one distribution and small
internal modules for the numerical core. The MCP companion has its own
installation and release workflow.

## Migration order

1. Characterize `CycleCount` addition across mismatched bins, timestamp order,
   warnings, and residual recovery. Extract those operations behind the
   existing facade.
2. Review the experimental ordered solver interface before committing to
   its public contract. Agree on units, which material laws it accepts, what a
   stop result means, and how resource limits are configured.
3. Exercise the boundaries with a second material law and a source-backed
   geometry case. A new law should change the rate calculation without changing
   how geometry evaluates intensities or how a step is accepted.
4. Use numerical references and end-to-end benchmarks to decide what to move
   next. A streaming rainflow API would first need tests for agreement across
   chunk boundaries.

See the [notebook index](user/examples/08-example-notebooks.md) for short examples
of the current API and the
[ordered crack-growth guide](user/examples/07-ordered-crack-growth.md)
for supported geometries and load cases in Py-Fatigue 3.
