# 1. Choose a setup

Run Py-Fatigue calculations through an MCP client. Count or combine
cycles, evaluate material curves, plot native results, and generate
reproducible inputs for analysis.

| Setup | Use it when | Start here |
| --- | --- | --- |
| Local stdio | You want to run calculations on your machine, read CSV inputs, or export result files. | [Install and run locally](02-run-locally.md) |
| Hosted HTTPS | You have a bearer token and want to send inline data from a remote MCP client. | [Connect to hosted MCP](03-connect-hosted.md) |

## Examples

- [4. Analyze a stress history](04-analyze-stress-history.md)
- [5. Analyze and export a CSV](05-analyze-local-csv.md) (local stdio)
- [6. Recover cycles across windows](06-recover-lffd.md)
- [7. Propagate an SC07 round-bar crack](07-propagate-sc07.md)

See the [tool reference](08-tool-reference.md) for inputs, units, and limits.


## Usage prompts

- [9. Count, combine, and scale cycles](09-count-and-combine-cycles.md)
- [10. Evaluate and plot material curves](10-evaluate-and-plot-curves.md)
- [11. Plot geometries and crack growth](11-plot-geometries-and-growth.md)
- [12. Generate and analyze synthetic data](12-generate-and-analyze-data.md)

For a plot, ask your client to check for a native Py-Fatigue plot first.
The server returns a PNG image by default or an ECharts HTML resource on request.
