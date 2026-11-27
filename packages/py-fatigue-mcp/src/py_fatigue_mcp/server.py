"""Register the Py-Fatigue MCP tools and serve them over stdio."""

from mcp.server import MCPServer

from .crack_growth import ordered_crack_growth
from .stress_life import miner_damage, rainflow_count

mcp = MCPServer(
    "py-fatigue",
    instructions=(
        "Use MPa for stress and mm for lengths. Supply the analysis inputs "
        "and calibrated material parameters. Results contain up to five "
        "preview rows. Offer CSV export for complete results when the preview "
        "omits rows, and export only at the user's request."
    ),
)


mcp.add_tool(rainflow_count)
mcp.add_tool(miner_damage)
mcp.add_tool(ordered_crack_growth)


def main() -> None:
    """Run the local MCP server over stdio."""
    mcp.run()
