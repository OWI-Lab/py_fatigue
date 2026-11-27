"""Register the Py-Fatigue MCP tools and serve them over stdio."""

from mcp.server import MCPServer

from .crack_growth import ordered_crack_growth
from .cycle_count import cycle_count
from .geometry import geometry_stress_intensity
from .lffd import lffd_recovery
from .models import SERVER_INSTRUCTIONS
from .plotting import plot_py_fatigue
from .stress_life import miner_damage, rainflow_count, sn_curve
from .synthetic import generate_synthetic_data

mcp = MCPServer("py-fatigue", instructions=SERVER_INSTRUCTIONS)

for tool in (
    rainflow_count,
    cycle_count,
    miner_damage,
    sn_curve,
    lffd_recovery,
    ordered_crack_growth,
    geometry_stress_intensity,
    generate_synthetic_data,
):
    mcp.add_tool(tool)
mcp.add_tool(plot_py_fatigue, structured_output=False)


def main() -> None:
    """Run the local MCP server over stdio."""
    mcp.run()
