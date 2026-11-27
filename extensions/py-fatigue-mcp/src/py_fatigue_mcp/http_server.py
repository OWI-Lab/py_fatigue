"""Serve the inline fatigue tools over bearer-authenticated MCP HTTP."""

import os
from secrets import compare_digest

from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .crack_growth import (
    CrackGeometry,
    CrackGrowthSummary,
    NasgroParameters,
    SolverControls,
    InlineLoads,
    ordered_crack_growth,
)
from .cycle_count import (
    CycleCountSummary,
    InlineCycleInput,
    InlineStress,
    RainflowMethod,
    cycle_count,
)
from .geometry import geometry_stress_intensity
from .lffd import lffd_recovery
from .models import SERVER_INSTRUCTIONS
from .plot_inputs import InlinePlotRequest, PlotFormat
from .plotting import PlotContent, plot_py_fatigue, render_native_plot
from .stress_life import (
    InlineMinerSource,
    MinerSummary,
    RainflowSummary,
    SNParameters,
    local_miner_source,
    miner_damage,
    rainflow_count,
    sn_curve,
)
from .synthetic import SyntheticInput, SyntheticSummary, generate_synthetic_data

MAX_HTTP_BYTES = 1024 * 1024
TOKEN_ENV = "PY_FATIGUE_MCP_BEARER_TOKEN"


class RequireBearerToken:
    """Require the configured token on every HTTP request."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        self.app = app
        self.token = token.encode("ascii")

    async def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        if scope["type"] == "http":
            authorization = next(
                (
                    value
                    for key, value in scope["headers"]
                    if key == b"authorization"
                ),
                b"",
            )
            scheme, _, token = authorization.partition(b" ")
            if scheme.lower() != b"bearer" or not compare_digest(
                token, self.token
            ):
                response = PlainTextResponse(
                    "Bearer token required.",
                    status_code=401,
                    headers={"WWW-Authenticate": 'Bearer realm="py-fatigue"'},
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def create_app() -> Starlette:
    """Create an HTTP MCP app with bounded inline inputs and no file tools."""
    token = os.environ.get(TOKEN_ENV)
    if (
        not token
        or not token.isascii()
        or not token.isprintable()
        or " " in token
    ):
        raise RuntimeError(f"Set {TOKEN_ENV} to a nonempty ASCII bearer token.")

    server = MCPServer("py-fatigue", instructions=SERVER_INSTRUCTIONS)
    for tool in (lffd_recovery, sn_curve, geometry_stress_intensity):
        server.add_tool(tool)

    @server.tool(name="cycle_count")
    def native_count(
        stress: InlineCycleInput, method: RainflowMethod = "astm"
    ) -> CycleCountSummary:
        """Build native CycleCount from inline or synthetic MPa samples."""
        return cycle_count(stress, method)

    @server.tool(name="generate_synthetic_data")
    def generate(specification: SyntheticInput) -> SyntheticSummary:
        """Generate reusable stress samples or ordered tensile blocks."""
        return generate_synthetic_data(specification)

    @server.tool(
        name="plot_py_fatigue",
        description=plot_py_fatigue.__doc__,
        structured_output=False,
    )
    def plot(
        request: InlinePlotRequest, format: PlotFormat = "png"
    ) -> PlotContent:
        return render_native_plot(request, format)

    @server.tool(name="rainflow_count")
    def count(stress: InlineStress) -> RainflowSummary:
        """Count ASTM rainflow cycles from inline or synthetic stress in MPa."""
        return rainflow_count(stress)

    @server.tool(name="miner_damage")
    def damage(source: InlineMinerSource, curve: SNParameters) -> MinerSummary:
        """Calculate Miner damage with caller-supplied S–N parameters in MPa."""
        return miner_damage(local_miner_source(source), curve)

    @server.tool(name="ordered_crack_growth")
    def propagate(
        geometry: CrackGeometry,
        curve: NasgroParameters,
        loads: InlineLoads,
        controls: SolverControls,
    ) -> CrackGrowthSummary:
        """Run ordered NASGRO with tensile stresses in MPa and lengths in mm."""
        return ordered_crack_growth(geometry, curve, loads, controls)

    app = server.streamable_http_app(
        stateless_http=True,
        json_response=True,
        max_request_body_size=MAX_HTTP_BYTES,
        transport_security=TransportSecuritySettings(
            allowed_hosts=[
                "127.0.0.1",
                "127.0.0.1:*",
                "localhost",
                "localhost:*",
                "mcp.py-fatigue.dev",
                "mcp.py-fatigue.dev:443",
            ],
            allowed_origins=[],
        ),
    )
    app.add_middleware(RequireBearerToken, token=token)
    return app
