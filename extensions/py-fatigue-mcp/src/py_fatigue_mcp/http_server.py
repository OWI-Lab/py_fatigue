"""Serve the inline fatigue tools over bearer-authenticated MCP HTTP."""

import os
from secrets import compare_digest
from typing import Annotated, Literal

from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .crack_growth import (
    MAX_LOAD_BLOCKS,
    CrackGeometry,
    CrackGrowthSummary,
    NasgroParameters,
    SolverControls,
    TensileLoad,
    ordered_crack_growth,
)
from .models import InputModel, Number
from .stress_life import (
    CountedCycle,
    CountedCycles,
    MinerSummary,
    RainflowSummary,
    SNParameters,
    StressHistory,
    miner_damage,
    rainflow_count,
)

MAX_HTTP_ROWS = 100_000
MAX_HTTP_BYTES = 1024 * 1024
TOKEN_ENV = "PY_FATIGUE_MCP_BEARER_TOKEN"
InlineStress = Annotated[
    list[Number], Field(min_length=2, max_length=MAX_HTTP_ROWS)
]
InlineLoads = Annotated[
    list[TensileLoad], Field(min_length=1, max_length=MAX_LOAD_BLOCKS)
]


class InlineHistory(InputModel):
    """Stress samples in applied order, in MPa."""

    kind: Literal["stress_history"]
    stress: InlineStress


class InlineCycles(InputModel):
    """Stress ranges in MPa and their cycle weights."""

    kind: Literal["counted_cycles"]
    cycles: Annotated[list[CountedCycle], Field(max_length=MAX_HTTP_ROWS)]


InlineMinerSource = Annotated[
    InlineHistory | InlineCycles, Field(discriminator="kind")
]


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

    server = MCPServer("py-fatigue")

    @server.tool(name="rainflow_count")
    def count(stress: InlineStress) -> RainflowSummary:
        """Count ASTM rainflow cycles from inline stress samples in MPa."""
        return rainflow_count(stress)

    @server.tool(name="miner_damage")
    def damage(source: InlineMinerSource, curve: SNParameters) -> MinerSummary:
        """Calculate Miner damage with caller-supplied S–N parameters in MPa."""
        if isinstance(source, InlineHistory):
            local_source = StressHistory(kind=source.kind, stress=source.stress)
        else:
            local_source = CountedCycles(kind=source.kind, cycles=source.cycles)
        return miner_damage(local_source, curve)

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
