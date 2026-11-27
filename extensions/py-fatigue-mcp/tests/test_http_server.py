"""Verify the hosted MCP transport keeps file access out of its tool schemas."""

import json

import pytest
from starlette.testclient import TestClient

from py_fatigue_mcp.http_server import TOKEN_ENV, create_app


def test_http_requires_a_configured_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    with pytest.raises(RuntimeError, match=TOKEN_ENV):
        create_app()


def test_http_authentication_and_inline_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    accept = {"Accept": "application/json, text/event-stream"}
    authorized = {**accept, "Authorization": "Bearer test-token"}

    with TestClient(create_app(), base_url="http://127.0.0.1") as client:

        def request(
            method: str, headers: dict[str, str], params: dict | None = None
        ):
            return client.post(
                "/mcp",
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": method,
                    "params": params or {},
                },
            )

        for headers in (accept, {**accept, "Authorization": "Bearer invalid"}):
            denied = request("tools/list", headers)
            assert denied.status_code == 401
            assert denied.headers["WWW-Authenticate"].startswith("Bearer")

        discovered = request("tools/list", authorized).json()["result"]["tools"]
        assert {tool["name"] for tool in discovered} == {
            "rainflow_count",
            "miner_damage",
            "lffd_recovery",
            "ordered_crack_growth",
            "cycle_count",
            "sn_curve",
            "geometry_stress_intensity",
            "generate_synthetic_data",
            "plot_py_fatigue",
        }
        schemas = json.dumps([tool["inputSchema"] for tool in discovered])
        assert "csv_path" not in schemas
        assert "export_csv" not in schemas

        result = request(
            "tools/call",
            authorized,
            {
                "name": "rainflow_count",
                "arguments": {"stress": [0, 75, -15, 50, 0]},
            },
        ).json()["result"]
        assert not result.get("isError", False)
        assert result["structuredContent"]["total_cycles"] == 2

        recovery = request(
            "tools/call",
            authorized,
            {
                "name": "lffd_recovery",
                "arguments": {
                    "windows": [
                        [20, 26, 18, 35, 30, 43, 34, 45],
                        [48, 43, 53, 45, 54, 40, 45, 27],
                        [23, 30, 17, 23, 14, 21, 10, 23],
                    ]
                },
            },
        ).json()["result"]
        assert not recovery.get("isError", False)
        assert recovery["structuredContent"]["max_stress_range"] == 44

        foreign = request(
            "tools/list", {**authorized, "Host": "unrelated.example"}
        )
        assert foreign.status_code == 421
        foreign_origin = request(
            "tools/list", {**authorized, "Origin": "https://unrelated.example"}
        )
        assert foreign_origin.status_code == 403

        oversized = client.post(
            "/mcp",
            headers={**authorized, "Content-Type": "application/json"},
            content=b" " * (1024 * 1024 + 1),
        )
        assert oversized.status_code == 413


def test_http_native_workflow_and_plot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reuse synthetic and composite inputs over HTTP, without file access."""
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    headers = {
        "Accept": "application/json, text/event-stream",
        "Authorization": "Bearer test-token",
    }
    spec = {
        "kind": "synthetic_history",
        "fs": 16,
        "duration": 16,
        "seed": 11,
        "minimum": -30,
        "maximum": 80,
    }
    windows = {
        "kind": "cycle_count_windows",
        "windows": [{"stress": spec}, {"stress": dict(spec, seed=13)}],
        "resolve_residuals": True,
        "stress_concentration_factor": 1.2,
    }
    with TestClient(create_app(), base_url="http://127.0.0.1") as client:

        def rpc(method: str, params: dict) -> dict:
            response = client.post(
                "/mcp",
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": method,
                    "params": params,
                },
            )
            response.raise_for_status()
            return response.json()["result"]

        def call(name: str, **arguments) -> dict:
            result = rpc("tools/call", {"name": name, "arguments": arguments})
            assert not result.get("isError"), result
            return result

        initialized = rpc(
            "initialize",
            {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "tests", "version": "1"},
            },
        )
        assert (
            "For every plotting request, inspect plot_py_fatigue"
            in initialized["instructions"]
        )
        assert (
            call("generate_synthetic_data", specification=spec)[
                "structuredContent"
            ]["row_count"]
            == 256
        )
        counts = call("cycle_count", stress=windows)["structuredContent"]
        assert counts["lffd_solved"] and counts["max_stress_range"] > 100
        source = {"kind": "cycle_count", "stress": windows}
        curve = {"slope": 3, "intercept": 12}
        damage = call("miner_damage", source=source, curve=curve)[
            "structuredContent"
        ]
        assert damage["total_damage"] > 0
        plot = call(
            "plot_py_fatigue",
            request={"kind": "miner_damage", "source": source, "curve": curve},
        )
        assert plot["content"][0]["mimeType"] == "image/png"
        invalid = rpc(
            "tools/call",
            {
                "name": "plot_py_fatigue",
                "arguments": {
                    "request": {
                        "kind": "cycle_count",
                        "stress": {"csv_path": "hidden.csv"},
                    }
                },
            },
        )
        assert invalid["isError"]
