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
