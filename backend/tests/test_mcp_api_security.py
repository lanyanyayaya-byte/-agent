"""P6.3.3, P6.3.4 & P6.3.6 verification suite: MCP API, RBAC, secrets, lifecycle & fault isolation.

Tests run against real FastAPI ASGI apps, real SQLite V6 databases, real local stdio
MCP subprocesses, and real local Streamable HTTP MCP servers.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from app.main import create_app
from app.mcp.models import McpError
from app.mcp.policy import parse_mcp_servers_json
from app.models.deepseek import DeepSeekAdapter
from app.services.models import ModelService
from tests.fixtures.mcp.http_test_server import run_http_mcp_server

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "mcp"
STDIO_SERVER_SCRIPT = FIXTURES_DIR / "stdio_test_server.py"

UI_TOKEN = "ui-token-p63-verification-0001"
HOST_TOKEN = "host-token-p63-verification-0002"
UI_HEADERS = {"Authorization": f"Bearer {UI_TOKEN}"}
HOST_HEADERS = {"Authorization": f"Bearer {HOST_TOKEN}"}

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@asynccontextmanager
async def live_mcp_app(db_path: Path, *, models: ModelService | None = None):
    app = create_app(UI_TOKEN, HOST_TOKEN, database_path=db_path, models=models)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8000") as client:
            yield app, client


async def test_ui_host_token_isolation_no_arbitrary_call_api_and_patch_audit(tmp_path, caplog):
    db_path = tmp_path / "business.sqlite3"
    caplog.set_level("INFO", logger="knowledgeagent.sidecar")

    async with live_mcp_app(db_path) as (_app, client):
        # Create one server via Host channel so we have a valid server_id
        created = await client.post(
            "/api/v1/_host/mcp/servers",
            headers=HOST_HEADERS,
            json={
                "name": "AuthProbe",
                "transport": "streamable_http",
                "url": "https://probe.example.com/mcp",
            },
        )
        assert created.status_code == 201
        assert created.headers.get("cache-control") == "no-store"
        server_id = created.json()["server_id"]

        # UI Token MUST receive 403 FORBIDDEN on EVERY privileged _host MCP route!
        host_calls = [
            ("GET", f"/api/v1/_host/mcp/servers/{server_id}", None),
            ("POST", "/api/v1/_host/mcp/servers", {"name": "X", "transport": "streamable_http", "url": "https://x.test/mcp"}),
            ("PATCH", f"/api/v1/_host/mcp/servers/{server_id}", {"name": "Hijack"}),
            ("POST", f"/api/v1/_host/mcp/servers/{server_id}/trust", {"expected_config_revision": 1, "expected_fingerprint": "0" * 64}),
            ("POST", f"/api/v1/_host/mcp/servers/{server_id}/revoke-trust", None),
            ("DELETE", f"/api/v1/_host/mcp/servers/{server_id}", None),
            ("POST", f"/api/v1/_host/mcp/servers/{server_id}/connect", None),
            ("POST", f"/api/v1/_host/mcp/servers/{server_id}/disconnect", None),
            ("POST", f"/api/v1/_host/mcp/servers/{server_id}/refresh", None),
            ("PATCH", f"/api/v1/_host/mcp/servers/{server_id}/tools/t1", {"permission_state": "allowed"}),
            ("GET", "/api/v1/_host/mcp/credential-refs", None),
            ("PUT", f"/api/v1/_host/mcp/servers/{server_id}/credentials", {"secrets": {"Authorization": "Bearer abc"}}),
            ("DELETE", f"/api/v1/_host/mcp/servers/{server_id}/credentials", None),
            ("POST", "/api/v1/_host/mcp/restore-complete", None),
        ]
        for method, path, body in host_calls:
            resp = await client.request(method, path, headers=UI_HEADERS, json=body)
            assert resp.status_code == 403, f"UI token unexpectedly got {resp.status_code} on {method} {path}"
            assert resp.json()["error"]["code"] == "FORBIDDEN"

            unauth = await client.request(method, path, json=body)
            assert unauth.status_code == 401

        # Host Token cannot call UI routes
        for ui_path in ("/api/v1/mcp/servers", f"/api/v1/mcp/servers/{server_id}/tools", "/api/v1/mcp/summary"):
            assert (await client.get(ui_path, headers=HOST_HEADERS)).status_code == 403
            ok = await client.get(ui_path, headers=UI_HEADERS)
            assert ok.status_code == 200
            assert ok.headers.get("cache-control") == "no-store"

        # No arbitrary tool call endpoint exists on UI or Host channel
        for ui_forbidden_path in (
            f"/api/v1/mcp/servers/{server_id}/call",
            f"/api/v1/mcp/servers/{server_id}/tools/echo/call",
        ):
            assert (await client.post(ui_forbidden_path, headers=UI_HEADERS, json={})).status_code == 404
            assert (await client.post(ui_forbidden_path, headers=HOST_HEADERS, json={})).status_code == 404
        for host_forbidden_path in (
            f"/api/v1/_host/mcp/servers/{server_id}/call",
            f"/api/v1/_host/mcp/servers/{server_id}/tools/echo/call",
        ):
            # Host middleware fails closed with 403 for UI token even on unregistered _host paths.
            assert (await client.post(host_forbidden_path, headers=UI_HEADERS, json={})).status_code == 403
            assert (await client.post(host_forbidden_path, headers=HOST_HEADERS, json={})).status_code == 404

        # Verify PATCH request auditing records method="PATCH" (not "OTHER")
        patched = await client.patch(
            f"/api/v1/_host/mcp/servers/{server_id}",
            headers=HOST_HEADERS,
            json={"name": "AuthProbeRenamed"},
        )
        assert patched.status_code == 200
        http_events = [
            json.loads(rec.getMessage())
            for rec in caplog.records
            if rec.getMessage().startswith("{")
        ]
        assert any(
            ev.get("event") == "http_request"
            and ev.get("method") == "PATCH"
            and ev.get("route") == "/api/v1/_host/mcp/servers/{server_id}"
            and ev.get("status") == 200
            for ev in http_events
        )


async def test_secret_bypass_rejection_json_import_non_execution_and_zero_leakage(tmp_path, caplog):
    db_path = tmp_path / "business.sqlite3"
    caplog.set_level("INFO", logger="knowledgeagent.sidecar")

    # 1. JSON import must NEVER execute any external command (verified via canary executable).
    canary_marker = tmp_path / "executed.marker"
    canary_script = tmp_path / "evil_mcp.sh"
    canary_script.write_text(f"#!/bin/sh\necho PWNED > {canary_marker}\n", encoding="utf-8")
    canary_script.chmod(canary_script.stat().st_mode | stat.S_IXUSR)

    import_secret = "sk-imported-secret-value-never-leak-987654321"
    raw_json = json.dumps({
        "mcpServers": {
            "CanaryStdio": {
                "command": str(canary_script),
                "args": ["--safe"],
                "env": {
                    "MODE": "readonly",
                    "BRAVE_API_KEY": import_secret,
                },
            }
        }
    })
    drafts = parse_mcp_servers_json(raw_json)
    assert len(drafts) == 1
    assert not canary_marker.exists(), "JSON import parsing executed the external program!"
    assert "BRAVE_API_KEY" in drafts[0].secret_env
    assert "BRAVE_API_KEY" not in drafts[0].plain_env
    assert drafts[0].plain_env == {"MODE": "readonly"}

    # Unknown fields and legacy SSE in JSON import must be rejected
    with pytest.raises(McpError) as exc:
        parse_mcp_servers_json(json.dumps({"mcpServers": {"A": {"command": "/bin/echo", "autoTrust": True}}}))
    assert exc.value.code == "MCP_INVALID_CONFIG"
    with pytest.raises(McpError) as exc:
        parse_mcp_servers_json(json.dumps({"mcpServers": {"A": {"type": "sse", "url": "https://a.test/sse"}}}))
    assert exc.value.code == "MCP_INVALID_CONFIG"

    async with live_mcp_app(db_path) as (_app, client):
        # 2. Secret smuggling bypasses rejected by API
        bypass_payloads = [
            # Secret in HTTP URL query parameter
            {
                "name": "BypassQuery",
                "transport": "streamable_http",
                "url": "https://api.example.com/mcp?api_key=super-secret-query-1234",
            },
            # Secret in plain HTTP Authorization header
            {
                "name": "BypassHeader",
                "transport": "streamable_http",
                "url": "https://api.example.com/mcp",
                "header_config": {
                    "Authorization": {"kind": "plain", "value": "Bearer leaked-header-token-1234"},
                },
            },
            # Secret in stdio command args
            {
                "name": "BypassArgFlag",
                "transport": "stdio",
                "command": sys.executable,
                "args": ["--api-key=secret-arg-value-1234"],
            },
            # Secret token shape in plain env
            {
                "name": "BypassPlainEnv",
                "transport": "stdio",
                "command": sys.executable,
                "env_config": {
                    "CUSTOM_VAR": {"kind": "plain", "value": "sk-plaintext-env-secret-1234567890"},
                },
            },
        ]
        for bad in bypass_payloads:
            resp = await client.post("/api/v1/_host/mcp/servers", headers=HOST_HEADERS, json=bad)
            assert resp.status_code == 422, f"Bypass payload was not rejected: {bad}"
            assert resp.json()["error"]["code"] == "MCP_INVALID_CONFIG"

        # 3. Create legitimate server with secret env, inject secret into memory, and verify ZERO leakage
        stdio_secret = "mcp-ultra-secret-token-xyz-999888777"
        created = await client.post(
            "/api/v1/_host/mcp/servers",
            headers=HOST_HEADERS,
            json={
                "name": "SecretStdio",
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(STDIO_SERVER_SCRIPT)],
                "env_config": {
                    "PLAIN_TAG": {"kind": "plain", "value": "safe"},
                    "SERVER_TOKEN": {"kind": "secret", "secret_key": "SERVER_TOKEN"},
                },
            },
        )
        assert created.status_code == 201
        server_id = created.json()["server_id"]
        assert created.json()["enabled"] is False
        assert created.json()["trust_state"] == "untrusted"
        assert created.json()["credentials_complete"] is False
        assert created.json()["missing_secret_keys"] == ["SERVER_TOKEN"]

        # Inject secret into Sidecar memory
        put_cred = await client.put(
            f"/api/v1/_host/mcp/servers/{server_id}/credentials",
            headers=HOST_HEADERS,
            json={"secrets": {"SERVER_TOKEN": stdio_secret}, "mode": "update"},
        )
        assert put_cred.status_code == 200
        assert put_cred.json()["credentials_complete"] is True
        assert stdio_secret not in put_cred.text

        # Undeclared secret key must be rejected
        bad_cred = await client.put(
            f"/api/v1/_host/mcp/servers/{server_id}/credentials",
            headers=HOST_HEADERS,
            json={"secrets": {"UNDECLARED_KEY": "some-val"}, "mode": "update"},
        )
        assert bad_cred.status_code == 422

        # UI and Host views must NEVER contain plaintext secret
        ui_servers = await client.get("/api/v1/mcp/servers", headers=UI_HEADERS)
        host_server = await client.get(f"/api/v1/_host/mcp/servers/{server_id}", headers=HOST_HEADERS)
        assert stdio_secret not in ui_servers.text
        assert stdio_secret not in host_server.text
        assert ui_servers.json()[0]["env_config"]["SERVER_TOKEN"] == {
            "kind": "secret",
            "secret_key": "SERVER_TOKEN",
            "configured": True,
        }

        # Raw SQLite file bytes and WAL must NEVER contain plaintext secret!
        for candidate in (db_path, db_path.with_name(db_path.name + "-wal"), db_path.with_name(db_path.name + "-shm")):
            if candidate.exists():
                assert stdio_secret.encode("utf-8") not in candidate.read_bytes()

        # Diagnostic logs must NEVER contain plaintext secret!
        for rec in caplog.records:
            assert stdio_secret not in rec.getMessage()

        assert not canary_marker.exists()


async def test_real_stdio_and_http_lifecycle_trust_rotation_list_changed_and_cross_server_secret_isolation(tmp_path):
    db_path = tmp_path / "business.sqlite3"
    secret_a = "Bearer http-secret-alpha-111222333"
    secret_b = "stdio-secret-beta-444555666"
    secret_b_rotated = "stdio-secret-beta-rotated-777888999"

    with run_http_mcp_server(expected_auth=secret_a) as (base_url, _state):
        async with live_mcp_app(db_path) as (app, client):
            # Create HTTP Server A (requires Authorization secret) & Stdio Server B (requires SERVER_TOKEN secret)
            srv_a = (
                await client.post(
                    "/api/v1/_host/mcp/servers",
                    headers=HOST_HEADERS,
                    json={
                        "name": "HttpAlpha",
                        "transport": "streamable_http",
                        "url": f"{base_url}/mcp",
                        "header_config": {
                            "Authorization": {"kind": "secret", "secret_key": "Authorization"},
                        },
                    },
                )
            ).json()
            srv_b = (
                await client.post(
                    "/api/v1/_host/mcp/servers",
                    headers=HOST_HEADERS,
                    json={
                        "name": "StdioBeta",
                        "transport": "stdio",
                        "command": sys.executable,
                        "args": [str(STDIO_SERVER_SCRIPT)],
                        "env_config": {
                            "SERVER_TOKEN": {"kind": "secret", "secret_key": "SERVER_TOKEN"},
                        },
                    },
                )
            ).json()
            id_a, id_b = srv_a["server_id"], srv_b["server_id"]

            # Untrusted server CANNOT connect
            assert (await client.post(f"/api/v1/_host/mcp/servers/{id_b}/connect", headers=HOST_HEADERS)).status_code == 409

            # Trust Server B & enable, but secrets are still missing -> connect MUST fail with MCP_CREDENTIALS_MISSING!
            trusted_b = await client.post(
                f"/api/v1/_host/mcp/servers/{id_b}/trust",
                headers=HOST_HEADERS,
                json={
                    "expected_config_revision": srv_b["config_revision"],
                    "expected_fingerprint": srv_b["current_fingerprint"],
                    "enable": True,
                },
            )
            assert trusted_b.status_code == 200
            missing_connect = await client.post(f"/api/v1/_host/mcp/servers/{id_b}/connect", headers=HOST_HEADERS)
            assert missing_connect.status_code == 409
            assert missing_connect.json()["error"]["code"] == "MCP_CREDENTIALS_MISSING"

            # Inject distinct secrets for Server A and Server B
            assert (
                await client.put(
                    f"/api/v1/_host/mcp/servers/{id_a}/credentials",
                    headers=HOST_HEADERS,
                    json={"secrets": {"Authorization": secret_a}, "mode": "update"},
                )
            ).status_code == 200
            assert (
                await client.put(
                    f"/api/v1/_host/mcp/servers/{id_b}/credentials",
                    headers=HOST_HEADERS,
                    json={"secrets": {"SERVER_TOKEN": secret_b}, "mode": "update"},
                )
            ).status_code == 200

            # Trust & connect both servers
            assert (
                await client.post(
                    f"/api/v1/_host/mcp/servers/{id_a}/trust",
                    headers=HOST_HEADERS,
                    json={
                        "expected_config_revision": srv_a["config_revision"],
                        "expected_fingerprint": srv_a["current_fingerprint"],
                        "enable": True,
                    },
                )
            ).status_code == 200
            conn_a = await client.post(f"/api/v1/_host/mcp/servers/{id_a}/connect", headers=HOST_HEADERS)
            conn_b = await client.post(f"/api/v1/_host/mcp/servers/{id_b}/connect", headers=HOST_HEADERS)
            assert conn_a.status_code == 200 and conn_a.json()["runtime"]["state"] == "connected"
            assert conn_b.status_code == 200 and conn_b.json()["runtime"]["state"] == "connected"

            # Verify cross-server secret isolation: Server B's stdio env has secret_b, NEVER secret_a!
            env_res = await app.state.mcp_manager.call_tool(id_b, "dump_env", {})
            child_env = json.loads(env_res.text)
            assert child_env["SERVER_TOKEN"] == secret_b
            assert secret_a not in json.dumps(child_env)

            # All newly discovered tools default to denied / not effective_allowed
            tools_b = (await client.get(f"/api/v1/mcp/servers/{id_b}/tools", headers=UI_HEADERS)).json()
            assert len(tools_b) >= 4
            assert all(t["permission_state"] == "denied" and t["effective_allowed"] is False for t in tools_b)

            # Authorize echo_text on Server B
            echo_tool = next(t for t in tools_b if t["original_tool_name"] == "echo_text")
            auth_resp = await client.patch(
                f"/api/v1/_host/mcp/servers/{id_b}/tools/{echo_tool['tool_id']}",
                headers=HOST_HEADERS,
                json={
                    "permission_state": "allowed",
                    "expected_definition_hash": echo_tool["definition_hash"],
                    "expected_config_revision": 1,
                },
            )
            assert auth_resp.status_code == 200
            assert auth_resp.json()["effective_allowed"] is True

            summary = (await client.get("/api/v1/mcp/summary", headers=UI_HEADERS)).json()
            assert summary["connected_servers"] == 2
            assert summary["effective_allowed_tools"] == 1

            # Trigger real tools/list_changed on Server B: echo_text description changes -> old approval IMMEDIATELY invalidated!
            before_ver = app.state.mcp_manager.get_diagnostics(id_b).catalog_version
            await app.state.mcp_manager.call_tool(id_b, "mutate_tools", {})
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                if app.state.mcp_manager.get_diagnostics(id_b).catalog_version > before_ver:
                    break
                await asyncio.sleep(0.02)

            tools_after_drift = (await client.get(f"/api/v1/mcp/servers/{id_b}/tools", headers=UI_HEADERS)).json()
            echo_after = next(t for t in tools_after_drift if t["original_tool_name"] == "echo_text")
            assert echo_after["definition_hash"] != echo_tool["definition_hash"]
            assert echo_after["permission_state"] == "denied"
            assert echo_after["definition_changed"] is True
            assert echo_after["effective_allowed"] is False

            # Credential value rotation MUST disconnect the active session immediately
            rot = await client.put(
                f"/api/v1/_host/mcp/servers/{id_b}/credentials",
                headers=HOST_HEADERS,
                json={"secrets": {"SERVER_TOKEN": secret_b_rotated}, "mode": "update"},
            )
            assert rot.status_code == 200
            view_after_rot = (await client.get(f"/api/v1/_host/mcp/servers/{id_b}", headers=HOST_HEADERS)).json()
            assert view_after_rot["runtime"]["state"] == "disconnected"

            # Reconnect uses the rotated secret
            reconn = await client.post(f"/api/v1/_host/mcp/servers/{id_b}/connect", headers=HOST_HEADERS)
            assert reconn.status_code == 200 and reconn.json()["runtime"]["state"] == "connected"
            env_res_2 = await app.state.mcp_manager.call_tool(id_b, "dump_env", {})
            assert json.loads(env_res_2.text)["SERVER_TOKEN"] == secret_b_rotated

            # Security-critical config change resets trust & disconnects immediately
            sec_edit = await client.patch(
                f"/api/v1/_host/mcp/servers/{id_b}",
                headers=HOST_HEADERS,
                json={"args": [str(STDIO_SERVER_SCRIPT), "--extra-flag"]},
            )
            assert sec_edit.status_code == 200
            assert sec_edit.json()["trust_state"] == "untrusted"
            assert sec_edit.json()["config_revision"] == 2
            assert sec_edit.json()["runtime"]["state"] == "disconnected"


async def test_startup_recovery_fault_isolation_and_normal_chat_unblocked(tmp_path):
    db_path = tmp_path / "business.sqlite3"

    # Phase 1: Configure 4 servers with distinct states, then shut down cleanly
    async with live_mcp_app(db_path) as (_app, client):
        # 1) Enabled + trusted + valid stdio (no secret needed) -> SHOULD auto-connect
        s_ok = (
            await client.post(
                "/api/v1/_host/mcp/servers",
                headers=HOST_HEADERS,
                json={
                    "name": "GoodStdio",
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": [str(STDIO_SERVER_SCRIPT)],
                },
            )
        ).json()
        await client.post(
            f"/api/v1/_host/mcp/servers/{s_ok['server_id']}/trust",
            headers=HOST_HEADERS,
            json={
                "expected_config_revision": s_ok["config_revision"],
                "expected_fingerprint": s_ok["current_fingerprint"],
                "enable": True,
            },
        )

        # 2) Enabled + trusted + secret required (will NOT be replayed on restart) -> MUST NOT auto-connect
        s_missing_cred = (
            await client.post(
                "/api/v1/_host/mcp/servers",
                headers=HOST_HEADERS,
                json={
                    "name": "MissingCredStdio",
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": [str(STDIO_SERVER_SCRIPT)],
                    "env_config": {"API_TOKEN": {"kind": "secret", "secret_key": "API_TOKEN"}},
                },
            )
        ).json()
        await client.post(
            f"/api/v1/_host/mcp/servers/{s_missing_cred['server_id']}/trust",
            headers=HOST_HEADERS,
            json={
                "expected_config_revision": s_missing_cred["config_revision"],
                "expected_fingerprint": s_missing_cred["current_fingerprint"],
                "enable": True,
            },
        )

        # 3) Enabled + trusted + broken HTTP endpoint -> fails independently, MUST NOT block GoodStdio or chat
        s_broken = (
            await client.post(
                "/api/v1/_host/mcp/servers",
                headers=HOST_HEADERS,
                json={
                    "name": "BrokenHttp",
                    "transport": "streamable_http",
                    "url": "http://127.0.0.1:1/unreachable-mcp",
                },
            )
        ).json()
        await client.post(
            f"/api/v1/_host/mcp/servers/{s_broken['server_id']}/trust",
            headers=HOST_HEADERS,
            json={
                "expected_config_revision": s_broken["config_revision"],
                "expected_fingerprint": s_broken["current_fingerprint"],
                "enable": True,
            },
        )

        # 4) Untrusted server -> MUST NOT auto-connect
        s_untrusted = (
            await client.post(
                "/api/v1/_host/mcp/servers",
                headers=HOST_HEADERS,
                json={
                    "name": "UntrustedStdio",
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": [str(STDIO_SERVER_SCRIPT)],
                },
            )
        ).json()

    # Phase 2: Restart app against the same SQLite DB
    async with live_mcp_app(db_path) as (app, client):
        # Before restore-complete, NO server is connected and normal session/chat APIs respond immediately!
        status_resp = await client.get("/api/v1/status", headers=UI_HEADERS)
        assert status_resp.status_code == 200
        sess_resp = await client.post("/api/v1/sessions", headers=UI_HEADERS, json={"title": "正常聊天不受影响"})
        assert sess_resp.status_code == 201

        # Trigger restore-complete -> non-blocking 204
        restore_resp = await client.post("/api/v1/_host/mcp/restore-complete", headers=HOST_HEADERS)
        assert restore_resp.status_code == 204

        # Wait for background recovery task to finish
        if app.state.mcp_service._recovery_task is not None:
            await asyncio.wait_for(app.state.mcp_service._recovery_task, timeout=10.0)

        views = {v["server_id"]: v for v in (await client.get("/api/v1/mcp/servers", headers=UI_HEADERS)).json()}
        assert views[s_ok["server_id"]]["runtime"]["state"] == "connected"
        assert views[s_missing_cred["server_id"]]["runtime"]["state"] == "disconnected"
        assert views[s_missing_cred["server_id"]]["credentials_complete"] is False
        assert views[s_broken["server_id"]]["runtime"]["state"] == "error"
        assert views[s_untrusted["server_id"]]["runtime"]["state"] == "disconnected"

        # Normal chat session listing remains completely healthy despite BrokenHttp error
        sessions = await client.get("/api/v1/sessions", headers=UI_HEADERS)
        assert sessions.status_code == 200
        assert len(sessions.json()) == 1


async def test_catalog_stale_during_refresh_and_safe_degradation_on_refresh_failure(tmp_path):
    """Verify P6.3.2 #8: stale catalog blocks tool usage during list_changed & degrades safely on refresh failure."""
    db_path = tmp_path / "business.sqlite3"
    async with live_mcp_app(db_path) as (app, client):
        srv = (
            await client.post(
                "/api/v1/_host/mcp/servers",
                headers=HOST_HEADERS,
                json={
                    "name": "StaleGuardStdio",
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": [str(STDIO_SERVER_SCRIPT)],
                },
            )
        ).json()
        sid = srv["server_id"]
        await client.post(
            f"/api/v1/_host/mcp/servers/{sid}/trust",
            headers=HOST_HEADERS,
            json={
                "expected_config_revision": srv["config_revision"],
                "expected_fingerprint": srv["current_fingerprint"],
                "enable": True,
            },
        )
        assert (await client.post(f"/api/v1/_host/mcp/servers/{sid}/connect", headers=HOST_HEADERS)).status_code == 200

        tools = (await client.get(f"/api/v1/mcp/servers/{sid}/tools", headers=UI_HEADERS)).json()
        echo_tool = next(t for t in tools if t["original_tool_name"] == "echo_text")
        await client.patch(
            f"/api/v1/_host/mcp/servers/{sid}/tools/{echo_tool['tool_id']}",
            headers=HOST_HEADERS,
            json={
                "permission_state": "allowed",
                "expected_definition_hash": echo_tool["definition_hash"],
                "expected_config_revision": 1,
            },
        )
        assert app.state.mcp_manager.list_tools(sid)

        # Force the live connection's list_tools() to fail during refresh and trigger refresh
        slot = app.state.mcp_manager._slots[sid]
        actor = slot.actor
        assert actor is not None and actor._conn is not None
        orig_list_tools = actor._conn.list_tools

        async def _failing_list_tools():
            raise McpError("MCP_DISCOVERY_FAILED", "模拟刷新失败")

        actor._conn.list_tools = _failing_list_tools  # type: ignore[method-assign]
        ref_resp = await client.post(f"/api/v1/_host/mcp/servers/{sid}/refresh", headers=HOST_HEADERS)
        assert ref_resp.status_code == 502
        assert ref_resp.json()["error"]["code"] == "MCP_DISCOVERY_FAILED"

        # Safe degradation: catalog is marked stale, live tools empty, call_tool rejected, effective_allowed=False!
        assert app.state.mcp_manager.is_catalog_stale(sid) is True
        assert app.state.mcp_manager.list_tools(sid) == ()
        with pytest.raises(McpError) as exc:
            await app.state.mcp_manager.call_tool(sid, "echo_text", {"text": "blocked"})
        assert exc.value.code == "MCP_SERVER_UNAVAILABLE"
        ui_tools = (await client.get(f"/api/v1/mcp/servers/{sid}/tools", headers=UI_HEADERS)).json()
        assert all(t["effective_allowed"] is False for t in ui_tools)

        # Recovery: restoring list_tools and calling refresh clears stale state cleanly!
        actor._conn.list_tools = orig_list_tools  # type: ignore[method-assign]
        recover_resp = await client.post(f"/api/v1/_host/mcp/servers/{sid}/refresh", headers=HOST_HEADERS)
        assert recover_resp.status_code == 200
        assert app.state.mcp_manager.is_catalog_stale(sid) is False
        recovered_echo = next(t for t in recover_resp.json() if t["original_tool_name"] == "echo_text")
        assert recovered_echo["effective_allowed"] is True
