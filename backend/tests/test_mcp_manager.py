"""Comprehensive real-transport verification suite for P6.2 (MCP Policy, Client, Manager).

Uses real local stdio subprocesses and real local Uvicorn/Starlette Streamable HTTP
servers — no mocked transport success results.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from app.mcp.manager import McpManager
from app.mcp.models import McpError, McpRuntimeLimits, McpServerConfig
from app.mcp.policy import (
    build_stdio_environment,
    compute_config_trust_fingerprint,
    compute_tool_definition_hash,
    is_high_risk_command,
    redact_text,
    resolve_stdio_executable,
    validate_http_url,
    validate_resolved_ip,
    validate_server_config,
)
from tests.fixtures.mcp.http_test_server import run_http_mcp_server

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "mcp"
STDIO_SERVER_SCRIPT = FIXTURES_DIR / "stdio_test_server.py"

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _stdio_config(
    server_id: str = "stdio-main",
    *,
    name: str = "Local Stdio Test",
    extra_args: tuple[str, ...] = (),
    env: dict[str, str] | None = None,
    secret_env: dict[str, str] | None = None,
    enabled: bool = True,
) -> McpServerConfig:
    return McpServerConfig(
        server_id=server_id,
        name=name,
        transport="stdio",
        command=sys.executable,
        args=(str(STDIO_SERVER_SCRIPT), *extra_args),
        env=env or {},
        secret_env=secret_env or {},
        enabled=enabled,
    )


def _http_config(
    base_url: str,
    server_id: str = "http-main",
    *,
    path: str = "/mcp",
    name: str = "Local HTTP Test",
    headers: dict[str, str] | None = None,
    secret_headers: dict[str, str] | None = None,
    allow_private_network: bool = False,
) -> McpServerConfig:
    return McpServerConfig(
        server_id=server_id,
        name=name,
        transport="streamable_http",
        url=f"{base_url}{path}",
        headers=headers or {},
        secret_headers=secret_headers or {},
        allow_private_network=allow_private_network,
    )


def _read_markers(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Policy & Security Boundary Unit Verification
# ---------------------------------------------------------------------------


def test_policy_validation_hashing_and_redaction(tmp_path):
    # Mutual exclusion between stdio and streamable_http
    with pytest.raises(McpError) as exc:
        validate_server_config(
            McpServerConfig(
                server_id="bad-1",
                name="Bad",
                transport="stdio",
                command=sys.executable,
                url="http://127.0.0.1:8000/mcp",
            )
        )
    assert exc.value.code == "MCP_INVALID_CONFIG"

    # Non-loopback plain HTTP must be rejected
    with pytest.raises(McpError) as exc:
        validate_http_url("http://example.com/mcp")
    assert exc.value.code == "MCP_NETWORK_POLICY_VIOLATION"

    # Embedded URL credentials must be rejected
    with pytest.raises(McpError) as exc:
        validate_http_url("https://user:pass@example.com/mcp")
    assert exc.value.code == "MCP_NETWORK_POLICY_VIOLATION"

    # Link-local / cloud metadata IP must always be rejected
    with pytest.raises(McpError) as exc:
        validate_resolved_ip("169.254.169.254", scheme="https", hostname="metadata.internal", allow_private_network=True)
    assert exc.value.code == "MCP_NETWORK_POLICY_VIOLATION"

    # Private RFC1918 IP rejected by default on non-loopback
    with pytest.raises(McpError) as exc:
        validate_resolved_ip("192.168.1.20", scheme="https", hostname="lan.example", allow_private_network=False)
    assert exc.value.code == "MCP_NETWORK_POLICY_VIOLATION"

    # Plain HTTP resolved to non-loopback IP rejected (anti-DNS-rebinding)
    with pytest.raises(McpError) as exc:
        validate_resolved_ip("93.184.216.34", scheme="http", hostname="localhost", allow_private_network=True)
    assert exc.value.code == "MCP_NETWORK_POLICY_VIOLATION"

    # Forbidden env prefixes rejected
    with pytest.raises(McpError) as exc:
        validate_server_config(
            McpServerConfig(
                server_id="bad-env",
                name="Bad Env",
                transport="stdio",
                command=sys.executable,
                env={"DEEPSEEK_API_KEY": "nope"},
            )
        )
    assert exc.value.code == "MCP_INVALID_CONFIG"

    # Display name rename does NOT change trust fingerprint
    cfg_a = _stdio_config("stable-id", name="Display Name A")
    cfg_b = _stdio_config("stable-id", name="Display Name B")
    assert compute_config_trust_fingerprint(cfg_a) == compute_config_trust_fingerprint(cfg_b)

    # Tool definition hash changes when description/schema changes
    h1 = compute_tool_definition_hash(
        name="t", title=None, description="v1", input_schema={"type": "object"}, output_schema=None, annotations={}
    )
    h2 = compute_tool_definition_hash(
        name="t", title=None, description="v2", input_schema={"type": "object"}, output_schema=None, annotations={}
    )
    assert h1 != h2

    # Shell-free resolution & high-risk detection
    resolved = resolve_stdio_executable(sys.executable)
    assert resolved.is_absolute()
    assert is_high_risk_command(Path("/opt/homebrew/bin/npx"), ("-y", "pkg"))
    assert is_high_risk_command(resolved, ("-c", "print(1)"))
    assert not is_high_risk_command(resolved, (str(STDIO_SERVER_SCRIPT),))

    # Redaction strips known secrets and bearer/sk tokens
    redacted = redact_text(
        "failed with Bearer abcdef12345678 and sk-testsecret999999 and my-super-secret",
        secrets=("my-super-secret",),
    )
    assert "abcdef12345678" not in redacted
    assert "sk-testsecret999999" not in redacted
    assert "my-super-secret" not in redacted

    # Environment allowlist excludes host secrets and SHELL
    fake_host_env = {
        "HOME": str(tmp_path),
        "USER": "tester",
        "SHELL": "/bin/zsh",
        "DEEPSEEK_API_KEY": "sk-host-deepseek-secret",
        "OPENAI_API_KEY": "sk-host-openai-secret",
        "SILICONFLOW_API_KEY": "sk-host-siliconflow-secret",
        "KNOWLEDGEAGENT_HOST_TOKEN": "host-token-secret",
        "DYLD_INSERT_LIBRARIES": "/tmp/evil.dylib",
    }
    child_env = build_stdio_environment(_stdio_config(env={"CUSTOM_PLAIN": "ok"}), host_environ=fake_host_env)
    assert child_env["CUSTOM_PLAIN"] == "ok"
    assert child_env["HOME"] == str(tmp_path)
    for forbidden in (
        "SHELL",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "SILICONFLOW_API_KEY",
        "KNOWLEDGEAGENT_HOST_TOKEN",
        "DYLD_INSERT_LIBRARIES",
    ):
        assert forbidden not in child_env


# ---------------------------------------------------------------------------
# 1, 2, 3, 4, 16: Real stdio lifecycle, pagination, text+structured, env isolation
# ---------------------------------------------------------------------------


async def test_stdio_real_connect_pagination_calls_and_env_isolation(monkeypatch):
    # Inject fake model/host secrets into parent process environment
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-parent-deepseek-leak-check")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-parent-openai-leak-check")
    monkeypatch.setenv("SILICONFLOW_API_KEY", "sk-parent-siliconflow-leak-check")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-parent-leak-check")
    monkeypatch.setenv("KNOWLEDGEAGENT_HOST_TOKEN", "host-token-leak-check")
    monkeypatch.setenv("SHELL", "/bin/zsh")

    manager = McpManager()
    try:
        cfg = _stdio_config(
            "stdio-e2e",
            env={"PLAIN_MARKER": "hello-plain"},
            secret_env={"SERVER_TOKEN": "mcp-only-secret-42"},
        )
        await manager.register_server(cfg)
        diag = await manager.connect("stdio-e2e")
        assert diag.state == "connected"
        assert diag.generation == 1
        assert diag.protocol_version == "2025-11-25"
        assert diag.server_name_reported == "local-stdio-test-server"
        assert diag.child_pid is not None and _pid_alive(diag.child_pid)

        # Verify 2-page cursor pagination discovered tools from BOTH page 1 and page 2
        tools = manager.list_tools("stdio-e2e")
        names = {t.original_tool_name for t in tools}
        assert {"echo_text", "structured_calc", "dump_env", "slow_tool", "mutate_tools", "spawn_grandchild"} <= names
        assert len(tools) == 10

        # Normal text tool call
        echo = await manager.call_tool("stdio-e2e", "echo_text", {"text": "knowledgeagent"})
        assert not echo.is_error
        assert echo.text == "echo:knowledgeagent"
        assert echo.generation == 1

        # Structured content + non-text (image) omission
        calc = await manager.call_tool("stdio-e2e", "structured_calc", {"a": 7, "b": 5})
        assert not calc.is_error
        assert calc.text == "sum=12"
        assert calc.structured_content == {"a": 7, "b": 5, "sum": 12}
        assert calc.omitted_non_text_types == ("image",)

        # Tool logical error (isError=true) preserved cleanly without faking success
        err_res = await manager.call_tool("stdio-e2e", "return_error", {})
        assert err_res.is_error is True
        assert "simulated tool failure" in err_res.text

        # Real child process environment isolation check (#16)
        env_res = await manager.call_tool("stdio-e2e", "dump_env", {})
        child_env = env_res.structured_content["env"]
        assert child_env["PLAIN_MARKER"] == "hello-plain"
        assert child_env["SERVER_TOKEN"] == "mcp-only-secret-42"
        for leaked in (
            "DEEPSEEK_API_KEY",
            "OPENAI_API_KEY",
            "SILICONFLOW_API_KEY",
            "LANGSMITH_API_KEY",
            "KNOWLEDGEAGENT_HOST_TOKEN",
            "SHELL",
        ):
            assert leaked not in child_env

        pid = diag.child_pid
    finally:
        await manager.shutdown()

    assert not _pid_alive(pid)


# ---------------------------------------------------------------------------
# 1, 2, 3, 4, 10: Real Streamable HTTP (JSON + SSE + auth + dynamic list_changed)
# ---------------------------------------------------------------------------


async def test_streamable_http_real_connect_pagination_sse_auth_and_list_changed():
    secret_token = "Bearer mcp-http-secret-token-999"
    catalog_events: list[tuple[str, int, int]] = []

    with run_http_mcp_server(expected_auth=secret_token) as (base_url, state):
        manager = McpManager(
            on_catalog_updated=lambda sid, tools, ver: catalog_events.append((sid, len(tools), ver))
        )
        try:
            await manager.register_server(
                _http_config(base_url, "http-e2e", secret_headers={"Authorization": secret_token})
            )
            diag = await manager.connect("http-e2e")
            assert diag.state == "connected"
            assert diag.server_name_reported == "local-http-test-server"
            assert diag.tool_count == 8  # 3 on page 1 + 5 on page 2

            # JSON response tool call
            echo = await manager.call_tool("http-e2e", "http_echo", {"text": "streamable"})
            assert not echo.is_error
            assert echo.text == "http_echo:streamable"

            # SSE response with structured content + omitted resource_link
            sse_res = await manager.call_tool("http-e2e", "http_structured", {"x": 9})
            assert not sse_res.is_error
            assert sse_res.text == "squared=81"
            assert sse_res.structured_content == {"x": 9, "squared": 81}
            assert sse_res.omitted_non_text_types == ("resource_link",)

            # Logical error
            err_res = await manager.call_tool("http-e2e", "http_error", {})
            assert err_res.is_error is True

            # Dynamic tool list change notification over GET SSE stream (#10)
            before_hashes = {t.original_tool_name: t.definition_hash for t in manager.list_tools("http-e2e")}
            before_ver = manager.get_diagnostics("http-e2e").catalog_version
            # Allow GET SSE stream to register on server
            for _ in range(30):
                if state.sse_queues:
                    break
                await asyncio.sleep(0.02)

            await manager.call_tool("http-e2e", "http_mutate_tools", {})
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                if manager.get_diagnostics("http-e2e").catalog_version > before_ver:
                    break
                await asyncio.sleep(0.02)

            after_tools = manager.list_tools("http-e2e")
            after_hashes = {t.original_tool_name: t.definition_hash for t in after_tools}
            assert "http_dynamic_tool" in after_hashes
            assert after_hashes["http_echo"] != before_hashes["http_echo"]
            dyn = await manager.call_tool("http-e2e", "http_dynamic_tool", {})
            assert dyn.text == "http_dynamic_ok"
            assert len(catalog_events) >= 2
        finally:
            await manager.shutdown()

        # Verify session termination DELETE was sent on close
        assert len(state.terminated_sessions) >= 1


# ---------------------------------------------------------------------------
# 5, 6, 9, 15: Multi-server concurrency, fault isolation, repeated connect/disconnect, cross-task safety
# ---------------------------------------------------------------------------


async def test_multi_server_isolation_repeated_connect_and_cross_task_safety():
    with run_http_mcp_server() as (base_url, _state):
        limits = McpRuntimeLimits(connect_timeout_seconds=3.0, call_timeout_seconds=3.0)
        manager = McpManager(limits=limits)
        try:
            await manager.register_server(_stdio_config("stdio-ok"))
            await manager.register_server(_http_config(base_url, "http-ok"))
            await manager.register_server(
                McpServerConfig(
                    server_id="stdio-broken",
                    name="Broken Stdio",
                    transport="stdio",
                    command="/nonexistent/mcp/binary",
                )
            )

            # Connect from separate tasks to exercise cross-task safety (#15)
            t1 = asyncio.create_task(manager.connect("stdio-ok"))
            t2 = asyncio.create_task(manager.connect("http-ok"))
            d1, d2 = await asyncio.gather(t1, t2)
            assert d1.state == "connected"
            assert d2.state == "connected"

            # Broken server fails independently without hurting healthy servers (#6)
            with pytest.raises(McpError) as exc:
                await manager.connect("stdio-broken")
            assert exc.value.code == "MCP_EXECUTABLE_NOT_FOUND"
            assert manager.get_diagnostics("stdio-broken").state == "error"

            # Concurrent calls across healthy stdio & HTTP servers from different tasks (#5, #15)
            r1, r2 = await asyncio.gather(
                asyncio.create_task(manager.call_tool("stdio-ok", "echo_text", {"text": "s1"})),
                asyncio.create_task(manager.call_tool("http-ok", "http_echo", {"text": "h1"})),
            )
            assert r1.text == "echo:s1"
            assert r2.text == "http_echo:h1"

            # Idempotent connect returns same generation; reconnect increments generation (#9)
            same = await asyncio.create_task(manager.connect("stdio-ok"))
            assert same.generation == d1.generation
            old_pid = d1.child_pid
            reconnected = await asyncio.create_task(manager.reconnect("stdio-ok"))
            assert reconnected.generation == d1.generation + 1
            assert reconnected.child_pid != old_pid
            assert old_pid is not None and not _pid_alive(old_pid)

            # Repeated idempotent disconnects from separate tasks (zero CancelScope errors)
            await asyncio.create_task(manager.disconnect("stdio-ok"))
            await asyncio.create_task(manager.disconnect("stdio-ok"))
            assert manager.get_diagnostics("stdio-ok").state == "disconnected"
            assert reconnected.child_pid is not None and not _pid_alive(reconnected.child_pid)

            # HTTP sibling still healthy after stdio disconnect
            r3 = await manager.call_tool("http-ok", "http_echo", {"text": "still-alive"})
            assert r3.text == "http_echo:still-alive"
        finally:
            await asyncio.create_task(manager.shutdown())


# ---------------------------------------------------------------------------
# 7, 8: Timeout (zero auto-retry), queued cancellation, and in-flight cancellation
# ---------------------------------------------------------------------------


async def test_timeout_zero_retry_and_queued_plus_inflight_cancellation(tmp_path):
    marker = tmp_path / "stdio_markers.jsonl"
    limits = McpRuntimeLimits(
        connect_timeout_seconds=5.0,
        call_timeout_seconds=0.35,
        max_queued_calls_per_server=2,
    )
    manager = McpManager(limits=limits)
    try:
        await manager.register_server(
            _stdio_config("stdio-ctl", extra_args=("--marker-file", str(marker)))
        )
        await manager.connect("stdio-ctl")

        # 7. Tool timeout -> MCP_TOOL_TIMEOUT + uncertain_outcome=True + zero auto-retry
        with pytest.raises(McpError) as exc:
            await manager.call_tool("stdio-ctl", "slow_tool", {"delay": 2.0, "tag": "timeout-case"}, timeout_seconds=0.25)
        assert exc.value.code == "MCP_TOOL_TIMEOUT"
        assert exc.value.uncertain_outcome is True

        # Wait briefly for courtesy cancel notification to reach server
        await asyncio.sleep(0.1)
        markers = _read_markers(marker)
        started_timeout = [m for m in markers if m["event"] == "call_started" and m["args"].get("tag") == "timeout-case"]
        assert len(started_timeout) == 1  # Strictly NO automatic retry!

        # 8. In-flight cancellation AND queued cancellation
        t_inflight = asyncio.create_task(
            manager.call_tool("stdio-ctl", "slow_tool", {"delay": 3.0, "tag": "inflight-cancel"}, timeout_seconds=5.0)
        )
        # Wait until inflight call is actually executing on server
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if any(m["event"] == "call_started" and m["args"].get("tag") == "inflight-cancel" for m in _read_markers(marker)):
                break
            await asyncio.sleep(0.02)

        t_queued = asyncio.create_task(
            manager.call_tool("stdio-ctl", "slow_tool", {"delay": 0.1, "tag": "queued-cancel"}, timeout_seconds=5.0)
        )
        t_queued_2 = asyncio.create_task(
            manager.call_tool("stdio-ctl", "echo_text", {"text": "after-cancels"}, timeout_seconds=5.0)
        )
        await asyncio.sleep(0.03)

        # Queue is now full (max_queued_calls_per_server=2) -> 3rd queued call gets MCP_SERVER_BUSY
        with pytest.raises(McpError) as busy_exc:
            await manager.call_tool("stdio-ctl", "echo_text", {"text": "overflow"})
        assert busy_exc.value.code == "MCP_SERVER_BUSY"

        # Cancel the queued call BEFORE it ever executes, and cancel the in-flight call
        t_queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t_queued

        t_inflight.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t_inflight

        # Subsequent queued call still succeeds cleanly on the SAME connection!
        after = await t_queued_2
        assert after.text == "echo:after-cancels"

        await asyncio.sleep(0.05)
        final_markers = _read_markers(marker)
        # Queued cancelled call NEVER reached the MCP server
        assert not any(m["event"] == "call_started" and m["args"].get("tag") == "queued-cancel" for m in final_markers)
        # In-flight cancelled call received courtesy notifications/cancelled
        assert any(m["event"] == "notification_cancelled_received" for m in final_markers)
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# 10 (stdio): Dynamic tools/list_changed refresh on stdio
# ---------------------------------------------------------------------------


async def test_stdio_dynamic_tools_list_changed_refreshes_catalog():
    manager = McpManager()
    try:
        await manager.register_server(_stdio_config("stdio-dyn"))
        await manager.connect("stdio-dyn")
        v1 = manager.get_diagnostics("stdio-dyn").catalog_version
        old_hashes = {t.original_tool_name: t.definition_hash for t in manager.list_tools("stdio-dyn")}
        assert "dynamic_tool" not in old_hashes

        await manager.call_tool("stdio-dyn", "mutate_tools", {})
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if manager.get_diagnostics("stdio-dyn").catalog_version > v1:
                break
            await asyncio.sleep(0.02)

        new_hashes = {t.original_tool_name: t.definition_hash for t in manager.list_tools("stdio-dyn")}
        assert "dynamic_tool" in new_hashes
        assert new_hashes["echo_text"] != old_hashes["echo_text"]
        res = await manager.call_tool("stdio-dyn", "dynamic_tool", {})
        assert res.text == "dynamic_ok"
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# 11, 12: Oversized response limit (stdio + HTTP JSON + HTTP SSE) and malformed responses
# ---------------------------------------------------------------------------


async def test_oversized_and_malformed_responses_stdio_and_http():
    limits = McpRuntimeLimits(max_transport_bytes=4096, call_timeout_seconds=3.0)

    # stdio oversized
    mgr_stdio = McpManager(limits=limits)
    try:
        await mgr_stdio.register_server(_stdio_config("stdio-size"))
        await mgr_stdio.connect("stdio-size")
        with pytest.raises(McpError) as exc:
            await mgr_stdio.call_tool("stdio-size", "oversized_response", {"size_bytes": 10_000})
        assert exc.value.code == "MCP_RESPONSE_TOO_LARGE"
    finally:
        await mgr_stdio.shutdown()

    # stdio malformed JSON
    mgr_bad = McpManager(limits=limits)
    try:
        await mgr_bad.register_server(_stdio_config("stdio-malformed"))
        await mgr_bad.connect("stdio-malformed")
        with pytest.raises(McpError) as exc:
            await mgr_bad.call_tool("stdio-malformed", "malformed_response", {})
        assert exc.value.code == "MCP_MALFORMED_RESPONSE"
    finally:
        await mgr_bad.shutdown()

    # Streamable HTTP oversized JSON (Content-Length) + oversized SSE (chunked) + malformed JSON
    with run_http_mcp_server() as (base_url, _state):
        mgr_http = McpManager(limits=limits)
        try:
            await mgr_http.register_server(_http_config(base_url, "http-size-json"))
            await mgr_http.connect("http-size-json")
            with pytest.raises(McpError) as exc:
                await mgr_http.call_tool("http-size-json", "http_oversized_json", {"size_bytes": 10_000})
            assert exc.value.code == "MCP_RESPONSE_TOO_LARGE"

            await mgr_http.register_server(_http_config(base_url, "http-size-sse"))
            await mgr_http.connect("http-size-sse")
            with pytest.raises(McpError) as exc:
                await mgr_http.call_tool("http-size-sse", "http_oversized_sse", {"size_bytes": 10_000})
            assert exc.value.code == "MCP_RESPONSE_TOO_LARGE"

            await mgr_http.register_server(_http_config(base_url, "http-malformed"))
            await mgr_http.connect("http-malformed")
            with pytest.raises(McpError) as exc:
                await mgr_http.call_tool("http-malformed", "http_malformed", {})
            assert exc.value.code == "MCP_MALFORMED_RESPONSE"
        finally:
            await mgr_http.shutdown()


# ---------------------------------------------------------------------------
# Streamable HTTP security: redirect rejection & TOCTOU-free DNS IP validation
# ---------------------------------------------------------------------------


async def test_http_rejects_redirects_and_dns_rebinding_in_connect_tcp(monkeypatch):
    with run_http_mcp_server() as (base_url, state):
        manager = McpManager(limits=McpRuntimeLimits(connect_timeout_seconds=3.0))
        try:
            # Redirect must NOT be followed and auth header must never hit redirect target
            await manager.register_server(
                _http_config(base_url, "http-redir", path="/redirect", secret_headers={"Authorization": "Bearer topsecret"})
            )
            with pytest.raises(McpError) as exc:
                await manager.connect("http-redir")
            assert exc.value.code == "MCP_NETWORK_POLICY_VIOLATION"
            assert state.redirect_target_hits == 0

            # Simulate DNS rebinding on http://localhost -> resolves to non-loopback 169.254.169.254
            real_getaddrinfo = socket.getaddrinfo

            def poisoned_getaddrinfo(host, port, *args, **kwargs):
                if host == "localhost":
                    return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("169.254.169.254", port))]
                return real_getaddrinfo(host, port, *args, **kwargs)

            monkeypatch.setattr(socket, "getaddrinfo", poisoned_getaddrinfo)
            port = int(base_url.rsplit(":", 1)[1])
            await manager.register_server(_http_config(f"http://localhost:{port}", "http-rebind"))
            with pytest.raises(McpError) as exc:
                await manager.connect("http-rebind")
            assert exc.value.code == "MCP_NETWORK_POLICY_VIOLATION"
        finally:
            await manager.shutdown()


# ---------------------------------------------------------------------------
# 13, 14: Abnormal child exit, stderr redaction, stubborn process group kill & parallel shutdown
# ---------------------------------------------------------------------------


async def test_abnormal_child_exit_stderr_redaction_and_process_group_cleanup():
    limits = McpRuntimeLimits(
        stdio_eof_wait_seconds=0.25,
        stdio_sigterm_wait_seconds=0.25,
        shutdown_total_timeout_seconds=2.5,
    )
    manager = McpManager(limits=limits)
    try:
        # Server 1: crashes abnormally after logging a secret to stderr
        await manager.register_server(
            _stdio_config("stdio-crash", secret_env={"MY_SECRET": "super-secret-value-777"})
        )
        # Server 2: healthy sibling + spawns a grandchild + ignores EOF & SIGTERM
        await manager.register_server(
            _stdio_config("stdio-stubborn", extra_args=("--ignore-eof-and-sigterm",))
        )

        d_crash, d_stubborn = await asyncio.gather(
            manager.connect("stdio-crash"),
            manager.connect("stdio-stubborn"),
        )
        crash_pid = d_crash.child_pid
        stubborn_pid = d_stubborn.child_pid
        assert crash_pid is not None and stubborn_pid is not None

        # Spawn grandchild inside stubborn server's process group
        gc_res = await manager.call_tool("stdio-stubborn", "spawn_grandchild", {})
        grandchild_pid = int(gc_res.structured_content["grandchild_pid"])
        assert _pid_alive(grandchild_pid)

        # Trigger abnormal crash on Server 1 with secret in stderr
        with pytest.raises(McpError):
            await manager.call_tool(
                "stdio-crash",
                "crash_now",
                {"stderr_msg": "crash error with super-secret-value-777 and sk-abcdef1234567890"},
            )

        # Wait briefly for actor to settle into error state
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if manager.get_diagnostics("stdio-crash").state == "error":
                break
            await asyncio.sleep(0.02)

        crash_diag = manager.get_diagnostics("stdio-crash")
        assert crash_diag.state == "error"
        assert crash_diag.last_error_code is not None
        assert not _pid_alive(crash_pid)
        joined_stderr = " ".join(crash_diag.stderr_tail)
        assert "super-secret-value-777" not in joined_stderr
        assert "sk-abcdef1234567890" not in joined_stderr
        assert "[REDACTED]" in joined_stderr

        # Sibling stubborn server is completely unaffected by Server 1's crash
        ok_res = await manager.call_tool("stdio-stubborn", "echo_text", {"text": "isolated"})
        assert ok_res.text == "echo:isolated"

        # Parallel shutdown must terminate BOTH stubborn child AND its grandchild within budget
        t0 = time.monotonic()
        await manager.shutdown()
        elapsed = time.monotonic() - t0
        assert elapsed < limits.shutdown_total_timeout_seconds
        assert not _pid_alive(stubborn_pid)
        assert not _pid_alive(grandchild_pid)
    finally:
        await manager.shutdown()


async def test_http_timeout_zero_retry_and_cancellation():
    with run_http_mcp_server() as (base_url, state):
        limits = McpRuntimeLimits(call_timeout_seconds=0.35, max_queued_calls_per_server=2)
        manager = McpManager(limits=limits)
        try:
            await manager.register_server(_http_config(base_url, "http-ctl"))
            await manager.connect("http-ctl")

            # Timeout on Streamable HTTP -> MCP_TOOL_TIMEOUT, zero auto-retry
            with pytest.raises(McpError) as exc:
                await manager.call_tool("http-ctl", "http_slow", {"delay": 2.0, "tag": "t1"}, timeout_seconds=0.25)
            assert exc.value.code == "MCP_TOOL_TIMEOUT"
            assert exc.value.uncertain_outcome is True
            assert state.call_counts.get("http_slow", 0) == 1

            # In-flight cancel on Streamable HTTP -> courtesy cancel sent, subsequent call works
            t_slow = asyncio.create_task(
                manager.call_tool("http-ctl", "http_slow", {"delay": 3.0, "tag": "c1"}, timeout_seconds=5.0)
            )
            await asyncio.sleep(0.08)
            t_slow.cancel()
            with pytest.raises(asyncio.CancelledError):
                await t_slow

            await asyncio.sleep(0.05)
            assert len(state.cancelled_request_ids) >= 1
            after = await manager.call_tool("http-ctl", "http_echo", {"text": "post-cancel"})
            assert after.text == "http_echo:post-cancel"
        finally:
            await manager.shutdown()


async def test_simulated_sidecar_sigkill_orphan_risk_evidence(tmp_path):
    """Empirically verify child behavior when a parent Sidecar is abruptly SIGKILLed.

    Demonstrates honestly that:
      - Well-behaved stdio servers exit automatically on stdin pipe EOF after parent SIGKILL.
      - Stubborn servers that ignore EOF/SIGTERM inside their own process group (start_new_session=True)
        CAN survive an abrupt parent SIGKILL and therefore require host-level (Rust) PGID supervision
        in P6.3/P6.6 rather than relying solely on Python teardown or stdin EOF.
    """
    pid_file = tmp_path / "child_pids.json"
    driver = f"""
import asyncio, json, sys
from pathlib import Path
from app.mcp.manager import McpManager
from app.mcp.models import McpServerConfig

async def main():
    mgr = McpManager()
    await mgr.register_server(McpServerConfig(
        server_id="normal", name="Normal", transport="stdio",
        command=sys.executable, args=[{str(STDIO_SERVER_SCRIPT)!r}],
    ))
    await mgr.register_server(McpServerConfig(
        server_id="stubborn", name="Stubborn", transport="stdio",
        command=sys.executable, args=[{str(STDIO_SERVER_SCRIPT)!r}, "--ignore-eof-and-sigterm"],
    ))
    d1 = await mgr.connect("normal")
    d2 = await mgr.connect("stubborn")
    Path({str(pid_file)!r}).write_text(json.dumps({{"normal": d1.child_pid, "stubborn": d2.child_pid}}))
    await asyncio.sleep(30)

asyncio.run(main())
"""
    backend_root = Path(__file__).resolve().parent.parent
    sidecar_sim = subprocess.Popen(
        [sys.executable, "-c", driver],
        cwd=str(backend_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    stubborn_pid: int | None = None
    normal_pid: int | None = None
    try:
        deadline = time.monotonic() + 8.0
        while not pid_file.exists() and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        assert pid_file.exists(), "Simulated sidecar did not start MCP servers in time"
        pids = json.loads(pid_file.read_text(encoding="utf-8"))
        normal_pid = int(pids["normal"])
        stubborn_pid = int(pids["stubborn"])
        assert _pid_alive(normal_pid) and _pid_alive(stubborn_pid)

        # Simulate Rust wait_or_terminate() escalating to killpg(sidecar_pid, SIGKILL)
        os.killpg(os.getpgid(sidecar_sim.pid), signal.SIGKILL)
        sidecar_sim.wait(timeout=3.0)

        # Well-behaved stdio child exits via stdin EOF
        eof_deadline = time.monotonic() + 3.0
        while _pid_alive(normal_pid) and time.monotonic() < eof_deadline:
            await asyncio.sleep(0.05)
        assert not _pid_alive(normal_pid)

        # Honest empirical finding: stubborn child in its own session survives parent SIGKILL
        await asyncio.sleep(0.15)
        assert _pid_alive(stubborn_pid)
    finally:
        if sidecar_sim.poll() is None:
            sidecar_sim.kill()
            sidecar_sim.wait(timeout=2.0)
        for pid in (normal_pid, stubborn_pid):
            if pid is not None and _pid_alive(pid):
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except OSError:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except OSError:
                        pass
        # Confirm test harness left zero orphans behind
        if stubborn_pid is not None:
            reap_deadline = time.monotonic() + 2.0
            while _pid_alive(stubborn_pid) and time.monotonic() < reap_deadline:
                await asyncio.sleep(0.02)
            assert not _pid_alive(stubborn_pid)

