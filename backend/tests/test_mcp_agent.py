"""P6.4.5 comprehensive verification suite: MCP Adapter, Timeouts, RunTools, and Agent Integration.

Test layers are explicitly separated:
- Layer 1 (Unit): `mcp/adapter.py` alias stability, schema normalization/rejection,
  jsonschema pre-execution validation (offline only), result truncation & separation,
  tool/schema count limits, and §6.1 credential rotation revocation.
- Layer 2 & 3 (Agent Integration with scripted local OpenAI-compatible provider + REAL
  stdio & Streamable HTTP MCP servers): plain chat, old-client default-off MCP, no-grant
  MCP tool calling, unauthorized tool filtering, cross-server same-name isolation, rename
  stability, native+MCP shared budget, multi-turn chained calls, invalid arg rejection,
  pre-dispatch queued permission revocation, list_changed drift invalidation, run cancel
  with ordered audit settlement, distinct model/connect/MCP/Run timeouts, isError
  checkpoint safety & multi-call short-circuit, RemoteProtocolError retry without tool
  replay, intermediate+final checkpoint & log zero-leakage, history/SSE compatibility,
  and §6.2 RAG/file external data egress boundary + idempotency.
"""

from __future__ import annotations

import asyncio
import json
import socket
import sqlite3
import sys
import time
from contextlib import asynccontextmanager, closing, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from uuid import UUID, uuid4

import httpx
import pytest
import uvicorn
from pydantic import SecretStr

from app.main import create_app
from app.mcp.adapter import (
    DEEPSEEK_TOOL_NAME_PATTERN,
    MAX_MCP_RESULT_BYTES,
    MAX_MCP_RESULT_CHARS,
    adapt_mcp_call_result,
    build_deepseek_tool_schema,
    make_mcp_tool_alias,
    normalize_mcp_input_schema,
    validate_mcp_tool_arguments,
)
from app.mcp.models import McpContentBlock, McpError, McpToolCallResult
from app.models import ModelError
from app.models.deepseek import DEEPSEEK_BASE_URL, DeepSeekAdapter, DeepSeekSettings
from app.services.grants import GrantRegistry
from app.services.models import ModelService
from app.storage.checkpoints import CHECKPOINT_DATABASE_NAME
from app.tools.protocol import ToolError
from app.tools.registry import RunTools
from test_chat import LocalProviderTransport, chunk, frames, new_session, next_event, rows, run_path
from test_checkpoints import safe_id
from tests.fixtures.mcp.http_test_server import run_http_mcp_server

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "mcp"
STDIO_SERVER_SCRIPT = FIXTURES_DIR / "stdio_test_server.py"

UI_TOKEN = "ui-token-p64-agent-test-0001"
HOST_TOKEN = "host-token-p64-agent-test-0002"
UI_HEADERS = {"Authorization": f"Bearer {UI_TOKEN}"}
HOST_HEADERS = {"Authorization": f"Bearer {HOST_TOKEN}"}

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


class ScriptedProvider:
    """Controllable local OpenAI-compatible SSE/JSON provider for deterministic Agent integration tests."""

    def __init__(self, handler):
        self._handler = handler
        self.requests: list[dict] = []

    def handle(self, body: dict):
        self.requests.append(body)
        return self._handler(body, len(self.requests))


@contextmanager
def provider_transport(provider: ScriptedProvider):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def do_POST(self):
            assert self.path == "/chat/completions"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            outcome = provider.handle(body)
            mode = outcome.get("mode", "sse")

            if outcome.get("delay_before_headers"):
                time.sleep(outcome["delay_before_headers"])

            if mode == "json":
                payload = json.dumps(outcome["body"], ensure_ascii=False).encode("utf-8")
                self.send_response(outcome.get("status", 200))
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            if mode == "drop_mid_stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                first = chunk({"content": "partial-before-drop"})
                self.wfile.write(f"{len(first):x}\r\n".encode() + first + b"\r\n")
                self.wfile.flush()
                # Abruptly close socket without terminating chunked stream -> RemoteProtocolError
                self.close_connection = True
                return

            if mode == "gated_stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for part in outcome["before_gate"]:
                    payload = chunk({"content": part})
                    self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
                    self.wfile.flush()
                gate = outcome["gate"]
                assert gate.wait(timeout=5.0), "Timed out waiting for live message_delta before stream end"
                for part in outcome["after_gate"]:
                    payload = chunk({"content": part})
                    self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
                    self.wfile.flush()
                tail = chunk({}, "stop") + b"data: [DONE]\n\n"
                self.wfile.write(f"{len(tail):x}\r\n".encode() + tail + b"\r\n0\r\n\r\n")
                self.wfile.flush()
                return

            # Default SSE response
            pieces: list[bytes] = []
            if "tool_calls" in outcome:
                for idx, call in enumerate(outcome["tool_calls"]):
                    args_str = json.dumps(call["arguments"], ensure_ascii=False)
                    mid = max(1, len(args_str) // 2)
                    pieces.append(
                        chunk({
                            "tool_calls": [{
                                "index": idx,
                                "id": call.get("id", f"call_{idx}_{uuid4().hex[:6]}"),
                                "type": "function",
                                "function": {"name": call["name"], "arguments": args_str[:mid]},
                            }]
                        })
                    )
                    pieces.append(
                        chunk({
                            "tool_calls": [{
                                "index": idx,
                                "function": {"arguments": args_str[mid:]},
                            }]
                        })
                    )
                pieces.append(chunk({}, "tool_calls"))
            else:
                text = outcome.get("text", "完成。")
                pieces.append(chunk({"content": text}))
                pieces.append(chunk({}, outcome.get("finish_reason", "stop")))
            pieces.append(b"data: [DONE]\n\n")

            raw = b"".join(pieces)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield LocalProviderTransport(server.server_port)
    finally:
        server.shutdown()
        worker.join(timeout=3)


@asynccontextmanager
async def live_agent_app(
    db_path: Path,
    transport,
    *,
    model_timeout_seconds: float = 15.0,
    max_run_seconds: float = 60.0,
    mcp_tool_timeout_seconds: float | None = None,
):
    settings = DeepSeekSettings(model="deepseek-chat", timeout_seconds=model_timeout_seconds)
    models = ModelService(DeepSeekAdapter(settings, transport=transport))
    models.configure_deepseek(SecretStr("sk-p64-test-key-000000000001"), DEEPSEEK_BASE_URL)
    app = create_app(UI_TOKEN, HOST_TOKEN, database_path=db_path, models=models)
    app.state.runs._max_run_seconds = max_run_seconds  # noqa: SLF001
    app.state.runs._mcp_tool_timeout_seconds = mcp_tool_timeout_seconds  # noqa: SLF001
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, access_log=False, proxy_headers=False, lifespan="on"))
        serve_task = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async with asyncio.timeout(5):
                while not server.started:
                    await asyncio.sleep(0.01)
            async with httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{port}",
                headers=UI_HEADERS,
                trust_env=False,
                timeout=15.0,
            ) as client:
                yield app, client
        finally:
            server.should_exit = True
            await serve_task


async def _setup_stdio_server(
    client: httpx.AsyncClient,
    *,
    name: str,
    extra_args: list[str] | None = None,
    env_config: dict | None = None,
    secrets: dict[str, str] | None = None,
    approve_tools: tuple[str, ...] = (),
) -> tuple[dict, dict[str, dict]]:
    created = (
        await client.post(
            "/api/v1/_host/mcp/servers",
            headers=HOST_HEADERS,
            json={
                "name": name,
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(STDIO_SERVER_SCRIPT), *(extra_args or [])],
                "env_config": env_config or {},
            },
        )
    ).json()
    sid = created["server_id"]
    if secrets:
        r = await client.put(
            f"/api/v1/_host/mcp/servers/{sid}/credentials",
            headers=HOST_HEADERS,
            json={"secrets": secrets, "mode": "update"},
        )
        assert r.status_code == 200
    r = await client.post(
        f"/api/v1/_host/mcp/servers/{sid}/trust",
        headers=HOST_HEADERS,
        json={
            "expected_config_revision": created["config_revision"],
            "expected_fingerprint": created["current_fingerprint"],
            "enable": True,
        },
    )
    assert r.status_code == 200
    conn = await client.post(f"/api/v1/_host/mcp/servers/{sid}/connect", headers=HOST_HEADERS)
    assert conn.status_code == 200 and conn.json()["runtime"]["state"] == "connected"

    tools = (await client.get(f"/api/v1/mcp/servers/{sid}/tools", headers=UI_HEADERS)).json()
    by_name = {t["original_tool_name"]: t for t in tools}
    for tool_name in approve_tools:
        tool = by_name[tool_name]
        ar = await client.patch(
            f"/api/v1/_host/mcp/servers/{sid}/tools/{tool['tool_id']}",
            headers=HOST_HEADERS,
            json={
                "permission_state": "allowed",
                "expected_definition_hash": tool["definition_hash"],
                "expected_config_revision": created["config_revision"],
            },
        )
        assert ar.status_code == 200 and ar.json()["effective_allowed"] is True
        by_name[tool_name] = ar.json()
    return conn.json(), by_name


def _read_markers(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ===========================================================================
# LAYER 1: Unit Tests — Adapter, Schema Validation, Truncation, Rotation
# ===========================================================================


def test_adapter_alias_stability_collision_isolation_schema_validation_and_truncation():
    # 1. Alias anchored ONLY to (server_id, original_tool_name) — independent of server display name
    a1 = make_mcp_tool_alias("srv-alpha-1", "echo_text")
    a2 = make_mcp_tool_alias("srv-beta-2", "echo_text")
    assert a1 != a2
    assert DEEPSEEK_TOOL_NAME_PATTERN.fullmatch(a1)
    assert DEEPSEEK_TOOL_NAME_PATTERN.fullmatch(a2)
    assert not a1.startswith(("list_files", "read_text", "search_knowledge"))

    # Special chars and long names remain valid <=128 chars
    weird = make_mcp_tool_alias("srv-1", "search.v2/中文:" + "x" * 150)
    assert DEEPSEEK_TOOL_NAME_PATTERN.fullmatch(weird)
    assert len(weird) <= 128

    # 2. Local $defs/$ref expansion preserves security constraints (required, enum, minimum, additionalProperties)
    raw_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "$defs": {
            "Mode": {"type": "string", "enum": ["fast", "deep"]},
        },
        "properties": {
            "query": {"type": "string", "minLength": 1, "maxLength": 100},
            "mode": {"$ref": "#/$defs/Mode"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    normalized = normalize_mcp_input_schema(raw_schema)
    assert normalized["type"] == "object"
    assert normalized["properties"]["mode"]["enum"] == ["fast", "deep"]
    assert normalized["required"] == ["query"]
    assert normalized["additionalProperties"] is False
    assert "$defs" not in normalized and "$schema" not in normalized

    ds_schema = build_deepseek_tool_schema(
        alias=a1,
        server_name="Docs",
        original_tool_name="echo_text",
        description="A" * 2000,
        normalized_input_schema=normalized,
    )
    assert ds_schema["type"] == "function"
    assert ds_schema["function"]["name"] == a1
    assert len(ds_schema["function"]["description"]) <= 1024

    # 3. Incompatible schemas explicitly refused (never silently stripped)
    for bad_schema in (
        {"type": "array", "items": {"type": "string"}},
        {"type": "object", "properties": {"x": {"$ref": "https://evil.example/schema.json"}}},
        {
            "type": "object",
            "$defs": {"N": {"type": "object", "properties": {"n": {"$ref": "#/$defs/N"}}}},
            "properties": {"root": {"$ref": "#/$defs/N"}},
        },
    ):
        with pytest.raises(McpError) as exc:
            normalize_mcp_input_schema(bad_schema)
        assert exc.value.code == "MCP_SCHEMA_INCOMPATIBLE"

    # 4. Pre-execution jsonschema validation against ORIGINAL schema (offline only, 32 KiB cap)
    assert validate_mcp_tool_arguments(raw_schema, {"query": "hello", "mode": "fast"}) == {
        "query": "hello",
        "mode": "fast",
    }
    for bad_args in (
        "not-a-dict",
        {},  # missing required 'query'
        {"query": "ok", "mode": "invalid-enum"},
        {"query": "ok", "limit": 99},
        {"query": "ok", "injected": True},  # additionalProperties: False
        {"query": "x" * 40_000},  # > 32 KiB
    ):
        with pytest.raises(ToolError) as exc:
            validate_mcp_tool_arguments(raw_schema, bad_args)
        assert exc.value.code == "VALIDATION_ERROR"

    # Remote $ref blocked during argument validation as well
    with pytest.raises(ToolError) as exc:
        validate_mcp_tool_arguments(
            {"type": "object", "properties": {"x": {"$ref": "http://127.0.0.1:9999/remote.json"}}},
            {"x": 1},
        )
    assert exc.value.code == "VALIDATION_ERROR"

    # 5. Result adaptation: text + structured_content + non-text placeholder + 16k char / 24k byte cap + redacted summary
    raw_res = McpToolCallResult(
        server_id="srv-1",
        original_tool_name="inspect",
        generation=1,
        is_error=False,
        content_blocks=(
            McpContentBlock(kind="text", text="RAW-SECRET-CONTENT-" + "中" * 10_000, original_type="text"),
            McpContentBlock(kind="omitted_non_text", original_type="resource_link", mime_type="text/plain"),
        ),
        text="",
        structured_content={"k": "v"},
        omitted_non_text_types=("resource_link",),
        duration_ms=42,
    )
    adapted = adapt_mcp_call_result(
        call_id="c1",
        server_id="srv-1",
        server_name="Demo",
        original_tool_name="inspect",
        raw_result=raw_res,
    )
    assert adapted.ok is True
    assert adapted.truncated is True
    assert isinstance(adapted.content, dict)
    body_text = adapted.content["text"]
    assert len(body_text) <= MAX_MCP_RESULT_CHARS
    assert len(body_text.encode("utf-8")) <= MAX_MCP_RESULT_BYTES
    assert "[输出已按安全上限截断]" in body_text
    # Redacted summary MUST NOT leak raw response content
    assert "RAW-SECRET-CONTENT" not in adapted.display_summary
    assert "已截断" in adapted.display_summary
    assert "非文本内容" in adapted.display_summary


async def test_credential_rotation_revokes_tool_approvals_while_startup_restore_preserves_them(tmp_path):
    """Verify §6.1: actual secret change disconnects & revokes tool permissions without storing secret hashes."""
    db_path = tmp_path / "business.sqlite3"
    provider = ScriptedProvider(lambda _b, _n: {"text": "ok"})
    with provider_transport(provider) as transport:
        async with live_agent_app(db_path, transport) as (_app, client):
            srv, tools = await _setup_stdio_server(
                client,
                name="CredRotateSrv",
                env_config={"SERVER_TOKEN": {"kind": "secret", "secret_key": "SERVER_TOKEN"}},
                secrets={"SERVER_TOKEN": "token-account-alpha-1111"},
                approve_tools=("echo_text",),
            )
            sid = srv["server_id"]
            assert tools["echo_text"]["permission_state"] == "allowed"

            # Submitting the SAME secret value is a no-op -> preserves approval
            same_resp = await client.put(
                f"/api/v1/_host/mcp/servers/{sid}/credentials",
                headers=HOST_HEADERS,
                json={"secrets": {"SERVER_TOKEN": "token-account-alpha-1111"}, "mode": "update"},
            )
            assert same_resp.status_code == 200
            t_after_same = {t["original_tool_name"]: t for t in (await client.get(f"/api/v1/mcp/servers/{sid}/tools")).json()}
            assert t_after_same["echo_text"]["permission_state"] == "allowed"

            # Normal startup restore (`mode="restore"`) NEVER revokes approvals
            restore_resp = await client.put(
                f"/api/v1/_host/mcp/servers/{sid}/credentials",
                headers=HOST_HEADERS,
                json={"secrets": {"SERVER_TOKEN": "token-account-alpha-1111"}, "mode": "restore"},
            )
            assert restore_resp.status_code == 200
            t_after_restore = {t["original_tool_name"]: t for t in (await client.get(f"/api/v1/mcp/servers/{sid}/tools")).json()}
            assert t_after_restore["echo_text"]["permission_state"] == "allowed"

            # Actual credential value change (e.g. different account/scope) MUST disconnect AND revoke tool approvals!
            rot_resp = await client.put(
                f"/api/v1/_host/mcp/servers/{sid}/credentials",
                headers=HOST_HEADERS,
                json={"secrets": {"SERVER_TOKEN": "token-account-beta-2222"}, "mode": "update"},
            )
            assert rot_resp.status_code == 200
            srv_after_rot = (await client.get(f"/api/v1/_host/mcp/servers/{sid}", headers=HOST_HEADERS)).json()
            assert srv_after_rot["runtime"]["state"] == "disconnected"
            t_after_rot = {t["original_tool_name"]: t for t in (await client.get(f"/api/v1/mcp/servers/{sid}/tools")).json()}
            assert t_after_rot["echo_text"]["permission_state"] == "denied"
            assert t_after_rot["echo_text"]["effective_allowed"] is False

            # Zero plaintext secrets in SQLite file
            raw_db = db_path.read_bytes()
            assert b"token-account-alpha-1111" not in raw_db
            assert b"token-account-beta-2222" not in raw_db


# ===========================================================================
# LAYER 2 & 3: Agent Integration with REAL stdio & Streamable HTTP MCP Servers
# ===========================================================================


async def test_plain_chat_old_client_default_off_and_no_grant_mcp_multi_turn_with_rename_and_same_name_isolation(
    tmp_path, caplog
):
    """Covers scenarios 1, 2, 3, 4, 5, 6, 8, 16, 17:
    - Plain chat & old client omitting `mcp` default to MCP disabled
    - No file grants -> MCP works, list_files/read_text NOT exposed
    - Connected unauthorized tools NOT exposed
    - Same-named tool (`echo_text`) isolated across two real servers (stdio + Streamable HTTP)
    - Server display rename keeps model tool alias identical
    - Multi-turn chained tool calls -> final answer, full V6 audit & SSE metadata
    - Zero raw sensitive args or raw MCP bodies in ALL intermediate & final checkpoints and logs
    """
    db_path = tmp_path / "business.sqlite3"
    cp_path = tmp_path / CHECKPOINT_DATABASE_NAME
    caplog.set_level("INFO", logger="knowledgeagent.sidecar")

    secret_arg_marker = "SENSITIVE-USER-QUERY-ARG-XYZ987"
    raw_mcp_body_marker = f"echo:{secret_arg_marker}"

    def decide(body: dict, call_idx: int):
        tools_sent = body.get("tools") or []
        tool_names = [t["function"]["name"] for t in tools_sent]
        last_msg = body["messages"][-1]

        # Turn 1 (plain chat / old client without mcp): NO tools sent
        if not tools_sent:
            return {"text": "普通聊天回复。"}

        # Multi-turn MCP flow: call Server A echo_text -> then Server B structured_calc -> final answer
        if last_msg["role"] == "user":
            alias_a = next(n for n in tool_names if n.endswith("__echo_text") and "alpha" not in n or n == target_aliases["a_echo"])
            return {
                "tool_calls": [
                    {"id": "call_step_1", "name": target_aliases["a_echo"], "arguments": {"text": secret_arg_marker}},
                ]
            }
        tool_msgs = [m for m in body["messages"] if m["role"] == "tool"]
        if len(tool_msgs) == 1:
            # Verify raw body arrived in memory for DeepSeek!
            assert raw_mcp_body_marker in tool_msgs[0]["content"]
            return {
                "tool_calls": [
                    {"id": "call_step_2", "name": target_aliases["b_echo"], "arguments": {"text": "from-server-b"}},
                    {"id": "call_step_3", "name": target_aliases["http_struct"], "arguments": {"x": 7}},
                ]
            }
        assert "echo:from-server-b" in tool_msgs[1]["content"]
        assert "squared=49" in tool_msgs[2]["content"]
        return {"text": "综合三个 MCP 工具结果：A 已回显，B 已回显，7 的平方为 49。"}

    target_aliases: dict[str, str] = {}
    provider = ScriptedProvider(decide)

    with run_http_mcp_server() as (http_base, _http_state):
        with provider_transport(provider) as transport:
            async with live_agent_app(db_path, transport) as (_app, client):
                # Connect two real stdio servers (both have `echo_text`, only authorize `echo_text`, NOT `dump_env`!)
                srv_a, _ = await _setup_stdio_server(client, name="StdioAlpha", approve_tools=("echo_text",))
                srv_b, _ = await _setup_stdio_server(client, name="StdioBeta", approve_tools=("echo_text",))

                # Connect a real Streamable HTTP server and authorize ONLY `http_structured`
                created_http = (
                    await client.post(
                        "/api/v1/_host/mcp/servers",
                        headers=HOST_HEADERS,
                        json={"name": "HttpGamma", "transport": "streamable_http", "url": f"{http_base}/mcp"},
                    )
                ).json()
                id_http = created_http["server_id"]
                await client.post(
                    f"/api/v1/_host/mcp/servers/{id_http}/trust",
                    headers=HOST_HEADERS,
                    json={
                        "expected_config_revision": created_http["config_revision"],
                        "expected_fingerprint": created_http["current_fingerprint"],
                        "enable": True,
                    },
                )
                assert (await client.post(f"/api/v1/_host/mcp/servers/{id_http}/connect", headers=HOST_HEADERS)).status_code == 200
                http_tools = {t["original_tool_name"]: t for t in (await client.get(f"/api/v1/mcp/servers/{id_http}/tools")).json()}
                hs = http_tools["http_structured"]
                await client.patch(
                    f"/api/v1/_host/mcp/servers/{id_http}/tools/{hs['tool_id']}",
                    headers=HOST_HEADERS,
                    json={
                        "permission_state": "allowed",
                        "expected_definition_hash": hs["definition_hash"],
                        "expected_config_revision": created_http["config_revision"],
                    },
                )

                alias_a_before = make_mcp_tool_alias(srv_a["server_id"], "echo_text")
                alias_b = make_mcp_tool_alias(srv_b["server_id"], "echo_text")
                alias_http = make_mcp_tool_alias(id_http, "http_structured")
                assert alias_a_before != alias_b

                # Rename StdioAlpha display name -> alias MUST stay identical!
                rename = await client.patch(
                    f"/api/v1/_host/mcp/servers/{srv_a['server_id']}",
                    headers=HOST_HEADERS,
                    json={"name": "RenamedStdioAlpha"},
                )
                assert rename.status_code == 200
                alias_a_after = make_mcp_tool_alias(srv_a["server_id"], "echo_text")
                assert alias_a_after == alias_a_before
                target_aliases.update({"a_echo": alias_a_after, "b_echo": alias_b, "http_struct": alias_http})

                session = await new_session(client)

                # 1 & 2. Old client omitting `mcp` -> MCP disabled by default, plain chat streams normally!
                async with client.stream(
                    "POST", run_path(session), json={"client_request_id": str(uuid4()), "message": "你好"}
                ) as resp:
                    ev_plain = [e async for e in frames(resp) if isinstance(e, dict)]
                assert ev_plain[-1]["type"] == "run_completed"
                assert not provider.requests[-1].get("tools")

                # Explicit `mcp: {"enabled": false}` -> also disables MCP
                async with client.stream(
                    "POST",
                    run_path(session),
                    json={"client_request_id": str(uuid4()), "message": "再见", "mcp": {"enabled": False}},
                ) as resp:
                    ev_off = [e async for e in frames(resp) if isinstance(e, dict)]
                assert ev_off[-1]["type"] == "run_completed"
                assert not provider.requests[-1].get("tools")

                # 3, 4, 5, 6, 8. Explicit `mcp: {"enabled": true}` with NO file grants!
                req_id = str(uuid4())
                async with client.stream(
                    "POST",
                    run_path(session),
                    json={"client_request_id": req_id, "message": "调用三个 MCP 工具", "mcp": {"enabled": True}},
                ) as resp:
                    assert resp.status_code == 200
                    ev_mcp = [e async for e in frames(resp) if isinstance(e, dict)]

                assert ev_mcp[-1]["type"] == "run_completed"
                mcp_run_id = ev_mcp[-1]["run_id"]

                # Check exposed tool schemas sent to model:
                # - ONLY the 3 authorized MCP tools (NOT unauthorized dump_env/http_echo, NOT list_files/read_text!)
                first_mcp_req = provider.requests[2]
                exposed_names = {t["function"]["name"] for t in first_mcp_req["tools"]}
                assert exposed_names == {alias_a_after, alias_b, alias_http}
                assert "list_files" not in exposed_names and "read_text" not in exposed_names

                # Check SSE incremental MCP metadata on tool_started & tool_completed
                started_events = [e for e in ev_mcp if e["type"] == "tool_started"]
                completed_events = [e for e in ev_mcp if e["type"] == "tool_completed"]
                assert len(started_events) == 3 and len(completed_events) == 3
                assert started_events[0]["data"]["source"] == "mcp"
                assert started_events[0]["data"]["server_id"] == srv_a["server_id"]
                assert started_events[0]["data"]["server_name"] == "RenamedStdioAlpha"
                assert started_events[0]["data"]["original_tool_name"] == "echo_text"
                assert isinstance(completed_events[0]["data"]["duration_ms"], int)
                assert secret_arg_marker not in json.dumps(ev_mcp, ensure_ascii=False)
                assert raw_mcp_body_marker not in json.dumps(ev_mcp, ensure_ascii=False)

                # Idempotency: duplicate client_request_id with flipped mcp.enabled returns 409 DUPLICATE_CLIENT_REQUEST
                dup = await client.post(
                    run_path(session),
                    json={"client_request_id": req_id, "message": "调用三个 MCP 工具", "mcp": {"enabled": False}},
                )
                assert dup.status_code == 409
                assert dup.json()["error"]["code"] == "DUPLICATE_CLIENT_REQUEST"
                assert dup.json()["error"]["run_id"] == mcp_run_id

                # History API (`GET /sessions/{id}/runs`) returns V6 MCP tool_calls audit metadata!
                runs_page = (await client.get(f"/api/v1/sessions/{session}/runs")).json()
                mcp_run_row = next(r for r in runs_page["items"] if r["run_id"] == mcp_run_id)
                assert len(mcp_run_row["tool_calls"]) == 3
                tc0 = mcp_run_row["tool_calls"][0]
                assert tc0["source"] == "mcp"
                assert tc0["server_id"] == srv_a["server_id"]
                assert tc0["server_name"] == "RenamedStdioAlpha"
                assert tc0["original_tool_name"] == "echo_text"
                assert isinstance(tc0["duration_ms"], int) and tc0["duration_ms"] >= 0

                # 16. Verify ZERO sensitive args & ZERO raw MCP body in ALL intermediate & final checkpoints and DB/logs!
                with closing(sqlite3.connect(cp_path)) as cpdb:
                    all_cp_blobs = cpdb.execute("SELECT checkpoint, metadata FROM checkpoints").fetchall()
                    all_write_blobs = cpdb.execute("SELECT value FROM writes").fetchall()
                assert len(all_cp_blobs) >= 3  # includes intermediate checkpoints!
                for blob in [*all_cp_blobs, *all_write_blobs]:
                    for col in blob:
                        if isinstance(col, bytes):
                            assert secret_arg_marker.encode() not in col
                            assert raw_mcp_body_marker.encode() not in col
                        elif isinstance(col, str):
                            assert secret_arg_marker not in col
                            assert raw_mcp_body_marker not in col
                raw_biz = db_path.read_bytes()
                assert secret_arg_marker.encode() not in raw_biz
                assert raw_mcp_body_marker.encode() not in raw_biz
                for rec in caplog.records:
                    assert secret_arg_marker not in rec.getMessage()
                    assert raw_mcp_body_marker not in rec.getMessage()


async def test_invalid_args_is_error_multi_call_short_circuit_and_checkpoint_preservation(tmp_path):
    """Covers scenarios 9 & 14:
    - Invalid MCP arguments rejected pre-execution with VALIDATION_ERROR (never hits server!)
    - isError=true fails Run, does NOT advance last_safe_checkpoint_id, short-circuits remaining batch calls,
      and does NOT break next turn or another session.
    """
    db_path = tmp_path / "business.sqlite3"
    marker_file = tmp_path / "markers.jsonl"
    phase = {"step": "seed"}

    def decide(body: dict, _n: int):
        if phase["step"] == "seed":
            return {"text": "基准回答。"}
        if phase["step"] == "invalid_args":
            if body["messages"][-1]["role"] == "user":
                # Missing required `text` field -> must fail jsonschema pre-execution!
                return {"tool_calls": [{"id": "c_bad", "name": aliases["echo"], "arguments": {"wrong": 123}}]}
            return {"text": "fallback"}
        if phase["step"] == "is_error_batch":
            if body["messages"][-1]["role"] == "user":
                # Batch of 2 calls: first returns isError=true, second MUST be short-circuited (never sent to server!)
                return {
                    "tool_calls": [
                        {"id": "c_err", "name": aliases["err"], "arguments": {}},
                        {"id": "c_skip", "name": aliases["echo"], "arguments": {"text": "MUST-NOT-REACH-SERVER"}},
                    ]
                }
            return {"text": "should-not-commit"}
        return {"text": "后续对话正常。"}

    aliases: dict[str, str] = {}
    provider = ScriptedProvider(decide)
    with provider_transport(provider) as transport:
        async with live_agent_app(db_path, transport) as (_app, client):
            srv, _ = await _setup_stdio_server(
                client,
                name="FailCheckSrv",
                extra_args=["--marker-file", str(marker_file)],
                approve_tools=("echo_text", "return_error"),
            )
            aliases["echo"] = make_mcp_tool_alias(srv["server_id"], "echo_text")
            aliases["err"] = make_mcp_tool_alias(srv["server_id"], "return_error")

            session = await new_session(client)
            # Establish a safe base checkpoint
            async with client.stream(
                "POST", run_path(session), json={"client_request_id": str(uuid4()), "message": "基准问题"}
            ) as resp:
                ev = [e async for e in frames(resp) if isinstance(e, dict)]
            assert ev[-1]["type"] == "run_completed"
            base_cp = safe_id(db_path, session)
            assert base_cp is not None

            # 9. Invalid arguments rejected BEFORE reaching external MCP server
            phase["step"] = "invalid_args"
            async with client.stream(
                "POST",
                run_path(session),
                json={"client_request_id": str(uuid4()), "message": "测试非法参数", "mcp": {"enabled": True}},
            ) as resp:
                ev_bad = [e async for e in frames(resp) if isinstance(e, dict)]
            assert ev_bad[-1]["type"] == "error"
            assert ev_bad[-1]["data"]["code"] == "VALIDATION_ERROR"
            assert safe_id(db_path, session) == base_cp
            assert _read_markers(marker_file) == []

            # 14. isError=true short-circuits second batch call & preserves base_cp
            phase["step"] = "is_error_batch"
            async with client.stream(
                "POST",
                run_path(session),
                json={"client_request_id": str(uuid4()), "message": "测试外部错误短路", "mcp": {"enabled": True}},
            ) as resp:
                ev_err = [e async for e in frames(resp) if isinstance(e, dict)]
            assert ev_err[-1]["type"] == "error"
            assert ev_err[-1]["data"]["code"] == "MCP_TOOL_RETURNED_ERROR"
            assert safe_id(db_path, session) == base_cp
            # Verify second call `echo_text` NEVER reached the real MCP server!
            markers = _read_markers(marker_file)
            assert not any(m.get("tool") == "echo_text" for m in markers)

            # Next turn on the same session still succeeds cleanly!
            phase["step"] = "after"
            async with client.stream(
                "POST", run_path(session), json={"client_request_id": str(uuid4()), "message": "恢复正常"}
            ) as resp:
                ev_ok = [e async for e in frames(resp) if isinstance(e, dict)]
            assert ev_ok[-1]["type"] == "run_completed"
            assert safe_id(db_path, session) != base_cp


async def test_queued_permission_revocation_prevents_dispatch_and_list_changed_invalidates_snapshot(tmp_path):
    """Covers scenarios 10 & 11:
    - Queued MCP call whose permission is revoked while waiting in server queue NEVER dispatches to external server!
    - Dynamic `tools/list_changed` mutation invalidates snapshot before next model inference / dispatch!
    """
    db_path = tmp_path / "business.sqlite3"
    marker_file = tmp_path / "queue_markers.jsonl"

    provider = ScriptedProvider(lambda _b, _n: {"text": "ok"})
    with provider_transport(provider) as transport:
        async with live_agent_app(db_path, transport) as (app, client):
            srv, tools = await _setup_stdio_server(
                client,
                name="QueueFenceSrv",
                extra_args=["--marker-file", str(marker_file)],
                approve_tools=("slow_tool", "echo_text", "mutate_tools"),
            )
            sid = srv["server_id"]
            snaps = {s.original_tool_name: s for s in app.state.mcp_service.snapshot_authorized_tools()}

            # 10. Start slow_tool (holds serial server actor), enqueue echo_text behind it, then REVOKE echo_text!
            t_slow = asyncio.create_task(
                app.state.mcp_service.execute_authorized_tool(snaps["slow_tool"], {"delay": 0.4, "tag": "blocker"})
            )
            # Wait until slow_tool actually starts on the server
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                if any(m.get("event") == "call_started" and m.get("args", {}).get("tag") == "blocker" for m in _read_markers(marker_file)):
                    break
                await asyncio.sleep(0.02)

            # Enqueue echo_text behind slow_tool (passes pre-enqueue check because still allowed right now)
            t_queued = asyncio.create_task(
                app.state.mcp_service.execute_authorized_tool(snaps["echo_text"], {"text": "NEVER-DISPATCH"})
            )
            await asyncio.sleep(0.05)

            # Revoke echo_text permission in DB while it is sitting in McpManager's queue!
            rev_resp = await client.patch(
                f"/api/v1/_host/mcp/servers/{sid}/tools/{tools['echo_text']['tool_id']}",
                headers=HOST_HEADERS,
                json={"permission_state": "denied"},
            )
            assert rev_resp.status_code == 200

            # Note: revoking permission cancels server calls AND before_dispatch rejects dequeued call!
            with pytest.raises(McpError) as exc:
                await t_queued
            assert exc.value.code in ("MCP_PERMISSION_DENIED", "MCP_TOOL_CANCELLED")
            await asyncio.gather(t_slow, return_exceptions=True)

            # Prove echo_text NEVER reached the external stdio server!
            markers = _read_markers(marker_file)
            assert not any(m.get("args", {}).get("text") == "NEVER-DISPATCH" for m in markers)

            # 11. Re-approve echo_text, snapshot it, then trigger real tools/list_changed -> old snapshot MUST fail!
            tools_now = {t["original_tool_name"]: t for t in (await client.get(f"/api/v1/mcp/servers/{sid}/tools")).json()}
            await client.patch(
                f"/api/v1/_host/mcp/servers/{sid}/tools/{tools_now['echo_text']['tool_id']}",
                headers=HOST_HEADERS,
                json={
                    "permission_state": "allowed",
                    "expected_definition_hash": tools_now["echo_text"]["definition_hash"],
                    "expected_config_revision": 1,
                },
            )
            snaps2 = {s.original_tool_name: s for s in app.state.mcp_service.snapshot_authorized_tools()}
            old_echo_snap = snaps2["echo_text"]
            before_ver = app.state.mcp_manager.get_diagnostics(sid).catalog_version

            # Mutate tools on real stdio server -> triggers real notifications/tools/list_changed
            await app.state.mcp_manager.call_tool(sid, "mutate_tools", {})
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                if app.state.mcp_manager.get_diagnostics(sid).catalog_version > before_ver:
                    break
                await asyncio.sleep(0.02)

            with pytest.raises(McpError) as exc:
                await app.state.mcp_service.execute_authorized_tool(old_echo_snap, {"text": "stale-def"})
            assert exc.value.code == "MCP_PERMISSION_DENIED"


async def test_run_cancellation_during_mcp_call_settles_audit_cleanly_before_run_cancel(tmp_path):
    """Covers scenario 12: cancelling Run mid-MCP-call sends courtesy cancel & leaves zero stuck `running` tool_calls."""
    db_path = tmp_path / "business.sqlite3"
    marker_file = tmp_path / "cancel_markers.jsonl"
    aliases: dict[str, str] = {}

    def decide(body: dict, _n: int):
        return {"tool_calls": [{"id": "c_slow", "name": aliases["slow"], "arguments": {"delay": 5.0, "tag": "cancel-me"}}]}

    provider = ScriptedProvider(decide)
    with provider_transport(provider) as transport:
        async with live_agent_app(db_path, transport) as (_app, client):
            srv, _ = await _setup_stdio_server(
                client,
                name="CancelSrv",
                extra_args=["--marker-file", str(marker_file)],
                approve_tools=("slow_tool",),
            )
            aliases["slow"] = make_mcp_tool_alias(srv["server_id"], "slow_tool")
            session = await new_session(client)

            async with client.stream(
                "POST",
                run_path(session),
                json={"client_request_id": str(uuid4()), "message": "运行慢工具并取消", "mcp": {"enabled": True}},
            ) as resp:
                assert resp.status_code == 200
                ev_iter = frames(resp)
                started = await next_event(ev_iter)
                run_id = started["run_id"]
                tool_started = await next_event(ev_iter)
                assert tool_started["type"] == "tool_started"
                assert tool_started["data"]["source"] == "mcp"

                # Wait until the real stdio server started slow_tool
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline:
                    if any(m.get("event") == "call_started" for m in _read_markers(marker_file)):
                        break
                    await asyncio.sleep(0.02)

                cancel_resp = await client.post(f"/api/v1/runs/{run_id}/cancel")
                assert cancel_resp.status_code == 202

                rest = [e async for e in ev_iter if isinstance(e, dict)]
                assert rest[-1]["type"] == "run_cancelled"

            # Verify zero stuck `running` tool_calls in SQLite and proper `cancelled` status + RUN_CANCELLED code
            tc_rows = rows(
                db_path,
                "SELECT status, error_code, source, duration_ms FROM tool_calls WHERE run_id = ?",
                (run_id,),
            )
            assert len(tc_rows) == 1
            assert tc_rows[0][0] == "cancelled"
            assert tc_rows[0][1] == "RUN_CANCELLED"
            assert tc_rows[0][2] == "mcp"
            assert tc_rows[0][3] is not None and tc_rows[0][3] >= 0

            # Verify courtesy notifications/cancelled reached real stdio server
            await asyncio.sleep(0.1)
            assert any(m.get("event") == "notification_cancelled_received" for m in _read_markers(marker_file))


async def test_distinct_timeouts_model_vs_mcp_tool_vs_run_total(tmp_path):
    """Covers scenario 13:
    1) Per-request model timeout (0.3s) does NOT abort a 0.5s MCP tool call!
    2) Slow MCP tool exceeding mcp_tool_timeout_seconds (0.2s) returns MCP_TOOL_TIMEOUT (NOT MODEL_TIMEOUT!).
    3) Slow model response exceeding model_timeout_seconds returns MODEL_TIMEOUT.
    4) Total Run exceeding max_run_seconds (0.35s) returns RUN_TIMEOUT.
    """
    db_path = tmp_path / "business.sqlite3"
    mode = {"case": "mcp_longer_than_model_timeout"}
    aliases: dict[str, str] = {}

    def decide(body: dict, _n: int):
        if mode["case"] == "mcp_longer_than_model_timeout":
            if body["messages"][-1]["role"] == "user":
                return {"tool_calls": [{"id": "c1", "name": aliases["slow"], "arguments": {"delay": 0.45}}]}
            return {"text": "MCP 耗时超过单次模型超时但仍成功！"}
        if mode["case"] == "mcp_timeout":
            return {"tool_calls": [{"id": "c2", "name": aliases["slow"], "arguments": {"delay": 2.0}}]}
        if mode["case"] == "model_timeout":
            return {"delay_before_headers": 0.6, "text": "too-late"}
        if mode["case"] == "run_timeout":
            if body["messages"][-1]["role"] == "user":
                return {"tool_calls": [{"id": "c3", "name": aliases["slow"], "arguments": {"delay": 0.25}}]}
            return {"tool_calls": [{"id": "c4", "name": aliases["slow"], "arguments": {"delay": 0.25}}]}
        return {"text": "ok"}

    provider = ScriptedProvider(decide)
    with provider_transport(provider) as transport:
        # Case 1: model_timeout=0.3s, MCP tool takes 0.45s -> MUST SUCCEED (timeout is per-LLM-call, not whole Run!)
        async with live_agent_app(
            db_path, transport, model_timeout_seconds=0.3, max_run_seconds=10.0, mcp_tool_timeout_seconds=2.0
        ) as (app, client):
            srv, _ = await _setup_stdio_server(client, name="TimeoutSrv", approve_tools=("slow_tool",))
            aliases["slow"] = make_mcp_tool_alias(srv["server_id"], "slow_tool")
            session = await new_session(client)

            async with client.stream(
                "POST", run_path(session), json={"client_request_id": str(uuid4()), "message": "c1", "mcp": {"enabled": True}}
            ) as resp:
                ev1 = [e async for e in frames(resp) if isinstance(e, dict)]
            assert ev1[-1]["type"] == "run_completed"

            # Case 2: MCP tool exceeds mcp_tool_timeout_seconds (0.2s) -> MCP_TOOL_TIMEOUT (NOT MODEL_TIMEOUT!)
            app.state.runs._mcp_tool_timeout_seconds = 0.2  # noqa: SLF001
            mode["case"] = "mcp_timeout"
            async with client.stream(
                "POST", run_path(session), json={"client_request_id": str(uuid4()), "message": "c2", "mcp": {"enabled": True}}
            ) as resp:
                ev2 = [e async for e in frames(resp) if isinstance(e, dict)]
            assert ev2[-1]["type"] == "error"
            assert ev2[-1]["data"]["code"] == "MCP_TOOL_TIMEOUT"

            # Case 3: Slow model response exceeds model_timeout_seconds (0.3s) -> MODEL_TIMEOUT
            mode["case"] = "model_timeout"
            async with client.stream(
                "POST", run_path(session), json={"client_request_id": str(uuid4()), "message": "c3"}
            ) as resp:
                ev3 = [e async for e in frames(resp) if isinstance(e, dict)]
            assert ev3[-1]["type"] == "error"
            assert ev3[-1]["data"]["code"] == "MODEL_TIMEOUT"

            # Case 4: Cumulative multi-turn execution exceeds max_run_seconds (0.35s) -> RUN_TIMEOUT
            app.state.runs._mcp_tool_timeout_seconds = 2.0  # noqa: SLF001
            app.state.runs._max_run_seconds = 0.35  # noqa: SLF001
            mode["case"] = "run_timeout"
            async with client.stream(
                "POST", run_path(session), json={"client_request_id": str(uuid4()), "message": "c4", "mcp": {"enabled": True}}
            ) as resp:
                ev4 = [e async for e in frames(resp) if isinstance(e, dict)]
            assert ev4[-1]["type"] == "error"
            assert ev4[-1]["data"]["code"] == "RUN_TIMEOUT"


async def test_remote_protocol_error_retry_never_replays_mcp_tool(tmp_path):
    """Covers scenario 15: stream disconnect after MCP tool retries ONLY the LLM request, never the MCP tool!"""
    db_path = tmp_path / "business.sqlite3"
    marker_file = tmp_path / "retry_markers.jsonl"
    aliases: dict[str, str] = {}

    def decide(body: dict, call_idx: int):
        if call_idx == 1:
            return {"tool_calls": [{"id": "c_once", "name": aliases["echo"], "arguments": {"text": "run-once-only"}}]}
        if call_idx == 2:
            # First post-tool model stream drops mid-stream -> RemoteProtocolError
            return {"mode": "drop_mid_stream"}
        # Retry (non-streaming JSON) succeeds!
        return {
            "mode": "json",
            "body": {
                "id": "chatcmpl-retry",
                "model": "deepseek-chat",
                "created": 1,
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "重试后回答成功。"},
                }],
            },
        }

    provider = ScriptedProvider(decide)
    with provider_transport(provider) as transport:
        async with live_agent_app(db_path, transport) as (_app, client):
            srv, _ = await _setup_stdio_server(
                client,
                name="RetrySrv",
                extra_args=["--marker-file", str(marker_file)],
                approve_tools=("echo_text",),
            )
            aliases["echo"] = make_mcp_tool_alias(srv["server_id"], "echo_text")
            session = await new_session(client)

            async with client.stream(
                "POST",
                run_path(session),
                json={"client_request_id": str(uuid4()), "message": "测试流断重试", "mcp": {"enabled": True}},
            ) as resp:
                ev = [e async for e in frames(resp) if isinstance(e, dict)]
            assert ev[-1]["type"] == "run_completed"
            assert len(provider.requests) == 3  # 1 tool proposal + 1 dropped stream + 1 non-stream retry
            # Verify echo_text was executed on the real MCP server EXACTLY ONCE!
            echo_calls = [m for m in _read_markers(marker_file) if m.get("tool") == "echo_text"]
            assert len(echo_calls) == 1


async def test_egress_boundary_native_plus_mcp_shared_budget_and_rag_collaboration(tmp_path):
    """Covers scenarios 7 & 18:
    - Mixing RAG or native file grants with MCP defaults to 403 MCP_EGRESS_AUTHORIZATION_REQUIRED
      and creates ZERO orphan running Runs!
    - Explicit `allow_external_data_egress: true` allows native file tools + MCP in the same Run,
      sharing the unified 8-call budget (9th call rejected with RATE_LIMITED).
    - Native tool SSE events preserve exact original key set while MCP events carry incremental metadata.
    """
    db_path = tmp_path / "business.sqlite3"
    note_file = tmp_path / "secret_note.txt"
    note_file.write_text("本地文件第一行内容\n", encoding="utf-8")

    aliases: dict[str, str] = {}
    grant_holder: dict[str, str] = {}
    turn_mode = {"kind": "mixed_ok"}

    def decide(body: dict, _n: int):
        tool_msgs = [m for m in body["messages"] if m["role"] == "tool"]
        if turn_mode["kind"] == "mixed_ok":
            if not tool_msgs:
                return {
                    "tool_calls": [
                        {"id": "c_nat", "name": "read_text", "arguments": {"grant_id": grant_holder["id"]}},
                        {"id": "c_mcp", "name": aliases["echo"], "arguments": {"text": "public-query"}},
                    ]
                }
            return {"text": "已混合调用原生文件工具与 MCP 工具。"}
        if turn_mode["kind"] == "budget_exhaust":
            # Propose 5 calls in round 1 (1 native + 4 MCP) and 4 calls in round 2 -> total 9 > MAX_TOOL_CALLS (8)!
            if not tool_msgs:
                return {
                    "tool_calls": [
                        {"id": "b1", "name": "read_text", "arguments": {"grant_id": grant_holder["id"]}},
                        {"id": "b2", "name": aliases["echo"], "arguments": {"text": "1"}},
                        {"id": "b3", "name": aliases["echo"], "arguments": {"text": "2"}},
                        {"id": "b4", "name": aliases["echo"], "arguments": {"text": "3"}},
                        {"id": "b5", "name": aliases["echo"], "arguments": {"text": "4"}},
                    ]
                }
            return {
                "tool_calls": [
                    {"id": "b6", "name": aliases["echo"], "arguments": {"text": "5"}},
                    {"id": "b7", "name": aliases["echo"], "arguments": {"text": "6"}},
                    {"id": "b8", "name": aliases["echo"], "arguments": {"text": "7"}},
                    {"id": "b9", "name": aliases["echo"], "arguments": {"text": "8"}},
                ]
            }
        return {"text": "ok"}

    provider = ScriptedProvider(decide)
    with provider_transport(provider) as transport:
        async with live_agent_app(db_path, transport) as (_app, client):
            srv, _ = await _setup_stdio_server(client, name="MixSrv", approve_tools=("echo_text",))
            aliases["echo"] = make_mcp_tool_alias(srv["server_id"], "echo_text")
            session = await new_session(client)

            # 18a. RAG + MCP without allow_external_data_egress -> 403 BEFORE creating a running Run!
            rag_blocked = await client.post(
                run_path(session),
                json={
                    "client_request_id": str(uuid4()),
                    "message": "同时查教材和外部 MCP",
                    "knowledge": {"enabled": True},
                    "mcp": {"enabled": True},
                },
            )
            assert rag_blocked.status_code == 403
            assert rag_blocked.json()["error"]["code"] == "MCP_EGRESS_AUTHORIZATION_REQUIRED"
            assert rows(db_path, "SELECT count(*) FROM runs WHERE session_id = ?", (session,)) == [(0,)]

            # Grant a local file to the session
            grant = (
                await client.post(
                    f"/api/v1/_host/sessions/{session}/grants",
                    headers=HOST_HEADERS,
                    json={
                        "scope": "file",
                        "path": str(note_file),
                        "operations": ["read"],
                        "lifetime": "application_session",
                    },
                )
            ).json()
            grant_holder["id"] = grant["grant_id"]

            # 18b. Native file grant + MCP without allow_external_data_egress -> 403, ZERO running Runs!
            file_blocked = await client.post(
                run_path(session),
                json={
                    "client_request_id": str(uuid4()),
                    "message": "读取本地文件并调用 MCP",
                    "mcp": {"enabled": True},
                },
            )
            assert file_blocked.status_code == 403
            assert file_blocked.json()["error"]["code"] == "MCP_EGRESS_AUTHORIZATION_REQUIRED"
            assert rows(db_path, "SELECT count(*) FROM runs WHERE session_id = ?", (session,)) == [(0,)]

            # 18c. Native file tools ALONE (without MCP) continue to work normally without egress flag!
            # 7a. Explicit allow_external_data_egress=True enables mixed native + MCP execution!
            async with client.stream(
                "POST",
                run_path(session),
                json={
                    "client_request_id": str(uuid4()),
                    "message": "混合调用",
                    "mcp": {"enabled": True, "allow_external_data_egress": True},
                },
            ) as resp:
                assert resp.status_code == 200
                ev_mix = [e async for e in frames(resp) if isinstance(e, dict)]
            assert ev_mix[-1]["type"] == "run_completed"

            # Verify native tool SSE event keeps exact original keys, while MCP event has incremental metadata
            starts = [e["data"] for e in ev_mix if e["type"] == "tool_started"]
            assert set(starts[0].keys()) == {"tool_call_id", "tool_name"}
            assert starts[0]["tool_name"] == "read_text"
            assert starts[1]["source"] == "mcp" and starts[1]["original_tool_name"] == "echo_text"

            # 7b. Shared 8-call budget across native + MCP: 5 + 4 = 9 -> RATE_LIMITED!
            turn_mode["kind"] = "budget_exhaust"
            async with client.stream(
                "POST",
                run_path(session),
                json={
                    "client_request_id": str(uuid4()),
                    "message": "超预算",
                    "mcp": {"enabled": True, "allow_external_data_egress": True},
                },
            ) as resp:
                ev_budget = [e async for e in frames(resp) if isinstance(e, dict)]
            assert ev_budget[-1]["type"] == "error"
            assert ev_budget[-1]["data"]["code"] == "RATE_LIMITED"


def test_tool_limit_exceeded_rejects_before_creating_run():
    """Verify P6.4.3 §4.3: exceeding 32 exposed tools or 128 KiB schemas raises MCP_TOOL_LIMIT_EXCEEDED pre-admission."""
    grants = GrantRegistry()
    session_id = uuid4()

    class FakeMcpService:
        def __init__(self, count: int):
            self._count = count

        def snapshot_authorized_tools(self):
            from app.mcp.adapter import McpAuthorizedToolSnapshot

            items = []
            for i in range(self._count):
                alias = make_mcp_tool_alias("srv-big", f"tool_{i}")
                schema = {"type": "object", "properties": {"q": {"type": "string"}}}
                items.append(
                    McpAuthorizedToolSnapshot(
                        alias=alias,
                        tool_id=f"tid-{i}",
                        server_id="srv-big",
                        server_name_snapshot="BigSrv",
                        original_tool_name=f"tool_{i}",
                        description="desc",
                        raw_input_schema=schema,
                        normalized_input_schema=schema,
                        model_schema=build_deepseek_tool_schema(
                            alias=alias,
                            server_name="BigSrv",
                            original_tool_name=f"tool_{i}",
                            description="desc",
                            normalized_input_schema=schema,
                        ),
                        definition_hash="a" * 64,
                        config_revision=1,
                        generation=1,
                        catalog_version=1,
                    )
                )
            return tuple(items)

    with pytest.raises(ModelError) as exc:
        RunTools.build_snapshot(
            grants,
            session_id,
            mcp_service=FakeMcpService(33),  # type: ignore[arg-type]
            mcp_enabled=True,
        )
    assert exc.value.code == "MCP_TOOL_LIMIT_EXCEEDED"


async def test_cross_turn_historical_file_rag_and_legacy_egress_protection(tmp_path):
    """Verify P6.5 §4.3 cross-turn historical context external data egress protection:
    1) Session that read a local file on Turn 1 STILL requires allow_external_data_egress
       on Turn 2 even AFTER the file grant is revoked.
    2) Session that ran textbook RAG on Turn 1 STILL requires allow_external_data_egress
       on Turn 2 even when RAG is turned OFF.
    3) Explicit allow_external_data_egress=True authorizes ONLY that single turn and never persists.
    4) Disabling MCP on a tainted session keeps plain chat working with zero regression.
    5) Legacy/unverifiable history (e.g., broken checkpoint chain or NULL final_checkpoint_id)
       fails closed and requires allow_external_data_egress=True.
    6) A fresh new session remains unaffected and can use MCP without egress authorization.
    """
    db_path = tmp_path / "business.sqlite3"
    note_file = tmp_path / "private_notes.txt"
    note_file.write_text("机密本地笔记：项目预算 999 万。\n", encoding="utf-8")

    grant_holder: dict[str, str] = {}
    aliases: dict[str, str] = {}

    def decide(body: dict, _n: int):
        tools_sent = body.get("tools") or []
        tool_names = [t["function"]["name"] for t in tools_sent]
        tool_msgs = [m for m in body["messages"] if m["role"] == "tool"]
        if "read_text" in tool_names and not tool_msgs:
            return {"tool_calls": [{"id": "c_read_1", "name": "read_text", "arguments": {"grant_id": grant_holder["id"]}}]}
        if aliases.get("echo") in tool_names and not tool_msgs:
            return {"tool_calls": [{"id": "c_mcp_1", "name": aliases["echo"], "arguments": {"text": "ping"}}]}
        return {"text": "已完成处理。"}

    provider = ScriptedProvider(decide)
    with provider_transport(provider) as transport:
        async with live_agent_app(db_path, transport) as (_app, client):
            srv, _ = await _setup_stdio_server(client, name="CrossTurnSrv", approve_tools=("echo_text",))
            aliases["echo"] = make_mcp_tool_alias(srv["server_id"], "echo_text")

            # --- Case 1: Historical local file read, then revoke grant, then try MCP ---
            file_session = await new_session(client)
            grant = (
                await client.post(
                    f"/api/v1/_host/sessions/{file_session}/grants",
                    headers=HOST_HEADERS,
                    json={
                        "scope": "file",
                        "path": str(note_file),
                        "operations": ["read"],
                        "lifetime": "application_session",
                    },
                )
            ).json()
            grant_holder["id"] = grant["grant_id"]

            async with client.stream(
                "POST",
                run_path(file_session),
                json={"client_request_id": str(uuid4()), "message": "读取机密笔记"},
            ) as resp:
                assert resp.status_code == 200
                ev1 = [e async for e in frames(resp) if isinstance(e, dict)]
            assert ev1[-1]["type"] == "run_completed"

            # Revoke the file grant completely
            rev = await client.delete(
                f"/api/v1/_host/grants/{grant['grant_id']}",
                params={"session_id": file_session},
                headers=HOST_HEADERS,
            )
            assert rev.status_code == 204
            assert (await client.get(f"/api/v1/sessions/{file_session}/grants")).json() == []

            # Turn 2 with MCP enabled but allow_external_data_egress=False -> MUST be rejected (403)!
            runs_before = rows(db_path, "SELECT count(*) FROM runs WHERE session_id = ?", (file_session,))[0][0]
            blocked = await client.post(
                run_path(file_session),
                json={
                    "client_request_id": str(uuid4()),
                    "message": "现在调用外部 MCP",
                    "mcp": {"enabled": True, "allow_external_data_egress": False},
                },
            )
            assert blocked.status_code == 403
            assert blocked.json()["error"]["code"] == "MCP_EGRESS_AUTHORIZATION_REQUIRED"
            assert rows(db_path, "SELECT count(*) FROM runs WHERE session_id = ?", (file_session,))[0][0] == runs_before

            # Disabling MCP on the tainted session works normally!
            async with client.stream(
                "POST",
                run_path(file_session),
                json={
                    "client_request_id": str(uuid4()),
                    "message": "关闭 MCP 普通聊天",
                    "mcp": {"enabled": False, "allow_external_data_egress": False},
                },
            ) as resp:
                assert resp.status_code == 200
                ev_plain = [e async for e in frames(resp) if isinstance(e, dict)]
            assert ev_plain[-1]["type"] == "run_completed"

            # Explicit one-turn authorization allows this turn...
            async with client.stream(
                "POST",
                run_path(file_session),
                json={
                    "client_request_id": str(uuid4()),
                    "message": "本轮允许外发调用 MCP",
                    "mcp": {"enabled": True, "allow_external_data_egress": True},
                },
            ) as resp:
                assert resp.status_code == 200
                ev_allow = [e async for e in frames(resp) if isinstance(e, dict)]
            assert ev_allow[-1]["type"] == "run_completed"

            # ...but does NOT persist to the next turn!
            blocked_again = await client.post(
                run_path(file_session),
                json={
                    "client_request_id": str(uuid4()),
                    "message": "下一轮未授权再次调用 MCP",
                    "mcp": {"enabled": True, "allow_external_data_egress": False},
                },
            )
            assert blocked_again.status_code == 403
            assert blocked_again.json()["error"]["code"] == "MCP_EGRESS_AUTHORIZATION_REQUIRED"

            # --- Case 2: Historical textbook RAG turn, then RAG OFF + MCP ON ---
            rag_session = await new_session(client)
            # Simulate a completed historical RAG turn (search_knowledge in tool_calls)
            async with client.stream(
                "POST",
                run_path(rag_session),
                json={"client_request_id": str(uuid4()), "message": "历史教材问答"},
            ) as resp:
                ev_rag_seed = [e async for e in frames(resp) if isinstance(e, dict)]
            assert ev_rag_seed[-1]["type"] == "run_completed"
            rag_run_id = ev_rag_seed[-1]["run_id"]
            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute(
                    """INSERT INTO tool_calls (
                           tool_call_id, session_id, run_id, tool_name, status, summary, started_at, finished_at, source
                       ) VALUES (?, ?, ?, 'search_knowledge', 'completed', '命中 2 条教材片段',
                                 '2026-11-25T00:00:00.000Z', '2026-11-25T00:00:01.000Z', 'native')""",
                    (str(uuid4()), rag_session, rag_run_id),
                )
                conn.commit()

            rag_hist_blocked = await client.post(
                run_path(rag_session),
                json={
                    "client_request_id": str(uuid4()),
                    "message": "关闭 RAG 后再开 MCP",
                    "mcp": {"enabled": True, "allow_external_data_egress": False},
                },
            )
            assert rag_hist_blocked.status_code == 403
            assert rag_hist_blocked.json()["error"]["code"] == "MCP_EGRESS_AUTHORIZATION_REQUIRED"

            # --- Case 3: Unverifiable / legacy history provenance fails closed ---
            legacy_session = await new_session(client)
            with closing(sqlite3.connect(db_path)) as conn:
                # Point session to an opaque/unverified checkpoint not backed by completed runs
                conn.execute(
                    "UPDATE sessions SET last_safe_checkpoint_id = 'legacy-unverified-cp-001' WHERE session_id = ?",
                    (legacy_session,),
                )
                conn.commit()

            legacy_blocked = await client.post(
                run_path(legacy_session),
                json={
                    "client_request_id": str(uuid4()),
                    "message": "旧版本会话尝试调用 MCP",
                    "mcp": {"enabled": True, "allow_external_data_egress": False},
                },
            )
            assert legacy_blocked.status_code == 403
            assert legacy_blocked.json()["error"]["code"] == "MCP_EGRESS_AUTHORIZATION_REQUIRED"

            # --- Case 4: Fresh new session has no protected history -> MCP succeeds directly ---
            clean_session = await new_session(client)
            async with client.stream(
                "POST",
                run_path(clean_session),
                json={
                    "client_request_id": str(uuid4()),
                    "message": "新会话直接调用 MCP",
                    "mcp": {"enabled": True, "allow_external_data_egress": False},
                },
            ) as resp:
                assert resp.status_code == 200
                ev_clean = [e async for e in frames(resp) if isinstance(e, dict)]
            assert ev_clean[-1]["type"] == "run_completed"


async def test_mcp_tool_reply_streams_deltas_in_real_time(tmp_path):
    """Verify post-tool replies emit message_delta live BEFORE the upstream LLM stream finishes."""
    import threading

    db_path = tmp_path / "business.sqlite3"
    aliases: dict[str, str] = {}
    release_tail = threading.Event()

    def decide(body: dict, call_idx: int):
        if call_idx == 1:
            return {"tool_calls": [{"id": "c_live", "name": aliases["echo"], "arguments": {"text": "stream-check"}}]}
        return {
            "mode": "gated_stream",
            "before_gate": ["第一段实时", "输出，"],
            "gate": release_tail,
            "after_gate": ["第二段后续完成。"],
        }

    provider = ScriptedProvider(decide)
    with provider_transport(provider) as transport:
        async with live_agent_app(db_path, transport) as (_app, client):
            srv, _ = await _setup_stdio_server(client, name="LiveStreamSrv", approve_tools=("echo_text",))
            aliases["echo"] = make_mcp_tool_alias(srv["server_id"], "echo_text")
            session = await new_session(client)

            deltas: list[str] = []
            events: list[dict] = []
            async with client.stream(
                "POST",
                run_path(session),
                json={"client_request_id": str(uuid4()), "message": "验证实时流式输出", "mcp": {"enabled": True}},
            ) as resp:
                assert resp.status_code == 200
                async for ev in frames(resp):
                    if not isinstance(ev, dict):
                        continue
                    events.append(ev)
                    if ev["type"] == "message_delta":
                        deltas.append(ev["data"]["delta"])
                        if "".join(deltas) == "第一段实时输出，":
                            release_tail.set()

            assert release_tail.is_set(), "Live deltas were NOT emitted before upstream stream finished"
            assert events[-1]["type"] == "run_completed"
            assert "".join(deltas) == "第一段实时输出，第二段后续完成。"
            msgs = (await client.get(f"/api/v1/sessions/{session}/messages")).json()
            assert msgs[-1]["content"] == "第一段实时输出，第二段后续完成。"
