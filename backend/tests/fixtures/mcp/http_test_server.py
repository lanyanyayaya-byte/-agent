"""Controllable real Streamable HTTP MCP test server for P6.2 verification.

Runs a genuine local Uvicorn + Starlette HTTP server on 127.0.0.1:<ephemeral port>
speaking MCP Streamable HTTP (protocol 2025-11-25) with both JSON and SSE responses,
Mcp-Session-Id lifecycle, GET SSE notifications (tools/list_changed), oversized
responses, malformed responses, auth header checks, and redirect rejection tests.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from collections.abc import AsyncIterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response, StreamingResponse
from starlette.routing import Route

PROTOCOL_VERSION = "2025-11-25"


@dataclass
class HttpServerState:
    expected_auth: str | None = None
    mutated: bool = False
    call_counts: dict[str, int] = field(default_factory=dict)
    cancelled_request_ids: list[Any] = field(default_factory=list)
    terminated_sessions: list[str] = field(default_factory=list)
    received_auth_headers: list[str | None] = field(default_factory=list)
    redirect_target_hits: int = 0
    sse_queues: list[asyncio.Queue[dict[str, Any] | None]] = field(default_factory=list)
    active_cancels: dict[Any, asyncio.Event] = field(default_factory=dict)
    sessions: set[str] = field(default_factory=set)

    def _tools_page1(self) -> list[dict[str, Any]]:
        desc = "HTTP echo tool (v2 updated)." if self.mutated else "HTTP echo tool."
        return [
            {
                "name": "http_echo",
                "title": "HTTP Echo",
                "description": desc,
                "inputSchema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
                "annotations": {"readOnlyHint": True},
            },
            {
                "name": "http_structured",
                "description": "Return structured JSON + text via SSE response.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                },
            },
            {
                "name": "http_error",
                "description": "Return isError=true.",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]

    def _tools_page2(self) -> list[dict[str, Any]]:
        tools = [
            {
                "name": "http_slow",
                "description": "Sleep for delay seconds; supports cancellation.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "delay": {"type": "number"},
                        "tag": {"type": "string"},
                    },
                },
            },
            {
                "name": "http_mutate_tools",
                "description": "Mutate tool list and push notifications/tools/list_changed.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "http_oversized_json",
                "description": "Return oversized JSON response.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"size_bytes": {"type": "integer"}},
                },
            },
            {
                "name": "http_oversized_sse",
                "description": "Return oversized SSE chunked response without Content-Length.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"size_bytes": {"type": "integer"}},
                },
            },
            {
                "name": "http_malformed",
                "description": "Return invalid JSON with HTTP 200 application/json.",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]
        if self.mutated:
            tools.append(
                {
                    "name": "http_dynamic_tool",
                    "description": "Dynamically added HTTP tool.",
                    "inputSchema": {"type": "object", "properties": {}},
                }
            )
        return tools


def build_http_mcp_app(state: HttpServerState) -> Starlette:
    async def mcp_endpoint(request: Request) -> Response:
        auth = request.headers.get("authorization")
        state.received_auth_headers.append(auth)
        if state.expected_auth is not None and auth != state.expected_auth:
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        if request.method == "DELETE":
            sid = request.headers.get("mcp-session-id")
            if sid:
                state.terminated_sessions.append(sid)
                state.sessions.discard(sid)
            return Response(status_code=204)

        if request.method == "GET":
            queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
            state.sse_queues.append(queue)

            async def event_gen() -> AsyncIterator[bytes]:
                try:
                    while True:
                        try:
                            item = await asyncio.wait_for(queue.get(), timeout=1.0)
                        except TimeoutError:
                            yield b": keepalive\n\n"
                            continue
                        if item is None:
                            return
                        data = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
                        yield f"event: message\ndata: {data}\n\n".encode("utf-8")
                finally:
                    if queue in state.sse_queues:
                        state.sse_queues.remove(queue)

            return StreamingResponse(event_gen(), media_type="text/event-stream")

        # POST
        body = await request.json()
        method = body.get("method")
        req_id = body.get("id")
        params = body.get("params") or {}

        if method == "initialize":
            session_id = str(uuid4())
            state.sessions.add(session_id)
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": PROTOCOL_VERSION,
                        "capabilities": {"tools": {"listChanged": True}},
                        "serverInfo": {"name": "local-http-test-server", "version": "1.0.0"},
                    },
                },
                headers={"mcp-session-id": session_id},
            )

        if method == "notifications/initialized":
            return Response(status_code=202)

        if method == "notifications/cancelled":
            target_id = params.get("requestId")
            state.cancelled_request_ids.append(target_id)
            ev = state.active_cancels.get(target_id)
            if ev is not None:
                ev.set()
            return Response(status_code=202)

        if method == "tools/list":
            cursor = params.get("cursor")
            if cursor is None:
                return JSONResponse(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {"tools": state._tools_page1(), "nextCursor": "http-page-2"},
                    }
                )
            if cursor == "http-page-2":
                return JSONResponse(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {"tools": state._tools_page2()},
                    }
                )
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32602, "message": "Invalid cursor"},
                }
            )

        if method == "tools/call":
            tool_name = params.get("name", "")
            args = params.get("arguments") or {}
            state.call_counts[tool_name] = state.call_counts.get(tool_name, 0) + 1
            count = state.call_counts[tool_name]

            if tool_name == "http_echo":
                return JSONResponse(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [{"type": "text", "text": f"http_echo:{args.get('text', '')}"}],
                            "isError": False,
                        },
                    }
                )

            if tool_name == "http_structured":
                x = int(args.get("x", 0))
                payload = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {"type": "text", "text": f"squared={x * x}"},
                            {"type": "resource_link", "uri": "file:///etc/passwd", "name": "ignored"},
                        ],
                        "structuredContent": {"x": x, "squared": x * x},
                        "isError": False,
                    },
                }

                async def sse_reply() -> AsyncIterator[bytes]:
                    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                    yield f"event: message\ndata: {encoded}\n\n".encode("utf-8")

                return StreamingResponse(sse_reply(), media_type="text/event-stream")

            if tool_name == "http_error":
                return JSONResponse(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [{"type": "text", "text": "http tool logical error"}],
                            "isError": True,
                        },
                    }
                )

            if tool_name == "http_slow":
                delay = float(args.get("delay", 1.0))
                tag = str(args.get("tag", ""))
                cancel_ev = asyncio.Event()
                state.active_cancels[req_id] = cancel_ev
                try:
                    try:
                        await asyncio.wait_for(cancel_ev.wait(), timeout=delay)
                        # Cancelled while waiting
                        return JSONResponse(
                            {
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "error": {"code": -32800, "message": "Cancelled"},
                            }
                        )
                    except TimeoutError:
                        return JSONResponse(
                            {
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "result": {
                                    "content": [{"type": "text", "text": f"http_slow_done:{tag}:{count}"}],
                                    "isError": False,
                                },
                            }
                        )
                finally:
                    state.active_cancels.pop(req_id, None)

            if tool_name == "http_mutate_tools":
                state.mutated = True
                notif = {"jsonrpc": "2.0", "method": "notifications/tools/list_changed"}
                for q in list(state.sse_queues):
                    q.put_nowait(notif)
                return JSONResponse(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [{"type": "text", "text": "http_mutated"}],
                            "isError": False,
                        },
                    }
                )

            if tool_name == "http_oversized_json":
                size_bytes = int(args.get("size_bytes", 16384))
                return JSONResponse(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [{"type": "text", "text": "Y" * size_bytes}],
                            "isError": False,
                        },
                    }
                )

            if tool_name == "http_oversized_sse":
                size_bytes = int(args.get("size_bytes", 16384))

                async def sse_huge() -> AsyncIterator[bytes]:
                    chunk_text = "Z" * 1024
                    payload = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [{"type": "text", "text": "Z" * size_bytes}],
                            "isError": False,
                        },
                    }
                    encoded = f"event: message\ndata: {json.dumps(payload)}\n\n".encode("utf-8")
                    for idx in range(0, len(encoded), len(chunk_text)):
                        yield encoded[idx : idx + len(chunk_text)]

                return StreamingResponse(sse_huge(), media_type="text/event-stream")

            if tool_name == "http_malformed":
                return Response(content=b"<<<MALFORMED JSON>>>", media_type="application/json", status_code=200)

            if tool_name == "http_dynamic_tool":
                return JSONResponse(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [{"type": "text", "text": "http_dynamic_ok"}],
                            "isError": False,
                        },
                    }
                )

            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
                }
            )

        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown method: {method}"},
            }
        )

    async def redirect_endpoint(_request: Request) -> Response:
        return RedirectResponse(url="/redirect-target", status_code=302)

    async def redirect_target(_request: Request) -> Response:
        state.redirect_target_hits += 1
        return JSONResponse({"leaked": True})

    return Starlette(
        routes=[
            Route("/mcp", mcp_endpoint, methods=["GET", "POST", "DELETE"]),
            Route("/redirect", redirect_endpoint, methods=["GET", "POST"]),
            Route("/redirect-target", redirect_target, methods=["GET", "POST"]),
        ]
    )


@contextmanager
def run_http_mcp_server(expected_auth: str | None = None):
    """Run a real Uvicorn Streamable HTTP MCP server in a background thread."""
    state = HttpServerState(expected_auth=expected_auth)
    app = build_http_mcp_app(state)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(64)
    port = sock.getsockname()[1]

    config = uvicorn.Config(app, log_level="critical", access_log=False, lifespan="off")
    server = uvicorn.Server(config)

    loop_holder: list[asyncio.AbstractEventLoop] = []

    def _serve() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop_holder.append(loop)
        try:
            loop.run_until_complete(server.serve(sockets=[sock]))
        finally:
            loop.close()

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()

    deadline = time.monotonic() + 5.0
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        raise RuntimeError("Local HTTP MCP test server failed to start")

    try:
        yield f"http://127.0.0.1:{port}", state
    finally:
        if loop_holder and not loop_holder[0].is_closed():
            for q in list(state.sse_queues):
                try:
                    loop_holder[0].call_soon_threadsafe(q.put_nowait, None)
                except Exception:
                    pass
        server.should_exit = True
        thread.join(timeout=3.0)
        sock.close()
