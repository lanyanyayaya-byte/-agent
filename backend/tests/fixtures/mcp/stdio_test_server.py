"""Controllable real stdio MCP test server for P6.2 verification.

Speaks real newline-delimited JSON-RPC 2.0 MCP (protocol 2025-11-25) over stdin/stdout.
Uses only Python stdlib so subprocess startup is fast and deterministic.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "2025-11-25"


def main() -> int:
    ignore_eof_and_sigterm = "--ignore-eof-and-sigterm" in sys.argv
    marker_path = None
    for idx, arg in enumerate(sys.argv):
        if arg == "--marker-file" and idx + 1 < len(sys.argv):
            marker_path = Path(sys.argv[idx + 1])

    if ignore_eof_and_sigterm:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

    write_lock = threading.Lock()
    state_lock = threading.Lock()
    call_counts: dict[str, int] = {}
    cancelled_ids: set[Any] = set()
    active_events: dict[Any, threading.Event] = {}
    mutated = False
    grandchildren: list[subprocess.Popen[Any]] = []

    def send(payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        with write_lock:
            sys.stdout.write(line)
            sys.stdout.flush()

    def record_marker(event: dict[str, Any]) -> None:
        if marker_path is None:
            return
        with state_lock:
            with marker_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def base_tools_page1() -> list[dict[str, Any]]:
        desc = "Echo text back (updated v2)." if mutated else "Echo text back."
        return [
            {
                "name": "echo_text",
                "title": "Echo Text",
                "description": desc,
                "inputSchema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
                "annotations": {"readOnlyHint": True},
            },
            {
                "name": "structured_calc",
                "description": "Return text, structured content, and an image block.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                },
            },
            {
                "name": "return_error",
                "description": "Return a tool result with isError=true.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "dump_env",
                "description": "Return child process environment keys/values.",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]

    def base_tools_page2() -> list[dict[str, Any]]:
        tools = [
            {
                "name": "slow_tool",
                "description": "Sleep until duration elapses or cancellation arrives.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "delay": {"type": "number"},
                        "tag": {"type": "string"},
                    },
                },
            },
            {
                "name": "mutate_tools",
                "description": "Dynamically update tools and emit notifications/tools/list_changed.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "oversized_response",
                "description": "Return a response larger than size_bytes.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"size_bytes": {"type": "integer"}},
                },
            },
            {
                "name": "malformed_response",
                "description": "Emit malformed non-JSON bytes on stdout.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "crash_now",
                "description": "Log to stderr and terminate abnormally.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"stderr_msg": {"type": "string"}},
                },
            },
            {
                "name": "spawn_grandchild",
                "description": "Spawn a sleeping grandchild process in the same process group.",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]
        if mutated:
            tools.append(
                {
                    "name": "dynamic_tool",
                    "description": "Newly added tool after list_changed.",
                    "inputSchema": {"type": "object", "properties": {}},
                }
            )
        return tools

    def handle_call(req_id: Any, params: dict[str, Any]) -> None:
        nonlocal mutated
        name = params.get("name", "")
        args = params.get("arguments") or {}
        with state_lock:
            call_counts[name] = call_counts.get(name, 0) + 1
            count = call_counts[name]
        record_marker({"event": "call_started", "tool": name, "count": count, "args": args})

        if name == "echo_text":
            text = str(args.get("text", ""))
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"echo:{text}"}],
                        "isError": False,
                    },
                }
            )
            return

        if name == "structured_calc":
            a = int(args.get("a", 0))
            b = int(args.get("b", 0))
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {"type": "text", "text": f"sum={a + b}"},
                            {"type": "image", "data": "AAAA", "mimeType": "image/png"},
                        ],
                        "structuredContent": {"a": a, "b": b, "sum": a + b},
                        "isError": False,
                    },
                }
            )
            return

        if name == "return_error":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": "simulated tool failure"}],
                        "isError": True,
                    },
                }
            )
            return

        if name == "dump_env":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(dict(os.environ))}],
                        "structuredContent": {"env": dict(os.environ)},
                        "isError": False,
                    },
                }
            )
            return

        if name == "slow_tool":
            delay = float(args.get("delay", 1.0))
            tag = str(args.get("tag", ""))
            cancel_ev = threading.Event()
            with state_lock:
                if req_id in cancelled_ids:
                    cancel_ev.set()
                active_events[req_id] = cancel_ev

            def worker() -> None:
                try:
                    was_cancelled = cancel_ev.wait(timeout=delay)
                    if was_cancelled:
                        record_marker({"event": "call_cancelled", "id": req_id, "tag": tag})
                        return
                    record_marker({"event": "call_finished", "id": req_id, "tag": tag})
                    send(
                        {
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": {
                                "content": [{"type": "text", "text": f"slow_done:{tag}:{count}"}],
                                "isError": False,
                            },
                        }
                    )
                finally:
                    with state_lock:
                        active_events.pop(req_id, None)

            threading.Thread(target=worker, daemon=True).start()
            return

        if name == "mutate_tools":
            mutated = True
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": "mutated"}],
                        "isError": False,
                    },
                }
            )
            send({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})
            return

        if name == "oversized_response":
            size_bytes = int(args.get("size_bytes", 16384))
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": "X" * size_bytes}],
                        "isError": False,
                    },
                }
            )
            return

        if name == "malformed_response":
            with write_lock:
                sys.stdout.write("<<<NOT VALID JSON-RPC>>>\n")
                sys.stdout.flush()
            return

        if name == "crash_now":
            msg = str(args.get("stderr_msg", "fatal stderr detail"))
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()
            os._exit(42)

        if name == "spawn_grandchild":
            proc = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            grandchildren.append(proc)
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": str(proc.pid)}],
                        "structuredContent": {"grandchild_pid": proc.pid, "child_pid": os.getpid()},
                        "isError": False,
                    },
                }
            )
            return

        if name == "dynamic_tool":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": "dynamic_ok"}],
                        "isError": False,
                    },
                }
            )
            return

        send(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {name}"},
            }
        )

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = msg.get("method")
        req_id = msg.get("id")
        params = msg.get("params") or {}

        if method == "initialize":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": PROTOCOL_VERSION,
                        "capabilities": {"tools": {"listChanged": True}},
                        "serverInfo": {"name": "local-stdio-test-server", "version": "1.0.0"},
                    },
                }
            )
        elif method == "notifications/initialized":
            continue
        elif method == "notifications/cancelled":
            target_id = params.get("requestId")
            record_marker({"event": "notification_cancelled_received", "requestId": target_id})
            with state_lock:
                cancelled_ids.add(target_id)
                ev = active_events.get(target_id)
                if ev is not None:
                    ev.set()
        elif method == "tools/list":
            cursor = params.get("cursor")
            if cursor is None:
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {"tools": base_tools_page1(), "nextCursor": "page-2"},
                    }
                )
            elif cursor == "page-2":
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {"tools": base_tools_page2()},
                    }
                )
            else:
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32602, "message": "Invalid cursor"},
                    }
                )
        elif method == "tools/call":
            handle_call(req_id, params)
        elif req_id is not None:
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }
            )

    if ignore_eof_and_sigterm:
        # Deliberately stay alive after stdin EOF to test Stage 3 SIGKILL process-group cleanup.
        time.sleep(30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
