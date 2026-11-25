"""Controlled MCP Client transport and session wrapper built on official `mcp` 2.x SDK.

Reuses official `mcp.client.session.ClientSession`, `mcp.client.streamable_http.streamable_http_client`,
and `mcp_types` wire models, while adding the security and lifecycle boundaries that
the raw SDK does not enforce on its own:
  1. stdio: strict env allowlist (no SHELL/API keys), pre-parse line byte limit,
     bounded redacted stderr ring buffer, child PID tracking, and fast bounded
     POSIX process-group teardown (EOF -> SIGTERM -> SIGKILL).
  2. Streamable HTTP: TOCTOU-free DNS resolution + IP policy validation inside
     httpcore2's `connect_tcp()`, disabled redirects & env proxies, origin pinning,
     and pre-parse streaming response byte cap.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from subprocess import PIPE
from time import monotonic
from typing import Any

import anyio
import anyio.lowlevel
import anyio.to_thread
import httpcore2
import httpx2
import mcp_types as types
from anyio.abc import Process
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp import ClientSession, MCPError
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.message import SessionMessage
from pydantic import ValidationError

from app.mcp.models import (
    McpContentBlock,
    McpError,
    McpRuntimeLimits,
    McpServerConfig,
    McpToolCallResult,
    McpToolDefinition,
)
from app.mcp.policy import (
    ValidatedHttpTarget,
    build_stdio_environment,
    compute_tool_definition_hash,
    redact_text,
    resolve_stdio_executable,
    validate_http_url,
    validate_resolved_ip,
    validate_server_config,
)
from app.mcp.process_registry import McpProcessRegistry

logger = logging.getLogger("knowledgeagent.mcp.client")

CLIENT_INFO = types.Implementation(name="KnowledgeAgent", version="0.3.0")

# JSON-RPC / MCP error codes defined by mcp_types
_PARSE_ERROR = getattr(types, "PARSE_ERROR", -32700)
_REQUEST_TIMEOUT = getattr(types, "REQUEST_TIMEOUT", -32001)
_CONNECTION_CLOSED = getattr(types, "CONNECTION_CLOSED", -32000)

_SIZE_SENTINEL = "[MCP_RESPONSE_TOO_LARGE]"
_POLICY_SENTINEL = "[MCP_NETWORK_POLICY_VIOLATION]"


class _StderrRingBuffer:
    """Thread/task-safe bounded ring buffer for redacted stdio stderr lines."""

    def __init__(self, *, max_lines: int, max_bytes: int, secrets: tuple[str, ...]) -> None:
        self._max_lines = max_lines
        self._max_bytes = max_bytes
        self._secrets = secrets
        self._lines: deque[str] = deque()
        self._bytes = 0

    def append(self, raw_line: str) -> None:
        cleaned = redact_text(raw_line, secrets=self._secrets, max_chars=400)
        if not cleaned:
            return
        line_bytes = len(cleaned.encode("utf-8"))
        self._lines.append(cleaned)
        self._bytes += line_bytes
        while len(self._lines) > self._max_lines or (self._bytes > self._max_bytes and len(self._lines) > 1):
            removed = self._lines.popleft()
            self._bytes -= len(removed.encode("utf-8"))

    def snapshot(self) -> tuple[str, ...]:
        return tuple(self._lines)


@dataclass
class _TransportState:
    resolved_command: Path | None = None
    child_pid: int | None = None
    stderr: _StderrRingBuffer | None = None
    last_error: McpError | None = None
    closed: bool = False
    _on_closed: Callable[[McpError | None], None] | None = field(default=None, repr=False)

    def record_error(self, error: McpError) -> None:
        if self.last_error is None:
            self.last_error = error

    def consume_error(self) -> McpError | None:
        error = self.last_error
        self.last_error = None
        return error

    def notify_closed(self, error: McpError | None = None) -> None:
        if error is not None:
            self.record_error(error)
        if not self.closed:
            self.closed = True
            if self._on_closed is not None:
                self._on_closed(self.last_error)


class _ResponseTooLargeStreamError(httpx2.StreamError):
    """Raised mid-stream when an HTTP/SSE response exceeds max_transport_bytes."""

    def __init__(self, max_bytes: int) -> None:
        super().__init__(f"{_SIZE_SENTINEL} Response exceeded {max_bytes} bytes")


class _ImmediateOverflowStream(httpx2.AsyncByteStream):
    """Fails immediately on read when Content-Length already exceeds max_transport_bytes."""

    def __init__(self, max_bytes: int, state: _TransportState) -> None:
        self._max_bytes = max_bytes
        self._state = state

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self._state.record_error(
            McpError("MCP_RESPONSE_TOO_LARGE", f"MCP HTTP 响应声明体积超过 {self._max_bytes} 字节上限。")
        )
        raise _ResponseTooLargeStreamError(self._max_bytes)
        yield b""  # pragma: no cover

    async def aclose(self) -> None:
        return None


class _BoundedAsyncByteStream(httpx2.AsyncByteStream):
    """Enforces a strict byte ceiling while streaming HTTP/SSE response bodies."""

    def __init__(
        self,
        inner: httpx2.AsyncByteStream,
        *,
        max_bytes: int,
        state: _TransportState,
    ) -> None:
        self._inner = inner
        self._max_bytes = max_bytes
        self._state = state
        self._total = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._inner:
                self._total += len(chunk)
                if self._total > self._max_bytes:
                    self._state.record_error(
                        McpError("MCP_RESPONSE_TOO_LARGE", f"MCP HTTP 响应超过 {self._max_bytes} 字节上限。")
                    )
                    await self.aclose()
                    raise _ResponseTooLargeStreamError(self._max_bytes)
                yield chunk
        except _ResponseTooLargeStreamError:
            raise
        except httpx2.StreamError:
            raise

    async def aclose(self) -> None:
        with suppress(Exception):
            await self._inner.aclose()


class _PolicyValidatedNetworkBackend(httpcore2.AsyncNetworkBackend):
    """Resolves DNS and validates every IP inside connect_tcp() to prevent DNS rebinding."""

    def __init__(self, target: ValidatedHttpTarget, state: _TransportState) -> None:
        self._target = target
        self._state = state
        self._inner = httpcore2.AnyIOBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore2.AsyncNetworkStream:
        normalized_host = host.strip().lower()
        if normalized_host != self._target.hostname or port != self._target.port:
            err = McpError("MCP_NETWORK_POLICY_VIOLATION", "HTTP 连接目标与已配置端点不一致。")
            self._state.record_error(err)
            raise httpcore2.ConnectError(f"{_POLICY_SENTINEL} {err.message}") from err

        try:
            addrinfos = await anyio.to_thread.run_sync(
                socket.getaddrinfo,
                normalized_host,
                port,
                socket.AF_UNSPEC,
                socket.SOCK_STREAM,
            )
        except OSError as exc:
            raise httpcore2.ConnectError(f"DNS lookup failed for {normalized_host}") from exc

        validated_ips: list[str] = []
        seen: set[str] = set()
        for _family, _socktype, _proto, _canonname, sockaddr in addrinfos:
            ip_str = str(sockaddr[0])
            # Strip IPv6 scope IDs if present
            if "%" in ip_str:
                ip_str = ip_str.split("%", 1)[0]
            try:
                validate_resolved_ip(
                    ip_str,
                    scheme=self._target.scheme,
                    hostname=normalized_host,
                    allow_private_network=self._target.allow_private_network,
                )
            except McpError as err:
                self._state.record_error(err)
                raise httpcore2.ConnectError(f"{_POLICY_SENTINEL} {err.message}") from err
            if ip_str not in seen:
                seen.add(ip_str)
                validated_ips.append(ip_str)

        if not validated_ips:
            raise httpcore2.ConnectError(f"No valid IP addresses resolved for {normalized_host}")

        last_exc: Exception | None = None
        for ip_str in validated_ips:
            try:
                return await self._inner.connect_tcp(
                    host=ip_str,
                    port=port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except (httpcore2.ConnectError, httpcore2.ConnectTimeout) as exc:
                last_exc = exc
        assert last_exc is not None
        raise last_exc

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Any = None,
    ) -> httpcore2.AsyncNetworkStream:
        err = McpError("MCP_NETWORK_POLICY_VIOLATION", "禁止通过 Unix Socket 连接 HTTP MCP 服务。")
        self._state.record_error(err)
        raise httpcore2.ConnectError(f"{_POLICY_SENTINEL} {err.message}")

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


class _PolicyBoundedHttpTransport(httpx2.AsyncBaseTransport):
    """Wraps httpx2.AsyncHTTPTransport with origin pinning, DNS-validated TCP, and byte caps."""

    def __init__(
        self,
        target: ValidatedHttpTarget,
        *,
        max_bytes: int,
        state: _TransportState,
    ) -> None:
        self._target = target
        self._max_bytes = max_bytes
        self._state = state
        self._inner = httpx2.AsyncHTTPTransport(trust_env=False, retries=0)
        # Bind the pool's network backend to our TOCTOU-free DNS validator while
        # preserving httpx2's TLS context and original-hostname SNI verification.
        self._inner._pool._network_backend = _PolicyValidatedNetworkBackend(target, state)  # noqa: SLF001

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        req_scheme = request.url.scheme.lower()
        req_host = (request.url.host or "").lower()
        req_port = request.url.port or (443 if req_scheme == "https" else 80)
        if req_scheme != self._target.scheme or req_host != self._target.hostname or req_port != self._target.port:
            err = McpError("MCP_NETWORK_POLICY_VIOLATION", "禁止跨源或降级访问非配置 HTTP 端点。")
            self._state.record_error(err)
            raise httpx2.ConnectError(f"{_POLICY_SENTINEL} {err.message}")

        response = await self._inner.handle_async_request(request)

        # Reject redirect responses explicitly (follow_redirects=False is also set on client).
        if 300 <= response.status_code < 400:
            await response.aclose()
            err = McpError("MCP_NETWORK_POLICY_VIOLATION", f"禁止自动跟随 HTTP 重定向（HTTP {response.status_code}）。")
            self._state.record_error(err)
            raise httpx2.ConnectError(f"{_POLICY_SENTINEL} {err.message}")

        content_length = response.headers.get("content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                declared = -1
            if declared > self._max_bytes:
                await response.aclose()
                err = McpError("MCP_RESPONSE_TOO_LARGE", f"MCP HTTP 响应声明体积超过 {self._max_bytes} 字节上限。")
                self._state.record_error(err)
                return httpx2.Response(
                    status_code=response.status_code,
                    headers=response.headers,
                    stream=_ImmediateOverflowStream(self._max_bytes, self._state),
                    extensions=response.extensions,
                )

        return httpx2.Response(
            status_code=response.status_code,
            headers=response.headers,
            stream=_BoundedAsyncByteStream(response.stream, max_bytes=self._max_bytes, state=self._state),
            extensions=response.extensions,
        )

    async def aclose(self) -> None:
        await self._inner.aclose()


async def _terminate_stdio_process(
    process: Process,
    *,
    eof_wait_seconds: float,
    sigterm_wait_seconds: float,
) -> None:
    """Bounded 3-stage POSIX process-group cleanup: stdin EOF -> SIGTERM -> SIGKILL."""
    if process.stdin is not None:
        with suppress(Exception):
            await process.stdin.aclose()

    if process.returncode is not None:
        return

    with anyio.move_on_after(eof_wait_seconds):
        with suppress(ProcessLookupError):
            await process.wait()
        if process.returncode is not None:
            return

    pid = process.pid
    pgid: int | None = None
    if pid:
        with suppress(ProcessLookupError, PermissionError, OSError):
            pgid = os.getpgid(pid)

    # Stage 2: SIGTERM to the isolated process group.
    if pgid is not None:
        with suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(pgid, signal.SIGTERM)
    else:
        with suppress(ProcessLookupError, OSError):
            process.terminate()

    with anyio.move_on_after(sigterm_wait_seconds):
        with suppress(ProcessLookupError):
            await process.wait()
        if pgid is not None:
            while True:
                try:
                    os.killpg(pgid, 0)
                    await anyio.sleep(0.02)
                except ProcessLookupError:
                    return
        elif process.returncode is not None:
            return

    # Stage 3: SIGKILL to the entire process group and reap.
    if pgid is not None:
        with suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(pgid, signal.SIGKILL)
    with suppress(ProcessLookupError, OSError):
        process.kill()
    with anyio.move_on_after(0.5):
        with suppress(ProcessLookupError):
            await process.wait()


@asynccontextmanager
async def _bounded_stdio_transport(
    config: McpServerConfig,
    limits: McpRuntimeLimits,
    state: _TransportState,
    *,
    generation: int = 0,
    process_registry: McpProcessRegistry | None = None,
) -> AsyncIterator[
    tuple[
        MemoryObjectReceiveStream[SessionMessage | Exception],
        MemoryObjectSendStream[SessionMessage],
    ]
]:
    """Controlled stdio transport with env isolation, pre-parse byte cap, stderr ring buffer, and PGID cleanup."""
    assert config.command is not None
    resolved_exe = resolve_stdio_executable(config.command)
    state.resolved_command = resolved_exe
    isolated_env = build_stdio_environment(config)
    stderr_buf = _StderrRingBuffer(
        max_lines=limits.max_stderr_lines,
        max_bytes=limits.max_stderr_bytes,
        secrets=config.all_secret_values(),
    )
    state.stderr = stderr_buf

    if process_registry is not None:
        process_registry.mark_launch_pending(config.server_id, generation, resolved_exe)
    try:
        process = await anyio.open_process(
            [str(resolved_exe), *config.args],
            env=isolated_env,
            cwd=config.cwd,
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        if process_registry is not None:
            process_registry.clear_launch_pending(config.server_id, generation)
        raise McpError("MCP_EXECUTABLE_NOT_FOUND", f"找不到可执行文件: {resolved_exe}") from exc
    except OSError as exc:
        if process_registry is not None:
            process_registry.clear_launch_pending(config.server_id, generation)
        raise McpError("MCP_CONNECT_FAILED", f"无法启动 stdio MCP 子进程: {type(exc).__name__}") from exc

    state.child_pid = process.pid
    # Synchronously record kernel process identity before yielding to the event loop.
    if process_registry is not None and process.pid is not None:
        process_registry.register_process(config.server_id, generation, process.pid, resolved_exe)
    read_writer, read_stream = anyio.create_memory_object_stream[SessionMessage | Exception](0)
    write_stream, write_reader = anyio.create_memory_object_stream[SessionMessage](0)
    shutting_down = False
    writer_done = anyio.Event()

    async def stderr_reader() -> None:
        if process.stderr is None:
            return
        buf = bytearray()
        try:
            async for chunk in process.stderr:
                buf.extend(chunk)
                while b"\n" in buf:
                    idx = buf.index(b"\n")
                    raw_line = bytes(buf[:idx])
                    del buf[: idx + 1]
                    stderr_buf.append(raw_line.decode("utf-8", errors="replace"))
                if len(buf) > 4096:
                    stderr_buf.append(bytes(buf).decode("utf-8", errors="replace"))
                    buf.clear()
            if buf:
                stderr_buf.append(bytes(buf).decode("utf-8", errors="replace"))
        except (anyio.ClosedResourceError, anyio.BrokenResourceError, OSError):
            pass

    async def stdout_reader() -> None:
        assert process.stdout is not None
        max_bytes = limits.max_transport_bytes
        buf = bytearray()
        try:
            async with read_writer:
                async for chunk in process.stdout:
                    # Scan chunks without allowing an unbounded line to exhaust memory.
                    start = 0
                    while True:
                        nl = chunk.find(b"\n", start)
                        if nl == -1:
                            remainder = chunk[start:]
                            if len(buf) + len(remainder) > max_bytes:
                                state.notify_closed(
                                    McpError("MCP_RESPONSE_TOO_LARGE", f"MCP stdio 响应行超过 {max_bytes} 字节上限。")
                                )
                                return
                            buf.extend(remainder)
                            break

                        part = chunk[start:nl]
                        if len(buf) + len(part) > max_bytes:
                            state.notify_closed(
                                McpError("MCP_RESPONSE_TOO_LARGE", f"MCP stdio 响应行超过 {max_bytes} 字节上限。")
                            )
                            return
                        line_bytes = bytes(buf) + part if buf else part
                        buf.clear()
                        start = nl + 1

                        stripped = line_bytes.strip()
                        if not stripped:
                            continue
                        try:
                            message = types.jsonrpc_message_adapter.validate_json(stripped, by_name=False)
                        except (ValidationError, ValueError):
                            state.notify_closed(McpError("MCP_MALFORMED_RESPONSE", "MCP stdio 返回了无效的 JSON-RPC 消息。"))
                            return
                        try:
                            await read_writer.send(SessionMessage(message))
                        except (anyio.ClosedResourceError, anyio.BrokenResourceError):
                            return
                if buf.strip():
                    # Trailing unterminated frame at EOF
                    try:
                        message = types.jsonrpc_message_adapter.validate_json(bytes(buf.strip()), by_name=False)
                    except (ValidationError, ValueError):
                        state.notify_closed(McpError("MCP_MALFORMED_RESPONSE", "MCP stdio 流以不完整的畸形数据结束。"))
                        return
                    with suppress(anyio.ClosedResourceError, anyio.BrokenResourceError):
                        await read_writer.send(SessionMessage(message))
        except (anyio.ClosedResourceError, anyio.BrokenResourceError, OSError):
            pass
        finally:
            if not shutting_down:
                state.notify_closed(McpError("MCP_PROTOCOL_ERROR", "MCP stdio 子进程已意外退出或输出流中断。"))

    async def stdin_writer() -> None:
        assert process.stdin is not None
        try:
            async with write_reader:
                async for session_message in write_reader:
                    encoded = session_message.message.model_dump_json(by_alias=True, exclude_unset=True)
                    await process.stdin.send((encoded + "\n").encode("utf-8"))
        except (anyio.ClosedResourceError, anyio.BrokenResourceError, OSError):
            with suppress(Exception):
                await read_writer.aclose()
        finally:
            writer_done.set()

    async with anyio.create_task_group() as tg:
        tg.start_soon(stderr_reader)
        tg.start_soon(stdout_reader)
        tg.start_soon(stdin_writer)
        try:
            yield read_stream, write_stream
        finally:
            shutting_down = True
            with anyio.CancelScope(shield=True):
                read_stream.close()
                write_stream.close()
                with anyio.move_on_after(0.2):
                    await writer_done.wait()
                await _terminate_stdio_process(
                    process,
                    eof_wait_seconds=limits.stdio_eof_wait_seconds,
                    sigterm_wait_seconds=limits.stdio_sigterm_wait_seconds,
                )
                if process_registry is not None:
                    process_registry.unregister_process(config.server_id, generation, process.pid)
                for res in (read_stream, write_stream, read_writer, write_reader):
                    with suppress(Exception):
                        await res.aclose()
                with suppress(Exception):
                    await process.aclose()
                tg.cancel_scope.cancel()


@asynccontextmanager
async def _controlled_http_transport(
    config: McpServerConfig,
    limits: McpRuntimeLimits,
    state: _TransportState,
) -> AsyncIterator[
    tuple[
        MemoryObjectReceiveStream[SessionMessage | Exception],
        MemoryObjectSendStream[SessionMessage],
    ]
]:
    """Controlled Streamable HTTP transport reusing official SDK with DNS/IP & byte-cap guards."""
    assert config.url is not None
    target = validate_http_url(config.url, allow_private_network=config.allow_private_network)
    merged_headers: dict[str, str] = {}
    merged_headers.update(config.headers)
    merged_headers.update(config.secret_headers)

    transport = _PolicyBoundedHttpTransport(
        target,
        max_bytes=limits.max_transport_bytes,
        state=state,
    )
    timeout = httpx2.Timeout(
        connect=min(limits.connect_timeout_seconds, 10.0),
        read=max(limits.call_timeout_seconds, limits.connect_timeout_seconds) + 5.0,
        write=min(limits.connect_timeout_seconds, 15.0),
        pool=10.0,
    )
    async with httpx2.AsyncClient(
        headers=merged_headers,
        timeout=timeout,
        follow_redirects=False,
        trust_env=False,
        transport=transport,
    ) as http_client:
        async with streamable_http_client(
            target.url,
            http_client=http_client,
            terminate_on_close=True,
            max_sse_event_size=limits.max_transport_bytes,
        ) as streams:
            yield streams


def _normalize_tool(server_id: str, tool: types.Tool) -> McpToolDefinition:
    input_schema = dict(tool.input_schema) if isinstance(tool.input_schema, dict) else {"type": "object"}
    output_schema = dict(tool.output_schema) if isinstance(tool.output_schema, dict) else None
    annotations: dict[str, Any] = {}
    if tool.annotations is not None:
        annotations = tool.annotations.model_dump(by_alias=True, exclude_none=True)
    description = tool.description or ""
    definition_hash = compute_tool_definition_hash(
        name=tool.name,
        title=tool.title,
        description=description,
        input_schema=input_schema,
        output_schema=output_schema,
        annotations=annotations,
    )
    return McpToolDefinition(
        server_id=server_id,
        original_tool_name=tool.name,
        title=tool.title,
        description=description,
        input_schema=input_schema,
        output_schema=output_schema,
        annotations=annotations,
        definition_hash=definition_hash,
    )


def _normalize_call_result(
    *,
    server_id: str,
    tool_name: str,
    generation: int,
    raw: types.CallToolResult,
    duration_ms: int,
) -> McpToolCallResult:
    blocks: list[McpContentBlock] = []
    texts: list[str] = []
    omitted: list[str] = []

    for item in raw.content:
        block_type = getattr(item, "type", type(item).__name__)
        if isinstance(item, types.TextContent) or block_type == "text":
            text_value = getattr(item, "text", "")
            blocks.append(McpContentBlock(kind="text", text=text_value, original_type="text"))
            texts.append(text_value)
        else:
            mime_type = getattr(item, "mime_type", None)
            omitted.append(str(block_type))
            blocks.append(
                McpContentBlock(
                    kind="omitted_non_text",
                    text=None,
                    original_type=str(block_type),
                    mime_type=str(mime_type) if mime_type is not None else None,
                )
            )

    structured = dict(raw.structured_content) if isinstance(raw.structured_content, dict) else None
    return McpToolCallResult(
        server_id=server_id,
        original_tool_name=tool_name,
        generation=generation,
        is_error=bool(raw.is_error),
        content_blocks=tuple(blocks),
        text="\n".join(texts),
        structured_content=structured,
        omitted_non_text_types=tuple(omitted),
        duration_ms=duration_ms,
    )


def _map_sdk_exception(
    exc: Exception,
    *,
    config: McpServerConfig,
    state: _TransportState,
    default_code: str = "MCP_PROTOCOL_ERROR",
) -> McpError:
    transport_error = state.consume_error()
    if transport_error is not None:
        return transport_error
    if isinstance(exc, McpError):
        return exc

    secrets = config.all_secret_values()
    raw_msg = str(exc)
    if _SIZE_SENTINEL in raw_msg:
        return McpError("MCP_RESPONSE_TOO_LARGE", "MCP 响应体积超过安全上限，已中止读取。")
    if _POLICY_SENTINEL in raw_msg:
        return McpError("MCP_NETWORK_POLICY_VIOLATION", redact_text(raw_msg.replace(_POLICY_SENTINEL, ""), secrets=secrets))

    if isinstance(exc, MCPError):
        if exc.code == _REQUEST_TIMEOUT:
            return McpError("MCP_TOOL_TIMEOUT", uncertain_outcome=True)
        if exc.code == _PARSE_ERROR:
            return McpError("MCP_MALFORMED_RESPONSE", "MCP 服务返回了无法解析的 JSON-RPC 响应。")
        if exc.code == _CONNECTION_CLOSED:
            return McpError("MCP_PROTOCOL_ERROR", "MCP 连接已中断。")
        safe = redact_text(exc.message, secrets=secrets)
        return McpError("MCP_PROTOCOL_ERROR", f"MCP 协议错误 ({exc.code}): {safe}")

    if isinstance(exc, ValidationError):
        return McpError("MCP_MALFORMED_RESPONSE", "MCP 响应字段不符合协议格式。")

    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        if default_code == "MCP_CONNECT_TIMEOUT":
            return McpError("MCP_CONNECT_TIMEOUT")
        return McpError("MCP_TOOL_TIMEOUT", uncertain_outcome=True)

    if isinstance(exc, (httpx2.HTTPError, httpcore2.NetworkError, OSError)):
        safe = redact_text(raw_msg, secrets=secrets)
        code = "MCP_CONNECT_FAILED" if default_code == "MCP_CONNECT_TIMEOUT" else "MCP_PROTOCOL_ERROR"
        return McpError(code, safe or None)

    return McpError(default_code, redact_text(raw_msg, secrets=secrets) or None)  # type: ignore[arg-type]


class McpClientConnection:
    """Active, initialized MCP ClientSession owned exclusively by one server actor task."""

    def __init__(
        self,
        *,
        config: McpServerConfig,
        limits: McpRuntimeLimits,
        generation: int,
        session: ClientSession,
        init_result: types.InitializeResult,
        state: _TransportState,
    ) -> None:
        self._config = config
        self._limits = limits
        self.generation = generation
        self._session = session
        self.init_result = init_result
        self._state = state

    @property
    def protocol_version(self) -> str:
        return self.init_result.protocol_version

    @property
    def server_name_reported(self) -> str | None:
        return self.init_result.server_info.name if self.init_result.server_info else None

    @property
    def server_version_reported(self) -> str | None:
        return self.init_result.server_info.version if self.init_result.server_info else None

    @property
    def resolved_command(self) -> str | None:
        return str(self._state.resolved_command) if self._state.resolved_command is not None else None

    @property
    def child_pid(self) -> int | None:
        return self._state.child_pid

    @property
    def stderr_tail(self) -> tuple[str, ...]:
        return self._state.stderr.snapshot() if self._state.stderr is not None else ()

    @property
    def is_closed(self) -> bool:
        return self._state.closed

    async def list_tools(self) -> tuple[tuple[McpToolDefinition, ...], bool]:
        """Discover tools across paginated tools/list calls within configured bounds."""
        tools: list[McpToolDefinition] = []
        seen_names: set[str] = set()
        seen_cursors: set[str] = set()
        cursor: str | None = None
        truncated = False

        try:
            async with asyncio.timeout(self._limits.discovery_timeout_seconds):
                for _page in range(self._limits.max_discovery_pages):
                    params = types.PaginatedRequestParams(cursor=cursor) if cursor is not None else None
                    page_result = await self._session.list_tools(params=params)
                    for raw_tool in page_result.tools:
                        if raw_tool.name in seen_names:
                            continue
                        seen_names.add(raw_tool.name)
                        tools.append(_normalize_tool(self._config.server_id, raw_tool))
                        if len(tools) >= self._limits.max_discovered_tools:
                            truncated = True
                            return tuple(tools), truncated
                    next_cursor = page_result.next_cursor
                    if not next_cursor:
                        return tuple(tools), truncated
                    if next_cursor in seen_cursors:
                        raise McpError("MCP_MALFORMED_RESPONSE", "MCP tools/list 返回了重复分页游标。")
                    seen_cursors.add(next_cursor)
                    cursor = next_cursor
                # More pages exist beyond max_discovery_pages
                truncated = True
                return tuple(tools), truncated
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise _map_sdk_exception(exc, config=self._config, state=self._state, default_code="MCP_DISCOVERY_FAILED") from None

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> McpToolCallResult:
        """Invoke tools/call with bounded timeout and automatic SDK courtesy cancellation."""
        effective_timeout = self._limits.call_timeout_seconds if timeout_seconds is None else timeout_seconds
        started = monotonic()
        try:
            raw = await self._session.call_tool(
                tool_name,
                arguments=arguments or {},
                read_timeout_seconds=effective_timeout,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise _map_sdk_exception(exc, config=self._config, state=self._state, default_code="MCP_PROTOCOL_ERROR") from None

        if not isinstance(raw, types.CallToolResult):
            raise McpError("MCP_MALFORMED_RESPONSE", "MCP tools/call 未返回标准 CallToolResult。")

        duration_ms = max(0, round((monotonic() - started) * 1000))
        return _normalize_call_result(
            server_id=self._config.server_id,
            tool_name=tool_name,
            generation=self.generation,
            raw=raw,
            duration_ms=duration_ms,
        )


@asynccontextmanager
async def open_mcp_connection(
    config: McpServerConfig,
    *,
    limits: McpRuntimeLimits,
    generation: int,
    on_tools_changed: Callable[[], None] | None = None,
    on_transport_closed: Callable[[McpError | None], None] | None = None,
    process_registry: McpProcessRegistry | None = None,
) -> AsyncIterator[McpClientConnection]:
    """Open, initialize, and cleanly tear down a single MCP connection in the caller's task.

    MUST be entered and exited within the same asyncio/AnyIO task (the server's actor task)
    to satisfy AnyIO's strict cancel-scope task affinity.
    """
    validate_server_config(config)
    state = _TransportState(_on_closed=on_transport_closed)

    async def handle_message(message: Any) -> None:
        if isinstance(message, types.ToolListChangedNotification):
            if on_tools_changed is not None:
                on_tools_changed()

    try:
        async with AsyncExitStack() as stack:
            if config.transport == "stdio":
                read_stream, write_stream = await stack.enter_async_context(
                    _bounded_stdio_transport(
                        config,
                        limits,
                        state,
                        generation=generation,
                        process_registry=process_registry,
                    )
                )
            else:
                read_stream, write_stream = await stack.enter_async_context(
                    _controlled_http_transport(config, limits, state)
                )

            # Explicitly omit sampling, elicitation, and roots callbacks so those
            # client capabilities remain disabled in P6.
            session = await stack.enter_async_context(
                ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=limits.call_timeout_seconds,
                    message_handler=handle_message,
                    client_info=CLIENT_INFO,
                )
            )
            async with asyncio.timeout(limits.connect_timeout_seconds):
                init_result = await session.initialize()

            yield McpClientConnection(
                config=config,
                limits=limits,
                generation=generation,
                session=session,
                init_result=init_result,
                state=state,
            )
    except asyncio.CancelledError:
        raise
    except McpError:
        raise
    except Exception as exc:
        raise _map_sdk_exception(exc, config=config, state=state, default_code="MCP_CONNECT_TIMEOUT") from None
