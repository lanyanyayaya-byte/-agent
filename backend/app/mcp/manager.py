"""Multi-server MCP connection and lifecycle manager.

Each connected MCP Server is owned by a dedicated `_ServerActor` asyncio task
that enters and exits its AnyIO transport + ClientSession contexts in the exact
same task, preventing cross-task CancelScope teardown errors.

P6.2 uses an in-memory server registry and in-memory tool catalog so this core
engine runs independently of P6.3 persistence/Keychain and P6.4 Agent integration.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.mcp.client import McpClientConnection, open_mcp_connection
from app.mcp.models import (
    McpDiagnostics,
    McpError,
    McpErrorCode,
    McpRuntimeLimits,
    McpServerConfig,
    McpToolCallResult,
    McpToolDefinition,
    ServerRuntimeState,
)
from app.mcp.policy import compute_config_trust_fingerprint, redact_text, validate_server_config


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass
class _CallRequest:
    tool_name: str
    arguments: dict[str, Any]
    timeout_seconds: float | None
    future: asyncio.Future[McpToolCallResult]
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    cancelled: bool = False


@dataclass
class _RefreshRequest:
    future: asyncio.Future[tuple[McpToolDefinition, ...]]


class _ServerActor:
    """Dedicated single-task owner of one MCP server connection generation."""

    def __init__(
        self,
        *,
        slot: _ServerSlot,
        config: McpServerConfig,
        generation: int,
        limits: McpRuntimeLimits,
        connect_semaphore: asyncio.Semaphore,
        on_catalog_updated: Callable[[str, tuple[McpToolDefinition, ...], int], None] | None,
    ) -> None:
        self._slot = slot
        self._config = config
        self.generation = generation
        self._limits = limits
        self._connect_semaphore = connect_semaphore
        self._on_catalog_updated = on_catalog_updated

        self._loop = asyncio.get_running_loop()
        self.ready: asyncio.Future[McpDiagnostics] = self._loop.create_future()
        self._stop_event = asyncio.Event()
        self._wakeup_event = asyncio.Event()
        self._transport_closed_event = asyncio.Event()
        self._transport_close_error: McpError | None = None
        self._pending_list_changed = False

        self._call_queue: deque[_CallRequest] = deque()
        self._refresh_queue: deque[_RefreshRequest] = deque()
        self._active_call: _CallRequest | None = None
        self._conn: McpClientConnection | None = None
        self.task: asyncio.Task[None] = asyncio.create_task(
            self._run(),
            name=f"mcp-actor:{config.server_id}:g{generation}",
        )

    def _is_current(self) -> bool:
        return self._slot.generation == self.generation

    def request_stop(self) -> None:
        self._stop_event.set()
        self._wakeup_event.set()
        if self._active_call is not None:
            self._active_call.cancelled = True
            self._active_call.cancel_event.set()

    def enqueue_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        timeout_seconds: float | None,
    ) -> _CallRequest:
        if self._stop_event.is_set() or self._transport_closed_event.is_set() or self.task.done():
            raise McpError("MCP_SERVER_UNAVAILABLE")
        # Count only non-cancelled queued items against the queue cap.
        live_queued = sum(1 for req in self._call_queue if not req.cancelled)
        if live_queued >= self._limits.max_queued_calls_per_server:
            raise McpError("MCP_SERVER_BUSY")
        req = _CallRequest(
            tool_name=tool_name,
            arguments=arguments,
            timeout_seconds=timeout_seconds,
            future=self._loop.create_future(),
        )
        self._call_queue.append(req)
        self._wakeup_event.set()
        return req

    def enqueue_refresh(self) -> asyncio.Future[tuple[McpToolDefinition, ...]]:
        if self._stop_event.is_set() or self._transport_closed_event.is_set() or self.task.done():
            raise McpError("MCP_SERVER_UNAVAILABLE")
        req = _RefreshRequest(future=self._loop.create_future())
        self._refresh_queue.append(req)
        self._wakeup_event.set()
        return req.future

    def _on_tools_changed(self) -> None:
        if self._stop_event.is_set() or not self._is_current():
            return
        self._pending_list_changed = True
        self._wakeup_event.set()

    def _on_transport_closed(self, error: McpError | None) -> None:
        if self._transport_close_error is None:
            self._transport_close_error = error
        self._transport_closed_event.set()
        self._wakeup_event.set()

    def _fail_pending_work(self, error: McpError) -> None:
        while self._call_queue:
            req = self._call_queue.popleft()
            if not req.future.done():
                if req.cancelled:
                    req.future.set_exception(McpError("MCP_TOOL_CANCELLED"))
                else:
                    req.future.set_exception(error)
        while self._refresh_queue:
            req = self._refresh_queue.popleft()
            if not req.future.done():
                req.future.set_exception(error)

    def _sync_stderr(self) -> None:
        if self._conn is not None and self._is_current():
            self._slot.stderr_tail = self._conn.stderr_tail

    async def _run(self) -> None:
        final_error: McpError | None = None
        acquired = False
        try:
            await self._connect_semaphore.acquire()
            acquired = True
            if self._stop_event.is_set():
                if not self.ready.done():
                    self.ready.set_exception(McpError("MCP_SERVER_UNAVAILABLE", "连接已取消。"))
                return

            async with open_mcp_connection(
                self._config,
                limits=self._limits,
                generation=self.generation,
                on_tools_changed=self._on_tools_changed,
                on_transport_closed=self._on_transport_closed,
            ) as conn:
                self._conn = conn
                if self._is_current():
                    self._slot.resolved_command = conn.resolved_command
                    self._slot.child_pid = conn.child_pid
                    self._slot.protocol_version = conn.protocol_version
                    self._slot.server_name_reported = conn.server_name_reported
                    self._slot.server_version_reported = conn.server_version_reported

                tools, truncated = await conn.list_tools()
                self._sync_stderr()

                # Release the connect concurrency slot once handshake + initial discovery finish.
                self._connect_semaphore.release()
                acquired = False

                if self._stop_event.is_set():
                    if not self.ready.done():
                        self.ready.set_exception(McpError("MCP_SERVER_UNAVAILABLE", "连接已取消。"))
                    return

                if self._is_current():
                    self._slot.tools = tools
                    self._slot.tools_truncated = truncated
                    self._slot.catalog_version += 1
                    self._slot.state = "connected"
                    self._slot.connected_at = _utc_now()
                    self._slot.last_error_code = None
                    self._slot.last_error_message = None
                    if self._on_catalog_updated is not None:
                        with suppress(Exception):
                            self._on_catalog_updated(
                                self._config.server_id,
                                tools,
                                self._slot.catalog_version,
                            )

                diag = self._slot.snapshot()
                if not self.ready.done():
                    self.ready.set_result(diag)

                await self._serve_loop(conn)

        except asyncio.CancelledError:
            final_error = McpError("MCP_SERVER_UNAVAILABLE", "MCP 连接任务已取消。")
            if not self.ready.done():
                self.ready.set_exception(final_error)
            raise
        except McpError as exc:
            final_error = exc
            if not self.ready.done():
                self.ready.set_exception(exc)
        except Exception as exc:
            safe_msg = redact_text(str(exc), secrets=self._config.all_secret_values())
            final_error = McpError("MCP_CONNECT_FAILED", safe_msg or None)
            if not self.ready.done():
                self.ready.set_exception(final_error)
        finally:
            if acquired:
                self._connect_semaphore.release()
            self._sync_stderr()
            self._conn = None
            if final_error is None and self._transport_closed_event.is_set() and not self._stop_event.is_set():
                final_error = self._transport_close_error or McpError("MCP_PROTOCOL_ERROR", "MCP 服务连接已中断。")

            pending_err = final_error or McpError("MCP_SERVER_UNAVAILABLE", "MCP 服务已断开连接。")
            self._fail_pending_work(pending_err)

            if self._is_current():
                self._slot.child_pid = None
                self._slot.connected_at = None
                if self._stop_event.is_set():
                    self._slot.state = "disconnected"
                    self._slot.tools = ()
                    self._slot.tools_truncated = False
                else:
                    self._slot.state = "error"
                    self._slot.tools = ()
                    self._slot.tools_truncated = False
                    if final_error is not None:
                        self._slot.last_error_code = final_error.code
                        self._slot.last_error_message = redact_text(
                            final_error.message,
                            secrets=self._config.all_secret_values(),
                        )
                if self._slot.actor is self:
                    self._slot.actor = None

    async def _serve_loop(self, conn: McpClientConnection) -> None:
        while not self._stop_event.is_set() and not self._transport_closed_event.is_set() and not conn.is_closed:
            if self._refresh_queue or self._pending_list_changed:
                await self._handle_refresh(conn)
                continue

            if self._call_queue:
                req = self._call_queue.popleft()
                if req.cancelled or req.future.done():
                    if not req.future.done():
                        req.future.set_exception(McpError("MCP_TOOL_CANCELLED"))
                    continue
                await self._execute_call(conn, req)
                continue

            self._wakeup_event.clear()
            # Re-check conditions after clearing to avoid missed wakeups.
            if (
                self._stop_event.is_set()
                or self._transport_closed_event.is_set()
                or conn.is_closed
                or self._call_queue
                or self._refresh_queue
                or self._pending_list_changed
            ):
                continue
            await self._wakeup_event.wait()

    async def _handle_refresh(self, conn: McpClientConnection) -> None:
        waiters: list[_RefreshRequest] = []
        while self._refresh_queue:
            waiters.append(self._refresh_queue.popleft())
        self._pending_list_changed = False

        try:
            tools, truncated = await conn.list_tools()
            self._sync_stderr()
        except McpError as exc:
            self._sync_stderr()
            for waiter in waiters:
                if not waiter.future.done():
                    waiter.future.set_exception(exc)
            if exc.code in ("MCP_PROTOCOL_ERROR", "MCP_RESPONSE_TOO_LARGE", "MCP_MALFORMED_RESPONSE"):
                self._transport_close_error = exc
                self._transport_closed_event.set()
            return

        if self._is_current():
            self._slot.tools = tools
            self._slot.tools_truncated = truncated
            self._slot.catalog_version += 1
            if self._on_catalog_updated is not None:
                with suppress(Exception):
                    self._on_catalog_updated(
                        self._config.server_id,
                        tools,
                        self._slot.catalog_version,
                    )
        for waiter in waiters:
            if not waiter.future.done():
                waiter.future.set_result(tools)

    async def _execute_call(self, conn: McpClientConnection, req: _CallRequest) -> None:
        self._active_call = req
        call_task = asyncio.create_task(
            conn.call_tool(
                req.tool_name,
                req.arguments,
                timeout_seconds=req.timeout_seconds,
            )
        )
        cancel_wait = asyncio.create_task(req.cancel_event.wait())
        stop_wait = asyncio.create_task(self._stop_event.wait())
        try:
            done, pending = await asyncio.wait(
                {call_task, cancel_wait, stop_wait},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if call_task in done:
                try:
                    result = call_task.result()
                except asyncio.CancelledError:
                    if not req.future.done():
                        req.future.set_exception(McpError("MCP_TOOL_CANCELLED", uncertain_outcome=True))
                except McpError as exc:
                    if not req.future.done():
                        req.future.set_exception(exc)
                    # Fatal transport-level errors force a clean disconnect to error state.
                    if exc.code in ("MCP_RESPONSE_TOO_LARGE", "MCP_MALFORMED_RESPONSE") or conn.is_closed:
                        self._transport_close_error = exc
                        self._transport_closed_event.set()
                except Exception as exc:
                    safe = redact_text(str(exc), secrets=self._config.all_secret_values())
                    if not req.future.done():
                        req.future.set_exception(McpError("MCP_PROTOCOL_ERROR", safe or None))
                else:
                    if not req.future.done():
                        req.future.set_result(result)
            else:
                # Caller cancelled or server is stopping: cancel call_task so SDK emits
                # courtesy notifications/cancelled without tearing down the actor task.
                call_task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await asyncio.wait_for(asyncio.shield(call_task), timeout=1.0)
                if not req.future.done():
                    if self._stop_event.is_set() and not req.cancelled:
                        req.future.set_exception(McpError("MCP_SERVER_UNAVAILABLE", "MCP 服务正在断开连接。"))
                    else:
                        req.future.set_exception(McpError("MCP_TOOL_CANCELLED", uncertain_outcome=True))
            for task in pending:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        finally:
            if call_task.done() and not call_task.cancelled():
                with suppress(Exception):
                    call_task.exception()
            self._sync_stderr()
            self._active_call = None


@dataclass
class _ServerSlot:
    config: McpServerConfig
    fingerprint: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    state: ServerRuntimeState = "disconnected"
    generation: int = 0
    actor: _ServerActor | None = None
    protocol_version: str | None = None
    server_name_reported: str | None = None
    server_version_reported: str | None = None
    resolved_command: str | None = None
    child_pid: int | None = None
    tools: tuple[McpToolDefinition, ...] = ()
    catalog_version: int = 0
    tools_truncated: bool = False
    last_error_code: McpErrorCode | None = None
    last_error_message: str | None = None
    stderr_tail: tuple[str, ...] = ()
    connected_at: str | None = None

    def snapshot(self) -> McpDiagnostics:
        return McpDiagnostics(
            server_id=self.config.server_id,
            name=self.config.name,
            transport=self.config.transport,
            state=self.state,
            generation=self.generation,
            protocol_version=self.protocol_version,
            server_name_reported=self.server_name_reported,
            server_version_reported=self.server_version_reported,
            resolved_command=self.resolved_command,
            child_pid=self.child_pid,
            tool_count=len(self.tools),
            catalog_version=self.catalog_version,
            tools_truncated=self.tools_truncated,
            last_error_code=self.last_error_code,
            last_error_message=self.last_error_message,
            stderr_tail=self.stderr_tail,
            connected_at=self.connected_at,
        )


class McpManager:
    """Manages multiple MCP servers with isolated per-server actors and in-memory catalog."""

    def __init__(
        self,
        *,
        limits: McpRuntimeLimits | None = None,
        on_catalog_updated: Callable[[str, tuple[McpToolDefinition, ...], int], None] | None = None,
    ) -> None:
        self.limits = limits or McpRuntimeLimits()
        self._on_catalog_updated = on_catalog_updated
        self._slots: dict[str, _ServerSlot] = {}
        self._connect_semaphore = asyncio.Semaphore(self.limits.max_connect_concurrency)
        self._shutting_down = False

    def _require_slot(self, server_id: str) -> _ServerSlot:
        slot = self._slots.get(server_id)
        if slot is None:
            raise McpError("MCP_SERVER_UNAVAILABLE", f"未找到 MCP 服务: {server_id}")
        return slot

    def _active_server_count(self) -> int:
        return sum(1 for slot in self._slots.values() if slot.state in ("connecting", "connected"))

    async def register_server(self, config: McpServerConfig) -> McpDiagnostics:
        """Register or update an MCP server configuration in memory."""
        if self._shutting_down:
            raise McpError("MCP_SERVER_UNAVAILABLE", "MCP Manager 正在关闭。")
        validate_server_config(config)
        fingerprint = compute_config_trust_fingerprint(config)

        existing = self._slots.get(config.server_id)
        if existing is None:
            if len(self._slots) >= self.limits.max_servers:
                raise McpError("MCP_INVALID_CONFIG", f"最多允许配置 {self.limits.max_servers} 个 MCP 服务。")
            slot = _ServerSlot(config=config, fingerprint=fingerprint)
            self._slots[config.server_id] = slot
            return slot.snapshot()

        async with existing.lock:
            security_changed = existing.fingerprint != fingerprint or existing.config.secret_env != config.secret_env or existing.config.secret_headers != config.secret_headers
            existing.config = config
            existing.fingerprint = fingerprint
            if security_changed and existing.actor is not None:
                await self._disconnect_locked(existing)
            return existing.snapshot()

    async def remove_server(self, server_id: str) -> None:
        """Disconnect (if active) and remove a server from the in-memory registry."""
        slot = self._slots.get(server_id)
        if slot is None:
            return
        async with slot.lock:
            await self._disconnect_locked(slot)
            self._slots.pop(server_id, None)

    def list_servers(self) -> list[McpDiagnostics]:
        return [slot.snapshot() for slot in self._slots.values()]

    def get_diagnostics(self, server_id: str) -> McpDiagnostics:
        return self._require_slot(server_id).snapshot()

    def list_tools(self, server_id: str) -> tuple[McpToolDefinition, ...]:
        slot = self._require_slot(server_id)
        if slot.state != "connected":
            return ()
        return slot.tools

    async def connect(self, server_id: str) -> McpDiagnostics:
        """Idempotently connect a registered MCP server and discover its tools."""
        if self._shutting_down:
            raise McpError("MCP_SERVER_UNAVAILABLE", "MCP Manager 正在关闭。")
        slot = self._require_slot(server_id)
        async with slot.lock:
            if not slot.config.enabled:
                raise McpError("MCP_SERVER_UNAVAILABLE", "该 MCP 服务已停用。")
            if slot.state == "connected" and slot.actor is not None and not slot.actor.task.done():
                return slot.snapshot()
            if slot.state == "connecting" and slot.actor is not None and not slot.actor.task.done():
                actor = slot.actor
            else:
                if slot.actor is not None:
                    await self._disconnect_locked(slot)
                if self._active_server_count() >= self.limits.max_active_servers:
                    raise McpError("MCP_SERVER_BUSY", f"同时活跃的 MCP 服务已达上限（{self.limits.max_active_servers}）。")
                slot.generation += 1
                slot.state = "connecting"
                slot.last_error_code = None
                slot.last_error_message = None
                slot.stderr_tail = ()
                actor = _ServerActor(
                    slot=slot,
                    config=slot.config,
                    generation=slot.generation,
                    limits=self.limits,
                    connect_semaphore=self._connect_semaphore,
                    on_catalog_updated=self._on_catalog_updated,
                )
                slot.actor = actor

        # Wait outside the per-server lock so disconnect() can cancel an in-flight connect().
        return await asyncio.shield(actor.ready)

    async def disconnect(self, server_id: str) -> McpDiagnostics:
        """Idempotently disconnect an MCP server and reclaim its resources."""
        slot = self._require_slot(server_id)
        async with slot.lock:
            await self._disconnect_locked(slot)
            return slot.snapshot()

    async def reconnect(self, server_id: str) -> McpDiagnostics:
        """Force a clean disconnect followed by a new generation connect."""
        await self.disconnect(server_id)
        return await self.connect(server_id)

    async def _disconnect_locked(self, slot: _ServerSlot) -> None:
        actor = slot.actor
        if actor is None:
            slot.state = "disconnected"
            slot.tools = ()
            slot.tools_truncated = False
            slot.child_pid = None
            slot.connected_at = None
            return
        slot.state = "disconnecting"
        actor.request_stop()
        try:
            await asyncio.wait_for(actor.task, timeout=self.limits.shutdown_total_timeout_seconds)
        except TimeoutError:
            actor.task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(actor.task, timeout=1.0)
        except (asyncio.CancelledError, Exception):
            pass
        finally:
            if slot.actor is actor:
                slot.actor = None
            slot.state = "disconnected"
            slot.tools = ()
            slot.tools_truncated = False
            slot.child_pid = None
            slot.connected_at = None

    async def refresh_tools(self, server_id: str) -> tuple[McpToolDefinition, ...]:
        """Explicitly refresh a connected server's tool catalog."""
        if self._shutting_down:
            raise McpError("MCP_SERVER_UNAVAILABLE", "MCP Manager 正在关闭。")
        slot = self._require_slot(server_id)
        actor = slot.actor
        if slot.state != "connected" or actor is None or actor.task.done():
            raise McpError("MCP_SERVER_UNAVAILABLE")
        future = actor.enqueue_refresh()
        return await future

    async def call_tool(
        self,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> McpToolCallResult:
        """Invoke a tool on a connected server (serial per server, zero auto-retry)."""
        if self._shutting_down:
            raise McpError("MCP_SERVER_UNAVAILABLE", "MCP Manager 正在关闭。")
        slot = self._require_slot(server_id)
        actor = slot.actor
        if slot.state != "connected" or actor is None or actor.task.done():
            raise McpError("MCP_SERVER_UNAVAILABLE")

        req = actor.enqueue_call(
            tool_name=tool_name,
            arguments=dict(arguments) if arguments is not None else {},
            timeout_seconds=timeout_seconds,
        )
        try:
            return await asyncio.shield(req.future)
        except asyncio.CancelledError:
            req.cancelled = True
            req.cancel_event.set()
            # Let the actor observe cancellation and emit courtesy notifications/cancelled.
            with suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(req.future, timeout=1.0)
            raise

    async def shutdown(self) -> None:
        """Concurrently stop all active server actors within the configured shutdown budget."""
        self._shutting_down = True
        actors: list[tuple[_ServerSlot, _ServerActor]] = []
        for slot in list(self._slots.values()):
            if slot.actor is not None:
                slot.state = "disconnecting"
                slot.actor.request_stop()
                actors.append((slot, slot.actor))
            else:
                slot.state = "disconnected"
                slot.tools = ()

        if not actors:
            return

        tasks = [actor.task for _, actor in actors]
        done, pending = await asyncio.wait(tasks, timeout=self.limits.shutdown_total_timeout_seconds)
        for task in done:
            with suppress(asyncio.CancelledError, Exception):
                task.result()

        if pending:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        for slot, actor in actors:
            if slot.actor is actor:
                slot.actor = None
            slot.state = "disconnected"
            slot.tools = ()
            slot.tools_truncated = False
            slot.child_pid = None
            slot.connected_at = None
