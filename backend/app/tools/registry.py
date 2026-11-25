import asyncio
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from threading import Event
from time import monotonic
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID, uuid4

from pydantic import ValidationError

from app.mcp.adapter import (
    McpAuthorizedToolSnapshot,
    adapt_mcp_call_result,
    validate_mcp_tool_arguments,
)
from app.mcp.models import McpError
from app.models import ModelError
from app.services.grants import GrantRegistry
from app.storage.business import BusinessStore, Run, SessionEgressRisk
from app.tools.files import FileAccess, FileArgs, ListFilesArgs, ReadTextArgs, list_files, read_text
from app.tools.protocol import ERROR_MESSAGES, ToolCall, ToolError, ToolEvent, ToolResult, encode_json

if TYPE_CHECKING:
    from app.services.mcp import McpService

MAX_TOOL_CALLS = 8
MAX_EXPOSED_TOOLS = 32
MAX_EXPOSED_SCHEMA_BYTES = 128 * 1024
MAX_TOOL_CONTEXT_CHARS = 64_000
MAX_TOOL_CONTEXT_BYTES = 96_000

_MCP_ERROR_TO_TOOL_CODE: dict[str, str] = {
    "MCP_SERVER_UNAVAILABLE": "MCP_SERVER_UNAVAILABLE",
    "MCP_PERMISSION_DENIED": "MCP_PERMISSION_DENIED",
    "MCP_UNTRUSTED_SERVER": "MCP_PERMISSION_DENIED",
    "MCP_CREDENTIALS_MISSING": "MCP_PERMISSION_DENIED",
    "MCP_TOOL_TIMEOUT": "MCP_TOOL_TIMEOUT",
    "MCP_TOOL_RETURNED_ERROR": "MCP_TOOL_RETURNED_ERROR",
    "MCP_TOOL_CANCELLED": "RUN_CANCELLED",
    "MCP_RESPONSE_TOO_LARGE": "MCP_RESPONSE_TOO_LARGE",
    "MCP_MALFORMED_RESPONSE": "MCP_MALFORMED_RESPONSE",
    "MCP_PROTOCOL_ERROR": "MCP_PROTOCOL_ERROR",
    "MCP_SERVER_BUSY": "MCP_SERVER_BUSY",
}


@dataclass(frozen=True)
class ToolSpec:
    description: str
    args: type[FileArgs]
    operation: Literal["read"]
    execute: Callable[[ToolCall, FileArgs, FileAccess], ToolResult]


TOOLS = {
    "list_files": ToolSpec(
        "列出授权目录中当前相对目录的 TXT、Markdown 文件和普通子目录。"
        "relative_dir 留空时列出根目录。使用 next_offset 分页。",
        ListFilesArgs, "read", list_files,
    ),
    "read_text": ToolSpec(
        "读取授权 TXT 或 Markdown。目录授权必须传入相对文件路径。"
        "单文件授权须将 relative_path 留空。使用从 0 开始的 offset_line 和 next_offset_line 分段读取。",
        ReadTextArgs, "read", read_text,
    ),
}


def _native_model_schemas() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "type": "function",
            "function": {
                "name": name,
                "description": spec.description,
                "parameters": spec.args.model_json_schema(),
            },
        }
        for name, spec in TOOLS.items()
    )


@dataclass(frozen=True)
class RunToolSnapshot:
    """Immutable per-run tool snapshot frozen prior to Run admission."""

    native_grant_ids: frozenset[UUID]
    native_authorizations: tuple[dict[str, Any], ...]
    mcp_tools: dict[str, McpAuthorizedToolSnapshot]
    mcp_enabled: bool
    allow_external_data_egress: bool
    model_schemas: tuple[dict[str, Any], ...]
    total_schema_bytes: int

    @property
    def has_native_tools(self) -> bool:
        return bool(self.native_authorizations)

    @property
    def has_mcp_tools(self) -> bool:
        return bool(self.mcp_tools)

    @property
    def has_tools(self) -> bool:
        return self.has_native_tools or self.has_mcp_tools


class RunTools:
    """One run's immutable native + MCP tool snapshot, live authorization checks, shared budget, and audit."""

    def __init__(
        self,
        grants: GrantRegistry,
        store: BusinessStore,
        run: Run,
        emit: Callable[[ToolEvent, dict[str, Any]], None],
        *,
        mcp_service: "McpService | None" = None,
        snapshot: RunToolSnapshot | None = None,
        max_tool_calls: int = MAX_TOOL_CALLS,
        max_context_chars: int = MAX_TOOL_CONTEXT_CHARS,
        max_context_bytes: int = MAX_TOOL_CONTEXT_BYTES,
        mcp_tool_timeout_seconds: float | None = None,
    ):
        self._grants = grants
        self._store = store
        self._run = run
        self._session = UUID(run.session_id)
        self._emit = emit
        self._mcp_service = mcp_service
        self._mcp_tool_timeout_seconds = mcp_tool_timeout_seconds
        self._snapshot = snapshot or self.build_snapshot(grants, self._session)
        self._grant_ids = self._snapshot.native_grant_ids
        self.authorizations = list(self._snapshot.native_authorizations)
        self.mcp_tools: dict[str, McpAuthorizedToolSnapshot] = dict(self._snapshot.mcp_tools)
        self._remaining_chars = max_context_chars
        self._remaining_bytes = max_context_bytes
        self.remaining_calls = max_tool_calls
        self.failure: ToolResult | None = None

    @classmethod
    def build_snapshot(
        cls,
        grants: GrantRegistry,
        session_id: UUID,
        *,
        mcp_service: "McpService | None" = None,
        mcp_enabled: bool = False,
        allow_external_data_egress: bool = False,
        knowledge_enabled: bool = False,
        session_egress_risk: SessionEgressRisk | None = None,
        max_exposed_tools: int = MAX_EXPOSED_TOOLS,
        max_exposed_schema_bytes: int = MAX_EXPOSED_SCHEMA_BYTES,
    ) -> RunToolSnapshot:
        """Build and validate a frozen tool snapshot BEFORE creating a running Run."""
        views = grants.list(session_id)
        grant_ids = frozenset(view.grant_id for view in views)
        native_authorizations = tuple(
            {
                "grant_id": str(view.grant_id),
                "scope": view.scope,
                "display_name": view.display_name,
                "operations": view.operations,
            }
            for view in views
        )

        mcp_map: dict[str, McpAuthorizedToolSnapshot] = {}
        if mcp_enabled and mcp_service is not None:
            for item in mcp_service.snapshot_authorized_tools():
                mcp_map[item.alias] = item

        # Security boundary (§6.2 + P6.5 §4.3): By default, forbid mixing protected local context
        # (current-turn RAG, active local file grants, OR prior-turn file/RAG/unverifiable history)
        # with external MCP tools unless the user explicitly authorizes external data egress.
        history_risk = session_egress_risk or SessionEgressRisk()
        has_current_risk = bool(knowledge_enabled or native_authorizations)
        has_history_risk = history_risk.requires_egress_authorization
        if mcp_map and (has_current_risk or has_history_risk) and not allow_external_data_egress:
            if history_risk.has_unverifiable_history:
                msg = (
                    "当前会话包含旧版本或来源不可验证的历史对话上下文，默认禁止同轮调用外部 MCP；"
                    "请开启新对话、关闭本轮 MCP，或显式授权本轮外部数据外发。"
                )
            elif has_history_risk and not has_current_risk:
                if history_risk.has_historical_file_context and history_risk.has_historical_rag_context:
                    msg = (
                        "当前会话的历史对话包含已读取的本地文件与教材检索内容，默认禁止后续同会话调用外部 MCP；"
                        "请开启新对话、关闭本轮 MCP，或显式授权本轮外部数据外发。"
                    )
                elif history_risk.has_historical_rag_context:
                    msg = (
                        "当前会话的历史对话包含教材知识库检索内容，默认禁止后续同会话调用外部 MCP；"
                        "请开启新对话、关闭本轮 MCP，或显式授权本轮外部数据外发。"
                    )
                else:
                    msg = (
                        "当前会话的历史对话包含已读取的本地文件内容，默认禁止后续同会话调用外部 MCP；"
                        "请开启新对话、关闭本轮 MCP，或显式授权本轮外部数据外发。"
                    )
            elif knowledge_enabled and native_authorizations:
                msg = (
                    "本轮同时包含教材知识库、本地文件授权与外部 MCP 工具，默认禁止向外部 MCP 传输受保护的本地上下文；"
                    "请关闭外部 MCP 或显式授权允许外部数据外发。"
                )
            elif knowledge_enabled:
                msg = (
                    "本轮同时开启了教材知识库与外部 MCP 工具，默认禁止向外部 MCP 传输受保护的教材上下文；"
                    "请关闭其中一项或显式授权允许外部数据外发。"
                )
            else:
                msg = (
                    "当前会话存在本地文件读取授权，默认禁止同轮启用外部 MCP 工具以防受保护文件内容外发；"
                    "请撤销文件授权、关闭本轮 MCP，或显式授权允许外部数据外发。"
                )
            raise ModelError("MCP_EGRESS_AUTHORIZATION_REQUIRED", msg)

        schemas: list[dict[str, Any]] = []
        if native_authorizations:
            schemas.extend(_native_model_schemas())
        if mcp_map:
            schemas.extend(item.model_schema for item in mcp_map.values())

        total_schema_bytes = len(encode_json(schemas).encode("utf-8")) if schemas else 0
        if len(schemas) > max_exposed_tools:
            raise ModelError(
                "MCP_TOOL_LIMIT_EXCEEDED",
                f"本轮可用工具总数（{len(schemas)}）超过安全上限（{max_exposed_tools}），请在 MCP 服务页精简已授权工具。",
            )
        if total_schema_bytes > max_exposed_schema_bytes:
            raise ModelError(
                "MCP_TOOL_LIMIT_EXCEEDED",
                f"本轮工具参数 Schema 总大小超过 {max_exposed_schema_bytes // 1024} KiB 上限，请精简已授权工具。",
            )

        return RunToolSnapshot(
            native_grant_ids=grant_ids,
            native_authorizations=native_authorizations,
            mcp_tools=mcp_map,
            mcp_enabled=mcp_enabled,
            allow_external_data_egress=allow_external_data_egress,
            model_schemas=tuple(schemas),
            total_schema_bytes=total_schema_bytes,
        )

    @property
    def has_native_tools(self) -> bool:
        return self._snapshot.has_native_tools

    @property
    def has_mcp_tools(self) -> bool:
        return self._snapshot.has_mcp_tools

    @property
    def has_tools(self) -> bool:
        return self._snapshot.has_tools

    @property
    def allow_external_data_egress(self) -> bool:
        return self._snapshot.allow_external_data_egress

    def model_schemas(self) -> list[dict[str, Any]]:
        """Return only the tools enabled in this Run's frozen snapshot."""
        return list(self._snapshot.model_schemas)

    def mcp_catalog_metadata(self) -> list[dict[str, str]]:
        """Redacted, untrusted-data catalog metadata included in the tool-graph system prompt."""
        return [
            {
                "tool_alias": snap.alias,
                "server_name": snap.server_name_snapshot,
                "original_tool_name": snap.original_tool_name,
            }
            for snap in self.mcp_tools.values()
        ]

    def prepare(self, name: str, arguments: object) -> ToolCall:
        if name in TOOLS and self.has_native_tools:
            return ToolCall(
                str(uuid4()),
                name,
                arguments,
                self._run.session_id,
                self._run.run_id,
                source="native",
            )
        mcp_snap = self.mcp_tools.get(name)
        if mcp_snap is not None:
            return ToolCall(
                str(uuid4()),
                mcp_snap.alias,
                arguments,
                self._run.session_id,
                self._run.run_id,
                source="mcp",
                server_id=mcp_snap.server_id,
                server_name=mcp_snap.server_name_snapshot,
                original_tool_name=mcp_snap.original_tool_name,
                definition_hash=mcp_snap.definition_hash,
            )
        return ToolCall(
            str(uuid4()),
            "unknown_tool",
            arguments,
            self._run.session_id,
            self._run.run_id,
            source="native",
        )

    def require_active_authorizations(self, completed_calls: Iterable[ToolCall]) -> None:
        """Re-verify every completed native grant and MCP tool authorization before next LLM inference / retry."""
        for call in completed_calls:
            if call.source == "mcp":
                snap = self.mcp_tools.get(call.tool_name)
                if snap is None or self._mcp_service is None:
                    raise ModelError("MCP_PERMISSION_DENIED", ERROR_MESSAGES["MCP_PERMISSION_DENIED"])
                try:
                    self._mcp_service.verify_tool_snapshot_authorized(snap)
                except McpError as exc:
                    mapped = _MCP_ERROR_TO_TOOL_CODE.get(exc.code, "MCP_PERMISSION_DENIED")
                    raise ModelError(mapped, ERROR_MESSAGES.get(mapped, exc.message)) from None
            else:
                spec = TOOLS.get(call.tool_name)
                if spec is None:
                    raise ModelError("FILE_PERMISSION_DENIED", ERROR_MESSAGES["FILE_PERMISSION_DENIED"])
                args = spec.args.model_validate(call.arguments)
                if self._grants.find_read(self._session, UUID(args.grant_id)) is None:
                    raise ModelError("FILE_PERMISSION_DENIED", ERROR_MESSAGES["FILE_PERMISSION_DENIED"])

    # Backward-compatible alias for existing callers/tests.
    require_active_grants = require_active_authorizations

    def abort_after_prior_failure(self, call: ToolCall) -> ToolResult:
        """Short-circuit a remaining call in a multi-tool batch after an earlier failure."""
        return ToolResult.failure(
            call.tool_call_id,
            "MCP_CALL_ABORTED",
            source=call.source,
            server_id=call.server_id,
            server_name=call.server_name,
            original_tool_name=call.original_tool_name,
        )

    async def execute(self, call: ToolCall) -> ToolResult:
        if self.remaining_calls <= 0:
            raise ModelError("RATE_LIMITED", ERROR_MESSAGES["RATE_LIMITED"])
        self.remaining_calls -= 1
        if call.source == "mcp":
            return await self._execute_mcp(call)
        return await self._execute_native_wrapper(call)

    async def _execute_native_wrapper(self, call: ToolCall) -> ToolResult:
        cancelled = Event()
        task = asyncio.create_task(self._execute_native(call, cancelled))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled.set()
            # Let the bounded worker close its descriptors and settle the audit before run cancellation.
            await asyncio.gather(task, return_exceptions=True)
            raise

    async def _execute_native(self, call: ToolCall, cancelled: Event) -> ToolResult:
        call_id = call.tool_call_id
        try:
            await asyncio.to_thread(self._store.start_tool, self._run, call_id, call.tool_name)
            self._emit("tool_started", {"tool_call_id": call_id, "tool_name": call.tool_name})
            result = await asyncio.to_thread(self._evaluate, call, cancelled)
            if cancelled.is_set():
                result = ToolResult.failure(call_id, "RUN_CANCELLED")
            status = "completed" if result.ok else "cancelled" if cancelled.is_set() else "failed"
            await asyncio.to_thread(
                self._store.finish_tool, self._run, call_id, status, result.display_summary, result.error_code,
            )
        except Exception:
            raise ModelError("INTERNAL_ERROR", ERROR_MESSAGES["INTERNAL_ERROR"]) from None

        if result.ok:
            self._emit("tool_completed", {"tool_call_id": call_id, "summary": result.display_summary})
        else:
            self.failure = self.failure or result
            # Preserve exact native tool_failed key set for backward compatibility (test_file_tools.py:142).
            self._emit("tool_failed", {
                "tool_call_id": call_id, "code": result.error_code, "message": result.display_summary,
            })
        return result

    def _evaluate(self, call: ToolCall, cancelled: Event) -> ToolResult:
        try:
            if cancelled.is_set():
                raise ToolError("RUN_CANCELLED")
            spec = TOOLS.get(call.tool_name)
            if spec is None or not self.has_native_tools:
                raise ToolError("TOOL_NOT_FOUND")
            args = spec.args.model_validate(call.arguments)
            if spec.operation != "read" or (call.session_id, call.run_id) != (self._run.session_id, self._run.run_id):
                raise ToolError("FILE_PERMISSION_DENIED")
            grant_id = UUID(args.grant_id)
            grant = self._grants.find_read(self._session, grant_id)
            if grant is None or grant_id not in self._grant_ids:
                raise ToolError("FILE_PERMISSION_DENIED")

            def check_active() -> None:
                if cancelled.is_set():
                    raise ToolError("RUN_CANCELLED")
                if self._grants.find_read(self._session, grant_id) is not grant:
                    raise ToolError("FILE_PERMISSION_DENIED")

            result = spec.execute(call, args, FileAccess(grant, check_active))
            encoded = result.for_model()
            byte_count = len(encoded.encode("utf-8"))
            if len(encoded) > self._remaining_chars or byte_count > self._remaining_bytes:
                raise ToolError("RATE_LIMITED")
            # Linearization point for concurrent revoke: results accepted before this check
            # may already reach the provider. A revocation that wins this check discards all data.
            check_active()
            self._remaining_chars -= len(encoded)
            self._remaining_bytes -= byte_count
            return result
        except ValidationError:
            code = "VALIDATION_ERROR"
        except ToolError as error:
            code = error.code
        except OSError:
            code = "FILE_PERMISSION_DENIED"
        except Exception:
            code = "INTERNAL_ERROR"
        return ToolResult.failure(call.tool_call_id, code)

    def _mcp_meta(self, call: ToolCall, *, duration_ms: int | None = None) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "source": "mcp",
            "server_id": call.server_id,
            "server_name": call.server_name,
            "original_tool_name": call.original_tool_name,
        }
        if duration_ms is not None:
            meta["duration_ms"] = duration_ms
        return meta

    async def _settle_mcp_audit(
        self,
        call: ToolCall,
        status: Literal["completed", "failed", "cancelled"],
        summary: str,
        error_code: str | None,
        duration_ms: int,
    ) -> None:
        await asyncio.to_thread(
            self._store.finish_tool,
            self._run,
            call.tool_call_id,
            status,
            summary,
            error_code,
            duration_ms=duration_ms,
        )

    async def _execute_mcp(self, call: ToolCall) -> ToolResult:
        call_id = call.tool_call_id
        started = monotonic()
        audit_started = False

        try:
            await asyncio.to_thread(
                self._store.start_tool,
                self._run,
                call_id,
                call.tool_name,
                source="mcp",
                server_id=call.server_id,
                server_name_snapshot=call.server_name,
                original_tool_name=call.original_tool_name,
                definition_hash=call.definition_hash,
            )
            audit_started = True
            self._emit(
                "tool_started",
                {
                    "tool_call_id": call_id,
                    "tool_name": call.tool_name,
                    **self._mcp_meta(call),
                },
            )
        except Exception:
            raise ModelError("INTERNAL_ERROR", ERROR_MESSAGES["INTERNAL_ERROR"]) from None

        try:
            result = await self._evaluate_mcp(call, started)
        except asyncio.CancelledError:
            duration_ms = max(0, round((monotonic() - started) * 1000))
            cancelled_summary = "MCP 工具调用已取消。"
            if audit_started:
                # Guarantee audit row reaches 'cancelled' in SQLite BEFORE CancelledError unwinds to RunService._settle().
                settle_task = asyncio.create_task(
                    self._settle_mcp_audit(call, "cancelled", cancelled_summary, "RUN_CANCELLED", duration_ms)
                )
                try:
                    await asyncio.shield(settle_task)
                except asyncio.CancelledError:
                    await asyncio.gather(settle_task, return_exceptions=True)
                except Exception:
                    pass
                self._emit(
                    "tool_failed",
                    {
                        "tool_call_id": call_id,
                        "code": "RUN_CANCELLED",
                        "message": cancelled_summary,
                        **self._mcp_meta(call, duration_ms=duration_ms),
                    },
                )
            raise

        duration_ms = result.duration_ms if result.duration_ms is not None else max(0, round((monotonic() - started) * 1000))
        status: Literal["completed", "failed"] = "completed" if result.ok else "failed"
        try:
            await self._settle_mcp_audit(call, status, result.display_summary, result.error_code, duration_ms)
        except Exception:
            raise ModelError("INTERNAL_ERROR", ERROR_MESSAGES["INTERNAL_ERROR"]) from None

        if result.ok:
            self._emit(
                "tool_completed",
                {
                    "tool_call_id": call_id,
                    "summary": result.display_summary,
                    **self._mcp_meta(call, duration_ms=duration_ms),
                },
            )
        else:
            self.failure = self.failure or result
            self._emit(
                "tool_failed",
                {
                    "tool_call_id": call_id,
                    "code": result.error_code,
                    "message": result.display_summary,
                    **self._mcp_meta(call, duration_ms=duration_ms),
                },
            )
        return result

    async def _evaluate_mcp(self, call: ToolCall, started: float) -> ToolResult:
        def _fail(code: str, summary: str | None = None) -> ToolResult:
            elapsed = max(0, round((monotonic() - started) * 1000))
            return ToolResult.failure(
                call.tool_call_id,
                code,
                display_summary=summary,
                source="mcp",
                server_id=call.server_id,
                server_name=call.server_name,
                original_tool_name=call.original_tool_name,
                duration_ms=elapsed,
            )

        if (call.session_id, call.run_id) != (self._run.session_id, self._run.run_id):
            return _fail("MCP_PERMISSION_DENIED")

        snap = self.mcp_tools.get(call.tool_name)
        if snap is None or self._mcp_service is None:
            return _fail("TOOL_NOT_FOUND")

        try:
            # Strict pre-execution argument validation against original unmodified input_schema (no remote $ref).
            validated_args = validate_mcp_tool_arguments(snap.raw_input_schema, call.arguments)

            # Execute exclusively via McpService (pre-enqueue, pre-dispatch, and post-result authorization checks).
            raw_result = await self._mcp_service.execute_authorized_tool(
                snap,
                validated_args,
                timeout_seconds=self._mcp_tool_timeout_seconds,
            )

            adapted = adapt_mcp_call_result(
                call_id=call.tool_call_id,
                server_id=snap.server_id,
                server_name=snap.server_name_snapshot,
                original_tool_name=snap.original_tool_name,
                raw_result=raw_result,
            )
            if not adapted.ok:
                return adapted

            encoded = adapted.for_model()
            byte_count = len(encoded.encode("utf-8"))
            if len(encoded) > self._remaining_chars or byte_count > self._remaining_bytes:
                return _fail("RATE_LIMITED")

            # Final linearization point before deducting shared budget and publishing to model.
            self._mcp_service.verify_tool_snapshot_authorized(snap)
            self._remaining_chars -= len(encoded)
            self._remaining_bytes -= byte_count
            return adapted
        except asyncio.CancelledError:
            raise
        except ToolError as exc:
            return _fail(exc.code)
        except McpError as exc:
            mapped = _MCP_ERROR_TO_TOOL_CODE.get(exc.code, "MCP_PROTOCOL_ERROR")
            return _fail(mapped)
        except Exception:
            return _fail("INTERNAL_ERROR")
