import asyncio
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from threading import Event
from typing import Literal
from uuid import UUID, uuid4

from pydantic import ValidationError

from app.models import ModelError
from app.services.grants import GrantRegistry
from app.storage.business import BusinessStore, Run
from app.tools.files import FileAccess, FileArgs, ListFilesArgs, ReadTextArgs, list_files, read_text
from app.tools.protocol import ERROR_MESSAGES, ToolCall, ToolError, ToolEvent, ToolResult

MAX_TOOL_CALLS = 8
MAX_TOOL_CONTEXT_CHARS = 64_000
MAX_TOOL_CONTEXT_BYTES = 96_000


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


class RunTools:
    """One run's trusted grant snapshot, live checks, execution budget and audit."""

    def __init__(self, grants: GrantRegistry, store: BusinessStore, run: Run,
                 emit: Callable[[ToolEvent, dict[str, str]], None]):
        self._grants = grants
        self._store = store
        self._run = run
        self._session = UUID(run.session_id)
        self._emit = emit
        views = grants.list(self._session)
        self._grant_ids = frozenset(view.grant_id for view in views)
        self.authorizations = [
            {"grant_id": str(view.grant_id), "scope": view.scope, "display_name": view.display_name,
             "operations": view.operations}
            for view in views
        ]
        self._remaining_chars = MAX_TOOL_CONTEXT_CHARS
        self._remaining_bytes = MAX_TOOL_CONTEXT_BYTES
        self.remaining_calls = MAX_TOOL_CALLS
        self.failure: ToolResult | None = None

    @staticmethod
    def model_schemas() -> list[dict]:
        return [
            {"type": "function", "function": {"name": name, "description": spec.description,
                                              "parameters": spec.args.model_json_schema()}}
            for name, spec in TOOLS.items()
        ]

    def prepare(self, name: str, arguments: object) -> ToolCall:
        return ToolCall(str(uuid4()), name if name in TOOLS else "unknown_tool",
                        arguments, self._run.session_id, self._run.run_id)

    def require_active_grants(self, completed_calls: Iterable[ToolCall]) -> None:
        for call in completed_calls:
            args = TOOLS[call.tool_name].args.model_validate(call.arguments)
            if self._grants.find_read(self._session, UUID(args.grant_id)) is None:
                raise ModelError("FILE_PERMISSION_DENIED", ERROR_MESSAGES["FILE_PERMISSION_DENIED"])

    async def execute(self, call: ToolCall) -> ToolResult:
        if self.remaining_calls <= 0:
            raise ModelError("RATE_LIMITED", ERROR_MESSAGES["RATE_LIMITED"])
        self.remaining_calls -= 1
        cancelled = Event()
        task = asyncio.create_task(self._execute(call, cancelled))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled.set()
            # Let the bounded worker close its descriptors and settle the audit before run cancellation.
            await asyncio.gather(task, return_exceptions=True)
            raise

    async def _execute(self, call: ToolCall, cancelled: Event) -> ToolResult:
        call_id = call.tool_call_id
        try:
            await asyncio.to_thread(self._store.start_tool, self._run, call_id, call.tool_name)
            self._emit("tool_started", {"tool_call_id": call_id, "tool_name": call.tool_name})
            result = await asyncio.to_thread(self._evaluate, call, cancelled)
            if cancelled.is_set():
                result = ToolResult.failure(call_id, "RUN_CANCELLED")
            status = "completed" if result.ok else "cancelled" if cancelled.is_set() else "failed"
            await asyncio.to_thread(self._store.finish_tool, self._run, call_id, status,
                                    result.display_summary, result.error_code)
        except Exception:
            raise ModelError("INTERNAL_ERROR", ERROR_MESSAGES["INTERNAL_ERROR"]) from None

        if result.ok:
            self._emit("tool_completed", {"tool_call_id": call_id, "summary": result.display_summary})
        else:
            self.failure = self.failure or result
            self._emit("tool_failed", {"tool_call_id": call_id, "code": result.error_code,
                                       "message": result.display_summary})
        return result

    def _evaluate(self, call: ToolCall, cancelled: Event) -> ToolResult:
        try:
            if cancelled.is_set():
                raise ToolError("RUN_CANCELLED")
            spec = TOOLS.get(call.tool_name)
            if spec is None:
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
