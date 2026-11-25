import asyncio
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from threading import Event
from time import monotonic
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ValidationError

from app.agent.budgets import (
    LEGACY_MAX_TOOL_CALLS,
    LEGACY_MAX_TOOL_CONTEXT_BYTES,
    LEGACY_MAX_TOOL_CONTEXT_CHARS,
    MAX_CONSECUTIVE_REPEAT_FAILURES,
    SKILLS_MAX_CATEGORY_TOOL_CALLS,
    SKILLS_MAX_LEGACY_TOOL_CALLS,
    SKILLS_MAX_TOOL_CONTEXT_BYTES,
    SKILLS_MAX_TOOL_CONTEXT_CHARS,
    SKILLS_MAX_TOTAL_TOOL_CALLS,
)
from app.mcp.adapter import (
    McpAuthorizedToolSnapshot,
    adapt_mcp_call_result,
    validate_mcp_tool_arguments,
)
from app.mcp.models import McpError
from app.models import ModelError
from app.services.grants import GrantRegistry
from app.skills.models import SkillCatalogEntry
from app.storage.artifacts import WorkspaceArtifactStore
from app.storage.business import BusinessStore, Run, SessionEgressRisk
from app.tools.files import FileAccess, FileArgs, ListFilesArgs, ReadTextArgs, list_files, read_text
from app.tools.protocol import ERROR_MESSAGES, ToolCall, ToolCategory, ToolError, ToolEvent, ToolResult, encode_json
from app.tools.runtime_tools import ExecuteCommandArgs, RuntimeToolContext, runtime_tool_schemas
from app.tools.skill_tools import (
    ListSkillsArgs,
    LoadSkillArgs,
    ReadSkillFileArgs,
    SkillToolContext,
    skill_tool_schemas,
)
from app.tools.workspace_tools import (
    CopyWorkspaceFileArgs,
    ListWorkspaceArgs,
    PublishArtifactArgs,
    ReadWorkspaceFileArgs,
    WorkspaceToolContext,
    WriteWorkspaceFileArgs,
    workspace_tool_schemas,
)

if TYPE_CHECKING:
    from app.runtime.executor import RuntimeExecutor
    from app.services.mcp import McpService
    from app.skills.manager import SkillManager
    from app.workspace.artifacts import ArtifactService
    from app.workspace.service import WorkspaceService

MAX_TOOL_CALLS = LEGACY_MAX_TOOL_CALLS
MAX_SKILLS_TOTAL_TOOL_CALLS = SKILLS_MAX_TOTAL_TOOL_CALLS
MAX_SKILLS_CATEGORY_CALLS = SKILLS_MAX_CATEGORY_TOOL_CALLS
MAX_SKILLS_LEGACY_CALLS = SKILLS_MAX_LEGACY_TOOL_CALLS
MAX_EXPOSED_TOOLS = 32
MAX_EXPOSED_TOOLS_WITH_SKILLS = 44
MAX_EXPOSED_SCHEMA_BYTES = 128 * 1024
MAX_TOOL_CONTEXT_CHARS = LEGACY_MAX_TOOL_CONTEXT_CHARS
MAX_TOOL_CONTEXT_BYTES = LEGACY_MAX_TOOL_CONTEXT_BYTES
MAX_SKILLS_TOOL_CONTEXT_CHARS = SKILLS_MAX_TOOL_CONTEXT_CHARS
MAX_SKILLS_TOOL_CONTEXT_BYTES = SKILLS_MAX_TOOL_CONTEXT_BYTES

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

_SKILLS_TOOL_CATEGORIES: dict[str, tuple[ToolCategory, type[BaseModel]]] = {
    "list_skills": ("skill", ListSkillsArgs),
    "load_skill": ("skill", LoadSkillArgs),
    "read_skill_file": ("skill", ReadSkillFileArgs),
    "list_workspace": ("workspace", ListWorkspaceArgs),
    "read_file": ("workspace", ReadWorkspaceFileArgs),
    "write_file": ("workspace", WriteWorkspaceFileArgs),
    "copy_file": ("workspace", CopyWorkspaceFileArgs),
    "publish_artifact": ("workspace", PublishArtifactArgs),
    "execute_command": ("runtime", ExecuteCommandArgs),
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
    skills_enabled: bool = False
    skill_catalog: tuple[SkillCatalogEntry, ...] = ()
    attachment_ids: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()

    @property
    def has_native_tools(self) -> bool:
        return bool(self.native_authorizations)

    @property
    def has_mcp_tools(self) -> bool:
        return bool(self.mcp_tools)

    @property
    def has_skills_tools(self) -> bool:
        return self.skills_enabled

    @property
    def has_tools(self) -> bool:
        return self.has_native_tools or self.has_mcp_tools or self.has_skills_tools


class RunTools:
    """One run's immutable native + MCP + Skills/Workspace/Runtime tool snapshot, budget, and audit."""

    def __init__(
        self,
        grants: GrantRegistry,
        store: BusinessStore,
        run: Run,
        emit: Callable[[ToolEvent, dict[str, Any]], None],
        *,
        mcp_service: "McpService | None" = None,
        skill_manager: "SkillManager | None" = None,
        workspace_service: "WorkspaceService | None" = None,
        artifact_service: "ArtifactService | None" = None,
        runtime_executor: "RuntimeExecutor | None" = None,
        emit_run_event: Callable[[str, dict[str, Any]], None] | None = None,
        snapshot: RunToolSnapshot | None = None,
        max_tool_calls: int = MAX_TOOL_CALLS,
        max_context_chars: int | None = None,
        max_context_bytes: int | None = None,
        mcp_tool_timeout_seconds: float | None = None,
    ):
        self._grants = grants
        self._store = store
        self._run = run
        self._session = UUID(run.session_id)
        self._emit = emit
        self._emit_run_event = emit_run_event
        self._mcp_service = mcp_service
        self._mcp_tool_timeout_seconds = mcp_tool_timeout_seconds
        self._snapshot = snapshot or self.build_snapshot(grants, self._session)
        self._grant_ids = self._snapshot.native_grant_ids
        self.authorizations = list(self._snapshot.native_authorizations)
        self.mcp_tools: dict[str, McpAuthorizedToolSnapshot] = dict(self._snapshot.mcp_tools)

        # Skills / Workspace / Runtime contexts (created only when skills_enabled)
        self._skill_ctx: SkillToolContext | None = None
        self._workspace_ctx: WorkspaceToolContext | None = None
        self._runtime_ctx: RuntimeToolContext | None = None
        if self._snapshot.has_skills_tools:
            artifact_store = WorkspaceArtifactStore(store)
            if skill_manager is not None:
                self._skill_ctx = SkillToolContext(
                    manager=skill_manager,
                    artifact_store=artifact_store,
                    session_id=run.session_id,
                    run_id=run.run_id,
                    catalog=self._snapshot.skill_catalog,
                    emit_event=emit_run_event,
                )
            if workspace_service is not None and artifact_service is not None:
                self._workspace_ctx = WorkspaceToolContext(
                    workspace=workspace_service,
                    artifacts=artifact_service,
                    session_id=run.session_id,
                    run_id=run.run_id,
                    attachment_ids=self._snapshot.attachment_ids,
                    artifact_ids=self._snapshot.artifact_ids,
                    emit_event=emit_run_event,
                )
            if runtime_executor is not None and workspace_service is not None:
                self._runtime_ctx = RuntimeToolContext(
                    executor=runtime_executor,
                    workspace=workspace_service,
                    session_id=run.session_id,
                    run_id=run.run_id,
                )

        # Dual-bucket tool call budget:
        # - Non-Skills Run: preserves exact legacy 8 calls & 64k/96k context budget.
        # - Skills Run: 100 total calls (<=80 skill/workspace/runtime, <=20 legacy file+MCP) & 400k/600k context budget.
        if self._snapshot.has_skills_tools:
            self.remaining_calls = MAX_SKILLS_TOTAL_TOOL_CALLS
            self._remaining_legacy_calls = MAX_SKILLS_LEGACY_CALLS
            self._remaining_skill_calls = MAX_SKILLS_CATEGORY_CALLS
            default_chars = MAX_SKILLS_TOOL_CONTEXT_CHARS
            default_bytes = MAX_SKILLS_TOOL_CONTEXT_BYTES
        else:
            self.remaining_calls = max_tool_calls
            self._remaining_legacy_calls = max_tool_calls
            self._remaining_skill_calls = 0
            default_chars = MAX_TOOL_CONTEXT_CHARS
            default_bytes = MAX_TOOL_CONTEXT_BYTES

        self._remaining_chars = max_context_chars if max_context_chars is not None else default_chars
        self._remaining_bytes = max_context_bytes if max_context_bytes is not None else default_bytes
        self.failure: ToolResult | None = None
        self._last_error_signature: str | None = None
        self._repeat_error_count: int = 0

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
        skill_manager: "SkillManager | None" = None,
        skills_enabled: bool = False,
        selected_skill_ids: Sequence[str] | None = None,
        attachment_ids: Sequence[str] = (),
        artifact_ids: Sequence[str] = (),
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

        skill_catalog: tuple[SkillCatalogEntry, ...] = ()
        if skills_enabled and skill_manager is not None:
            skill_catalog = skill_manager.build_catalog_snapshot(selected_skill_ids=selected_skill_ids)

        att_tuple = tuple(attachment_ids)
        art_tuple = tuple(artifact_ids)

        # Security boundary (§6.2 + P6.5 §4.3 + P9 §12): By default, forbid mixing protected local context
        # (current-turn RAG, active local file grants, workspace attachments/artifacts/skills, OR prior-turn
        # file/RAG/unverifiable history) with external MCP tools unless explicitly authorized.
        history_risk = session_egress_risk or SessionEgressRisk()
        has_workspace_risk = bool(skills_enabled or att_tuple or art_tuple)
        has_current_risk = bool(knowledge_enabled or native_authorizations or has_workspace_risk)
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
            elif has_workspace_risk:
                msg = (
                    "本轮已启用 Skills/工作区文件能力并同时开启外部 MCP 工具，默认禁止向外部 MCP 传输本地工作区上下文；"
                    "请关闭外部 MCP 或显式授权允许外部数据外发。"
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
        if skills_enabled:
            schemas.extend(skill_tool_schemas())
            schemas.extend(workspace_tool_schemas())
            schemas.extend(runtime_tool_schemas())

        effective_tool_limit = MAX_EXPOSED_TOOLS_WITH_SKILLS if skills_enabled else max_exposed_tools
        if len(mcp_map) > max_exposed_tools or len(schemas) > effective_tool_limit:
            raise ModelError(
                "MCP_TOOL_LIMIT_EXCEEDED",
                f"本轮可用工具总数（{len(schemas)}）超过安全上限（{effective_tool_limit}），请在 MCP 服务页精简已授权工具。",
            )
        total_schema_bytes = len(encode_json(schemas).encode("utf-8")) if schemas else 0
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
            skills_enabled=skills_enabled,
            skill_catalog=skill_catalog,
            attachment_ids=att_tuple,
            artifact_ids=art_tuple,
        )

    @property
    def has_native_tools(self) -> bool:
        return self._snapshot.has_native_tools

    @property
    def has_mcp_tools(self) -> bool:
        return self._snapshot.has_mcp_tools

    @property
    def has_skills_tools(self) -> bool:
        return self._snapshot.has_skills_tools

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

    def skill_catalog_metadata(self) -> list[dict[str, Any]]:
        """Lightweight Progressive Disclosure skill catalog (name + description + skill_id only)."""
        return [entry.to_prompt_dict() for entry in self._snapshot.skill_catalog]

    def workspace_context_metadata(self) -> dict[str, Any]:
        if self._workspace_ctx is None:
            return {"run_work_dirs": ["work/", "output/"], "attachments": [], "published_artifacts": []}
        return self._workspace_ctx.workspace_context_metadata()

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
        if name in _SKILLS_TOOL_CATEGORIES and self.has_skills_tools:
            category, _ = _SKILLS_TOOL_CATEGORIES[name]
            return ToolCall(
                str(uuid4()),
                name,
                arguments,
                self._run.session_id,
                self._run.run_id,
                source="native",
                tool_category=category,
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
            tool_category="skill" if self.has_skills_tools else None,
        )

    def require_active_authorizations(self, completed_calls: Iterable[ToolCall]) -> None:
        """Re-verify completed native file grants & MCP tool authorizations before next LLM inference / retry."""
        for call in completed_calls:
            if call.tool_category in ("skill", "workspace", "runtime"):
                continue
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
        """Short-circuit a remaining call in a multi-tool batch after an earlier fatal failure."""
        return ToolResult.failure(
            call.tool_call_id,
            "MCP_CALL_ABORTED",
            source=call.source,
            server_id=call.server_id,
            server_name=call.server_name,
            original_tool_name=call.original_tool_name,
            tool_category=call.tool_category,
        )

    async def execute(self, call: ToolCall) -> ToolResult:
        if self.remaining_calls <= 0:
            raise ModelError("RATE_LIMITED", ERROR_MESSAGES["RATE_LIMITED"])
        if call.tool_category in ("skill", "workspace", "runtime"):
            if self._remaining_skill_calls <= 0:
                raise ModelError("RATE_LIMITED", ERROR_MESSAGES["RATE_LIMITED"])
            self.remaining_calls -= 1
            self._remaining_skill_calls -= 1
            return await self._execute_skills_tool(call)

        if self._remaining_legacy_calls <= 0:
            raise ModelError("RATE_LIMITED", ERROR_MESSAGES["RATE_LIMITED"])
        self.remaining_calls -= 1
        self._remaining_legacy_calls -= 1
        if call.source == "mcp":
            return await self._execute_mcp(call)
        return await self._execute_native_wrapper(call)

    # ------------------------------------------------------------------
    # Skills / Workspace / Runtime unified execution + recoverable failures
    # ------------------------------------------------------------------

    def _skills_meta(
        self,
        call: ToolCall,
        *,
        duration_ms: int | None = None,
        recoverable: bool | None = None,
    ) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "source": "native",
            "tool_category": call.tool_category,
        }
        if duration_ms is not None:
            meta["duration_ms"] = duration_ms
        if recoverable is not None:
            meta["recoverable"] = recoverable
        return meta

    async def _settle_skills_audit(
        self,
        call: ToolCall,
        status: Literal["completed", "failed", "cancelled"],
        summary: str,
        error_code: str | None,
        duration_ms: int,
        *,
        recoverable: bool = False,
    ) -> None:
        await asyncio.to_thread(
            self._store.finish_tool,
            self._run,
            call.tool_call_id,
            status,
            summary,
            error_code,
            duration_ms=duration_ms,
            recoverable=recoverable,
        )

    async def _execute_skills_tool(self, call: ToolCall) -> ToolResult:
        call_id = call.tool_call_id
        started = monotonic()
        audit_started = False

        try:
            await asyncio.to_thread(
                self._store.start_tool,
                self._run,
                call_id,
                call.tool_name,
                source="native",
                tool_category=call.tool_category,
            )
            audit_started = True
            self._emit(
                "tool_started",
                {
                    "tool_call_id": call_id,
                    "tool_name": call.tool_name,
                    **self._skills_meta(call),
                },
            )
        except Exception:
            raise ModelError("INTERNAL_ERROR", ERROR_MESSAGES["INTERNAL_ERROR"]) from None

        try:
            result = await self._evaluate_skills_tool(call)
        except asyncio.CancelledError:
            duration_ms = max(0, round((monotonic() - started) * 1000))
            cancelled_summary = "工具执行已取消。"
            if audit_started:
                settle_task = asyncio.create_task(
                    self._settle_skills_audit(call, "cancelled", cancelled_summary, "RUN_CANCELLED", duration_ms)
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
                        **self._skills_meta(call, duration_ms=duration_ms, recoverable=False),
                    },
                )
            raise

        duration_ms = result.duration_ms if result.duration_ms is not None else max(0, round((monotonic() - started) * 1000))

        # Track consecutive identical recoverable failures (different errors do NOT accumulate together).
        if result.ok:
            self._last_error_signature = None
            self._repeat_error_count = 0
        elif result.recoverable:
            sig = result.error_signature or f"{call.tool_name}:{result.error_code}"
            if sig == self._last_error_signature:
                self._repeat_error_count += 1
            else:
                self._last_error_signature = sig
                self._repeat_error_count = 1
            if self._repeat_error_count >= MAX_CONSECUTIVE_REPEAT_FAILURES:
                result = ToolResult.failure(
                    call_id,
                    "SKILL_REPEAT_FAILURE_LIMIT",
                    display_summary=(
                        f"同一错误连续失败 {self._repeat_error_count} 次，已停止自动重试：{result.display_summary}"
                    ),
                    content=result.content,
                    duration_ms=duration_ms,
                    tool_category=call.tool_category,
                    recoverable=False,
                    error_signature=sig,
                )
        else:
            self._last_error_signature = None
            self._repeat_error_count = 0

        # Context budget check for successful or recoverable results
        if result.ok or result.recoverable:
            encoded = result.for_model()
            byte_count = len(encoded.encode("utf-8"))
            if len(encoded) > self._remaining_chars or byte_count > self._remaining_bytes:
                result = ToolResult.failure(
                    call_id,
                    "RATE_LIMITED",
                    duration_ms=duration_ms,
                    tool_category=call.tool_category,
                    recoverable=False,
                )
            else:
                self._remaining_chars -= len(encoded)
                self._remaining_bytes -= byte_count

        status: Literal["completed", "failed"] = "completed" if result.ok else "failed"
        try:
            await self._settle_skills_audit(
                call,
                status,
                result.display_summary,
                result.error_code,
                duration_ms,
                recoverable=(not result.ok and result.recoverable),
            )
        except Exception:
            raise ModelError("INTERNAL_ERROR", ERROR_MESSAGES["INTERNAL_ERROR"]) from None

        if result.ok:
            self._emit(
                "tool_completed",
                {
                    "tool_call_id": call_id,
                    "summary": result.display_summary,
                    **self._skills_meta(call, duration_ms=duration_ms),
                },
            )
        else:
            # Recoverable failures are fed back to the model without aborting the Run!
            if not result.recoverable:
                self.failure = self.failure or result
            self._emit(
                "tool_failed",
                {
                    "tool_call_id": call_id,
                    "code": result.error_code,
                    "message": result.display_summary,
                    **self._skills_meta(call, duration_ms=duration_ms, recoverable=result.recoverable),
                },
            )
        return result

    async def _evaluate_skills_tool(self, call: ToolCall) -> ToolResult:
        entry = _SKILLS_TOOL_CATEGORIES.get(call.tool_name)
        if entry is None:
            return ToolResult.failure(
                call.tool_call_id,
                "TOOL_NOT_FOUND",
                display_summary=f"未注册的工具：{call.tool_name}",
                content={"error": f"工具 '{call.tool_name}' 未注册，请仅使用本轮提供的工具。"},
                tool_category=call.tool_category,
                recoverable=True,
                error_signature=f"tool_not_found:{call.tool_name}",
            )
        category, args_model = entry
        try:
            args = args_model.model_validate(call.arguments if isinstance(call.arguments, dict) else {})
        except ValidationError as exc:
            err_msg = str(exc.errors()[0].get("msg", "参数格式无效")) if exc.errors() else "参数格式无效"
            return ToolResult.failure(
                call.tool_call_id,
                "VALIDATION_ERROR",
                display_summary=f"工具 {call.tool_name} 参数无效：{err_msg}",
                content={"error": f"参数校验失败：{exc}"},
                tool_category=category,
                recoverable=True,
                error_signature=f"validation:{call.tool_name}:{err_msg}",
            )

        if category == "skill":
            if self._skill_ctx is None:
                return ToolResult.failure(
                    call.tool_call_id, "INTERNAL_ERROR",
                    display_summary="技能服务未初始化。", tool_category="skill", recoverable=False,
                )
            if call.tool_name == "list_skills":
                return await self._skill_ctx.list_skills(call, args)  # type: ignore[arg-type]
            if call.tool_name == "load_skill":
                return await self._skill_ctx.load_skill(call, args)  # type: ignore[arg-type]
            if call.tool_name == "read_skill_file":
                return await self._skill_ctx.read_skill_file(call, args)  # type: ignore[arg-type]

        if category == "workspace":
            if self._workspace_ctx is None:
                return ToolResult.failure(
                    call.tool_call_id, "INTERNAL_ERROR",
                    display_summary="工作区服务未初始化。", tool_category="workspace", recoverable=False,
                )
            if call.tool_name == "list_workspace":
                return await self._workspace_ctx.list_workspace(call, args)  # type: ignore[arg-type]
            if call.tool_name == "read_file":
                return await self._workspace_ctx.read_file(call, args)  # type: ignore[arg-type]
            if call.tool_name == "write_file":
                return await self._workspace_ctx.write_file(call, args)  # type: ignore[arg-type]
            if call.tool_name == "copy_file":
                return await self._workspace_ctx.copy_file(call, args)  # type: ignore[arg-type]
            if call.tool_name == "publish_artifact":
                return await self._workspace_ctx.publish_artifact(call, args)  # type: ignore[arg-type]

        if category == "runtime":
            if self._runtime_ctx is None:
                return ToolResult.failure(
                    call.tool_call_id, "INTERNAL_ERROR",
                    display_summary="运行环境执行器未初始化。", tool_category="runtime", recoverable=False,
                )
            if call.tool_name == "execute_command":
                return await self._runtime_ctx.execute_command(call, args)  # type: ignore[arg-type]

        return ToolResult.failure(
            call.tool_call_id, "TOOL_NOT_FOUND", tool_category=category, recoverable=True,
        )

    # ------------------------------------------------------------------
    # Legacy native file tools (`list_files` / `read_text`) — UNCHANGED semantics
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # MCP tools — UNCHANGED semantics
    # ------------------------------------------------------------------

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
            validated_args = validate_mcp_tool_arguments(snap.raw_input_schema, call.arguments)
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
