"""Agent tools for session workspace file operations and artifact publication."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import asdict
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.tools.protocol import ToolCall, ToolResult
from app.workspace.artifacts import ArtifactService
from app.workspace.service import WorkspaceError, WorkspaceService


class ListWorkspaceArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relative_path: Annotated[str, Field(max_length=512)] = ""
    scope: Literal["run", "session"] = "run"


class ReadWorkspaceFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relative_path: Annotated[str, Field(min_length=1, max_length=512)]
    scope: Literal["run", "session"] = "run"
    offset_line: Annotated[int, Field(ge=0, le=100_000)] = 0
    limit_lines: Annotated[int, Field(ge=1, le=500)] = 200


class WriteWorkspaceFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relative_path: Annotated[str, Field(min_length=1, max_length=512)]
    content: Annotated[str, Field(max_length=512_000)]
    overwrite: bool = True


class CopyWorkspaceFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_path: Annotated[str, Field(min_length=1, max_length=512)]
    dest_path: Annotated[str, Field(min_length=1, max_length=512)]
    overwrite: bool = True


class PublishArtifactArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relative_path: Annotated[str, Field(min_length=1, max_length=512)]
    title: Annotated[str, Field(max_length=120)] | None = None
    filename: Annotated[str, Field(max_length=200)] | None = None


class WorkspaceToolContext:
    """Per-run workspace and artifact operations exposed to the Agent."""

    def __init__(
        self,
        *,
        workspace: WorkspaceService,
        artifacts: ArtifactService,
        session_id: str,
        run_id: str,
        attachment_ids: tuple[str, ...] = (),
        artifact_ids: tuple[str, ...] = (),
        emit_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.workspace = workspace
        self.artifacts = artifacts
        self.session_id = session_id
        self.run_id = run_id
        self.attachment_ids = attachment_ids
        self.artifact_ids = artifact_ids
        self._emit_event = emit_event
        self.staged_artifacts: list[dict[str, Any]] = []

    def workspace_context_metadata(self) -> dict[str, Any]:
        """Compact workspace metadata injected into system prompt at Run start."""
        files = self.workspace.list_files(self.session_id)
        if self.attachment_ids:
            wanted = set(self.attachment_ids)
            selected_files = [f for f in files if f.file_id in wanted]
        else:
            selected_files = files

        published = self.artifacts.list_artifacts(self.session_id, include_non_published=False)
        if self.artifact_ids:
            wanted_arts = set(self.artifact_ids)
            selected_arts = [a for a in published if a.artifact_id in wanted_arts]
        else:
            selected_arts = published

        return {
            "run_work_dirs": ["work/", "output/"],
            "attachments": [
                {
                    "file_id": f.file_id,
                    "filename": f.original_name,
                    "relative_path": f.relative_path,
                    "media_type": f.media_type,
                    "size_bytes": f.size_bytes,
                }
                for f in selected_files[:20]
            ],
            "published_artifacts": [
                {
                    "artifact_id": a.artifact_id,
                    "filename": a.filename,
                    "title": a.title,
                    "relative_path": a.relative_path,
                    "media_type": a.media_type,
                    "size_bytes": a.size_bytes,
                    "file_exists": a.file_exists,
                }
                for a in selected_arts[:20]
            ],
        }

    async def list_workspace(self, call: ToolCall, args: ListWorkspaceArgs) -> ToolResult:
        try:
            run_scope = self.run_id if args.scope == "run" else None
            entries = await asyncio.to_thread(
                self.workspace.list_directory,
                self.session_id,
                run_id=run_scope,
                relative_path=args.relative_path,
            )
            payload: dict[str, Any] = {
                "scope": args.scope,
                "relative_path": args.relative_path or ".",
                "entries": [asdict(e) for e in entries],
            }
            if args.scope == "run" and not args.relative_path:
                meta = await asyncio.to_thread(self.workspace_context_metadata)
                payload["attachments"] = meta["attachments"]
                payload["published_artifacts"] = meta["published_artifacts"]
            return ToolResult(
                call.tool_call_id,
                True,
                payload,
                f"列出工作区 {len(entries)} 项。",
                tool_category="workspace",
            )
        except WorkspaceError as exc:
            return ToolResult.failure(
                call.tool_call_id,
                exc.code,
                display_summary=exc.message,
                content={"error": exc.message, "relative_path": args.relative_path},
                tool_category="workspace",
                recoverable=True,
                error_signature=f"list_workspace:{exc.code}:{args.relative_path}",
            )

    async def read_file(self, call: ToolCall, args: ReadWorkspaceFileArgs) -> ToolResult:
        def _do_read():
            run_scope = self.run_id if args.scope == "run" else None
            try:
                return self.workspace.read_file(
                    self.session_id,
                    args.relative_path,
                    run_id=run_scope,
                    offset_line=args.offset_line,
                    limit_lines=args.limit_lines,
                )
            except WorkspaceError as err:
                # Helpful fallback: if caller passed `inputs/...` or `artifacts/...` with default scope='run', retry on session scope.
                if args.scope == "run" and err.code == "WORKSPACE_NOT_FOUND" and (
                    args.relative_path.startswith("inputs/") or args.relative_path.startswith("artifacts/")
                ):
                    return self.workspace.read_file(
                        self.session_id,
                        args.relative_path,
                        run_id=None,
                        offset_line=args.offset_line,
                        limit_lines=args.limit_lines,
                    )
                raise

        try:
            res = await asyncio.to_thread(_do_read)
        except WorkspaceError as exc:
            return ToolResult.failure(
                call.tool_call_id,
                exc.code,
                display_summary=exc.message,
                content={"error": exc.message, "relative_path": args.relative_path},
                tool_category="workspace",
                recoverable=True,
                error_signature=f"read_file:{exc.code}:{args.relative_path}",
            )

        data = asdict(res)
        if res.is_binary:
            data["hint"] = "该文件为二进制文件，请使用 copy_file 复制到 work/ 后通过 Python/Node 脚本读取或处理。"
            summary = f"检测到二进制文件 {res.relative_path}（{res.media_type}，{res.size_bytes} bytes）。"
        else:
            summary = f"读取工作区文件 {res.relative_path}（{res.returned_lines} 行）。"
        return ToolResult(
            call.tool_call_id,
            True,
            data,
            summary,
            truncated=res.truncated,
            next_offset_line=res.next_offset_line,
            tool_category="workspace",
        )

    async def write_file(self, call: ToolCall, args: WriteWorkspaceFileArgs) -> ToolResult:
        try:
            res = await asyncio.to_thread(
                self.workspace.write_file,
                self.session_id,
                args.relative_path,
                args.content,
                run_id=self.run_id,
                overwrite=args.overwrite,
            )
        except WorkspaceError as exc:
            return ToolResult.failure(
                call.tool_call_id,
                exc.code,
                display_summary=exc.message,
                content={"error": exc.message, "relative_path": args.relative_path},
                tool_category="workspace",
                recoverable=True,
                error_signature=f"write_file:{exc.code}:{args.relative_path}",
            )

        return ToolResult(
            call.tool_call_id,
            True,
            {
                "relative_path": res.relative_path,
                "size_bytes": res.size_bytes,
                "media_type": res.media_type,
            },
            f"已写入 {res.relative_path}（{res.size_bytes} bytes）。",
            tool_category="workspace",
        )

    async def copy_file(self, call: ToolCall, args: CopyWorkspaceFileArgs) -> ToolResult:
        try:
            dest = await asyncio.to_thread(
                self.workspace.copy_file,
                self.session_id,
                args.source_path,
                args.dest_path,
                run_id=self.run_id,
                overwrite=args.overwrite,
            )
        except WorkspaceError as exc:
            return ToolResult.failure(
                call.tool_call_id,
                exc.code,
                display_summary=exc.message,
                content={"error": exc.message, "source_path": args.source_path, "dest_path": args.dest_path},
                tool_category="workspace",
                recoverable=True,
                error_signature=f"copy_file:{exc.code}:{args.source_path}",
            )

        run_root = self.workspace.run_root(self.session_id, self.run_id).resolve()
        rel = dest.resolve().relative_to(run_root).as_posix()
        size = dest.stat().st_size
        return ToolResult(
            call.tool_call_id,
            True,
            {"source_path": args.source_path, "dest_path": rel, "size_bytes": size},
            f"已复制到 {rel}（{size} bytes）。",
            tool_category="workspace",
        )

    async def publish_artifact(self, call: ToolCall, args: PublishArtifactArgs) -> ToolResult:
        try:
            stored = await asyncio.to_thread(
                self.artifacts.stage_artifact,
                self.session_id,
                self.run_id,
                args.relative_path,
                title=args.title,
                filename=args.filename,
            )
        except WorkspaceError as exc:
            return ToolResult.failure(
                call.tool_call_id,
                exc.code,
                display_summary=exc.message,
                content={"error": exc.message, "relative_path": args.relative_path},
                tool_category="workspace",
                recoverable=True,
                error_signature=f"publish_artifact:{exc.code}:{args.relative_path}",
            )

        record = asdict(stored)
        self.staged_artifacts.append(record)
        if self._emit_event is not None:
            self._emit_event("artifact_created", {
                "artifact_id": stored.artifact_id,
                "filename": stored.filename,
                "title": stored.title,
                "media_type": stored.media_type,
                "size_bytes": stored.size_bytes,
                "status": stored.status,
            })
        return ToolResult(
            call.tool_call_id,
            True,
            {
                "artifact_id": stored.artifact_id,
                "filename": stored.filename,
                "title": stored.title,
                "media_type": stored.media_type,
                "size_bytes": stored.size_bytes,
                "relative_path": stored.relative_path,
                "status": stored.status,
            },
            f"已登记产物 {stored.filename}（{stored.size_bytes} bytes）。",
            tool_category="workspace",
        )


def workspace_tool_schemas() -> tuple[dict[str, Any], ...]:
    return (
        {
            "type": "function",
            "function": {
                "name": "list_workspace",
                "description": "列举当前任务工作目录（work/、output/）以及当前会话可用的附件（inputs/）和历史已发布产物（artifacts/）。",
                "parameters": ListWorkspaceArgs.model_json_schema(),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "读取工作区中的文本文件（如 .mmd、.md、.py、.json）；若是 .docx/.png 等二进制文件则返回文件类型和大小元数据。",
                "parameters": ReadWorkspaceFileArgs.model_json_schema(),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "在当前任务目录中创建或覆盖 UTF-8 文本文件（例如 work/script.py、work/format_spec.json、output/mindmap.mmd）。",
                "parameters": WriteWorkspaceFileArgs.model_json_schema(),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "copy_file",
                "description": "将工作区附件（inputs/...）或历史已发布产物（artifacts/...）复制到当前任务工作目录（如 work/source.docx）以保留原件不变。",
                "parameters": CopyWorkspaceFileArgs.model_json_schema(),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "publish_artifact",
                "description": "验证当前任务生成的结果文件（如 output/*.docx、output/*.svg、output/*.png、output/*.mmd）并登记为正式候选产物。",
                "parameters": PublishArtifactArgs.model_json_schema(),
            },
        },
    )
