"""Agent tools for Progressive Disclosure of Agent Skills (`list_skills`, `load_skill`, `read_skill_file`)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from app.skills.manager import SkillManager
from app.skills.models import SkillCatalogEntry, SkillError
from app.storage.artifacts import WorkspaceArtifactStore
from app.tools.protocol import ToolCall, ToolResult


class ListSkillsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoadSkillArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill_id: Annotated[str, Field(min_length=1, max_length=128)]


class ReadSkillFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill_id: Annotated[str, Field(min_length=1, max_length=128)]
    relative_path: Annotated[str, Field(min_length=1, max_length=512)]
    offset_line: Annotated[int, Field(ge=0, le=100_000)] = 0
    limit_lines: Annotated[int, Field(ge=1, le=500)] = 200


@dataclass
class ActivatedSkillRecord:
    skill_id: str
    name: str
    revision: str
    skill_dir: str
    resources: tuple[dict[str, Any], ...]


class SkillToolContext:
    """Per-run skill catalog snapshot, deduplicated activation cache, and event emitter."""

    def __init__(
        self,
        *,
        manager: SkillManager,
        artifact_store: WorkspaceArtifactStore,
        session_id: str,
        run_id: str,
        catalog: tuple[SkillCatalogEntry, ...],
        emit_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.manager = manager
        self.artifact_store = artifact_store
        self.session_id = session_id
        self.run_id = run_id
        self.catalog = catalog
        self._by_id: dict[str, SkillCatalogEntry] = {entry.skill_id: entry for entry in catalog}
        self._by_name: dict[str, SkillCatalogEntry] = {entry.name.lower(): entry for entry in catalog}
        self.activated: dict[str, ActivatedSkillRecord] = {}
        self._emit_event = emit_event

    def resolve_catalog_entry(self, skill_id_or_name: str) -> SkillCatalogEntry | None:
        key = skill_id_or_name.strip()
        if key in self._by_id:
            return self._by_id[key]
        return self._by_name.get(key.lower())

    async def list_skills(self, call: ToolCall, _args: ListSkillsArgs) -> ToolResult:
        items = [
            {
                **entry.to_prompt_dict(),
                "activated": entry.skill_id in self.activated,
            }
            for entry in self.catalog
        ]
        return ToolResult(
            call.tool_call_id,
            True,
            {"skills": items},
            f"列出 {len(items)} 个可用技能。",
            tool_category="skill",
        )

    async def load_skill(self, call: ToolCall, args: LoadSkillArgs) -> ToolResult:
        entry = self.resolve_catalog_entry(args.skill_id)
        if entry is None:
            return ToolResult.failure(
                call.tool_call_id,
                "SKILL_NOT_FOUND",
                display_summary=f"未找到可用技能：{args.skill_id}",
                content={"error": f"技能 '{args.skill_id}' 不存在或未在本轮启用。", "available_skill_ids": list(self._by_id)},
                tool_category="skill",
                recoverable=True,
                error_signature=f"load_skill:not_found:{args.skill_id}",
            )

        # Deduplicate repeated activations in the same Run to avoid wasting context tokens.
        existing = self.activated.get(entry.skill_id)
        if existing is not None:
            return ToolResult(
                call.tool_call_id,
                True,
                {
                    "skill_id": existing.skill_id,
                    "name": existing.name,
                    "revision": existing.revision,
                    "already_activated": True,
                    "skill_dir": existing.skill_dir,
                    "resources": list(existing.resources),
                    "note": "该技能已在本轮激活，请直接遵循前文已加载的完整技能指令，无需重复加载。",
                },
                f"技能 {existing.name} 已在当前轮次激活（跳过重复加载）。",
                tool_category="skill",
            )

        try:
            stored, doc, resources, disk_dir, dep_status, prep_result = await self.manager.activate_skill(
                entry.skill_id,
                revision=entry.revision,
                auto_prepare_builtin=True,
            )
        except SkillError as exc:
            return ToolResult.failure(
                call.tool_call_id,
                exc.code,
                display_summary=exc.message,
                content={"error": exc.message},
                tool_category="skill",
                recoverable=True,
                error_signature=f"load_skill:{exc.code}:{entry.skill_id}",
            )

        res_dicts = tuple(r.to_dict() for r in resources)
        self.activated[stored.skill_id] = ActivatedSkillRecord(
            skill_id=stored.skill_id,
            name=stored.name,
            revision=entry.revision,
            skill_dir=str(disk_dir.resolve()),
            resources=res_dicts,
        )

        # Persist activation record in SQLite `run_skill_activations`
        await asyncio.to_thread(
            self.artifact_store.record_skill_activation,
            session_id=self.session_id,
            run_id=self.run_id,
            skill_id=stored.skill_id,
            name_snapshot=stored.name,
            revision=entry.revision,
        )
        if self._emit_event is not None:
            self._emit_event("skill_activated", {
                "skill_id": stored.skill_id,
                "name": stored.name,
                "revision": entry.revision,
                "dependencies_ready": dep_status.ready,
                "prepared_dependencies": bool(prep_result and prep_result.steps),
            })

        content: dict[str, Any] = {
            "skill_id": stored.skill_id,
            "name": stored.name,
            "description": stored.description,
            "revision": entry.revision,
            "already_activated": False,
            "skill_dir": str(disk_dir.resolve()),
            "resources": list(res_dicts),
            "dependency_status": dep_status.to_dict(),
            "instructions": doc.body,
        }
        if prep_result and prep_result.steps:
            content["dependency_preparation"] = [
                {"step": s.step, "ok": s.ok, "message": s.message}
                for s in prep_result.steps
            ]
        summary = f"已加载技能「{stored.name}」（{len(resources)} 个资源文件）。"
        if prep_result and prep_result.steps:
            summary += " 已自动准备缺失运行依赖。" if prep_result.ok else " 依赖准备存在异常。"
        return ToolResult(
            call.tool_call_id,
            True,
            content,
            summary,
            tool_category="skill",
        )

    async def read_skill_file(self, call: ToolCall, args: ReadSkillFileArgs) -> ToolResult:
        entry = self.resolve_catalog_entry(args.skill_id)
        if entry is None:
            return ToolResult.failure(
                call.tool_call_id,
                "SKILL_NOT_FOUND",
                display_summary=f"未找到可用技能：{args.skill_id}",
                content={"error": f"技能 '{args.skill_id}' 不存在或未启用。"},
                tool_category="skill",
                recoverable=True,
                error_signature=f"read_skill_file:not_found:{args.skill_id}",
            )

        try:
            data = await asyncio.to_thread(
                self.manager.read_skill_file,
                entry.skill_id,
                args.relative_path,
                revision=entry.revision,
                offset_line=args.offset_line,
                limit_lines=args.limit_lines,
            )
        except SkillError as exc:
            return ToolResult.failure(
                call.tool_call_id,
                exc.code,
                display_summary=exc.message,
                content={"error": exc.message, "skill_id": entry.skill_id, "relative_path": args.relative_path},
                tool_category="skill",
                recoverable=True,
                error_signature=f"read_skill_file:{exc.code}:{args.relative_path}",
            )

        return ToolResult(
            call.tool_call_id,
            True,
            data,
            f"读取技能资源 {args.relative_path}。",
            truncated=bool(data.get("truncated", False)),
            next_offset_line=data.get("next_offset_line") if isinstance(data.get("next_offset_line"), int) else None,
            tool_category="skill",
        )


def skill_tool_schemas() -> tuple[dict[str, Any], ...]:
    return (
        {
            "type": "function",
            "function": {
                "name": "list_skills",
                "description": "列出本轮已启用的 Agent 技能名称、描述、ID 和激活状态。",
                "parameters": ListSkillsArgs.model_json_schema(),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "load_skill",
                "description": "按需激活指定技能，返回该技能完整的 SKILL.md 执行指令、技能目录绝对路径（skill_dir）及脚本/参考资料清单。",
                "parameters": LoadSkillArgs.model_json_schema(),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_skill_file",
                "description": "按需读取技能目录中的参考资料、配置模板或脚本文件（相对路径如 references/... 或 scripts/...）。",
                "parameters": ReadSkillFileArgs.model_json_schema(),
            },
        },
    )
