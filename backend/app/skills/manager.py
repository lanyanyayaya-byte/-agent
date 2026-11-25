"""SkillManager: discovery, builtin sync, local import, Progressive Disclosure, and resource access."""

from __future__ import annotations

import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from app.runtime.manager import PrepareResult, RuntimeManager, RuntimeStatus
from app.skills.dependencies import evaluate_dependencies, extract_dependency_spec, prepare_missing_dependencies
from app.skills.models import (
    ParsedSkillDocument,
    SkillCatalogEntry,
    SkillDependencyStatus,
    SkillDetailView,
    SkillError,
    SkillResourceEntry,
    SkillView,
    StoredSkill,
)
from app.skills.parser import (
    SKILL_FILENAME,
    compute_skill_revision,
    discover_skill_dirs,
    list_skill_resources,
    parse_skill_directory,
)
from app.storage.skills import SkillStore
from app.workspace.service import DEFAULT_READ_MAX_CHARS, LINE_ENDING, resolve_within, sanitize_filename

def _resolve_default_builtin_dir() -> Path:
    local = Path(__file__).resolve().parent / "builtin"
    if local.is_dir():
        return local
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        bundled = Path(meipass) / "app" / "skills" / "builtin"
        if bundled.is_dir():
            return bundled
    return local


DEFAULT_BUILTIN_DIR = _resolve_default_builtin_dir()
MAX_SKILL_RESOURCE_READ_BYTES = 1 * 1024 * 1024  # 1 MiB


def _copy_skill_tree(source_dir: Path, dest_dir: Path) -> None:
    """Copy a skill directory cleanly without hidden VCS or cache directories."""
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source_dir,
        dest_dir,
        symlinks=False,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".DS_Store", "*.pyc", "node_modules"),
    )


class SkillManager:
    """Manages local & built-in Agent Skills, revisioned files, dependencies, and activation."""

    def __init__(
        self,
        app_data_dir: Path,
        store: SkillStore,
        runtime_manager: RuntimeManager,
        *,
        builtin_dir: Path | None = None,
    ) -> None:
        self.app_data_dir = app_data_dir
        self.skills_root = app_data_dir / "skills"
        self.store = store
        self.runtime_manager = runtime_manager
        self.builtin_dir = builtin_dir or DEFAULT_BUILTIN_DIR

    def _skill_disk_dir(self, skill: StoredSkill) -> Path:
        return resolve_within(self.skills_root, skill.relative_dir, allow_missing=True)

    def _stored_revision_dir(self, skill_id: str, revision: str) -> Path | None:
        rev_root = self.skills_root / skill_id / "revisions" / revision
        if not rev_root.is_dir():
            return None
        for child in rev_root.iterdir():
            if child.is_dir() and (child / SKILL_FILENAME).is_file():
                return child
        return None

    def resolve_skill_dir(self, skill_id: str, *, revision: str | None = None) -> tuple[StoredSkill, Path]:
        stored = self.store.get_skill(skill_id)
        if stored is None:
            raise SkillError("SKILL_NOT_FOUND", f"未找到技能：{skill_id}")
        if revision and revision != stored.revision:
            pinned = self._stored_revision_dir(skill_id, revision)
            if pinned is not None:
                return stored, pinned
        path = self._skill_disk_dir(stored)
        if not path.is_dir() or not (path / SKILL_FILENAME).is_file():
            raise SkillError("SKILL_FILES_MISSING", f"技能文件已从本地目录缺失：{stored.name}")
        return stored, path

    # ------------------------------------------------------------------
    # Startup sync of built-in skills (idempotent; preserves user enabled state)
    # ------------------------------------------------------------------

    def sync_builtin_skills(self) -> list[StoredSkill]:
        """Install or refresh built-in skills into `<app_data_dir>/skills/` while keeping user enable toggles."""
        if not self.builtin_dir.is_dir():
            return []
        self.skills_root.mkdir(parents=True, exist_ok=True)
        synced: list[StoredSkill] = []
        for source_dir in discover_skill_dirs(self.builtin_dir, max_depth=2):
            doc = parse_skill_directory(source_dir)
            revision = compute_skill_revision(source_dir)
            skill_id = f"builtin-{doc.name}"
            safe_dir_name = sanitize_filename(source_dir.name, fallback=doc.name)
            rel_dir = f"{skill_id}/revisions/{revision}/{safe_dir_name}"
            dest_dir = self.skills_root / rel_dir

            existing = self.store.get_skill(skill_id)
            if existing is None or existing.revision != revision or not dest_dir.is_dir():
                _copy_skill_tree(source_dir, dest_dir)

            record = self.store.upsert_skill(
                skill_id=skill_id,
                name=doc.name,
                description=doc.description,
                source_type="builtin",
                source_url=None,
                revision=revision,
                relative_dir=rel_dir,
                enabled=True,  # Built-in skills enabled by default on first install
                metadata=doc.metadata,
                preserve_enabled_if_exists=True,
            )
            synced.append(record)
        return synced

    # ------------------------------------------------------------------
    # Local directory import, enable/disable, delete
    # ------------------------------------------------------------------

    def import_local_directory(
        self,
        source_path: Path | str,
        *,
        enabled: bool = False,
        replace_existing_id: str | None = None,
    ) -> list[SkillView]:
        """Discover and install one or more standard Agent Skills from a local directory."""
        src = Path(source_path)
        if not src.is_absolute():
            raise SkillError("SKILL_INVALID_PATH", "本地技能导入目录必须为绝对路径。")
        if not src.is_dir():
            raise SkillError("SKILL_NOT_FOUND", f"未找到本地目录：{source_path}")

        skill_dirs = discover_skill_dirs(src, max_depth=3)
        if not skill_dirs:
            raise SkillError("SKILL_NOT_FOUND", "所选目录中未发现包含 SKILL.md 的技能。")

        self.skills_root.mkdir(parents=True, exist_ok=True)
        runtime_status = self.runtime_manager.probe_status()
        installed: list[SkillView] = []

        for sdir in skill_dirs:
            doc = parse_skill_directory(sdir)
            revision = compute_skill_revision(sdir)
            skill_id = replace_existing_id if (replace_existing_id and len(skill_dirs) == 1) else str(uuid4())
            safe_dir_name = sanitize_filename(sdir.name, fallback=doc.name)
            rel_dir = f"{skill_id}/revisions/{revision}/{safe_dir_name}"
            dest_dir = self.skills_root / rel_dir
            _copy_skill_tree(sdir, dest_dir)

            stored = self.store.upsert_skill(
                skill_id=skill_id,
                name=doc.name,
                description=doc.description,
                source_type="local",
                source_url=str(sdir.resolve()),
                revision=revision,
                relative_dir=rel_dir,
                enabled=enabled,
                metadata=doc.metadata,
                preserve_enabled_if_exists=False,
            )
            installed.append(self._build_skill_view(stored, runtime_status, parsed=doc, skill_dir=dest_dir))
        return installed

    def set_enabled(self, skill_id: str, enabled: bool) -> SkillView:
        updated = self.store.set_enabled(skill_id, enabled)
        if updated is None:
            raise SkillError("SKILL_NOT_FOUND", f"未找到技能：{skill_id}")
        runtime_status = self.runtime_manager.probe_status()
        return self._build_skill_view(updated, runtime_status)

    def delete_skill(self, skill_id: str) -> None:
        deleted = self.store.delete_skill(skill_id)
        if deleted is None:
            raise SkillError("SKILL_NOT_FOUND", f"未找到技能：{skill_id}")
        skill_dir = self.skills_root / skill_id
        if skill_dir.exists() and not skill_dir.is_symlink():
            shutil.rmtree(skill_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Queries & views
    # ------------------------------------------------------------------

    def _build_skill_view(
        self,
        stored: StoredSkill,
        runtime_status: RuntimeStatus,
        *,
        parsed: ParsedSkillDocument | None = None,
        skill_dir: Path | None = None,
    ) -> SkillView:
        disk_dir = skill_dir or self._skill_disk_dir(stored)
        warnings: list[str] = []
        if parsed is None:
            if disk_dir.is_dir() and (disk_dir / SKILL_FILENAME).is_file():
                try:
                    parsed = parse_skill_directory(disk_dir)
                except SkillError as exc:
                    warnings.append(exc.message)
            else:
                warnings.append("技能本地文件缺失。")

        if parsed is not None:
            warnings.extend(parsed.warnings)
            spec = extract_dependency_spec(disk_dir, parsed)
            dep_status = evaluate_dependencies(spec, runtime_status)
        else:
            dep_status = SkillDependencyStatus(
                ready=False,
                can_auto_prepare=False,
                summary="技能文件不可用。",
            )

        return SkillView(
            skill_id=stored.skill_id,
            name=stored.name,
            description=stored.description,
            source_type=stored.source_type,
            source_url=stored.source_url,
            revision=stored.revision,
            enabled=stored.enabled,
            ready=dep_status.ready,
            dependency_status=dep_status,
            warnings=tuple(warnings),
            created_at=stored.created_at,
            updated_at=stored.updated_at,
        )

    def list_skills(self) -> list[SkillView]:
        runtime_status = self.runtime_manager.probe_status()
        return [self._build_skill_view(s, runtime_status) for s in self.store.list_skills()]

    def get_skill_detail(self, skill_id: str) -> SkillDetailView:
        stored, disk_dir = self.resolve_skill_dir(skill_id)
        doc = parse_skill_directory(disk_dir)
        runtime_status = self.runtime_manager.probe_status()
        view = self._build_skill_view(stored, runtime_status, parsed=doc, skill_dir=disk_dir)
        resources = list_skill_resources(disk_dir)
        return SkillDetailView(
            skill=view,
            license=doc.license,
            compatibility=doc.compatibility,
            allowed_tools=doc.allowed_tools,
            extra_frontmatter=doc.extra_frontmatter,
            metadata=doc.metadata,
            instructions=doc.body,
            resources=resources,
        )

    # ------------------------------------------------------------------
    # Progressive Disclosure: Catalog -> Activation -> On-demand Resource
    # ------------------------------------------------------------------

    def build_catalog_snapshot(
        self,
        *,
        selected_skill_ids: Sequence[str] | None = None,
    ) -> tuple[SkillCatalogEntry, ...]:
        """Return ONLY enabled skills' lightweight metadata (name + description + skill_id)."""
        enabled_skills = self.store.list_skills(enabled_only=True)
        if selected_skill_ids:
            wanted = set(selected_skill_ids)
            enabled_skills = [s for s in enabled_skills if s.skill_id in wanted or s.name in wanted]
        if not enabled_skills:
            return ()
        runtime_status = self.runtime_manager.probe_status()
        entries: list[SkillCatalogEntry] = []
        for stored in enabled_skills:
            disk_dir = self._skill_disk_dir(stored)
            if not (disk_dir.is_dir() and (disk_dir / SKILL_FILENAME).is_file()):
                continue
            try:
                doc = parse_skill_directory(disk_dir)
                spec = extract_dependency_spec(disk_dir, doc)
                dep = evaluate_dependencies(spec, runtime_status)
                ready = dep.ready or dep.can_auto_prepare
            except SkillError:
                continue
            entries.append(SkillCatalogEntry(
                skill_id=stored.skill_id,
                name=stored.name,
                description=stored.description,
                revision=stored.revision,
                source_type=stored.source_type,
                ready=ready,
            ))
        return tuple(entries)

    async def activate_skill(
        self,
        skill_id: str,
        *,
        revision: str | None = None,
        auto_prepare_builtin: bool = True,
    ) -> tuple[StoredSkill, ParsedSkillDocument, tuple[SkillResourceEntry, ...], Path, SkillDependencyStatus, PrepareResult | None]:
        """Load full `SKILL.md` body and resource index; optionally prepare missing predefined builtin deps."""
        stored, disk_dir = self.resolve_skill_dir(skill_id, revision=revision)
        doc = parse_skill_directory(disk_dir)
        resources = list_skill_resources(disk_dir)
        spec = extract_dependency_spec(disk_dir, doc)

        prep_result: PrepareResult | None = None
        if auto_prepare_builtin and stored.source_type == "builtin":
            dep_status, prep_result = await prepare_missing_dependencies(spec, self.runtime_manager)
        else:
            runtime_status = self.runtime_manager.probe_status()
            dep_status = evaluate_dependencies(spec, runtime_status)

        return stored, doc, resources, disk_dir, dep_status, prep_result

    def read_skill_file(
        self,
        skill_id: str,
        relative_path: str,
        *,
        revision: str | None = None,
        offset_line: int = 0,
        limit_lines: int = 200,
        max_chars: int = DEFAULT_READ_MAX_CHARS,
    ) -> dict[str, object]:
        """Read a text resource file from inside a skill's revision directory."""
        _, disk_dir = self.resolve_skill_dir(skill_id, revision=revision)
        try:
            target = resolve_within(disk_dir, relative_path)
        except Exception as exc:
            raise SkillError("SKILL_RESOURCE_INVALID", f"无法访问技能资源：{exc}") from None
        if not target.is_file():
            raise SkillError("SKILL_RESOURCE_NOT_FOUND", f"未找到技能资源文件：{relative_path}")
        size = target.stat().st_size
        if size > MAX_SKILL_RESOURCE_READ_BYTES:
            raise SkillError("SKILL_RESOURCE_TOO_LARGE", "技能资源文件超过 1 MiB 读取上限。")

        raw = target.read_bytes()
        if b"\x00" in raw[:8192]:
            return {
                "relative_path": target.relative_to(disk_dir.resolve()).as_posix(),
                "is_binary": True,
                "size_bytes": size,
                "absolute_path": str(target),
            }

        decoded = raw.decode("utf-8-sig", errors="replace")
        lines = LINE_ENDING.sub("\n", decoded).split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        start = max(0, offset_line)
        limit = max(1, limit_lines)
        selected: list[str] = []
        total_chars = 0
        clipped = False
        for line in lines[start : start + limit]:
            remaining = max_chars - total_chars - (1 if selected else 0)
            if len(line) > remaining:
                if not selected:
                    selected.append(line[:max_chars])
                    clipped = True
                break
            selected.append(line)
            total_chars += len(line) + (1 if len(selected) > 1 else 0)
        end = start + len(selected)
        next_line = end if end < len(lines) else None
        return {
            "relative_path": target.relative_to(disk_dir.resolve()).as_posix(),
            "is_binary": False,
            "size_bytes": size,
            "content": "\n".join(selected),
            "total_lines": len(lines),
            "offset_line": start,
            "returned_lines": len(selected),
            "next_offset_line": next_line,
            "truncated": clipped or next_line is not None,
        }
