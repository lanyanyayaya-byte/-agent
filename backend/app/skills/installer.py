"""SkillInstaller: import preview, selective install, and update from GitHub repos, local folders, and ZIP archives.

Supports:
- Public GitHub repository root URLs (`https://github.com/<owner>/<repo>`)
- Public GitHub skill subdirectory URLs (`https://github.com/<owner>/<repo>/tree/<ref>/<subpath>`)
- Explicit `ref` and `subpath` overrides (resolves branch names containing slashes)
- Local directories (single skill or multi-skill collection)
- Local `.zip` archives (with path traversal / Zip Slip protection)

Uses local `git` (`git clone --depth 1`) first to capture exact commit SHA, falling back to
HTTP ZIP archive download via `httpx` + `zipfile` if `git` is unavailable.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import time
import zipfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from app.runtime.manager import RuntimeManager
from app.skills.dependencies import evaluate_dependencies, extract_dependency_spec
from app.skills.manager import SkillManager, _copy_skill_tree
from app.skills.models import SkillDependencyStatus, SkillError, SkillView
from app.skills.parser import (
    SKILL_FILENAME,
    compute_skill_revision,
    discover_skill_dirs,
    list_skill_resources,
    parse_skill_directory,
)
from app.workspace.service import sanitize_filename

ImportSourceKind = Literal["github", "local_dir", "local_zip"]

GITHUB_HOST = "github.com"
OWNER_REPO_RE = re.compile(r"^[A-Za-z0-9._-]{1,100}$")
STAGING_TTL_SECONDS = 3600
MAX_ZIP_EXTRACT_BYTES = 150 * 1024 * 1024  # 150 MiB
MAX_ZIP_FILES = 4000
CLONE_TIMEOUT_SECONDS = 120.0

_CLAUDE_TOOL_HINTS = ("MultiEdit", "WebFetch", "TodoWrite", "NotebookEdit", "Task(")


@dataclass(frozen=True, slots=True)
class ParsedGitHubUrl:
    owner: str
    repo: str
    repo_url: str
    ref: str | None = None
    subpath: str | None = None


@dataclass(frozen=True, slots=True)
class ImportCandidate:
    relative_path: str
    name: str
    description: str
    license: str | None
    compatibility: str | None
    resource_count: int
    dependency_status: SkillDependencyStatus
    required_python_packages: tuple[str, ...]
    required_node_tools: tuple[str, ...]
    required_system_commands: tuple[str, ...]
    warnings: tuple[str, ...]
    compatibility_notes: tuple[str, ...]
    existing_skill_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["dependency_status"] = self.dependency_status.to_dict()
        return data


@dataclass(frozen=True, slots=True)
class ImportPreviewResult:
    preview_id: str
    source_kind: ImportSourceKind
    source_url: str
    resolved_ref: str | None
    commit_sha: str | None
    subpath: str | None
    candidates: tuple[ImportCandidate, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "preview_id": self.preview_id,
            "source_kind": self.source_kind,
            "source_url": self.source_url,
            "resolved_ref": self.resolved_ref,
            "commit_sha": self.commit_sha,
            "subpath": self.subpath,
            "candidates": [c.to_dict() for c in self.candidates],
        }


def parse_github_url(
    raw_url: str,
    *,
    explicit_ref: str | None = None,
    explicit_subpath: str | None = None,
) -> ParsedGitHubUrl:
    """Parse a public GitHub repo root or `/tree/<ref>/<subpath>` URL."""
    text = raw_url.strip()
    if not text:
        raise SkillError("SKILL_INVALID_SOURCE", "请输入有效的 GitHub 仓库地址。")

    # Allow shorthand `owner/repo`
    if "://" not in text and text.count("/") == 1 and not text.startswith("/"):
        text = f"https://{GITHUB_HOST}/{text}"

    parsed = urlsplit(text)
    if parsed.scheme not in ("https", "http") or (parsed.hostname or "").lower() != GITHUB_HOST:
        raise SkillError("SKILL_INVALID_SOURCE", "仅支持公开 GitHub 仓库地址（https://github.com/owner/repo）。")

    segments = [seg for seg in parsed.path.split("/") if seg]
    if len(segments) < 2:
        raise SkillError("SKILL_INVALID_SOURCE", "GitHub 地址必须包含仓库所有者与仓库名（owner/repo）。")

    owner = segments[0]
    repo = segments[1].removesuffix(".git")
    if not OWNER_REPO_RE.fullmatch(owner) or not OWNER_REPO_RE.fullmatch(repo):
        raise SkillError("SKILL_INVALID_SOURCE", "GitHub 仓库名称格式无效。")

    ref = explicit_ref.strip() if isinstance(explicit_ref, str) and explicit_ref.strip() else None
    subpath = explicit_subpath.strip().strip("/") if isinstance(explicit_subpath, str) and explicit_subpath.strip() else None

    # Parse `/tree/<ref>/<optional subpath...>` or `/blob/<ref>/<subpath>/SKILL.md`
    if len(segments) >= 4 and segments[2] in ("tree", "blob"):
        url_ref = segments[3]
        url_rest = segments[4:]
        if segments[2] == "blob" and url_rest and url_rest[-1] == SKILL_FILENAME:
            url_rest = url_rest[:-1]
        if ref is None:
            ref = url_ref
        if subpath is None and url_rest:
            subpath = "/".join(url_rest)

    if subpath:
        if ".." in subpath.split("/") or subpath.startswith("/"):
            raise SkillError("SKILL_INVALID_SOURCE", "子目录路径不能包含 '..' 或绝对路径。")

    return ParsedGitHubUrl(
        owner=owner,
        repo=repo,
        repo_url=f"https://{GITHUB_HOST}/{owner}/{repo}",
        ref=ref,
        subpath=subpath,
    )


def _safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract a ZIP archive safely into `dest_dir`, blocking Zip Slip path traversal."""
    dest_resolved = dest_dir.resolve()
    total_bytes = 0
    file_count = 0
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                file_count += 1
                if file_count > MAX_ZIP_FILES:
                    raise SkillError("SKILL_IMPORT_FAILED", f"压缩包内文件数量超过上限（{MAX_ZIP_FILES}）。")
                member_name = info.filename.replace("\\", "/")
                if member_name.startswith("/") or ".." in Path(member_name).parts:
                    raise SkillError("SKILL_INVALID_PATH", f"压缩包包含非法路径：{info.filename}")
                # Skip macOS AppleDouble resource forks
                if any(part == "__MACOSX" or part.startswith("._") for part in Path(member_name).parts):
                    continue
                # Skip symlinks inside zip (external_attr mode 0o120000)
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    continue
                target = (dest_resolved / member_name).resolve()
                if not target.is_relative_to(dest_resolved):
                    raise SkillError("SKILL_INVALID_PATH", f"压缩包路径试图越界：{info.filename}")
                total_bytes += info.file_size
                if total_bytes > MAX_ZIP_EXTRACT_BYTES:
                    raise SkillError("SKILL_IMPORT_FAILED", "压缩包解压体积超过 150 MiB 上限。")
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src_fp, target.open("wb") as dst_fp:
                    shutil.copyfileobj(src_fp, dst_fp)
    except zipfile.BadZipFile:
        raise SkillError("SKILL_INVALID_SOURCE", "所选文件不是有效的 ZIP 压缩包。") from None


def _unwrap_single_root_dir(directory: Path) -> Path:
    """GitHub zipballs wrap content in a single `<repo>-<ref>/` directory; unwrap it cleanly."""
    if (directory / SKILL_FILENAME).is_file():
        return directory
    entries = [p for p in directory.iterdir() if not p.name.startswith(".") and p.name != "__MACOSX"]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return directory


class SkillInstaller:
    """Handles previewing, selectively installing, and updating Skills from GitHub, local folders, or ZIPs."""

    def __init__(self, manager: SkillManager, runtime_manager: RuntimeManager) -> None:
        self.manager = manager
        self.runtime_manager = runtime_manager
        self.staging_root = manager.skills_root / ".staging"

    def _cleanup_expired_staging(self) -> None:
        if not self.staging_root.is_dir():
            return
        now = time.time()
        for child in self.staging_root.iterdir():
            if not child.is_dir():
                continue
            try:
                age = now - child.stat().st_mtime
                if age > STAGING_TTL_SECONDS:
                    shutil.rmtree(child, ignore_errors=True)
            except OSError:
                continue

    async def _run_git(self, args: Sequence[str], *, cwd: Path, timeout_seconds: float = CLONE_TIMEOUT_SECONDS) -> tuple[int, str, str]:
        env = self.runtime_manager.build_clean_env(
            include_venv=False,
            env_overrides={"GIT_TERMINAL_PROMPT": "0"},
        )
        git_path = self.runtime_manager.probe_status().git.path
        if not git_path:
            return -1, "", "git not found"
        proc = await asyncio.create_subprocess_exec(
            git_path,
            *args,
            cwd=str(cwd),
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return -1, "", f"git 命令超时（>{timeout_seconds:.0f}s）。"
        return int(proc.returncode or 0), out_b.decode("utf-8", errors="replace"), err_b.decode("utf-8", errors="replace")

    async def _fetch_github_to_dir(
        self,
        parsed: ParsedGitHubUrl,
        dest_dir: Path,
    ) -> tuple[Path, str | None, str | None]:
        """Fetch a public GitHub repo into `dest_dir` via git clone (primary) or HTTP zip (fallback).

        Returns `(content_root, resolved_ref, commit_sha)`.
        """
        clone_dir = dest_dir / "repo"
        status = await asyncio.to_thread(self.runtime_manager.probe_status)

        if status.git.available and status.git.path:
            clone_args = ["clone", "--depth", "1"]
            if parsed.ref:
                clone_args.extend(["--branch", parsed.ref])
            clone_args.extend([f"{parsed.repo_url}.git", str(clone_dir)])
            rc, _out, err = await self._run_git(clone_args, cwd=dest_dir)
            if rc != 0 and parsed.ref:
                # Branch may be a commit SHA or tag that `--branch` couldn't directly clone; try default clone + fetch ref.
                shutil.rmtree(clone_dir, ignore_errors=True)
                rc2, _, _ = await self._run_git(["clone", "--depth", "1", f"{parsed.repo_url}.git", str(clone_dir)], cwd=dest_dir)
                if rc2 == 0:
                    rc_f, _, _ = await self._run_git(["fetch", "--depth", "1", "origin", parsed.ref], cwd=clone_dir)
                    if rc_f == 0:
                        rc, _, err = await self._run_git(["checkout", "FETCH_HEAD"], cwd=clone_dir)
            if rc == 0 and clone_dir.is_dir():
                _, sha_out, _ = await self._run_git(["rev-parse", "HEAD"], cwd=clone_dir)
                commit_sha = sha_out.strip() if re.fullmatch(r"[0-9a-f]{7,40}", sha_out.strip()) else None
                _, branch_out, _ = await self._run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=clone_dir)
                resolved_ref = parsed.ref or (branch_out.strip() if branch_out.strip() and branch_out.strip() != "HEAD" else None)
                return clone_dir, resolved_ref, commit_sha
            shutil.rmtree(clone_dir, ignore_errors=True)

        # Fallback: HTTP ZIP archive download via httpx
        zip_path = dest_dir / "archive.zip"
        extract_dir = dest_dir / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        candidate_refs = [parsed.ref] if parsed.ref else ["main", "master"]
        last_error = "无法下载 GitHub 仓库压缩包。"
        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
            for ref in candidate_refs:
                zip_url = f"{parsed.repo_url}/archive/{ref}.zip"
                try:
                    resp = await client.get(zip_url)
                except httpx.HTTPError as exc:
                    last_error = f"下载 GitHub 归档失败：{exc}"
                    continue
                if resp.status_code == 200:
                    zip_path.write_bytes(resp.content)
                    await asyncio.to_thread(_safe_extract_zip, zip_path, extract_dir)
                    content_root = _unwrap_single_root_dir(extract_dir)
                    return content_root, ref, None
                last_error = f"GitHub 归档返回 HTTP {resp.status_code}（ref={ref}）。"
        raise SkillError("SKILL_IMPORT_FAILED", last_error)

    def _scan_candidates(
        self,
        content_root: Path,
        *,
        subpath: str | None = None,
    ) -> tuple[ImportCandidate, ...]:
        search_root = content_root
        if subpath:
            target = (content_root / subpath).resolve()
            if not target.is_relative_to(content_root.resolve()) or not target.is_dir():
                raise SkillError("SKILL_NOT_FOUND", f"在仓库中未找到指定子目录：{subpath}")
            search_root = target

        skill_dirs = discover_skill_dirs(search_root, max_depth=4)
        if not skill_dirs:
            raise SkillError("SKILL_NOT_FOUND", "未在来源中发现包含 SKILL.md 的有效技能目录。")

        runtime_status = self.runtime_manager.probe_status()
        root_resolved = content_root.resolve()
        candidates: list[ImportCandidate] = []

        for sdir in skill_dirs:
            try:
                doc = parse_skill_directory(sdir)
            except SkillError:
                continue
            rel = sdir.resolve().relative_to(root_resolved).as_posix()
            resources = list_skill_resources(sdir)
            spec = extract_dependency_spec(sdir, doc)
            dep_status = evaluate_dependencies(spec, runtime_status)

            compat_notes = list(spec.notes)
            if any(hint in doc.body for hint in _CLAUDE_TOOL_HINTS):
                compat_notes.append("该技能包含其他平台工具名称提示，KnowledgeAgent 执行时将自动适配为本地通用工具（execute_command / read_file / write_file）。")

            existing = self.manager.store.find_by_name(doc.name)
            existing_id = existing[0].skill_id if existing else None

            candidates.append(ImportCandidate(
                relative_path=rel,
                name=doc.name,
                description=doc.description,
                license=doc.license,
                compatibility=doc.compatibility,
                resource_count=len(resources),
                dependency_status=dep_status,
                required_python_packages=spec.python_packages,
                required_node_tools=spec.node_tools,
                required_system_commands=spec.system_commands,
                warnings=doc.warnings,
                compatibility_notes=tuple(compat_notes),
                existing_skill_id=existing_id,
            ))

        if not candidates:
            raise SkillError("SKILL_PARSE_ERROR", "发现的 SKILL.md 均无法通过解析，请检查技能格式。")
        return tuple(candidates)

    async def preview_import(
        self,
        *,
        source_kind: ImportSourceKind,
        source: str,
        ref: str | None = None,
        subpath: str | None = None,
    ) -> ImportPreviewResult:
        """Stage a GitHub repo, local directory, or local ZIP and return discovered skill candidates."""
        await asyncio.to_thread(self._cleanup_expired_staging)
        preview_id = str(uuid4())
        stage_dir = self.staging_root / preview_id
        stage_dir.mkdir(parents=True, exist_ok=True)

        resolved_ref: str | None = None
        commit_sha: str | None = None
        effective_subpath = subpath.strip().strip("/") if isinstance(subpath, str) and subpath.strip() else None
        canonical_source_url = source.strip()

        try:
            if source_kind == "github":
                parsed = parse_github_url(source, explicit_ref=ref, explicit_subpath=effective_subpath)
                canonical_source_url = parsed.repo_url
                effective_subpath = parsed.subpath
                content_root, resolved_ref, commit_sha = await self._fetch_github_to_dir(parsed, stage_dir)
            elif source_kind == "local_dir":
                src_path = Path(source)
                if not src_path.is_absolute() or not src_path.is_dir():
                    raise SkillError("SKILL_INVALID_PATH", "请选择有效的本地技能文件夹。")
                canonical_source_url = str(src_path.resolve())
                content_root = stage_dir / "local"
                await asyncio.to_thread(_copy_skill_tree, src_path, content_root)
            elif source_kind == "local_zip":
                zip_path = Path(source)
                if not zip_path.is_absolute() or not zip_path.is_file():
                    raise SkillError("SKILL_INVALID_PATH", "请选择有效的本地 ZIP 技能压缩包。")
                canonical_source_url = str(zip_path.resolve())
                extracted = stage_dir / "extracted"
                extracted.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(_safe_extract_zip, zip_path, extracted)
                content_root = _unwrap_single_root_dir(extracted)
            else:
                raise SkillError("SKILL_INVALID_SOURCE", f"不支持的导入来源类型：{source_kind}")

            candidates = await asyncio.to_thread(self._scan_candidates, content_root, subpath=effective_subpath)

            meta = {
                "preview_id": preview_id,
                "source_kind": source_kind,
                "source_url": canonical_source_url,
                "resolved_ref": resolved_ref,
                "commit_sha": commit_sha,
                "subpath": effective_subpath,
                "content_root": str(content_root.resolve()),
                "candidates": [c.relative_path for c in candidates],
            }
            (stage_dir / "preview.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            return ImportPreviewResult(
                preview_id=preview_id,
                source_kind=source_kind,
                source_url=canonical_source_url,
                resolved_ref=resolved_ref,
                commit_sha=commit_sha,
                subpath=effective_subpath,
                candidates=candidates,
            )
        except Exception:
            shutil.rmtree(stage_dir, ignore_errors=True)
            raise

    async def install_from_preview(
        self,
        preview_id: str,
        selected_relative_paths: Sequence[str],
        *,
        enabled: bool = True,
        replace_existing: bool = True,
    ) -> list[SkillView]:
        """Install only the user-selected skills from a staged preview into `<app_data_dir>/skills/`."""
        if not preview_id or "/" in preview_id or ".." in preview_id:
            raise SkillError("SKILL_INVALID_SOURCE", "无效的导入预览 ID。")
        stage_dir = self.staging_root / preview_id
        meta_path = stage_dir / "preview.json"
        if not meta_path.is_file():
            raise SkillError("SKILL_PREVIEW_EXPIRED", "导入预览已过期或不存在，请重新获取技能列表。")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        content_root = Path(meta["content_root"]).resolve()
        allowed_paths = set(meta.get("candidates", []))
        wanted = [p.strip() for p in selected_relative_paths if p and p.strip()]
        if not wanted:
            raise SkillError("SKILL_INVALID_SOURCE", "请至少选择一个要安装的技能。")

        source_kind: ImportSourceKind = meta["source_kind"]
        source_url: str = meta["source_url"]
        resolved_ref: str | None = meta.get("resolved_ref")
        commit_sha: str | None = meta.get("commit_sha")

        def _do_install() -> list[SkillView]:
            self.manager.skills_root.mkdir(parents=True, exist_ok=True)
            runtime_status = self.runtime_manager.probe_status()
            installed: list[SkillView] = []

            for rel in wanted:
                if rel not in allowed_paths:
                    raise SkillError("SKILL_INVALID_PATH", f"所选技能路径不在预览候选列表中：{rel}")
                src_dir = (content_root / rel).resolve() if rel != "." else content_root
                if not src_dir.is_relative_to(content_root) or not (src_dir / SKILL_FILENAME).is_file():
                    raise SkillError("SKILL_NOT_FOUND", f"未找到候选技能目录：{rel}")

                doc = parse_skill_directory(src_dir)
                content_rev = compute_skill_revision(src_dir)
                revision = commit_sha[:12] if commit_sha else content_rev

                # Reuse existing skill_id when replacing same-name non-builtin skill
                skill_id = str(uuid4())
                if replace_existing:
                    existing = [s for s in self.manager.store.find_by_name(doc.name) if s.source_type != "builtin"]
                    if existing:
                        skill_id = existing[0].skill_id

                safe_dir_name = sanitize_filename(src_dir.name if rel != "." else doc.name, fallback=doc.name)
                rel_dir = f"{skill_id}/revisions/{revision}/{safe_dir_name}"
                dest_dir = self.manager.skills_root / rel_dir
                _copy_skill_tree(src_dir, dest_dir)

                # Copy top-level LICENSE from repo root if skill directory doesn't include one
                if not any(dest_dir.glob("LICENSE*")):
                    for lic in content_root.glob("LICENSE*"):
                        if lic.is_file():
                            shutil.copy2(lic, dest_dir / lic.name)
                            break

                stored_meta = dict(doc.metadata)
                stored_meta["_source"] = {
                    "kind": source_kind,
                    "repo_url": source_url,
                    "subpath": None if rel == "." else rel,
                    "ref": resolved_ref,
                    "commit_sha": commit_sha,
                    "content_revision": content_rev,
                }
                db_source_type = "github" if source_kind == "github" else "local"
                stored = self.manager.store.upsert_skill(
                    skill_id=skill_id,
                    name=doc.name,
                    description=doc.description,
                    source_type=db_source_type,
                    source_url=source_url,
                    revision=revision,
                    relative_dir=rel_dir,
                    enabled=enabled,
                    metadata=stored_meta,
                    preserve_enabled_if_exists=False,
                )
                installed.append(self.manager._build_skill_view(stored, runtime_status, parsed=doc, skill_dir=dest_dir))
            return installed

        try:
            return await asyncio.to_thread(_do_install)
        finally:
            shutil.rmtree(stage_dir, ignore_errors=True)

    async def update_skill(self, skill_id: str) -> SkillView:
        """Manually fetch a fresh revision for an installed GitHub/local skill and switch cleanly."""
        stored = self.manager.store.get_skill(skill_id)
        if stored is None:
            raise SkillError("SKILL_NOT_FOUND", f"未找到技能：{skill_id}")
        if stored.source_type == "builtin":
            raise SkillError("SKILL_UPDATE_UNSUPPORTED", "内置技能随应用版本自动同步，无需手动更新。")

        src_meta = stored.metadata.get("_source") if isinstance(stored.metadata, dict) else None
        source_kind: ImportSourceKind = (
            src_meta.get("kind", "github" if stored.source_type == "github" else "local_dir")
            if isinstance(src_meta, dict)
            else ("github" if stored.source_type == "github" else "local_dir")
        )
        source_url = (src_meta.get("repo_url") if isinstance(src_meta, dict) else None) or stored.source_url
        if not source_url:
            raise SkillError("SKILL_UPDATE_UNSUPPORTED", "该技能未记录更新来源地址。")
        subpath = src_meta.get("subpath") if isinstance(src_meta, dict) else None
        ref = src_meta.get("ref") if isinstance(src_meta, dict) else None

        preview = await self.preview_import(
            source_kind=source_kind,
            source=source_url,
            ref=ref,
            subpath=subpath,
        )
        stage_dir = self.staging_root / preview.preview_id
        try:
            # Pick matching candidate by subpath or skill name
            chosen = next(
                (c for c in preview.candidates if c.name.lower() == stored.name.lower()),
                preview.candidates[0],
            )
            views = await self.install_from_preview(
                preview.preview_id,
                [chosen.relative_path],
                enabled=stored.enabled,
                replace_existing=True,
            )
            return views[0]
        finally:
            shutil.rmtree(stage_dir, ignore_errors=True)
