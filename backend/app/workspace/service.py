"""Session-level persistent workspace and per-run working directory service.

Directory layout under `<app_data_dir>`:
```text
workspaces/
  <session_id>/
    inputs/
      <file_id>/<filename>      # Immutable user-imported original copies
    runs/
      <run_id>/
        work/                   # Per-run scripts, temp files, editable copies
        output/                 # Per-run candidate artifacts
    artifacts/
      <artifact_id>/<filename>  # Versioned durable artifacts
```

Boundary note:
Path validation inside `WorkspaceService` prevents accidental path traversal
and protects original inputs and published artifacts from in-place overwrite.
It is a tool-layer boundary, not an OS sandbox; general Python/Shell execution
runs with the current user's OS privileges.
"""

from __future__ import annotations

import mimetypes
import os
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from uuid import UUID, uuid4

from app.storage.artifacts import StoredWorkspaceFile, WorkspaceArtifactStore
from app.storage.business import SessionNotFound

MAX_IMPORT_BYTES = 100 * 1024 * 1024  # 100 MiB
MAX_TEXT_READ_BYTES = 2 * 1024 * 1024  # 2 MiB
DEFAULT_READ_MAX_CHARS = 32_000
MAX_LIST_ENTRIES = 500
MAX_PATH_BYTES = 1024
UNSAFE_CATEGORIES = {"Cc", "Cf", "Cs"}
LINE_ENDING = re.compile(r"\r\n|[\r\n]")

KNOWN_MEDIA_TYPES: dict[str, str] = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".pdf": "application/pdf",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".mmd": "text/plain; charset=utf-8",
    ".mermaid": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".markdown": "text/markdown; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".csv": "text/csv; charset=utf-8",
    ".yaml": "text/yaml; charset=utf-8",
    ".yml": "text/yaml; charset=utf-8",
    ".py": "text/x-python; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".ts": "text/typescript; charset=utf-8",
    ".sh": "text/x-shellscript; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".xml": "application/xml; charset=utf-8",
}

TEXT_EXTENSIONS: frozenset[str] = frozenset({
    ".txt", ".md", ".markdown", ".mmd", ".mermaid", ".json", ".jsonl",
    ".yaml", ".yml", ".toml", ".csv", ".tsv", ".xml", ".html", ".htm",
    ".css", ".svg", ".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx",
    ".sh", ".bash", ".zsh", ".sql", ".ini", ".cfg", ".conf", ".env",
    ".log", ".tex", ".bib", ".dot", ".gv",
})


class WorkspaceError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class WorkspaceEntry:
    relative_path: str
    name: str
    kind: str  # "file" | "directory"
    size_bytes: int | None
    media_type: str | None


@dataclass(frozen=True, slots=True)
class WorkspaceReadResult:
    relative_path: str
    is_binary: bool
    media_type: str
    size_bytes: int
    content: str | None = None
    total_lines: int | None = None
    offset_line: int | None = None
    returned_lines: int | None = None
    next_offset_line: int | None = None
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class WorkspaceWriteResult:
    relative_path: str
    size_bytes: int
    media_type: str
    absolute_path: str


def _validate_id(value: str, label: str) -> str:
    try:
        normalized = str(UUID(value))
    except (TypeError, ValueError):
        raise WorkspaceError("WORKSPACE_INVALID_ID", f"无效的 {label}。") from None
    if normalized != value:
        raise WorkspaceError("WORKSPACE_INVALID_ID", f"无效的 {label}。")
    return normalized


def sanitize_filename(name: str, fallback: str = "file") -> str:
    """Keep Unicode/Chinese filenames intact while removing separators and control chars."""
    base = Path(name.replace("\\", "/")).name
    cleaned = "".join(
        ch for ch in base
        if ch not in ("/", "\\", "\x00") and unicodedata.category(ch) not in UNSAFE_CATEGORIES
    ).strip()
    while cleaned in ("", ".", ".."):
        return fallback
    # Keep filename well within POSIX 255-byte component limit.
    encoded = cleaned.encode("utf-8")
    if len(encoded) > 200:
        stem = Path(cleaned).stem
        suffix = Path(cleaned).suffix
        suffix_bytes = suffix.encode("utf-8")[:40]
        max_stem_bytes = max(16, 200 - len(suffix_bytes))
        trimmed_stem = stem.encode("utf-8")[:max_stem_bytes].decode("utf-8", errors="ignore").rstrip()
        cleaned = f"{trimmed_stem or fallback}{suffix_bytes.decode('utf-8', errors='ignore')}"
    return cleaned


def guess_media_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in KNOWN_MEDIA_TYPES:
        return KNOWN_MEDIA_TYPES[suffix]
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


def parse_relative_parts(relative_path: str) -> tuple[str, ...]:
    """Parse and validate a POSIX-style relative path without traversal."""
    if not isinstance(relative_path, str):
        raise WorkspaceError("WORKSPACE_PATH_INVALID", "路径必须是字符串。")
    normalized = relative_path.strip()
    if not normalized or normalized == ".":
        return ()
    if normalized.startswith("/") or "\\" in normalized or PureWindowsPath(normalized).drive:
        raise WorkspaceError("WORKSPACE_PATH_INVALID", "必须使用工作区内的相对路径。")
    if len(normalized.encode("utf-8")) > MAX_PATH_BYTES:
        raise WorkspaceError("WORKSPACE_PATH_INVALID", "路径长度超过限制。")
    if any(unicodedata.category(ch) in UNSAFE_CATEGORIES for ch in normalized):
        raise WorkspaceError("WORKSPACE_PATH_INVALID", "路径包含非法控制字符。")
    parts = tuple(p for p in normalized.split("/") if p != "")
    if len(parts) > 32 or any(part in (".", "..") for part in parts):
        raise WorkspaceError("WORKSPACE_PATH_INVALID", "路径不允许包含 '..' 或无效段。")
    if any(len(part.encode("utf-8")) > 255 for part in parts):
        raise WorkspaceError("WORKSPACE_PATH_INVALID", "文件名段长度超过 255 字节。")
    return parts


def resolve_within(base_dir: Path, relative_path: str, *, allow_missing: bool = False) -> Path:
    """Resolve a relative path strictly inside `base_dir`, blocking symlink escapes."""
    parts = parse_relative_parts(relative_path)
    base_resolved = base_dir.resolve()
    candidate = base_resolved.joinpath(*parts) if parts else base_resolved

    # Check every existing ancestor and target for symlinks escaping base_dir.
    cur = base_resolved
    for part in parts:
        cur = cur / part
        if cur.is_symlink():
            target = cur.resolve()
            if not target.is_relative_to(base_resolved):
                raise WorkspaceError("WORKSPACE_PATH_INVALID", "不允许通过软链接访问工作区外部路径。")

    resolved = candidate.resolve()
    if not resolved.is_relative_to(base_resolved):
        raise WorkspaceError("WORKSPACE_PATH_INVALID", "目标路径超出工作区范围。")
    if not allow_missing and not resolved.exists():
        raise WorkspaceError("WORKSPACE_NOT_FOUND", f"未找到工作区路径：{relative_path or '.'}")
    return resolved


class WorkspaceService:
    """Manages session workspaces, input copies, per-run directories, and safe file operations."""

    def __init__(self, app_data_dir: Path, store: WorkspaceArtifactStore) -> None:
        self.app_data_dir = app_data_dir
        self.workspaces_root = app_data_dir / "workspaces"
        self.store = store

    # ------------------------------------------------------------------
    # Directory derivation (always derived from validated IDs)
    # ------------------------------------------------------------------

    def session_root(self, session_id: str) -> Path:
        sid = _validate_id(session_id, "session_id")
        return self.workspaces_root / sid

    def inputs_dir(self, session_id: str) -> Path:
        return self.session_root(session_id) / "inputs"

    def runs_dir(self, session_id: str) -> Path:
        return self.session_root(session_id) / "runs"

    def run_root(self, session_id: str, run_id: str) -> Path:
        rid = _validate_id(run_id, "run_id")
        return self.runs_dir(session_id) / rid

    def run_work_dir(self, session_id: str, run_id: str) -> Path:
        return self.run_root(session_id, run_id) / "work"

    def run_output_dir(self, session_id: str, run_id: str) -> Path:
        return self.run_root(session_id, run_id) / "output"

    def artifacts_dir(self, session_id: str) -> Path:
        return self.session_root(session_id) / "artifacts"

    def ensure_session_dirs(self, session_id: str) -> Path:
        root = self.session_root(session_id)
        for directory in (root, self.inputs_dir(session_id), self.runs_dir(session_id), self.artifacts_dir(session_id)):
            directory.mkdir(parents=True, exist_ok=True)
        return root

    def ensure_run_dirs(self, session_id: str, run_id: str) -> tuple[Path, Path]:
        self.ensure_session_dirs(session_id)
        work = self.run_work_dir(session_id, run_id)
        output = self.run_output_dir(session_id, run_id)
        work.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=True)
        return work, output

    # ------------------------------------------------------------------
    # Input file import (preserves original user file untouched)
    # ------------------------------------------------------------------

    def import_local_file(
        self,
        session_id: str,
        source_path: Path | str,
        *,
        filename: str | None = None,
        max_bytes: int = MAX_IMPORT_BYTES,
    ) -> StoredWorkspaceFile:
        sid = _validate_id(session_id, "session_id")
        try:
            self.store.require_session(sid)
        except SessionNotFound:
            raise WorkspaceError("WORKSPACE_SESSION_NOT_FOUND", "未找到该会话。") from None

        src = Path(source_path)
        if not src.is_absolute():
            raise WorkspaceError("WORKSPACE_PATH_INVALID", "导入源文件必须为绝对路径。")
        try:
            st = src.stat()
        except OSError:
            raise WorkspaceError("WORKSPACE_NOT_FOUND", "无法访问要导入的本地文件。") from None
        if not src.is_file():
            raise WorkspaceError("WORKSPACE_PATH_INVALID", "仅支持导入普通文件。")
        if st.st_size > max_bytes:
            raise WorkspaceError(
                "WORKSPACE_FILE_TOO_LARGE",
                f"文件大小超过上限（{max_bytes // (1024 * 1024)} MiB）。",
            )

        safe_name = sanitize_filename(filename or src.name, fallback="attachment")
        file_id = str(uuid4())
        rel_path = f"inputs/{file_id}/{safe_name}"
        dest_dir = self.inputs_dir(sid) / file_id
        dest_path = dest_dir / safe_name

        self.ensure_session_dirs(sid)
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dest_path)
            actual_size = dest_path.stat().st_size
            media_type = guess_media_type(safe_name)
            return self.store.add_workspace_file(
                file_id=file_id,
                session_id=sid,
                original_name=safe_name,
                relative_path=rel_path,
                media_type=media_type,
                size_bytes=actual_size,
            )
        except WorkspaceError:
            shutil.rmtree(dest_dir, ignore_errors=True)
            raise
        except Exception as exc:
            shutil.rmtree(dest_dir, ignore_errors=True)
            raise WorkspaceError("WORKSPACE_IO_ERROR", f"导入工作区文件失败：{exc}") from None

    def import_bytes(
        self,
        session_id: str,
        filename: str,
        data: bytes,
        *,
        media_type: str | None = None,
        max_bytes: int = MAX_IMPORT_BYTES,
    ) -> StoredWorkspaceFile:
        sid = _validate_id(session_id, "session_id")
        try:
            self.store.require_session(sid)
        except SessionNotFound:
            raise WorkspaceError("WORKSPACE_SESSION_NOT_FOUND", "未找到该会话。") from None

        if len(data) > max_bytes:
            raise WorkspaceError(
                "WORKSPACE_FILE_TOO_LARGE",
                f"文件大小超过上限（{max_bytes // (1024 * 1024)} MiB）。",
            )
        safe_name = sanitize_filename(filename, fallback="attachment")
        file_id = str(uuid4())
        rel_path = f"inputs/{file_id}/{safe_name}"
        dest_dir = self.inputs_dir(sid) / file_id
        dest_path = dest_dir / safe_name

        self.ensure_session_dirs(sid)
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            dest_path.write_bytes(data)
            resolved_media = media_type or guess_media_type(safe_name)
            return self.store.add_workspace_file(
                file_id=file_id,
                session_id=sid,
                original_name=safe_name,
                relative_path=rel_path,
                media_type=resolved_media,
                size_bytes=len(data),
            )
        except Exception as exc:
            shutil.rmtree(dest_dir, ignore_errors=True)
            raise WorkspaceError("WORKSPACE_IO_ERROR", f"写入工作区文件失败：{exc}") from None

    def list_files(self, session_id: str) -> list[StoredWorkspaceFile]:
        sid = _validate_id(session_id, "session_id")
        try:
            return self.store.list_workspace_files(sid)
        except SessionNotFound:
            raise WorkspaceError("WORKSPACE_SESSION_NOT_FOUND", "未找到该会话。") from None

    def get_file_path(self, session_id: str, file_id: str) -> tuple[StoredWorkspaceFile, Path]:
        sid = _validate_id(session_id, "session_id")
        fid = _validate_id(file_id, "file_id")
        try:
            record = self.store.get_workspace_file(sid, fid)
        except SessionNotFound:
            raise WorkspaceError("WORKSPACE_SESSION_NOT_FOUND", "未找到该会话。") from None
        if record is None:
            raise WorkspaceError("WORKSPACE_NOT_FOUND", "未找到该工作区附件。")
        path = resolve_within(self.session_root(sid), record.relative_path, allow_missing=True)
        return record, path

    def delete_file(self, session_id: str, file_id: str) -> None:
        sid = _validate_id(session_id, "session_id")
        fid = _validate_id(file_id, "file_id")
        try:
            deleted = self.store.delete_workspace_file(sid, fid)
        except SessionNotFound:
            raise WorkspaceError("WORKSPACE_SESSION_NOT_FOUND", "未找到该会话。") from None
        if deleted is None:
            raise WorkspaceError("WORKSPACE_NOT_FOUND", "未找到该工作区附件。")
        file_dir = self.inputs_dir(sid) / fid
        if file_dir.exists() and not file_dir.is_symlink():
            shutil.rmtree(file_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Workspace file operations (list / read / write / copy)
    # ------------------------------------------------------------------

    def _base_dir(self, session_id: str, run_id: str | None) -> Path:
        if run_id is not None:
            self.ensure_run_dirs(session_id, run_id)
            return self.run_root(session_id, run_id)
        self.ensure_session_dirs(session_id)
        return self.session_root(session_id)

    def list_directory(
        self,
        session_id: str,
        *,
        run_id: str | None = None,
        relative_path: str = "",
        max_entries: int = MAX_LIST_ENTRIES,
    ) -> list[WorkspaceEntry]:
        base = self._base_dir(session_id, run_id)
        target = resolve_within(base, relative_path)
        if not target.is_dir():
            raise WorkspaceError("WORKSPACE_PATH_INVALID", "目标路径不是目录。")
        base_resolved = base.resolve()
        entries: list[WorkspaceEntry] = []
        with os.scandir(target) as it:
            sorted_entries = sorted(it, key=lambda e: (not e.is_dir(follow_symlinks=False), e.name.casefold(), e.name))
            for entry in sorted_entries[:max_entries]:
                if entry.name.startswith("."):
                    continue
                entry_path = Path(entry.path)
                rel = entry_path.resolve().relative_to(base_resolved).as_posix()
                if entry.is_dir(follow_symlinks=False):
                    entries.append(WorkspaceEntry(
                        relative_path=rel,
                        name=entry.name,
                        kind="directory",
                        size_bytes=None,
                        media_type=None,
                    ))
                elif entry.is_file(follow_symlinks=False):
                    size = entry.stat(follow_symlinks=False).st_size
                    entries.append(WorkspaceEntry(
                        relative_path=rel,
                        name=entry.name,
                        kind="file",
                        size_bytes=size,
                        media_type=guess_media_type(entry.name),
                    ))
        return entries

    def read_file(
        self,
        session_id: str,
        relative_path: str,
        *,
        run_id: str | None = None,
        offset_line: int = 0,
        limit_lines: int = 200,
        max_chars: int = DEFAULT_READ_MAX_CHARS,
    ) -> WorkspaceReadResult:
        """Read text file with pagination, or return structured metadata for binary files."""
        base = self._base_dir(session_id, run_id)
        target = resolve_within(base, relative_path)
        if not target.is_file():
            raise WorkspaceError("WORKSPACE_PATH_INVALID", "目标路径不是普通文件。")

        size_bytes = target.stat().st_size
        media_type = guess_media_type(target.name)
        rel = target.relative_to(base.resolve()).as_posix()
        suffix = target.suffix.lower()

        # Quickly classify obvious binary files (DOCX, PNG, PDF, etc.) without trying to decode.
        if suffix not in TEXT_EXTENSIONS and not media_type.startswith("text/") and media_type != "application/json":
            return WorkspaceReadResult(
                relative_path=rel,
                is_binary=True,
                media_type=media_type,
                size_bytes=size_bytes,
            )

        if size_bytes > MAX_TEXT_READ_BYTES:
            raise WorkspaceError(
                "WORKSPACE_FILE_TOO_LARGE",
                f"文本文件超过单次读取上限（{MAX_TEXT_READ_BYTES // (1024 * 1024)} MiB），请通过脚本分段处理。",
            )

        raw = target.read_bytes()
        if b"\x00" in raw[:8192]:
            return WorkspaceReadResult(
                relative_path=rel,
                is_binary=True,
                media_type=media_type,
                size_bytes=size_bytes,
            )

        decoded: str | None = None
        for encoding in ("utf-8-sig", "gbk"):
            try:
                decoded = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        if decoded is None:
            return WorkspaceReadResult(
                relative_path=rel,
                is_binary=True,
                media_type=media_type,
                size_bytes=size_bytes,
            )

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
        return WorkspaceReadResult(
            relative_path=rel,
            is_binary=False,
            media_type=media_type,
            size_bytes=size_bytes,
            content="\n".join(selected),
            total_lines=len(lines),
            offset_line=start,
            returned_lines=len(selected),
            next_offset_line=next_line,
            truncated=clipped or next_line is not None,
        )

    def write_file(
        self,
        session_id: str,
        relative_path: str,
        content: str,
        *,
        run_id: str,
        overwrite: bool = True,
    ) -> WorkspaceWriteResult:
        """Create or update a UTF-8 text file inside a run's working directory."""
        run_base = self._base_dir(session_id, run_id)
        parts = parse_relative_parts(relative_path)
        if not parts:
            raise WorkspaceError("WORKSPACE_PATH_INVALID", "写入文件路径不能为空。")
        target = resolve_within(run_base, relative_path, allow_missing=True)
        if target.exists() and target.is_dir():
            raise WorkspaceError("WORKSPACE_PATH_INVALID", "目标路径是目录，无法作为文件写入。")
        if target.exists() and not overwrite:
            raise WorkspaceError("WORKSPACE_PATH_EXISTS", f"目标文件已存在：{relative_path}")

        target.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode("utf-8")
        target.write_bytes(data)
        rel = target.relative_to(run_base.resolve()).as_posix()
        return WorkspaceWriteResult(
            relative_path=rel,
            size_bytes=len(data),
            media_type=guess_media_type(target.name),
            absolute_path=str(target),
        )

    def copy_file(
        self,
        session_id: str,
        source_relative_path: str,
        dest_relative_path: str,
        *,
        run_id: str,
        overwrite: bool = True,
    ) -> Path:
        """Copy a file from anywhere inside the session workspace (e.g. inputs/ or artifacts/) into the run directory."""
        session_base = self.ensure_session_dirs(session_id)
        run_base = self._base_dir(session_id, run_id)

        # Allow source relative to run_root first, then session_root (for inputs/ and artifacts/).
        try:
            src = resolve_within(run_base, source_relative_path)
        except WorkspaceError:
            src = resolve_within(session_base, source_relative_path)
        if not src.is_file():
            raise WorkspaceError("WORKSPACE_PATH_INVALID", "复制源路径不是普通文件。")

        dest_parts = parse_relative_parts(dest_relative_path)
        if not dest_parts:
            raise WorkspaceError("WORKSPACE_PATH_INVALID", "复制目标路径不能为空。")
        dest = resolve_within(run_base, dest_relative_path, allow_missing=True)
        if dest.exists() and dest.is_dir():
            dest = dest / src.name
        if dest.exists() and not overwrite:
            raise WorkspaceError("WORKSPACE_PATH_EXISTS", f"目标文件已存在：{dest_relative_path}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return dest

    def copy_input_to_run(
        self,
        session_id: str,
        run_id: str,
        file_id: str,
        *,
        dest_name: str | None = None,
    ) -> Path:
        """Copy an imported session attachment into `runs/<run_id>/work/` without altering the original copy."""
        record, src_path = self.get_file_path(session_id, file_id)
        if not src_path.is_file():
            raise WorkspaceError("WORKSPACE_NOT_FOUND", f"附件文件已缺失：{record.original_name}")
        work_dir, _ = self.ensure_run_dirs(session_id, run_id)
        target_name = sanitize_filename(dest_name or record.original_name, fallback="input_file")
        dest = resolve_within(work_dir, target_name, allow_missing=True)
        shutil.copy2(src_path, dest)
        return dest

    def copy_artifact_to_run(
        self,
        session_id: str,
        run_id: str,
        artifact_id: str,
        *,
        dest_name: str | None = None,
    ) -> Path:
        """Copy a previously published session artifact into `runs/<run_id>/work/` for cross-turn editing."""
        sid = _validate_id(session_id, "session_id")
        aid = _validate_id(artifact_id, "artifact_id")
        try:
            artifact = self.store.get_artifact(sid, aid, include_non_published=False)
        except SessionNotFound:
            raise WorkspaceError("WORKSPACE_SESSION_NOT_FOUND", "未找到该会话。") from None
        if artifact is None:
            raise WorkspaceError("WORKSPACE_NOT_FOUND", "未找到该历史产物。")
        src_path = resolve_within(self.session_root(sid), artifact.relative_path, allow_missing=True)
        if not src_path.is_file():
            raise WorkspaceError("WORKSPACE_NOT_FOUND", f"历史产物文件已缺失：{artifact.filename}")
        work_dir, _ = self.ensure_run_dirs(sid, run_id)
        target_name = sanitize_filename(dest_name or artifact.filename, fallback="artifact_file")
        dest = resolve_within(work_dir, target_name, allow_missing=True)
        shutil.copy2(src_path, dest)
        return dest

    def delete_session_workspace(self, session_id: str) -> None:
        """Best-effort cleanup of a deleted session's workspace directory."""
        root = self.session_root(session_id)
        if root.exists() and not root.is_symlink():
            shutil.rmtree(root, ignore_errors=True)
