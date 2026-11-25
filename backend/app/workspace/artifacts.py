"""Versioned artifact staging, publication, lookup, and controlled content reading."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from app.storage.artifacts import WorkspaceArtifactStore
from app.storage.business import ArtifactStatus, SessionNotFound, StoredArtifact
from app.workspace.service import (
    WorkspaceError,
    WorkspaceService,
    guess_media_type,
    resolve_within,
    sanitize_filename,
)

MAX_ARTIFACT_BYTES = 100 * 1024 * 1024  # 100 MiB
MAX_PREVIEW_READ_BYTES = 20 * 1024 * 1024  # 20 MiB


@dataclass(frozen=True, slots=True)
class ArtifactView:
    artifact_id: str
    session_id: str
    run_id: str
    filename: str
    title: str | None
    relative_path: str
    media_type: str
    size_bytes: int
    status: ArtifactStatus
    created_at: str
    file_exists: bool

    @classmethod
    def from_stored(cls, stored: StoredArtifact, *, file_exists: bool) -> ArtifactView:
        return cls(
            artifact_id=stored.artifact_id,
            session_id=stored.session_id,
            run_id=stored.run_id,
            filename=stored.filename,
            title=stored.title,
            relative_path=stored.relative_path,
            media_type=stored.media_type,
            size_bytes=stored.size_bytes,
            status=stored.status,
            created_at=stored.created_at,
            file_exists=file_exists,
        )


class ArtifactService:
    """Stages candidate run outputs into durable versioned artifacts and serves published records."""

    def __init__(self, workspace: WorkspaceService, store: WorkspaceArtifactStore) -> None:
        self.workspace = workspace
        self.store = store

    def _artifact_disk_path(self, artifact: StoredArtifact) -> Path:
        return resolve_within(
            self.workspace.session_root(artifact.session_id),
            artifact.relative_path,
            allow_missing=True,
        )

    def stage_artifact(
        self,
        session_id: str,
        run_id: str,
        relative_path: str,
        *,
        title: str | None = None,
        filename: str | None = None,
        max_bytes: int = MAX_ARTIFACT_BYTES,
    ) -> StoredArtifact:
        """Validate a generated file under `runs/<run_id>/`, copy to `artifacts/<artifact_id>/`, and record `staged`."""
        run_base = self.workspace.run_root(session_id, run_id)
        if not run_base.is_dir():
            raise WorkspaceError("WORKSPACE_NOT_FOUND", "当前任务的工作目录不存在。")

        # Accept either explicit `output/foo.docx` / `work/foo.docx` or bare `foo.docx` inside `output/`.
        src: Path | None = None
        last_err: WorkspaceError | None = None
        for candidate in (relative_path, f"output/{relative_path}"):
            try:
                resolved = resolve_within(run_base, candidate)
                if resolved.is_file():
                    src = resolved
                    break
            except WorkspaceError as err:
                last_err = err
        if src is None:
            if last_err is not None and last_err.code == "WORKSPACE_PATH_INVALID":
                raise last_err
            raise WorkspaceError("WORKSPACE_NOT_FOUND", f"未找到可发布的候选产物文件：{relative_path}")
        if src.is_symlink():
            raise WorkspaceError("WORKSPACE_PATH_INVALID", "不允许将软链接注册为产物。")

        st = src.stat()
        if st.st_size <= 0:
            raise WorkspaceError("WORKSPACE_EMPTY_ARTIFACT", f"候选产物文件为空，无法发布：{src.name}")
        if st.st_size > max_bytes:
            raise WorkspaceError(
                "WORKSPACE_FILE_TOO_LARGE",
                f"产物文件大小超过上限（{max_bytes // (1024 * 1024)} MiB）。",
            )

        safe_name = sanitize_filename(filename or src.name, fallback="artifact")
        artifact_id = str(uuid4())
        rel_path = f"artifacts/{artifact_id}/{safe_name}"
        dest_dir = self.workspace.artifacts_dir(session_id) / artifact_id
        dest_path = dest_dir / safe_name

        self.workspace.ensure_session_dirs(session_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dest_path)
            size_bytes = dest_path.stat().st_size
            media_type = guess_media_type(safe_name)
            return self.store.create_staged_artifact(
                artifact_id=artifact_id,
                session_id=session_id,
                run_id=run_id,
                filename=safe_name,
                title=(title.strip() if isinstance(title, str) and title.strip() else None),
                relative_path=rel_path,
                media_type=media_type,
                size_bytes=size_bytes,
            )
        except SessionNotFound:
            shutil.rmtree(dest_dir, ignore_errors=True)
            raise WorkspaceError("WORKSPACE_SESSION_NOT_FOUND", "未找到该会话。") from None
        except ValueError as err:
            shutil.rmtree(dest_dir, ignore_errors=True)
            raise WorkspaceError("WORKSPACE_INVALID_RUN_STATE", str(err)) from None
        except WorkspaceError:
            shutil.rmtree(dest_dir, ignore_errors=True)
            raise
        except Exception as exc:
            shutil.rmtree(dest_dir, ignore_errors=True)
            raise WorkspaceError("WORKSPACE_IO_ERROR", f"登记产物失败：{exc}") from None

    def publish_run_artifacts(self, session_id: str, run_id: str) -> list[StoredArtifact]:
        try:
            return self.store.publish_run_artifacts(session_id, run_id)
        except SessionNotFound:
            raise WorkspaceError("WORKSPACE_SESSION_NOT_FOUND", "未找到该会话。") from None

    def discard_run_artifacts(self, session_id: str, run_id: str) -> int:
        try:
            return self.store.discard_run_artifacts(session_id, run_id)
        except SessionNotFound:
            raise WorkspaceError("WORKSPACE_SESSION_NOT_FOUND", "未找到该会话。") from None

    def list_artifacts(
        self,
        session_id: str,
        *,
        run_id: str | None = None,
        include_non_published: bool = False,
    ) -> list[ArtifactView]:
        try:
            records = self.store.list_artifacts(
                session_id,
                run_id=run_id,
                status=None if include_non_published else "published",
            )
        except SessionNotFound:
            raise WorkspaceError("WORKSPACE_SESSION_NOT_FOUND", "未找到该会话。") from None
        views: list[ArtifactView] = []
        for rec in records:
            disk_path = self._artifact_disk_path(rec)
            views.append(ArtifactView.from_stored(rec, file_exists=disk_path.is_file()))
        return views

    def get_artifact(
        self,
        session_id: str,
        artifact_id: str,
        *,
        include_non_published: bool = False,
    ) -> tuple[ArtifactView, Path]:
        try:
            stored = self.store.get_artifact(
                session_id,
                artifact_id,
                include_non_published=include_non_published,
            )
        except SessionNotFound:
            raise WorkspaceError("WORKSPACE_SESSION_NOT_FOUND", "未找到该会话。") from None
        if stored is None:
            raise WorkspaceError("WORKSPACE_NOT_FOUND", "未找到该产物记录。")
        disk_path = self._artifact_disk_path(stored)
        return ArtifactView.from_stored(stored, file_exists=disk_path.is_file()), disk_path

    def read_artifact_bytes(
        self,
        session_id: str,
        artifact_id: str,
        *,
        max_bytes: int = MAX_PREVIEW_READ_BYTES,
    ) -> tuple[ArtifactView, bytes]:
        view, disk_path = self.get_artifact(session_id, artifact_id, include_non_published=False)
        if not view.file_exists or not disk_path.is_file():
            raise WorkspaceError("WORKSPACE_ARTIFACT_MISSING", f"产物文件已从磁盘缺失：{view.filename}")
        size = disk_path.stat().st_size
        if size > max_bytes:
            raise WorkspaceError(
                "WORKSPACE_FILE_TOO_LARGE",
                f"产物超过在线预览大小上限（{max_bytes // (1024 * 1024)} MiB），请直接用系统应用打开或另存为。",
            )
        return view, disk_path.read_bytes()

    def cleanup_orphan_artifact_dirs(self, session_id: str) -> int:
        """Remove artifact directories on disk that are not referenced in SQLite."""
        try:
            known = self.store.list_known_artifact_ids(session_id)
        except SessionNotFound:
            return 0
        artifacts_dir = self.workspace.artifacts_dir(session_id)
        if not artifacts_dir.is_dir() or artifacts_dir.is_symlink():
            return 0
        removed = 0
        for child in artifacts_dir.iterdir():
            if child.is_dir() and not child.is_symlink() and child.name not in known:
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
        return removed
