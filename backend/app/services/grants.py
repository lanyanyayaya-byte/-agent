import os
import stat
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Literal
from uuid import UUID, uuid4

from app.security import ApiError
from app.storage.business import utc_now

GrantScope = Literal["file", "directory"]
GrantLifetime = Literal["application_session"]
TEXT_SUFFIXES = {".txt", ".md"}
MACOS_SYSTEM_LINKS = {"etc": "private/etc", "tmp": "private/tmp", "var": "private/var"}


@dataclass(frozen=True)
class GrantView:
    grant_id: UUID
    session_id: UUID
    scope: GrantScope
    display_name: str
    operations: tuple[Literal["read"], ...] = ("read",)


@dataclass(frozen=True)
class GrantRecord:
    view: GrantView
    path: Path
    device: int
    inode: int
    birthtime: float | None
    created_at: str
    lifetime: GrantLifetime = "application_session"


def _inspect_resource(raw_path: str, scope: GrantScope) -> tuple[Path, os.stat_result]:
    path = Path(raw_path)
    if not path.is_absolute() or ".." in path.parts or len(path.parts) < 2 or path.anchor == "//":
        raise ApiError(400, "INVALID_REQUEST", "请选择有效的本地文件或文件夹。")

    try:
        # macOS exposes these fixed system paths through links at the volume root.
        first = path.parts[1]
        alias = MACOS_SYSTEM_LINKS.get(first) if sys.platform == "darwin" and path.anchor == "/" else None
        if alias and Path("/", first).is_symlink():
            if os.readlink(Path("/", first)) != alias:
                raise OSError
            path = Path("/", alias, *path.parts[2:])

        current = Path(path.anchor)
        for part in path.parts[1:-1]:
            current /= part
            metadata = current.lstat()
            if not stat.S_ISDIR(metadata.st_mode) or getattr(metadata, "st_reparse_tag", 0):
                raise OSError
        metadata = (current / path.name).lstat()
        if getattr(metadata, "st_reparse_tag", 0):
            raise OSError
        if scope == "file":
            if not stat.S_ISREG(metadata.st_mode) or path.suffix.lower() not in TEXT_SUFFIXES:
                raise OSError
            access = os.R_OK
        else:
            if not stat.S_ISDIR(metadata.st_mode):
                raise OSError
            access = os.R_OK | os.X_OK

        resolved = path.resolve(strict=True)
        verified = resolved.lstat()
        if resolved != path or not os.path.samestat(metadata, verified) or not os.access(resolved, access):
            raise OSError
        return resolved, verified
    except (OSError, RuntimeError, ValueError):
        raise ApiError(403, "FILE_PERMISSION_DENIED", "无法授权此资源。请选择可访问的 TXT、Markdown 文件或文件夹。") from None


class GrantRegistry:
    """Active grants belong to one Sidecar instance and live only in memory."""

    def __init__(self) -> None:
        self._records: dict[UUID, GrantRecord] = {}
        self._lock = Lock()

    def create(self, session_id: UUID, scope: GrantScope, path: str) -> GrantView:
        resolved, metadata = _inspect_resource(path, scope)
        birthtime = getattr(metadata, "st_birthtime", None)
        name = "".join(char for char in resolved.name if unicodedata.category(char) not in {"Cc", "Cf", "Cs"})
        with self._lock:
            for record in self._records.values():
                if (record.view.session_id == session_id and record.view.scope == scope and record.path == resolved
                        and (record.device, record.inode, record.birthtime)
                        == (metadata.st_dev, metadata.st_ino, birthtime)):
                    return record.view
            view = GrantView(uuid4(), session_id, scope, name.strip() or ("文件" if scope == "file" else "文件夹"))
            self._records[view.grant_id] = GrantRecord(
                view, resolved, metadata.st_dev, metadata.st_ino, birthtime, utc_now(),
            )
            return view

    def list(self, session_id: UUID) -> list[GrantView]:
        with self._lock:
            return [record.view for record in self._records.values() if record.view.session_id == session_id]

    def find_read(self, session_id: UUID, grant_id: UUID) -> GrantRecord | None:
        with self._lock:
            record = self._records.get(grant_id)
            if record and record.view.session_id == session_id and "read" in record.view.operations:
                return record
            return None

    def revoke(self, grant_id: UUID, session_id: UUID | None) -> None:
        with self._lock:
            record = self._records.get(grant_id)
            if record is None or (session_id is not None and record.view.session_id != session_id):
                raise ApiError(404, "NOT_FOUND", "未找到该会话授权。")
            del self._records[grant_id]
