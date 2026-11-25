"""Managed stdio MCP subprocess registry for host-level orphan supervision.

When Sidecar launches a local stdio MCP server in its own POSIX process group
(`start_new_session=True`), a sudden `SIGKILL` to the Sidecar leaves Python no
chance to run `McpManager.shutdown()`. To close that gap safely without a
separate daemon, Sidecar writes an atomic per-process identity record scoped to
its own `instance_id` under `<app_data_dir>/runtime/mcp-procs/<instance_id>/`.

Security & safety guarantees:
- Records ONLY kernel process identity (pid, pgid, uid, kernel start timestamp
  from macOS `libproc`, canonical executable path, owning `instance_id` and
  `sidecar_pid`). Command arguments, env vars, and secrets are NEVER written.
- Atomic temp-file + `os.replace()` writes prevent partial/corrupted files.
- The Rust host verifies every identity field against live kernel state before
  signaling a process group, preventing PID/PGID reuse from harming unrelated
  processes, and never kills PIDs from stale/expired instances.
"""

from __future__ import annotations

import ctypes
import json
import os
import struct
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

PROC_PIDTBSDINFO = 3
PROC_BSDINFO_SIZE = 136
PROC_PIDPATHINFO_MAXSIZE = 4096

_LIBPROC: ctypes.CDLL | None = None
if sys.platform == "darwin":
    with suppress(OSError):
        _LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib")
        _LIBPROC.proc_pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        _LIBPROC.proc_pidinfo.restype = ctypes.c_int
        _LIBPROC.proc_pidpath.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        _LIBPROC.proc_pidpath.restype = ctypes.c_int


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    pid: int
    pgid: int
    uid: int
    start_tvsec: int | None
    start_tvusec: int | None
    executable_path: str


def query_process_identity(pid: int, fallback_executable: Path | str) -> ProcessIdentity:
    """Capture kernel-backed process identity (start time, pgid, uid, executable path)."""
    pgid = pid
    with suppress(OSError):
        pgid = os.getpgid(pid)
    uid = os.getuid() if hasattr(os, "getuid") else 0
    start_tvsec: int | None = None
    start_tvusec: int | None = None
    exe_path = str(Path(fallback_executable).resolve())

    if _LIBPROC is not None:
        buf = ctypes.create_string_buffer(PROC_BSDINFO_SIZE)
        size = _LIBPROC.proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, buf, PROC_BSDINFO_SIZE)
        if size == PROC_BSDINFO_SIZE:
            raw = buf.raw
            live_pid = struct.unpack_from("<I", raw, 12)[0]
            if live_pid == pid:
                uid = struct.unpack_from("<I", raw, 20)[0]
                pgid = struct.unpack_from("<I", raw, 100)[0]
                start_tvsec = struct.unpack_from("<Q", raw, 120)[0]
                start_tvusec = struct.unpack_from("<Q", raw, 128)[0]

        path_buf = ctypes.create_string_buffer(PROC_PIDPATHINFO_MAXSIZE)
        path_len = _LIBPROC.proc_pidpath(pid, path_buf, PROC_PIDPATHINFO_MAXSIZE)
        if path_len > 0:
            detected = os.fsdecode(path_buf.value)
            if detected:
                with suppress(OSError):
                    exe_path = str(Path(detected).resolve())

    return ProcessIdentity(
        pid=pid,
        pgid=pgid,
        uid=uid,
        start_tvsec=start_tvsec,
        start_tvusec=start_tvusec,
        executable_path=exe_path,
    )


def _safe_filename_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)[:96]


class McpProcessRegistry:
    """Writes/cleans per-instance managed MCP stdio process records for Rust host supervision."""

    def __init__(self, app_data_dir: Path, instance_id: str) -> None:
        self.app_data_dir = app_data_dir
        self.instance_id = instance_id
        self.sidecar_pid = os.getpid()
        self.runtime_root = app_data_dir / "runtime" / "mcp-procs"
        self.instance_dir = self.runtime_root / _safe_filename_component(instance_id)

    def _ensure_dir(self) -> bool:
        try:
            for directory in (self.app_data_dir / "runtime", self.runtime_root, self.instance_dir):
                if directory.is_symlink():
                    return False
                directory.mkdir(mode=0o700, parents=True, exist_ok=True)
                with suppress(OSError):
                    os.chmod(directory, 0o700)
            return not self.instance_dir.is_symlink() and self.instance_dir.is_dir()
        except OSError:
            return False

    def _atomic_write_json(self, target: Path, payload: dict[str, object]) -> None:
        if not self._ensure_dir():
            return
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        tmp_path = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        fd = -1
        try:
            fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            os.write(fd, encoded)
            os.close(fd)
            fd = -1
            os.replace(tmp_path, target)
        except OSError:
            if fd >= 0:
                with suppress(OSError):
                    os.close(fd)
            with suppress(OSError):
                tmp_path.unlink(missing_ok=True)

    def _pending_path(self, server_id: str, generation: int) -> Path:
        return self.instance_dir / f"pending-{_safe_filename_component(server_id)}-g{generation}.json"

    def _record_path(self, server_id: str, generation: int, pid: int) -> Path:
        return self.instance_dir / f"proc-{_safe_filename_component(server_id)}-g{generation}-{pid}.json"

    def mark_launch_pending(self, server_id: str, generation: int, executable_path: Path | str) -> None:
        payload: dict[str, object] = {
            "kind": "pending",
            "version": 1,
            "instance_id": self.instance_id,
            "sidecar_pid": self.sidecar_pid,
            "server_id": server_id,
            "generation": generation,
            "executable_path": str(Path(executable_path).resolve()),
            "started_at_ms": int(time.time() * 1000),
        }
        self._atomic_write_json(self._pending_path(server_id, generation), payload)

    def clear_launch_pending(self, server_id: str, generation: int) -> None:
        with suppress(OSError):
            self._pending_path(server_id, generation).unlink(missing_ok=True)

    def register_process(
        self,
        server_id: str,
        generation: int,
        pid: int,
        executable_path: Path | str,
    ) -> ProcessIdentity:
        identity = query_process_identity(pid, executable_path)
        payload: dict[str, object] = {
            "kind": "active",
            "version": 1,
            "instance_id": self.instance_id,
            "sidecar_pid": self.sidecar_pid,
            "server_id": server_id,
            "generation": generation,
            "pid": identity.pid,
            "pgid": identity.pgid,
            "uid": identity.uid,
            "start_tvsec": identity.start_tvsec,
            "start_tvusec": identity.start_tvusec,
            "executable_path": identity.executable_path,
            "registered_at_ms": int(time.time() * 1000),
        }
        self._atomic_write_json(self._record_path(server_id, generation, pid), payload)
        self.clear_launch_pending(server_id, generation)
        return identity

    def unregister_process(self, server_id: str, generation: int, pid: int | None) -> None:
        self.clear_launch_pending(server_id, generation)
        if pid is not None:
            with suppress(OSError):
                self._record_path(server_id, generation, pid).unlink(missing_ok=True)

    def clear_instance(self) -> None:
        if not self.instance_dir.exists() or self.instance_dir.is_symlink():
            return
        with suppress(OSError):
            for child in self.instance_dir.iterdir():
                if child.is_file() and not child.is_symlink():
                    with suppress(OSError):
                        child.unlink(missing_ok=True)
            self.instance_dir.rmdir()
