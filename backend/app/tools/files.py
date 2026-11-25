import os
import re
import stat
import sys
import unicodedata
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from ctypes import CDLL, Structure, byref, c_char, c_int32, c_uint32, c_uint64
from pathlib import Path, PureWindowsPath
from typing import Annotated
from uuid import UUID

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from app.services.grants import TEXT_SUFFIXES, GrantRecord
from app.tools.protocol import ToolCall, ToolError, ToolResult, encode_json

MAX_FILE_BYTES = 1_048_576
MAX_TEXT_CHARS = 20_000
MAX_DIRECTORY_ENTRIES = 2_048
MAX_DIRECTORY_BYTES = 8 * MAX_FILE_BYTES
MAX_LIST_CHARS = 12_000
READ_CHUNK = 65_536
MAX_PATH_BYTES = 1_023
HIDDEN = getattr(stat, "UF_HIDDEN", 0)
DATALESS = getattr(stat, "SF_DATALESS", 0x40000000)
UNSAFE_CATEGORIES = {"Cc", "Cf", "Cs"}
LINE_ENDING = re.compile(r"\r\n|[\r\n]")


def path_parts(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    parts = tuple(value.split("/"))
    if ("\\" in value or PureWindowsPath(value).drive or len(parts) > 32
            or any(part in {"", ".", ".."} for part in parts)
            or any(unicodedata.category(char) in UNSAFE_CATEGORIES for char in value)
            or len(value.encode()) > MAX_PATH_BYTES or any(len(part.encode()) > 255 for part in parts)):
        raise ToolError("FILE_PERMISSION_DENIED")
    return parts


def valid_grant_id(value: str) -> str:
    if str(UUID(value)) != value:
        raise ValueError("Invalid grant ID")
    return value


class FileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    grant_id: Annotated[str, Field(min_length=36, max_length=36), AfterValidator(valid_grant_id)]


class ListFilesArgs(FileArgs):
    relative_dir: Annotated[str, Field(max_length=MAX_PATH_BYTES)] = ""
    offset: Annotated[int, Field(ge=0, le=MAX_DIRECTORY_ENTRIES)] = 0
    limit: Annotated[int, Field(ge=1, le=100)] = 50


class ReadTextArgs(FileArgs):
    relative_path: Annotated[str, Field(max_length=MAX_PATH_BYTES)] = ""
    offset_line: Annotated[int, Field(ge=0, le=MAX_FILE_BYTES)] = 0
    limit_lines: Annotated[int, Field(ge=1, le=500)] = 200


class DarwinStatFS(Structure):
    # macOS sys/mount.h's 64-bit statfs ABI. Only f_flags is consumed.
    _fields_ = [
        ("f_bsize", c_uint32), ("f_iosize", c_int32), ("counts", c_uint64 * 5),
        ("f_fsid", c_int32 * 2), ("f_owner", c_uint32), ("f_type", c_uint32),
        ("f_flags", c_uint32), ("f_fssubtype", c_uint32), ("f_fstypename", c_char * 16),
        ("f_mntonname", c_char * 1024), ("f_mntfromname", c_char * 1024), ("reserved", c_uint32 * 8),
    ]


def require_local_volume(fd: int) -> None:
    mount = DarwinStatFS()
    if CDLL(None).fstatfs(fd, byref(mount)) != 0 or not mount.f_flags & 0x1000:  # MNT_LOCAL
        raise ToolError("FILE_PERMISSION_DENIED")


def identity(info: os.stat_result) -> tuple[int, int, float | None]:
    return info.st_dev, info.st_ino, getattr(info, "st_birthtime", None)


def revision(info: os.stat_result) -> tuple:
    return identity(info), info.st_mode, info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_flags


def descriptor_path(fd: int) -> Path:
    import fcntl

    return Path(os.fsdecode(fcntl.fcntl(fd, fcntl.F_GETPATH, bytes(MAX_PATH_BYTES + 1)).split(b"\0", 1)[0]))


def hidden(name: str, info: os.stat_result) -> bool:
    return name.startswith(".") or bool(info.st_flags & HIDDEN)


class FileAccess:
    def __init__(self, grant: GrantRecord, check_active: Callable[[], None]):
        self.grant = grant
        self.check_active = check_active

    def verify(self, fd: int, expected: Path, before: os.stat_result, *, unchanged: bool = False) -> None:
        self.check_active()
        # Verify the current descriptor path and identity after opening and before releasing data.
        actual = descriptor_path(fd)
        current = os.fstat(fd)
        if (actual != expected or not actual.is_relative_to(self.grant.path) or current.st_dev != self.grant.device
                or identity(current) != identity(before) or identity(expected.lstat()) != identity(before)
                or (unchanged and revision(current) != revision(before))):
            raise ToolError("FILE_PERMISSION_DENIED")

    def _open(self, stack: ExitStack, name: str | Path, before: os.stat_result, *,
              directory: bool, parent: int | None = None) -> int:
        if not (stat.S_ISDIR(before.st_mode) if directory else stat.S_ISREG(before.st_mode)):
            raise ToolError("FILE_PERMISSION_DENIED")
        if before.st_dev != self.grant.device or before.st_flags & DATALESS:
            raise ToolError("FILE_PERMISSION_DENIED")
        if not directory and before.st_nlink != 1:
            raise ToolError("FILE_PERMISSION_DENIED")
        if not directory and before.st_size > MAX_FILE_BYTES:
            raise ToolError("FILE_TOO_LARGE")
        self.check_active()
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK | os.O_NOFOLLOW_ANY
        fd = os.open(name, flags | (os.O_DIRECTORY if directory else 0), dir_fd=parent)
        stack.callback(os.close, fd)
        opened = os.fstat(fd)
        if identity(before) != identity(opened) or stat.S_IFMT(before.st_mode) != stat.S_IFMT(opened.st_mode):
            raise ToolError("FILE_PERMISSION_DENIED")
        if not directory and (opened.st_nlink != 1 or opened.st_flags & DATALESS):
            raise ToolError("FILE_PERMISSION_DENIED")
        if not directory and opened.st_size > MAX_FILE_BYTES:
            raise ToolError("FILE_TOO_LARGE")
        return fd

    @contextmanager
    def open(self, parts: tuple[str, ...], *, directory: bool) -> Iterator[tuple[int, Path, os.stat_result]]:
        if sys.platform != "darwin" or not hasattr(os, "O_NOFOLLOW_ANY"):
            raise ToolError("FILE_PERMISSION_DENIED")
        with ExitStack() as stack:
            self.check_active()
            path = self.grant.path
            info = path.lstat()
            if identity(info) != (self.grant.device, self.grant.inode, self.grant.birthtime):
                raise ToolError("FILE_PERMISSION_DENIED")
            fd = self._open(stack, path, info, directory=self.grant.view.scope == "directory")
            require_local_volume(fd)
            lineage = [(fd, path, info)]
            for index, part in enumerate(parts):
                self.verify(fd, path, info)
                info = os.stat(part, dir_fd=fd, follow_symlinks=False)
                if hidden(part, info):
                    raise ToolError("FILE_PERMISSION_DENIED")
                path = path / part
                fd = self._open(stack, part, info, directory=directory or index < len(parts) - 1, parent=fd)
                lineage.append((fd, path, info))
            for item in lineage:
                self.verify(*item)
            yield fd, path, info
            # Check the whole directory chain again before releasing any result.
            for item in lineage:
                self.verify(*item, unchanged=item[0] == fd)

    def text(self, fd: int, expected: Path, before: os.stat_result) -> str:
        self.verify(fd, expected, before, unchanged=True)
        if before.st_size > MAX_FILE_BYTES:
            raise ToolError("FILE_TOO_LARGE")
        raw = bytearray()
        while len(raw) <= MAX_FILE_BYTES:
            self.check_active()
            chunk = os.read(fd, min(READ_CHUNK, MAX_FILE_BYTES + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        if len(raw) > MAX_FILE_BYTES:
            raise ToolError("FILE_TOO_LARGE")
        self.verify(fd, expected, before, unchanged=True)
        try:
            try:
                decoded = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                decoded = raw.decode("gbk")
        except UnicodeDecodeError:
            raise ToolError("INVALID_TEXT_FILE") from None
        if any(unicodedata.category(char) == "Cc" and char not in "\t\r\n" for char in decoded):
            raise ToolError("INVALID_TEXT_FILE")
        return decoded


def read_text(call: ToolCall, args: ReadTextArgs, access: FileAccess) -> ToolResult:
    parts = path_parts(args.relative_path)
    if (access.grant.view.scope == "directory") != bool(parts):
        raise ToolError("VALIDATION_ERROR")
    filename = parts[-1] if parts else access.grant.path.name
    if Path(filename).suffix.lower() not in TEXT_SUFFIXES:
        raise ToolError("UNSUPPORTED_FILE_TYPE")
    with access.open(parts, directory=False) as (fd, expected, before):
        decoded = access.text(fd, expected, before)
    lines = LINE_ENDING.sub("\n", decoded).split("\n")
    if lines[-1] == "":
        lines.pop()
    selected: list[str] = []
    total = 0
    clipped = False
    for line in lines[args.offset_line:args.offset_line + args.limit_lines]:
        remaining = MAX_TEXT_CHARS - total - bool(selected)
        if len(line) > remaining:
            if not selected:
                selected.append(line[:MAX_TEXT_CHARS])
                clipped = True
            break
        selected.append(line)
        total += len(line) + int(len(selected) > 1)
    end = args.offset_line + len(selected)
    next_line = end if end < len(lines) else None
    summary = f"读取了 {len(selected)} 行。"
    if clipped:
        summary += "超长单行仅返回前 20,000 字符。"
    return ToolResult(call.tool_call_id, True, "\n".join(selected), summary,
                      truncated=clipped or next_line is not None, next_offset_line=next_line)


def list_files(call: ToolCall, args: ListFilesArgs, access: FileAccess) -> ToolResult:
    if access.grant.view.scope != "directory":
        raise ToolError("FILE_PERMISSION_DENIED")
    parts = path_parts(args.relative_dir)
    with access.open(parts, directory=True) as (fd, expected, before):
        items, has_more = _list_page(access, fd, expected, before, parts, args)
    content = {"items": items, "has_more": has_more, "next_offset": args.offset + len(items) if has_more else None}
    return ToolResult(call.tool_call_id, True, content, f"列出 {len(items)} 项。", truncated=has_more)


def _list_page(access: FileAccess, fd: int, expected: Path, before: os.stat_result,
               parts: tuple[str, ...], args: ListFilesArgs) -> tuple[list[dict], bool]:
    candidates = []
    with os.scandir(fd) as entries:
        for index, entry in enumerate(entries):
            access.check_active()
            if index >= MAX_DIRECTORY_ENTRIES:
                raise ToolError("DIRECTORY_LIMIT_EXCEEDED")
            info = os.stat(entry.name, dir_fd=fd, follow_symlinks=False)
            if hidden(entry.name, info) or info.st_dev != before.st_dev or info.st_flags & DATALESS:
                continue
            relative = "/".join((*parts, entry.name))
            try:
                path_parts(relative)
            except ToolError:
                continue
            if stat.S_ISDIR(info.st_mode):
                kind = "directory"
            elif (stat.S_ISREG(info.st_mode) and Path(entry.name).suffix.lower() in TEXT_SUFFIXES
                  and info.st_size <= MAX_FILE_BYTES and info.st_nlink == 1):
                kind = "file"
            else:
                continue
            candidates.append((entry.name, relative, kind, info))

    page: list[dict] = []
    seen = scanned_bytes = chars = 0
    for name, relative, kind, info in sorted(candidates, key=lambda item: (item[0].casefold(), item[0])):
        access.verify(fd, expected, before, unchanged=True)
        try:
            with ExitStack() as stack:
                item_fd = access._open(stack, name, info, directory=kind == "directory", parent=fd)
                if kind == "directory":
                    access.verify(item_fd, expected / name, info)
                else:
                    scanned_bytes += info.st_size
                    if scanned_bytes > MAX_DIRECTORY_BYTES:
                        raise ToolError("DIRECTORY_LIMIT_EXCEEDED")
                    access.text(item_fd, expected / name, info)
        except PermissionError:
            continue
        except ToolError as error:
            if error.code == "INVALID_TEXT_FILE":
                continue
            raise
        seen += 1
        if seen <= args.offset:
            continue
        item = {"relative_path": relative, "name": name, "kind": kind}
        chars += len(encode_json(item)) + 1
        if len(page) >= args.limit or chars > MAX_LIST_CHARS - 100:
            return page, True
        page.append(item)
    return page, False
