import codecs
import errno
import hashlib
import os
import re
import sqlite3
import stat
import struct
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, suppress
from itertools import islice
from pathlib import Path
from threading import Lock
from typing import BinaryIO
from uuid import UUID, uuid4
from zipfile import BadZipFile, ZipFile

from app.diagnostics import record
from app.security import ApiError
from app.storage.business import utc_now
from app.storage.library import (
    ORIGINALS_DIRECTORY, DocumentDetails, DocumentFormat, DocumentSummary, ImportResult, KnowledgeStore,
    KnowledgeStoreError,
)

if os.name == "posix":
    import fcntl

TEMPORARY_DIRECTORY = "temporary"
IMPORT_ID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
DOCUMENT_ID_PATTERN = r"^[0-9a-f]{32}$"
HASH_NAME = re.compile(r"[0-9a-f]{64}")
SCRATCH_PREFIX = ".incoming-"
SCRATCH_NAME = re.compile(r"\.incoming-[0-9a-f]{32}")
FORMATS: dict[str, DocumentFormat] = {"pdf": "pdf", "docx": "docx", "md": "md"}
MAX_FILE_BYTES = 100 * 1024 * 1024
COPY_BLOCK_BYTES = 256 * 1024
MAX_DOCX_ENTRIES = 10_000
MAX_DOCX_DIRECTORY_BYTES = 4 * 1024 * 1024
MAX_DOCX_EXPANDED_BYTES = 512 * 1024 * 1024
ZIP_END_RECORD = struct.Struct("<4s4H2IH")
DEFAULT_CLEANUP_LIMIT = 128


@contextmanager
def private_directory(name: str | Path, parent: int | None = None) -> Iterator[int]:
    fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
    try:
        info = os.fstat(fd)
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise KnowledgeStoreError("INVALID_KNOWLEDGE_DIRECTORY")
        yield fd
    finally:
        os.close(fd)


def private_file(info: os.stat_result) -> bool:
    return (stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_uid == os.geteuid()
            and stat.S_IMODE(info.st_mode) in (0o400, 0o600))


def file_revision(info: os.stat_result) -> tuple[int, ...]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def stat_leaf(directory: int, name: str) -> os.stat_result | None:
    with suppress(FileNotFoundError):
        return os.stat(name, dir_fd=directory, follow_symlinks=False)
    return None


def invalid_format() -> ApiError:
    return ApiError(415, "INVALID_TEXTBOOK_FORMAT", "教材文件与声明的格式不符，或超出该格式的安全限制。")


def file_too_large() -> ApiError:
    return ApiError(413, "TEXTBOOK_TOO_LARGE", "教材文件大小超过限制。")


def source_changed() -> ApiError:
    return ApiError(409, "IMPORT_SOURCE_CHANGED", "暂存文件在导入期间发生变化，请重新暂存后重试。")


def textbook_format(filename: str) -> DocumentFormat:
    stem, _, suffix = filename.rpartition(".")
    if (not filename.strip() or len(filename) > 255
            or re.search(r"[\\/\x00-\x1f\x7f\ud800-\udfff]", filename)):
        raise ApiError(422, "VALIDATION_ERROR", "文件名必须是有效的单个文件名。")
    if not stem.strip() or suffix.lower() not in FORMATS:
        raise invalid_format()
    return FORMATS[suffix.lower()]


def validate_docx(stream: BinaryIO, size: int) -> None:
    # Bound ZipFile's central directory allocation before it opens the archive. DOCX
    # imports do not decompress entries. ZIP64 and split archives are outside these limits.
    tail_size = min(size, 65_535 + ZIP_END_RECORD.size)
    stream.seek(size - tail_size)
    tail = stream.read(tail_size)
    index = tail.rfind(b"PK\x05\x06")
    if tail[-22:-18] == b"PK\x05\x06" and tail[-2:] == b"\0\0":
        index = len(tail) - ZIP_END_RECORD.size
    if index < 0 or index + ZIP_END_RECORD.size > len(tail):
        raise invalid_format()
    _, disk, directory_disk, disk_entries, entries, length, offset, comment = ZIP_END_RECORD.unpack_from(tail, index)
    end_record = size - tail_size + index
    if (disk or directory_disk or disk_entries != entries or not 1 <= entries <= MAX_DOCX_ENTRIES
            or length > MAX_DOCX_DIRECTORY_BYTES or offset + length != end_record
            or index + ZIP_END_RECORD.size + comment != len(tail)):
        raise invalid_format()
    if end_record >= 20:
        stream.seek(end_record - 20)
        if stream.read(4) == b"PK\x06\x07":
            raise invalid_format()
    try:
        with ZipFile(stream) as archive:
            members = archive.infolist()
            if (len(members) != entries or sum(member.file_size for member in members) > MAX_DOCX_EXPANDED_BYTES
                    or any(member.flag_bits & 1 for member in members)):
                raise invalid_format()
            for path in ("[Content_Types].xml", "_rels/.rels", "word/document.xml"):
                if archive.getinfo(path).file_size == 0:
                    raise invalid_format()
    except (BadZipFile, KeyError, NotImplementedError, UnicodeDecodeError, struct.error):
        raise invalid_format() from None


def copy_textbook(source: BinaryIO, destination: BinaryIO, format: DocumentFormat) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    decoder = codecs.getincrementaldecoder("utf-8-sig")() if format == "md" else None
    try:
        while block := source.read(COPY_BLOCK_BYTES):
            size += len(block)
            if size > MAX_FILE_BYTES:
                raise file_too_large()
            if decoder is not None:
                if b"\0" in block:
                    raise invalid_format()
                decoder.decode(block)
            digest.update(block)
            destination.write(block)
        if decoder is not None:
            decoder.decode(b"", final=True)
    except UnicodeDecodeError:
        raise invalid_format() from None

    if format == "pdf":
        destination.seek(0)
        if not re.fullmatch(rb"%PDF-(?:1\.[0-7]|2\.0)", destination.read(8)):
            raise invalid_format()
        destination.seek(max(0, size - 1024))
        if not destination.read(1024).rstrip(b" \t\r\n\f").endswith(b"%%EOF"):
            raise invalid_format()
    if format == "docx":
        destination.seek(0)
        if destination.read(4) != b"PK\x03\x04":
            raise invalid_format()
        validate_docx(destination, size)
    return digest.hexdigest(), size


def log_maintenance_failure(error: Exception) -> None:
    record(
        "knowledge_cleanup_error", exception_type=type(error).__name__,
        sqlite_error=getattr(error, "sqlite_errorname", None), errno=getattr(error, "errno", None),
    )


class LibraryService:
    def __init__(self, store: KnowledgeStore, instance_id: UUID):
        if os.name != "posix":
            raise KnowledgeStoreError("UNSUPPORTED_KNOWLEDGE_FILESYSTEM")
        self.store = store
        self.instance_name = str(instance_id)
        self.staging_dir = store.path.parent / TEMPORARY_DIRECTORY / self.instance_name
        self._scan = None
        self._scan_lock = Lock()

        with private_directory(store.path.parent) as root:
            with suppress(FileExistsError):
                os.mkdir(TEMPORARY_DIRECTORY, 0o700, dir_fd=root)
            with private_directory(TEMPORARY_DIRECTORY, root) as temporary:
                os.mkdir(self.instance_name, 0o700, dir_fd=temporary)
                self._staging_info = os.stat(self.instance_name, dir_fd=temporary, follow_symlinks=False)
                os.fsync(temporary)
            # The directory links must be durable as well as the final file entry.
            os.fsync(root)
        parent = os.open(store.path.parent.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)

    def close(self) -> None:
        with self._scan_lock:
            if self._scan is not None:
                self._scan.close()
                self._scan = None
        with suppress(OSError, KnowledgeStoreError), private_directory(self.store.path.parent) as root:
            with private_directory(TEMPORARY_DIRECTORY, root) as temporary:
                os.rmdir(self.instance_name, dir_fd=temporary)

    def list_documents(self) -> list[DocumentSummary]:
        return self.store.list_documents()

    def get_document(self, document_id: str) -> DocumentDetails:
        result = self.store.get_document(document_id)
        if result is None:
            raise ApiError(404, "TEXTBOOK_NOT_FOUND", "未找到该教材。")
        return result

    @staticmethod
    def _replay(db: sqlite3.Connection, import_id: str, request_sha256: str) -> ImportResult | None:
        receipt = db.execute(
            """SELECT i.request_sha256, i.document_id, i.document_version_id, i.outcome,
                      v.document_version_id IS NOT NULL
               FROM knowledge_imports AS i LEFT JOIN document_versions AS v
                   ON v.document_version_id = i.document_version_id AND v.document_id = i.document_id
               WHERE i.import_id = ?""", (import_id,),
        ).fetchone()
        if receipt is None:
            return None
        saved_hash, result_id, version_id, outcome, exists = receipt
        if saved_hash != request_sha256:
            raise ApiError(409, "IMPORT_ID_CONFLICT", "此导入编号已用于其他请求。")
        if not exists:
            raise ApiError(410, "TEXTBOOK_REMOVED", "此导入对应的教材已删除。")
        return ImportResult(import_id, result_id, version_id, outcome)

    @contextmanager
    def _scratch(self, originals: int) -> Iterator[tuple[BinaryIO, str]]:
        name = SCRATCH_PREFIX + uuid4().hex
        with ExitStack() as stack:
            # The cleaner takes the same writer reservation before inspecting scratch files.
            with self.store.write():
                fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=originals)
                stream = stack.enter_context(os.fdopen(fd, "w+b"))
                stack.callback(self._discard, originals, name)
                fcntl.flock(fd, fcntl.LOCK_EX)
            yield stream, name

    @staticmethod
    def _discard(directory: int, name: str, expected: os.stat_result | None = None) -> None:
        try:
            if expected is not None:
                current = stat_leaf(directory, name)
                if current is None or file_revision(current) != file_revision(expected):
                    return
            os.unlink(name, dir_fd=directory)
        except FileNotFoundError:
            pass
        except OSError as error:
            log_maintenance_failure(error)

    @contextmanager
    def _source(self, staged: int, import_id: str) -> Iterator[tuple[BinaryIO, os.stat_result]]:
        before = stat_leaf(staged, import_id)
        if before is None:
            raise ApiError(404, "IMPORT_SOURCE_NOT_FOUND", "未找到此 Sidecar 实例的暂存文件。")
        if not private_file(before):
            raise ApiError(415, "INVALID_IMPORT_SOURCE", "暂存文件必须是私有常规文件。")
        if before.st_size > MAX_FILE_BYTES:
            raise file_too_large()
        if before.st_size == 0:
            raise invalid_format()
        try:
            fd = os.open(import_id, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=staged)
        except OSError as error:
            if error.errno in (errno.ENOENT, errno.ELOOP):
                raise source_changed() from None
            raise
        with os.fdopen(fd, "rb") as source:
            opened = os.fstat(fd)
            if not private_file(opened) or file_revision(opened) != file_revision(before):
                raise source_changed()
            yield source, opened

    def import_textbook(self, import_id: str, filename: str, document_id: str | None = None) -> ImportResult:
        if not re.fullmatch(IMPORT_ID_PATTERN, import_id) or (
            document_id is not None and not re.fullmatch(DOCUMENT_ID_PATTERN, document_id)
        ):
            raise ApiError(422, "VALIDATION_ERROR", "导入编号或教材编号无效。")
        format = textbook_format(filename)
        request_sha256 = hashlib.sha256(f"{document_id or ''}\0{filename}".encode()).hexdigest()
        with self.store.connect() as db:
            replay = self._replay(db, import_id, request_sha256)
        if replay is not None:
            return replay

        with ExitStack() as stack:
            root = stack.enter_context(private_directory(self.store.path.parent))
            temporary = stack.enter_context(private_directory(TEMPORARY_DIRECTORY, root))
            staged = stack.enter_context(private_directory(self.instance_name, temporary))
            if not os.path.samestat(os.fstat(staged), self._staging_info):
                raise KnowledgeStoreError("INVALID_IMPORT_STAGING_DIRECTORY")
            originals = stack.enter_context(private_directory(ORIGINALS_DIRECTORY, root))
            try:
                with self._source(staged, import_id) as (source, before):
                    with self._scratch(originals) as (output, scratch_name):
                        sha256, size = copy_textbook(source, output, format)
                        output.flush()
                        os.fchmod(output.fileno(), 0o400)
                        os.fsync(output.fileno())
                        with self.store.write() as db:
                            result = self._replay(db, import_id, request_sha256)
                            if result is None:
                                current = stat_leaf(staged, import_id)
                                if (current is None or size != before.st_size
                                        or file_revision(current) != file_revision(before)
                                        or file_revision(os.fstat(source.fileno())) != file_revision(before)):
                                    raise source_changed()
                                result = self._register(
                                    db, originals, scratch_name, import_id, request_sha256, filename,
                                    format, document_id, sha256, size,
                                )
                self._discard(staged, import_id, before)
                return result
            except ApiError as error:
                if error.code in {"IMPORT_SOURCE_NOT_FOUND", "IMPORT_SOURCE_CHANGED"}:
                    # An identical concurrent request may have committed and consumed the staging file.
                    with self.store.connect() as db:
                        replay = self._replay(db, import_id, request_sha256)
                    if replay is not None:
                        return replay
                raise

    @staticmethod
    def _register(
        db: sqlite3.Connection, originals: int, scratch: str, import_id: str, request_sha256: str, filename: str,
        format: DocumentFormat, target_id: str | None, sha256: str, size: int,
    ) -> ImportResult:
        if target_id is None:
            existing = db.execute(
                """SELECT document_id, document_version_id FROM document_versions
                   WHERE content_sha256 = ? ORDER BY rowid LIMIT 1""", (sha256,),
            ).fetchone()
            if existing:
                document_id, version_id = existing
                outcome = "duplicate"
            else:
                document_id, = db.execute(
                    "INSERT INTO documents (name, format) VALUES (?, ?) RETURNING document_id", (filename, format),
                ).fetchone()
                version_id = None
                outcome = "imported"
        else:
            current = db.execute(
                """SELECT d.format, v.document_version_id, v.content_sha256 FROM documents AS d
                   LEFT JOIN document_versions AS v ON v.rowid = (
                       SELECT rowid FROM document_versions WHERE document_id = d.document_id ORDER BY rowid DESC LIMIT 1
                   ) WHERE d.document_id = ?""", (target_id,),
            ).fetchone()
            if current is None:
                raise ApiError(404, "TEXTBOOK_NOT_FOUND", "未找到该教材。")
            if current[0] != format:
                raise ApiError(409, "TEXTBOOK_FORMAT_MISMATCH", "更新文件须与该教材的原格式一致。")
            document_id = target_id
            version_id = current[1] if current[2] == sha256 else None
            outcome = "unchanged" if version_id else "updated"

        blob = db.execute("SELECT size_bytes FROM content_blobs WHERE sha256 = ?", (sha256,)).fetchone()
        existing_file = stat_leaf(originals, sha256)
        if blob is not None and (blob[0] != size or (
            existing_file is not None and (not private_file(existing_file) or existing_file.st_size != size)
        )):
            raise KnowledgeStoreError("INVALID_TEXTBOOK_BLOB")
        if blob is None or existing_file is None:
            # Unregistered entries may be left by a failed commit. Publish the verified
            # copy over that name, without following it, before any reference can commit.
            os.replace(scratch, sha256, src_dir_fd=originals, dst_dir_fd=originals)
        os.fsync(originals)
        if blob is None:
            db.execute("INSERT INTO content_blobs (sha256, size_bytes) VALUES (?, ?)", (sha256, size))
        if version_id is None:
            version_id, = db.execute(
                """INSERT INTO document_versions (document_id, content_sha256, size_bytes)
                   VALUES (?, ?, ?) RETURNING document_version_id""", (document_id, sha256, size),
            ).fetchone()
            if target_id is not None:
                db.execute("UPDATE documents SET updated_at = ? WHERE document_id = ?", (utc_now(), document_id))

        db.execute(
            """INSERT INTO knowledge_imports (import_id, request_sha256, document_id,
                                              document_version_id, outcome) VALUES (?, ?, ?, ?, ?)""",
            (import_id, request_sha256, document_id, version_id, outcome),
        )
        return ImportResult(import_id, document_id, version_id, outcome)

    def delete_document(self, document_id: str) -> None:
        hashes = self.store.detach_document(document_id)
        if not hashes:
            return
        # Metadata is already committed. A failure here leaves an unreferenced file for recovery.
        try:
            with private_directory(self.store.path.parent) as root, private_directory(ORIGINALS_DIRECTORY, root) as fd:
                with self.store.write(wait=False) as db:
                    for sha256 in hashes:
                        self._remove_unreferenced(db, fd, sha256)
                    os.fsync(fd)
        except (OSError, sqlite3.Error, KnowledgeStoreError) as error:
            log_maintenance_failure(error)

    @staticmethod
    def _remove_unreferenced(db: sqlite3.Connection, directory: int, name: str) -> bool:
        if db.execute("SELECT 1 FROM content_blobs WHERE sha256 = ?", (name,)).fetchone():
            return False
        info = stat_leaf(directory, name)
        if info is None or not private_file(info):
            return False
        os.unlink(name, dir_fd=directory)
        return True

    @staticmethod
    def _remove_abandoned_scratch(directory: int, name: str) -> bool:
        info = stat_leaf(directory, name)
        if info is None or not private_file(info):
            return False
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        except FileNotFoundError:
            return False
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            current = stat_leaf(directory, name)
            if current is None or not os.path.samestat(current, os.fstat(fd)):
                return False
            os.unlink(name, dir_fd=directory)
            return True
        finally:
            os.close(fd)

    def cleanup_orphans(self, limit: int = DEFAULT_CLEANUP_LIMIT) -> int:
        if limit < 1:
            raise ValueError("Cleanup limit must be positive")
        with self._scan_lock, private_directory(self.store.path.parent) as root:
            with private_directory(ORIGINALS_DIRECTORY, root) as directory, self.store.write(wait=False) as db:
                if self._scan is None:
                    self._scan = os.scandir(directory)
                removed = 0
                visited = 0
                for entry in islice(self._scan, limit):
                    visited += 1
                    if HASH_NAME.fullmatch(entry.name):
                        removed += self._remove_unreferenced(db, directory, entry.name)
                    elif SCRATCH_NAME.fullmatch(entry.name):
                        removed += self._remove_abandoned_scratch(directory, entry.name)
                if visited < limit:
                    self._scan.close()
                    self._scan = None
                if removed:
                    os.fsync(directory)
                return removed
