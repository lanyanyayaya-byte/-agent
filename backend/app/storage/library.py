import os
import sqlite3
import stat
from collections.abc import Iterator
from contextlib import closing, contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.storage.business import BUSY_TIMEOUT_MS
from app.storage.chunks_schema import SCHEMA_V4
from app.storage.embeddings_schema import SCHEMA_V5
from app.storage.index_jobs_schema import SCHEMA_V7
from app.storage.keyword_schema import SCHEMA_V6
from app.storage.keyword_terms import keyword_terms, keyword_title_terms
from app.storage.parsed_documents_schema import SCHEMA_V3

KNOWLEDGE_DIRECTORY = "knowledge"
KNOWLEDGE_DATABASE_NAME = "knowledge.db"
ORIGINALS_DIRECTORY = "originals"

SCHEMA_V1 = (
    # The importer will register assets after persisting the private copy. Index readiness is separate.
    """CREATE TABLE documents (
        document_id TEXT PRIMARY KEY NOT NULL DEFAULT (lower(hex(randomblob(16)))),
        name TEXT NOT NULL,
        format TEXT NOT NULL CHECK (format IN ('pdf', 'docx', 'md')),
        management_status TEXT NOT NULL DEFAULT 'imported' CHECK (management_status = 'imported'),
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
    )""",
    f"""CREATE TABLE content_blobs (
        sha256 TEXT PRIMARY KEY NOT NULL CHECK (
            typeof(sha256) = 'text' AND length(sha256) = 64
            AND instr(sha256, char(0)) = 0 AND sha256 NOT GLOB '*[^0-9a-f]*'
        ),
        size_bytes INTEGER NOT NULL CHECK (typeof(size_bytes) = 'integer' AND size_bytes >= 0),
        relative_path TEXT GENERATED ALWAYS AS ('{ORIGINALS_DIRECTORY}/' || sha256) VIRTUAL,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        UNIQUE (sha256, size_bytes)
    )""",
    """CREATE TABLE document_versions (
        document_version_id TEXT PRIMARY KEY NOT NULL DEFAULT (lower(hex(randomblob(16)))),
        document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
        content_sha256 TEXT NOT NULL,
        size_bytes INTEGER NOT NULL,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        FOREIGN KEY (content_sha256, size_bytes) REFERENCES content_blobs(sha256, size_bytes)
            ON DELETE RESTRICT ON UPDATE RESTRICT
    )""",
    "CREATE INDEX versions_by_document ON document_versions (document_id)",
    "CREATE INDEX versions_by_blob ON document_versions (content_sha256, size_bytes)",
    """CREATE TRIGGER keep_document_id BEFORE UPDATE OF document_id ON documents
       WHEN NEW.document_id IS NOT OLD.document_id
       BEGIN SELECT RAISE(ABORT, 'document ID is immutable'); END""",
    """CREATE TRIGGER keep_document_version
       BEFORE UPDATE OF document_version_id, document_id, content_sha256, size_bytes, created_at ON document_versions
       BEGIN SELECT RAISE(ABORT, 'document version is immutable'); END""",
    """CREATE TRIGGER reject_document_version_replacement BEFORE INSERT ON document_versions
       WHEN EXISTS (SELECT 1 FROM document_versions WHERE document_version_id = NEW.document_version_id)
       BEGIN SELECT RAISE(ABORT, 'document version ID already exists'); END""",
)
SCHEMA_V2 = (
    # Keep successful import IDs after deletion so a delayed retry cannot recreate a removed document.
    """CREATE TABLE knowledge_imports (
        import_id TEXT PRIMARY KEY NOT NULL,
        request_sha256 TEXT NOT NULL,
        document_id TEXT NOT NULL,
        document_version_id TEXT NOT NULL,
        outcome TEXT NOT NULL CHECK (outcome IN ('imported', 'duplicate', 'updated', 'unchanged'))
    )""",
)
MIGRATIONS = (SCHEMA_V1, SCHEMA_V2, SCHEMA_V3, SCHEMA_V4, SCHEMA_V5, SCHEMA_V6, SCHEMA_V7)
SCHEMA_VERSION = len(MIGRATIONS)

DocumentFormat = Literal["pdf", "docx", "md"]
ImportOutcome = Literal["imported", "duplicate", "updated", "unchanged"]


@dataclass(frozen=True)
class DocumentVersion:
    document_version_id: str
    content_sha256: str
    size_bytes: int
    created_at: str


@dataclass(frozen=True)
class DocumentSummary:
    document_id: str
    name: str
    format: DocumentFormat
    management_status: Literal["imported"]
    created_at: str
    updated_at: str
    version_count: int
    latest_size_bytes: int | None


@dataclass(frozen=True)
class DocumentDetails(DocumentSummary):
    versions: list[DocumentVersion]


@dataclass(frozen=True)
class ImportResult:
    import_id: str
    document_id: str
    document_version_id: str
    outcome: ImportOutcome


DOCUMENT_SUMMARY_SQL = """SELECT d.document_id, d.name, d.format, d.management_status, d.created_at, d.updated_at,
    (SELECT count(*) FROM document_versions AS v WHERE v.document_id = d.document_id),
    (SELECT v.size_bytes FROM document_versions AS v WHERE v.document_id = d.document_id ORDER BY v.rowid DESC LIMIT 1)
    FROM documents AS d"""


class KnowledgeStoreError(RuntimeError):
    """Initialization failure with a diagnostic code safe to log."""


class KnowledgeStore:
    """Private knowledge metadata store with operation-scoped SQLite connections."""

    def __init__(self, app_data_dir: Path):
        root = app_data_dir / KNOWLEDGE_DIRECTORY
        self.path = root / KNOWLEDGE_DATABASE_NAME
        self.originals_dir = root / ORIGINALS_DIRECTORY
        for directory in (root, self.originals_dir):
            directory.mkdir(mode=0o700, exist_ok=True)
            if not stat.S_ISDIR(directory.lstat().st_mode):
                raise KnowledgeStoreError("INVALID_KNOWLEDGE_DIRECTORY")
            directory.chmod(0o700)

        # SQLite uses the database permissions when creating WAL files on macOS.
        with suppress(FileExistsError):
            os.close(os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
        self._restrict_database_file(self.path)
        for suffix in ("-wal", "-shm", "-journal"):
            with suppress(FileNotFoundError):
                self._restrict_database_file(self.path.with_name(self.path.name + suffix))

        with self.connect() as connection:
            if connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] != "wal":
                raise KnowledgeStoreError("KNOWLEDGE_WAL_INITIALIZATION_FAILED")

            # Reserve the writer before reading the schema version. DDL and user_version commit together.
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                version = connection.execute("PRAGMA user_version").fetchone()[0]
                if not 0 <= version <= SCHEMA_VERSION:
                    raise KnowledgeStoreError("UNSUPPORTED_KNOWLEDGE_SCHEMA_VERSION")
                for target, statements in enumerate(MIGRATIONS[version:], start=version + 1):
                    for statement in statements:
                        connection.execute(statement)
                    connection.execute(f"PRAGMA user_version={target}")

    @staticmethod
    def _restrict_database_file(path: Path) -> None:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise KnowledgeStoreError("INVALID_KNOWLEDGE_DATABASE_FILE")
        path.chmod(0o600)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        uri = self.path.absolute().as_uri() + "?mode=rw"
        with closing(sqlite3.connect(
            uri, uri=True, timeout=BUSY_TIMEOUT_MS / 1000, isolation_level=None,
        )) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA synchronous=FULL")
            connection.create_function("knowledge_keyword_terms", 1, keyword_terms, deterministic=True)
            connection.create_function("knowledge_keyword_title", 1, keyword_title_terms, deterministic=True)
            yield connection

    @contextmanager
    def write(self, *, wait: bool = True) -> Iterator[sqlite3.Connection]:
        with self.connect() as db, db:
            if not wait:
                db.execute("PRAGMA busy_timeout=0")
            db.execute("BEGIN IMMEDIATE")
            yield db

    def list_documents(self) -> list[DocumentSummary]:
        with self.connect() as db:
            return [DocumentSummary(*row) for row in db.execute(
                DOCUMENT_SUMMARY_SQL + " ORDER BY d.updated_at DESC, d.document_id",
            )]

    def get_document(self, document_id: str) -> DocumentDetails | None:
        with self.connect() as db, db:
            db.execute("BEGIN")
            row = db.execute(DOCUMENT_SUMMARY_SQL + " WHERE d.document_id = ?", (document_id,)).fetchone()
            if row is None:
                return None
            versions = [DocumentVersion(*version) for version in db.execute(
                """SELECT document_version_id, content_sha256, size_bytes, created_at
                   FROM document_versions WHERE document_id = ? ORDER BY rowid""", (document_id,),
            )]
            return DocumentDetails(*row, versions=versions)

    def get_document_version(self, document_version_id: str) -> tuple[str, DocumentFormat, DocumentVersion] | None:
        with self.connect() as db:
            row = db.execute(
                """SELECT v.document_id, d.format, v.document_version_id, v.content_sha256, v.size_bytes, v.created_at
                   FROM document_versions AS v
                   JOIN documents AS d USING (document_id) WHERE v.document_version_id = ?""",
                (document_version_id,),
            ).fetchone()
            if row is None:
                return None
            document_id, format, *version = row
            return document_id, format, DocumentVersion(*version)

    def latest_version_identity(self, document_id: str) -> tuple[str, DocumentFormat] | None:
        with self.connect() as db:
            return db.execute("""SELECT v.document_version_id, d.format FROM document_versions AS v
                JOIN documents AS d USING (document_id) WHERE v.document_id = ?
                ORDER BY v.rowid DESC LIMIT 1""", (document_id,)).fetchone()

    def detach_document(self, document_id: str) -> list[str]:
        with self.write() as db:
            hashes = [sha256 for sha256, in db.execute(
                """SELECT DISTINCT v.content_sha256 FROM document_versions AS v WHERE v.document_id = ?
                   AND NOT EXISTS (SELECT 1 FROM document_versions AS other
                                   WHERE other.content_sha256 = v.content_sha256 AND other.document_id <> ?)""",
                (document_id, document_id),
            )]
            db.execute("DELETE FROM document_versions WHERE document_id = ?", (document_id,))
            db.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
            db.executemany("DELETE FROM content_blobs WHERE sha256 = ?", ((sha256,) for sha256 in hashes))
            return hashes
