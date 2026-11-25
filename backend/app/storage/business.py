import json
import os
import sqlite3
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

DATABASE_NAME = "knowledgeagent.sqlite3"
BUSY_TIMEOUT_MS = 5000
ACTIVE_STATUSES = "'running', 'awaiting_approval', 'cancelling'"
RunState = Literal["running", "awaiting_approval", "cancelling", "completed", "failed", "cancelled"]
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
INTERRUPTED_CODE = "RUN_INTERRUPTED"

SCHEMA_V1 = (
    """CREATE TABLE sessions (
        session_id TEXT PRIMARY KEY NOT NULL,
        title TEXT NOT NULL,
        last_safe_checkpoint_id TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""",
    f"""CREATE TABLE runs (
        run_id TEXT PRIMARY KEY NOT NULL,
        session_id TEXT NOT NULL REFERENCES sessions(session_id),
        client_request_id TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ({ACTIVE_STATUSES}, 'completed', 'failed', 'cancelled')),
        base_checkpoint_id TEXT,
        final_checkpoint_id TEXT CHECK (final_checkpoint_id IS NULL OR status = 'completed'),
        started_at TEXT NOT NULL,
        finished_at TEXT,
        error_code TEXT,
        error_message TEXT,
        UNIQUE (session_id, client_request_id),
        UNIQUE (session_id, run_id),
        CHECK ((finished_at IS NULL) = (status IN ({ACTIVE_STATUSES}))),
        CHECK ((error_code IS NOT NULL) = (status = 'failed')),
        CHECK ((error_message IS NOT NULL) = (status = 'failed'))
    )""",
    f"""CREATE UNIQUE INDEX one_active_run_per_session ON runs (session_id)
        WHERE status IN ({ACTIVE_STATUSES})""",
    """CREATE TABLE messages (
        message_id TEXT PRIMARY KEY NOT NULL,
        session_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
        content TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (session_id, run_id) REFERENCES runs (session_id, run_id),
        UNIQUE (run_id, role)
    )""",
)

# Messages already contain the run foreign key and are inserted only on commit.
MIGRATIONS = (
    SCHEMA_V1,
    (
        "CREATE INDEX sessions_by_activity ON sessions (updated_at DESC, session_id DESC)",
        "CREATE INDEX runs_by_session_start ON runs (session_id, started_at DESC, run_id DESC)",
    ),
    (
        # V1 reserved the nullable columns. Existing completed runs have no graph state.
        """CREATE TRIGGER require_final_checkpoint_on_insert BEFORE INSERT ON runs
           WHEN NEW.status = 'completed' AND coalesce(NEW.final_checkpoint_id, '') = ''
           BEGIN SELECT RAISE(ABORT, 'completed run requires a final checkpoint'); END""",
        """CREATE TRIGGER require_final_checkpoint_on_update BEFORE UPDATE OF status, final_checkpoint_id ON runs
           WHEN NEW.status = 'completed' AND coalesce(NEW.final_checkpoint_id, '') = ''
           BEGIN SELECT RAISE(ABORT, 'completed run requires a final checkpoint'); END""",
    ),
    (
        """CREATE TABLE tool_calls (
            tool_call_id TEXT PRIMARY KEY NOT NULL,
            session_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
            summary TEXT,
            error_code TEXT,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            FOREIGN KEY (session_id, run_id) REFERENCES runs (session_id, run_id),
            CHECK ((finished_at IS NULL) = (status = 'running')),
            CHECK ((summary IS NULL) = (status = 'running')),
            CHECK ((error_code IS NOT NULL) = (status IN ('failed', 'cancelled')))
        )""",
        "CREATE INDEX tools_by_run ON tool_calls (session_id, run_id, started_at)",
    ),
    (
        """CREATE TABLE run_citations (
            run_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
            evidence_id TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            chunk_set_id INTEGER NOT NULL CHECK (chunk_set_id > 0),
            chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
            document_id TEXT NOT NULL,
            document_version_id TEXT NOT NULL,
            textbook_name TEXT NOT NULL,
            format TEXT NOT NULL CHECK (format IN ('pdf', 'docx', 'md')),
            section_path TEXT NOT NULL,
            location_summary TEXT NOT NULL,
            location_precision TEXT NOT NULL,
            location_status TEXT NOT NULL,
            warnings TEXT NOT NULL,
            snippet TEXT NOT NULL,
            PRIMARY KEY (run_id, ordinal),
            FOREIGN KEY (session_id, run_id) REFERENCES runs (session_id, run_id)
        )""",
        "CREATE INDEX citations_by_session_run ON run_citations (session_id, run_id, ordinal)",
    ),
)
SCHEMA_VERSION = len(MIGRATIONS)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class Session:
    session_id: str
    title: str
    created_at: str


@dataclass(frozen=True)
class SessionSummary:
    session_id: str
    title: str
    updated_at: str
    active_run_id: str | None
    active_run_status: RunState | None


@dataclass(frozen=True)
class Message:
    message_id: str
    session_id: str
    run_id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: str
    status: Literal["committed"] = "committed"


@dataclass(frozen=True)
class ToolCallSummary:
    tool_call_id: str
    tool_name: str
    status: Literal["running", "completed", "failed", "cancelled"]
    summary: str | None
    error_code: str | None


@dataclass(frozen=True)
class StoredCitation:
    evidence_id: str
    chunk_id: str
    chunk_set_id: int
    chunk_index: int
    document_id: str
    document_version_id: str
    textbook_name: str
    format: Literal["pdf", "docx", "md"]
    section_path: tuple[str, ...]
    location_summary: str
    location_precision: str
    location_status: str
    warnings: tuple[dict[str, object], ...]
    snippet: str
    status: str = "resolved"
    historical: bool = False
    parse_spec_current: bool = True


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    session_id: str
    user_message_id: str
    status: RunState
    error_code: str | None
    started_at: str
    finished_at: str | None
    tool_calls: tuple[ToolCallSummary, ...] = ()
    citations: tuple[StoredCitation, ...] = ()


@dataclass(frozen=True)
class Run:
    session_id: str
    run_id: str
    client_request_id: str
    base_checkpoint_id: str | None


@dataclass(frozen=True)
class RunStatus:
    run_id: str
    session_id: str
    status: RunState
    started_at: str
    finished_at: str | None
    error_code: str | None
    pending_approval: None = None


class SessionNotFound(Exception):
    pass


class RunNotFound(Exception):
    pass


class RunConflict(Exception):
    def __init__(self, code: Literal["DUPLICATE_CLIENT_REQUEST", "RUN_ALREADY_ACTIVE"], run_id: str):
        super().__init__(code)
        self.code = code
        self.run_id = run_id


class DatabaseLockedError(BlockingIOError):
    pass


class BusinessStore:
    def __init__(self, path: Path, *, initialize: bool = True):
        self.path = path
        if initialize:
            self._initialize()

    def _initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            if connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] != "wal":
                raise RuntimeError("SQLite WAL initialization failed")
            with self._write(connection):
                version = connection.execute("PRAGMA user_version").fetchone()[0]
                if not 0 <= version <= SCHEMA_VERSION:
                    raise RuntimeError("Unsupported business database schema version")
                for target, statements in enumerate(MIGRATIONS[version:], start=version + 1):
                    for statement in statements:
                        connection.execute(statement)
                    connection.execute(f"PRAGMA user_version={target}")

    @contextmanager
    def own_runs(self) -> Iterator[None]:
        # Only a process with exclusive ownership can migrate or classify active rows as interrupted.
        # Keep the lock file in place so contenders always lock the same inode.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a+b") as lock:
            try:
                if os.name == "nt":
                    import msvcrt

                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise DatabaseLockedError("Business database is locked by another running instance") from error
            self._initialize()
            yield

    def recover_interrupted_runs(self) -> list[tuple[str, str]]:
        now = utc_now()
        with closing(self._connect()) as connection, self._write(connection):
            connection.execute(
                f"""UPDATE sessions SET updated_at = ? WHERE session_id IN (
                    SELECT session_id FROM runs WHERE status IN ({ACTIVE_STATUSES})
                )""",
                (now,),
            )
            recovered = connection.execute(
                f"""UPDATE runs SET status = 'failed', finished_at = ?, error_code = ?, error_message = ?
                    WHERE status IN ({ACTIVE_STATUSES}) RETURNING run_id, session_id""",
                (now, INTERRUPTED_CODE, "上一次后端运行已中断，无法恢复本次生成。"),
            ).fetchall()
            for run_id, session_id in recovered:
                self._finish_pending_tools(
                    connection, session_id, run_id, "failed", "工具调用已中断。", INTERRUPTED_CODE, now,
                )
        return recovered

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=BUSY_TIMEOUT_MS / 1000, isolation_level=None)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @staticmethod
    @contextmanager
    def _write(connection: sqlite3.Connection) -> Iterator[None]:
        # Reserve the writer before reading any data used to decide what to insert.
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield
            connection.commit()
        except BaseException:
            connection.rollback()
            raise

    def create_session(self, title: str) -> Session:
        session = Session(str(uuid4()), title, utc_now())
        with closing(self._connect()) as connection, self._write(connection):
            connection.execute(
                "INSERT INTO sessions (session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session.session_id, session.title, session.created_at, session.created_at),
            )
        return session

    def rename_session(self, session_id: str, title: str) -> Session:
        now = utc_now()
        with closing(self._connect()) as connection, self._write(connection):
            self._require_session(connection, session_id)
            row = connection.execute(
                """UPDATE sessions SET title = ?, updated_at = ?
                   WHERE session_id = ? RETURNING session_id, title, created_at""",
                (title, now, session_id),
            ).fetchone()
            return Session(*row)

    def delete_session(self, session_id: str) -> None:
        with closing(self._connect()) as connection, self._write(connection):
            self._require_session(connection, session_id)
            active = connection.execute(
                f"SELECT run_id FROM runs WHERE session_id = ? AND status IN ({ACTIVE_STATUSES})",
                (session_id,),
            ).fetchone()
            if active:
                raise RunConflict("RUN_ALREADY_ACTIVE", active[0])
            connection.execute("DELETE FROM run_citations WHERE session_id = ?", (session_id,))
            connection.execute("DELETE FROM tool_calls WHERE session_id = ?", (session_id,))
            connection.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            connection.execute("DELETE FROM runs WHERE session_id = ?", (session_id,))
            connection.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    def list_sessions(self) -> list[SessionSummary]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"""SELECT s.session_id, s.title, s.updated_at, r.run_id, r.status
                    FROM sessions AS s LEFT JOIN runs AS r
                      ON r.session_id = s.session_id AND r.status IN ({ACTIVE_STATUSES})
                    ORDER BY s.updated_at DESC, s.session_id DESC""",
            )
            return [SessionSummary(*row) for row in rows]

    def list_messages(self, session_id: str) -> list[Message]:
        with closing(self._connect()) as connection:
            self._require_session(connection, session_id)
            rows = connection.execute(
                """SELECT m.message_id, m.session_id, m.run_id, m.role, m.content, m.created_at
                   FROM runs AS r JOIN messages AS m ON m.run_id = r.run_id
                   WHERE r.session_id = ? AND (m.role = 'user' OR r.status = 'completed')
                   ORDER BY r.started_at, r.run_id, m.role DESC""",
                (session_id,),
            )
            return [Message(*row) for row in rows]

    def list_runs(
        self, session_id: str, limit: int, before: tuple[str, str] | None,
    ) -> list[RunSummary]:
        with closing(self._connect()) as connection:
            self._require_session(connection, session_id)
            boundary = "AND (r.started_at, r.run_id) < (?, ?)" if before else ""
            rows = connection.execute(
                f"""SELECT r.run_id, r.session_id, m.message_id, r.status, r.error_code,
                           r.started_at, r.finished_at
                    FROM runs AS r JOIN messages AS m ON m.run_id = r.run_id AND m.role = 'user'
                    WHERE r.session_id = ? {boundary}
                    ORDER BY r.started_at DESC, r.run_id DESC LIMIT ?""",
                (session_id, *(before or ()), limit),
            ).fetchall()
            if not rows:
                return []
            run_ids = [row[0] for row in rows]
            placeholders = ", ".join("?" for _ in run_ids)
            tools_by_run: dict[str, list[ToolCallSummary]] = defaultdict(list)
            for run_id, call_id, name, status, summary, error_code in connection.execute(
                f"""SELECT run_id, tool_call_id, tool_name, status, summary, error_code
                    FROM tool_calls WHERE session_id = ? AND run_id IN ({placeholders})
                    ORDER BY rowid""",
                (session_id, *run_ids),
            ):
                tools_by_run[run_id].append(ToolCallSummary(call_id, name, status, summary, error_code))

            citations_by_run: dict[str, list[StoredCitation]] = defaultdict(list)
            for (
                run_id, evidence_id, chunk_id, chunk_set_id, chunk_index, doc_id, ver_id,
                book_name, fmt, sec_path, loc_sum, loc_prec, loc_stat, warns, snippet,
            ) in connection.execute(
                f"""SELECT run_id, evidence_id, chunk_id, chunk_set_id, chunk_index,
                           document_id, document_version_id, textbook_name, format,
                           section_path, location_summary, location_precision, location_status,
                           warnings, snippet
                    FROM run_citations WHERE session_id = ? AND run_id IN ({placeholders})
                    ORDER BY ordinal""",
                (session_id, *run_ids),
            ):
                citations_by_run[run_id].append(StoredCitation(
                    evidence_id=evidence_id,
                    chunk_id=chunk_id,
                    chunk_set_id=chunk_set_id,
                    chunk_index=chunk_index,
                    document_id=doc_id,
                    document_version_id=ver_id,
                    textbook_name=book_name,
                    format=fmt,
                    section_path=tuple(json.loads(sec_path)),
                    location_summary=loc_sum,
                    location_precision=loc_prec,
                    location_status=loc_stat,
                    warnings=tuple(json.loads(warns)),
                    snippet=snippet,
                ))

            return [
                RunSummary(
                    *row,
                    tool_calls=tuple(tools_by_run.get(row[0], ())),
                    citations=tuple(citations_by_run.get(row[0], ())),
                )
                for row in rows
            ]

    def require_session(self, session_id: str) -> None:
        with closing(self._connect()) as connection:
            self._require_session(connection, session_id)

    @staticmethod
    def _require_session(connection: sqlite3.Connection, session_id: str) -> None:
        if connection.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)).fetchone() is None:
            raise SessionNotFound

    def start_run(
        self, session_id: str, client_request_id: str, message: str, *, accept_new_run: bool,
    ) -> Run | None:
        with closing(self._connect()) as connection, self._write(connection):
            session = connection.execute(
                "SELECT last_safe_checkpoint_id FROM sessions WHERE session_id = ?", (session_id,),
            ).fetchone()
            if session is None:
                raise SessionNotFound
            duplicate = connection.execute(
                "SELECT run_id FROM runs WHERE session_id = ? AND client_request_id = ?",
                (session_id, client_request_id),
            ).fetchone()
            if duplicate:
                raise RunConflict("DUPLICATE_CLIENT_REQUEST", duplicate[0])
            active = connection.execute(
                f"SELECT run_id FROM runs WHERE session_id = ? AND status IN ({ACTIVE_STATUSES})",
                (session_id,),
            ).fetchone()
            if active:
                raise RunConflict("RUN_ALREADY_ACTIVE", active[0])
            if not accept_new_run:
                return None

            run = Run(session_id, str(uuid4()), client_request_id, session[0])
            now = utc_now()
            connection.execute(
                """INSERT INTO runs (run_id, session_id, client_request_id, status, base_checkpoint_id, started_at)
                   VALUES (?, ?, ?, 'running', ?, ?)""",
                (run.run_id, session_id, client_request_id, run.base_checkpoint_id, now),
            )
            connection.execute(
                """INSERT INTO messages (message_id, session_id, run_id, role, content, created_at)
                   VALUES (?, ?, ?, 'user', ?, ?)""",
                (str(uuid4()), session_id, run.run_id, message, now),
            )
            connection.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?", (now, session_id))
        return run

    def get_run(self, run_id: str) -> RunStatus:
        with closing(self._connect()) as connection:
            return self._run_status(connection, "runs.run_id = ?", (run_id,))

    def lookup_run(self, session_id: str, client_request_id: str) -> RunStatus:
        with closing(self._connect()) as connection:
            return self._run_status(connection, "runs.session_id = ? AND runs.client_request_id = ?",
                                    (session_id, client_request_id))

    @staticmethod
    def _run_status(connection: sqlite3.Connection, predicate: str, values: tuple[str, ...]) -> RunStatus:
        row = connection.execute(
            f"""SELECT runs.run_id, runs.session_id, runs.status, runs.started_at, runs.finished_at, runs.error_code
                FROM runs JOIN sessions ON sessions.session_id = runs.session_id WHERE {predicate}""",
            values,
        ).fetchone()
        if row is None:
            raise RunNotFound
        return RunStatus(*row)

    def request_cancel(self, run: Run) -> RunStatus:
        with closing(self._connect()) as connection, self._write(connection):
            connection.execute(
                """UPDATE runs SET status = 'cancelling'
                   WHERE session_id = ? AND run_id = ? AND status = 'running'""",
                (run.session_id, run.run_id),
            )
            return self._run_status(connection, "runs.session_id = ? AND runs.run_id = ?",
                                    (run.session_id, run.run_id))

    def start_tool(self, run: Run, call_id: str, name: str) -> None:
        with closing(self._connect()) as connection, self._write(connection):
            changed = connection.execute(
                """INSERT INTO tool_calls (tool_call_id, session_id, run_id, tool_name, status, started_at)
                   SELECT ?, session_id, run_id, ?, 'running', ? FROM runs
                   WHERE session_id = ? AND run_id = ? AND status IN ('running', 'cancelling')""",
                (call_id, name, utc_now(), run.session_id, run.run_id),
            ).rowcount
            if changed != 1:
                raise RuntimeError("Tool call requires an active run")

    def finish_tool(self, run: Run, call_id: str, status: str, summary: str, error_code: str | None) -> None:
        with closing(self._connect()) as connection, self._write(connection):
            changed = connection.execute(
                """UPDATE tool_calls SET status = ?, summary = ?, error_code = ?, finished_at = ?
                   WHERE session_id = ? AND run_id = ? AND tool_call_id = ? AND status = 'running'""",
                (status, summary, error_code, utc_now(), run.session_id, run.run_id, call_id),
            ).rowcount
            if changed != 1:
                raise RuntimeError("Tool call has no active audit entry")

    def complete_run(
        self, run: Run, reply: str, final_checkpoint_id: str, citations: Sequence[StoredCitation] = (),
    ) -> str | None:
        if not final_checkpoint_id or final_checkpoint_id == run.base_checkpoint_id:
            raise ValueError("Run completion requires a new checkpoint")
        now, message_id = utc_now(), str(uuid4())
        with closing(self._connect()) as connection, self._write(connection):
            if connection.execute(
                """SELECT 1 FROM tool_calls WHERE session_id = ? AND run_id = ? AND status != 'completed'""",
                (run.session_id, run.run_id),
            ).fetchone():
                raise RuntimeError("Run completion requires successful tool calls")
            if not self._finish(connection, run, "completed", now, from_statuses=("running",),
                                final_checkpoint_id=final_checkpoint_id):
                return None
            connection.execute(
                """INSERT INTO messages (message_id, session_id, run_id, role, content, created_at)
                   VALUES (?, ?, ?, 'assistant', ?, ?)""",
                (message_id, run.session_id, run.run_id, reply, now),
            )
            for ordinal, item in enumerate(citations):
                connection.execute(
                    """INSERT INTO run_citations (
                        run_id, session_id, ordinal, evidence_id, chunk_id, chunk_set_id, chunk_index,
                        document_id, document_version_id, textbook_name, format, section_path,
                        location_summary, location_precision, location_status, warnings, snippet
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run.run_id, run.session_id, ordinal, item.evidence_id, item.chunk_id,
                        item.chunk_set_id, item.chunk_index, item.document_id, item.document_version_id,
                        item.textbook_name, item.format,
                        json.dumps(list(item.section_path), ensure_ascii=False, separators=(",", ":")),
                        item.location_summary, item.location_precision, item.location_status,
                        json.dumps(list(item.warnings), ensure_ascii=False, separators=(",", ":")),
                        item.snippet,
                    ),
                )
            updated = connection.execute(
                """UPDATE sessions SET last_safe_checkpoint_id = ?
                   WHERE session_id = ? AND last_safe_checkpoint_id IS ?""",
                (final_checkpoint_id, run.session_id, run.base_checkpoint_id),
            ).rowcount
            if updated != 1:
                raise RuntimeError("Session checkpoint changed during run")
        return message_id

    def fail_run(self, run: Run, code: str, message: str) -> bool:
        with closing(self._connect()) as connection, self._write(connection):
            return self._finish(connection, run, "failed", utc_now(), code, message,
                                from_statuses=("running", "cancelling"))

    def confirm_cancel(self, run: Run, *, shutdown: bool = False) -> bool:
        with closing(self._connect()) as connection, self._write(connection):
            return self._finish(connection, run, "cancelled", utc_now(),
                                from_statuses=("running",) if shutdown else ("cancelling",))

    @staticmethod
    def _finish(
        connection: sqlite3.Connection, run: Run, status: str, now: str,
        error_code: str | None = None, error_message: str | None = None, *, from_statuses: tuple[str, ...],
        final_checkpoint_id: str | None = None,
    ) -> bool:
        placeholders = ", ".join("?" for _ in from_statuses)
        changed = connection.execute(
            f"""UPDATE runs SET status = ?, finished_at = ?, error_code = ?, error_message = ?, final_checkpoint_id = ?
                WHERE session_id = ? AND run_id = ? AND status IN ({placeholders})""",
            (status, now, error_code, error_message, final_checkpoint_id, run.session_id, run.run_id, *from_statuses),
        ).rowcount
        if changed:
            if status == "cancelled":
                BusinessStore._finish_pending_tools(
                    connection, run.session_id, run.run_id, "cancelled", "工具调用已取消。", "RUN_CANCELLED", now,
                )
            elif status == "failed":
                BusinessStore._finish_pending_tools(
                    connection, run.session_id, run.run_id, "failed", "工具调用已中断。", INTERRUPTED_CODE, now,
                )
            connection.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?", (now, run.session_id))
        return changed == 1

    @staticmethod
    def _finish_pending_tools(
        connection: sqlite3.Connection, session_id: str, run_id: str, status: str,
        summary: str, error_code: str, now: str,
    ) -> None:
        connection.execute(
            """UPDATE tool_calls SET status = ?, summary = ?, error_code = ?, finished_at = ?
               WHERE session_id = ? AND run_id = ? AND status = 'running'""",
            (status, summary, error_code, now, session_id, run_id),
        )
