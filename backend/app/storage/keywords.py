import sqlite3
from threading import Event

from app.parsing.worker import check_cancelled
from app.storage.business import utc_now
from app.storage.chunks import PUBLISHED_SETS
from app.storage.index_jobs import PIPELINE_MANAGED_SET, IndexJobClaim, IndexJobError, IndexJobStore
from app.storage.keyword_schema import (
    CREATE_TRIGGERS, FTS_SHADOW_TABLES, INDEX_OBJECTS, INDEX_STORAGE, INDEX_TRIGGERS,
)
from app.storage.library import KnowledgeStore

INDEX_BATCH_SIZE = 256
COMPLETE_KEYWORD_ROWS = """(SELECT count(*) FROM chunks WHERE chunk_set_id = s.chunk_set_id)
    = (SELECT count(*) FROM keyword_index_rows WHERE chunk_set_id = s.chunk_set_id)"""
PENDING_SETS = f"""SELECT s.chunk_set_id {PUBLISHED_SETS}
    LEFT JOIN keyword_index_sets AS i USING (chunk_set_id)
    WHERE s.status = 'ready' AND i.indexed_at IS NULL AND NOT {PIPELINE_MANAGED_SET}
    ORDER BY s.chunk_set_id LIMIT ?"""


class KeywordIndexError(RuntimeError):
    """Keyword index operation failed with the given diagnostic code."""


def _index_state(db: sqlite3.Connection, set_id: int) -> tuple[int | None, str | None]:
    row = db.execute(f"""SELECT i.last_chunk_index, i.indexed_at {PUBLISHED_SETS}
        LEFT JOIN keyword_index_sets AS i USING (chunk_set_id)
        WHERE s.chunk_set_id = ? AND s.status = 'ready'""", (set_id,)).fetchone()
    if row is None:
        raise KeywordIndexError("KEYWORD_CHUNK_SET_UNAVAILABLE")
    return row


def _drop_fts(db: sqlite3.Connection) -> None:
    try:
        db.execute("DROP TABLE IF EXISTS keyword_fts")
    except sqlite3.OperationalError as error:
        # FTS5 needs a readable format record even to drop a virtual table.
        if (str(error) != "vtable constructor failed: keyword_fts"
                and not str(error).startswith("invalid fts5 file format")):
            raise
        db.execute("CREATE VIRTUAL TABLE temp.keyword_fts_template USING fts5(body)")
        db.execute("DROP TABLE IF EXISTS main.keyword_fts_config")
        db.execute("CREATE TABLE main.keyword_fts_config (k PRIMARY KEY, v) WITHOUT ROWID")
        db.execute("INSERT INTO main.keyword_fts_config SELECT k, v FROM temp.keyword_fts_template_config")
        db.execute("DROP TABLE keyword_fts")
        db.execute("DROP TABLE temp.keyword_fts_template")


class KeywordIndex:
    def __init__(self, store: KnowledgeStore):
        self.store = store

    def ensure_indexed(
        self, chunk_set_id: int, *, claim: IndexJobClaim | None = None, cancelled: Event | None = None,
    ) -> None:
        """Resume a published set in batches and expose it only after the final commit."""
        with self.store.connect() as db:
            if _index_state(db, chunk_set_id)[1] is not None:
                return
        while True:
            check_cancelled(cancelled)
            if not self._index_batch(chunk_set_id, claim=claim):
                return

    def _index_batch(self, set_id: int, *, claim: IndexJobClaim | None = None) -> bool:
        with self.store.write() as db:
            cursor, indexed_at = _index_state(db, set_id)
            if indexed_at is not None:
                return False
            jobs = IndexJobStore(self.store)
            jobs.require_index_publication(db, set_id, claim)
            if cursor is None:
                db.execute("INSERT INTO keyword_index_sets(chunk_set_id) VALUES (?)", (set_id,))
                cursor = -1
            rows = db.execute("""INSERT INTO keyword_index_rows(chunk_set_id, chunk_index)
                SELECT chunk_set_id, chunk_index FROM chunks
                WHERE chunk_set_id = ? AND chunk_index > ? ORDER BY chunk_index LIMIT ?
                RETURNING chunk_index""", (set_id, cursor, INDEX_BATCH_SIZE)).fetchall()
            cursor = max((index for index, in rows), default=cursor)
            more = db.execute("SELECT 1 FROM chunks WHERE chunk_set_id = ? AND chunk_index > ? LIMIT 1",
                              (set_id, cursor)).fetchone() is not None
            if not more:
                if db.execute(f"""SELECT 1 FROM chunk_sets AS s
                    WHERE s.chunk_set_id = ? AND {COMPLETE_KEYWORD_ROWS}""", (set_id,)).fetchone() is None:
                    raise KeywordIndexError("KEYWORD_INDEX_INCONSISTENT")
                jobs.require_index_publication(db, set_id, claim)
            db.execute("UPDATE keyword_index_sets SET last_chunk_index = ?, indexed_at = ? WHERE chunk_set_id = ?",
                       (cursor, None if more else utc_now(), set_id))
            return more

    def index_pending(self, *, limit: int = 32) -> tuple[int, ...]:
        """Index up to limit pending published sets, including sets migrated from v5."""
        if type(limit) is not int or not 1 <= limit <= 128:
            raise ValueError("Invalid pending set limit.")
        with self.store.connect() as db:
            set_ids = tuple(set_id for set_id, in db.execute(PENDING_SETS, (limit,)))
        indexed = []
        for set_id in set_ids:
            try:
                self.ensure_indexed(set_id)
            except IndexJobError as error:
                if str(error) != "INDEX_PUBLICATION_REQUIRES_CLAIM":
                    raise
            except KeywordIndexError as error:
                if str(error) != "KEYWORD_CHUNK_SET_UNAVAILABLE":
                    raise
            else:
                indexed.append(set_id)
        return tuple(indexed)

    def check_integrity(self) -> None:
        """Check schema, batch coverage, and FTS postings against published source text."""
        with self.store.write() as db:
            self._check(db)

    @staticmethod
    def _check(db: sqlite3.Connection) -> None:
        names = {name for name, in db.execute("SELECT name FROM sqlite_schema")}
        if not INDEX_OBJECTS <= names:
            raise KeywordIndexError("KEYWORD_INDEX_MISSING")
        if db.execute("""SELECT 1 FROM keyword_index_sets AS i
            LEFT JOIN chunk_sets AS s USING (chunk_set_id)
            LEFT JOIN parsed_documents AS p ON p.parsed_document_id = s.parsed_document_id
            WHERE s.status IS NOT 'ready' OR p.status IS NOT 'ready' LIMIT 1""").fetchone() or db.execute(
            """SELECT 1 FROM chunks AS c JOIN keyword_index_sets AS i USING (chunk_set_id)
               LEFT JOIN keyword_index_rows AS r USING (chunk_set_id, chunk_index)
               WHERE (i.indexed_at IS NOT NULL OR c.chunk_index <= i.last_chunk_index)
                     != (r.row_id IS NOT NULL) LIMIT 1"""
        ).fetchone() or db.execute("""SELECT 1 FROM keyword_index_rows AS r
            LEFT JOIN keyword_index_sets AS i USING (chunk_set_id)
            LEFT JOIN chunks AS c USING (chunk_set_id, chunk_index)
            WHERE i.chunk_set_id IS NULL OR c.chunk_id IS NULL LIMIT 1""").fetchone():
            raise KeywordIndexError("KEYWORD_INDEX_INCONSISTENT")
        try:
            db.execute("INSERT INTO keyword_fts(keyword_fts, rank) VALUES ('integrity-check', 1)")
        except sqlite3.DatabaseError as error:
            raise KeywordIndexError("KEYWORD_INDEX_CHECK_FAILED") from error

    def rebuild(self) -> None:
        """Repair published indexes and backfill legacy sets in one transaction."""
        with self.store.write() as db:
            db.execute("CREATE TEMP TABLE previous_keyword_publications (chunk_set_id INTEGER PRIMARY KEY)")
            if db.execute("SELECT 1 FROM sqlite_schema WHERE name = 'keyword_index_sets'").fetchone():
                db.execute("""INSERT INTO previous_keyword_publications
                    SELECT chunk_set_id FROM keyword_index_sets WHERE indexed_at IS NOT NULL""")
            for name in INDEX_TRIGGERS:
                db.execute(f"DROP TRIGGER IF EXISTS {name}")
            _drop_fts(db)
            # An externally removed virtual table can leave orphan FTS shadow objects.
            for name in FTS_SHADOW_TABLES:
                db.execute(f"DROP TABLE IF EXISTS {name}")
            db.execute("DROP VIEW IF EXISTS keyword_index_content")
            db.execute("DROP TABLE IF EXISTS keyword_index_rows")
            db.execute("DROP TABLE IF EXISTS keyword_index_sets")
            for statement in INDEX_STORAGE:
                db.execute(statement)
            db.execute(f"""INSERT INTO keyword_index_sets(chunk_set_id, indexed_at)
                SELECT s.chunk_set_id, ? {PUBLISHED_SETS} WHERE s.status = 'ready'
                AND (NOT {PIPELINE_MANAGED_SET}
                    OR s.chunk_set_id IN (SELECT chunk_set_id FROM previous_keyword_publications))""", (utc_now(),))
            db.execute("""INSERT INTO keyword_index_rows(chunk_set_id, chunk_index)
                SELECT c.chunk_set_id, c.chunk_index FROM keyword_index_sets AS i
                JOIN chunks AS c USING (chunk_set_id) ORDER BY c.chunk_set_id, c.chunk_index""")
            db.execute("INSERT INTO keyword_fts(keyword_fts) VALUES ('rebuild')")
            for statement in CREATE_TRIGGERS:
                db.execute(statement)
            self._check(db)
