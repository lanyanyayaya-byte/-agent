import json
import sqlite3
from collections.abc import Callable, Iterable, Iterator
from contextlib import suppress
from dataclasses import dataclass

from app.chunking.models import Chunk, ChunkSource, ChunkSourceSpan
from app.parsing.models import DocumentParseError
from app.storage.business import utc_now
from app.storage.index_jobs import IndexJobClaim, IndexJobStore
from app.storage.library import KnowledgeStore
from app.storage.parsed_documents import (
    BATCH_ROWS, MAX_PAGE_SIZE, WRITE_LEASE_SECONDS, collect_locations, iter_write_batches, read_ready_parse,
)

SET_FIELDS = """chunk_set_id, parsed_document_id, document_version_id, chunker_version, chunker_config_key,
    created_at, published_at, chunk_count, source_span_count"""
PUBLISHED_SETS = """FROM chunk_sets AS s JOIN parsed_documents AS p
                   ON p.parsed_document_id = s.parsed_document_id AND p.status = 'ready'"""
ACTIVE_SET = f"""SELECT {SET_FIELDS}, status, lease_expires_at > unixepoch() FROM chunk_sets
    WHERE parsed_document_id = ? AND chunker_version = ? AND chunker_config_key = ?
       AND status IN ('writing', 'ready')"""
EXPIRED_SET = """SELECT chunk_set_id FROM chunk_sets WHERE status IN ('writing', 'abandoned')
    AND lease_expires_at <= unixepoch() ORDER BY lease_expires_at, chunk_set_id LIMIT 1"""
CHUNK_FIELDS = "c.chunk_index, c.chunk_id, c.text, c.section_heading_block_id, c.section_path, c.title_context"
CHUNK_BY_ID = f"""SELECT s.chunk_set_id, s.parsed_document_id, s.chunker_version, s.chunker_config_key, {CHUNK_FIELDS}
    FROM chunks AS c JOIN chunk_sets AS s USING (chunk_set_id) WHERE c.chunk_id = ? AND s.status = 'ready'"""
REFERENCED_BLOCKS = """FROM chunk_sources AS s JOIN parsed_blocks AS b
    ON b.parsed_document_id = s.parsed_document_id AND b.block_id = s.source_block_id
    WHERE s.chunk_set_id = ?"""
INSERT_CHUNK = """INSERT INTO chunks (chunk_set_id, chunk_index, chunk_id, text, text_length,
    section_heading_block_id, section_path, title_context) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"""
INSERT_SOURCE = """INSERT INTO chunk_sources (chunk_set_id, chunk_index, source_index, parsed_document_id,
    source_block_id, block_start, block_end, chunk_start, chunk_end, is_overlap) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""


@dataclass(frozen=True, slots=True)
class ChunkSet:
    chunk_set_id: int
    parsed_document_id: int
    document_version_id: str
    chunker_version: str
    chunker_config_key: str
    created_at: str
    published_at: str
    chunk_count: int
    source_span_count: int


@dataclass(frozen=True, slots=True)
class ChunkPage:
    chunks: tuple[Chunk, ...]
    next_after_index: int | None


def _rows(set_id: int, parsed_id: int, version_id: str, chunker_version: str, config_key: str,
          chunks: Iterable[Chunk]) -> Iterator[tuple]:
    for index, chunk in enumerate(chunks):
        if (chunk.chunk_index, chunk.parsed_document_id, chunk.document_version_id,
                chunk.chunker_version, chunk.chunker_config_key) != (
                index, parsed_id, version_id, chunker_version, config_key) or not chunk.source_spans:
            raise DocumentParseError("CHUNK_RESULT_INVALID", "Chunk identity or source spans do not match the build.")
        yield ("chunk", set_id, index, chunk.chunk_id, chunk.text, len(chunk.text), chunk.section_heading_block_id,
               json.dumps(chunk.section_path, ensure_ascii=False, separators=(",", ":")),
               json.dumps(chunk.title_context, ensure_ascii=False, separators=(",", ":")))
        for position, span in enumerate(chunk.source_spans):
            yield ("source", set_id, index, position, parsed_id, span.block_id, span.block_start, span.block_end,
                   span.chunk_start, span.chunk_end, span.is_overlap)


def _read_spans(db: sqlite3.Connection, set_id: int, first: int, last: int) -> dict[int, list[ChunkSourceSpan]]:
    return _read_source_spans(db, "s.chunk_index BETWEEN ? AND ?", (set_id, first, last))


def read_chunk_source_spans(db: sqlite3.Connection, set_id: int,
                            indices: Iterable[int]) -> dict[int, list[ChunkSourceSpan]]:
    """Read exact chunk indices and their source locations in the caller's snapshot."""
    return _read_source_spans(db, "s.chunk_index IN (SELECT value FROM json_each(?))",
                              (set_id, json.dumps(list(indices))))


def _read_source_spans(db: sqlite3.Connection, index_filter: str, params: tuple) -> dict[int, list[ChunkSourceSpan]]:
    referenced = f"{REFERENCED_BLOCKS} AND {index_filter}"
    locations = collect_locations(db.execute(
        f"""WITH referenced AS (SELECT DISTINCT b.parsed_document_id, b.position {referenced})
            SELECT l.block_position, l.kind, l.details FROM referenced AS r JOIN source_locations AS l
              ON l.parsed_document_id = r.parsed_document_id AND l.block_position = r.position
            ORDER BY l.block_position, l.position""", params,
    ))
    locations_by_block = {position: tuple(items) for position, items in locations.items()}
    spans: dict[int, list[ChunkSourceSpan]] = {}
    for index, block_id, kind, block_start, block_end, chunk_start, chunk_end, overlap, block_position in db.execute(
        f"""SELECT s.chunk_index, s.source_block_id, b.kind, s.block_start, s.block_end, s.chunk_start, s.chunk_end,
                   s.is_overlap, b.position {referenced} ORDER BY s.chunk_index, s.source_index""", params,
    ):
        spans.setdefault(index, []).append(ChunkSourceSpan(
            block_id, kind, block_start, block_end, chunk_start, chunk_end,
            locations_by_block.get(block_position, ()), bool(overlap),
        ))
    return spans


def _read_chunks(db: sqlite3.Connection, set_id: int, parsed_id: int, chunker_version: str,
                 config_key: str, rows: list[tuple]) -> tuple[Chunk, ...]:
    if not rows:
        return ()
    parsed = read_ready_parse(db, parsed_id)
    if parsed is None:
        raise DocumentParseError("PARSED_DOCUMENT_UNAVAILABLE", "The chunk set's published parse is unavailable.")
    sha256, = db.execute("SELECT content_sha256 FROM document_versions WHERE document_version_id = ?",
                         (parsed.document_version_id,)).fetchone()
    source = ChunkSource(parsed.document_id, parsed.document_version_id, sha256, parsed_id, parsed.spec,
                         parsed.format, parsed.warnings)
    spans = _read_spans(db, set_id, rows[0][0], rows[-1][0])
    return tuple(Chunk(chunk_id, source, index, text, heading, tuple(json.loads(path)), tuple(json.loads(title)),
                       tuple(spans[index]), chunker_version, config_key)
                 for index, chunk_id, text, heading, path, title in rows)


class ChunkStore:
    """Store immutable chunk sets against exact parses in the knowledge library."""

    def __init__(self, store: KnowledgeStore):
        self.store = store

    def get_chunk_set(self, parsed_id: int, chunker_version: str, config_key: str) -> ChunkSet | None:
        with self.store.connect() as db:
            row = db.execute(ACTIVE_SET, (parsed_id, chunker_version, config_key)).fetchone()
        return ChunkSet(*row[:-2]) if row is not None and row[-2] == "ready" else None

    def list_chunks(self, set_id: int, *, limit: int = 100, after_index: int = -1) -> ChunkPage | None:
        """Read a keyset page, or return None for an unavailable set."""
        if not 1 <= limit <= MAX_PAGE_SIZE or after_index < -1:
            raise ValueError("Invalid chunk pagination parameters.")
        with self.store.connect() as db, db:
            db.execute("BEGIN")
            ready = db.execute("""SELECT parsed_document_id, chunker_version, chunker_config_key, chunk_count
                                  FROM chunk_sets WHERE chunk_set_id = ? AND status = 'ready'""", (set_id,)).fetchone()
            if ready is None:
                return None
            parsed_id, chunker_version, config_key, chunk_count = ready
            rows = db.execute(
                f"""SELECT {CHUNK_FIELDS} FROM chunks AS c WHERE c.chunk_set_id = ? AND c.chunk_index > ?
                    ORDER BY c.chunk_index LIMIT ?""", (set_id, after_index, limit),
            ).fetchall()
            chunks = _read_chunks(db, set_id, parsed_id, chunker_version, config_key, rows)
            next_after_index = rows[-1][0] if rows and rows[-1][0] + 1 < chunk_count else None
            return ChunkPage(chunks, next_after_index)

    def iter_chunks(self, set_id: int, *, page_size: int = 100) -> Iterator[Chunk]:
        cursor = -1
        while True:
            page = self.list_chunks(set_id, limit=page_size, after_index=cursor)
            if page is None:
                raise DocumentParseError("CHUNK_SET_UNAVAILABLE", "The requested published chunk set is unavailable.")
            yield from page.chunks
            if page.next_after_index is None:
                return
            cursor = page.next_after_index

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Return None when the ID no longer points to a published chunk."""
        with self.store.connect() as db, db:
            db.execute("BEGIN")
            row = db.execute(CHUNK_BY_ID, (chunk_id,)).fetchone()
            if row is None:
                return None
            set_id, parsed_id, version, config_key, *chunk = row
            return _read_chunks(db, set_id, parsed_id, version, config_key, [tuple(chunk)])[0]

    def get_chunk_sources(self, chunk_id: str) -> tuple[ChunkSourceSpan, ...] | None:
        with self.store.connect() as db, db:
            db.execute("BEGIN")
            row = db.execute("""SELECT c.chunk_set_id, c.chunk_index FROM chunks AS c JOIN chunk_sets AS s USING (chunk_set_id)
                                WHERE c.chunk_id = ? AND s.status = 'ready'""", (chunk_id,)).fetchone()
            if row is None:
                return None
            set_id, index = row
            return tuple(_read_spans(db, set_id, index, index)[index])

    def ensure(self, parsed_id: int, chunker_version: str, config_key: str,
               generate: Callable[[], Iterable[Chunk]], *, claim: IndexJobClaim | None = None) -> ChunkSet:
        """Build once per identity, with a retryable error while another writer holds the lease."""
        ready = self.get_chunk_set(parsed_id, chunker_version, config_key)
        if ready is not None:
            return ready
        with self.store.write() as db:
            owner_deadline = self._owner_deadline(db, claim)
            row = db.execute(ACTIVE_SET, (parsed_id, chunker_version, config_key)).fetchone()
            if row is not None:
                state, lease_active = row[-2:]
                if state == "ready":
                    return ChunkSet(*row[:-2])
                if lease_active:
                    raise DocumentParseError("CHUNK_BUILD_BUSY", "A chunk set with this configuration is being built.")
                db.execute("UPDATE chunk_sets SET status = 'abandoned' WHERE chunk_set_id = ?", (row[0],))
            reserved = db.execute(
                """INSERT INTO chunk_sets (parsed_document_id, document_version_id, chunker_version, chunker_config_key,
                       created_at, status, lease_expires_at)
                   SELECT parsed_document_id, document_version_id, ?, ?, ?, 'writing', min(unixepoch() + ?, ?)
                   FROM parsed_documents WHERE parsed_document_id = ? AND status = 'ready'
                   RETURNING chunk_set_id, document_version_id""",
                (chunker_version, config_key, utc_now(), WRITE_LEASE_SECONDS, owner_deadline, parsed_id),
            ).fetchone()
            if reserved is None:
                raise DocumentParseError("PARSED_DOCUMENT_UNAVAILABLE", "The requested published parse is unavailable.")
        set_id, version_id = reserved
        try:
            counts = self._save_batches(set_id, _rows(set_id, parsed_id, version_id, chunker_version,
                                                      config_key, generate()), claim)
            return self._publish(set_id, *counts, claim)
        except BaseException:
            # Process death skips this path. The last committed lease still expires.
            with suppress(sqlite3.Error), self.store.write(wait=False) as db:
                db.execute("""UPDATE chunk_sets SET status = 'abandoned', lease_expires_at = 0
                              WHERE chunk_set_id = ? AND status = 'writing'""", (set_id,))
            raise

    @staticmethod
    def _check_lease(changed: int) -> None:
        if changed != 1:
            raise DocumentParseError("CHUNK_WRITE_INTERRUPTED", "The staged chunk set lost its write lease.")

    def _owner_deadline(self, db: sqlite3.Connection, claim: IndexJobClaim | None) -> int:
        if claim is None:
            return 2**63 - 1
        # An unfinished chunk build must be reclaimable when the job lease expires.
        job = IndexJobStore(self.store).require_active_claim(db, claim)
        assert job.lease_expires_at is not None
        return job.lease_expires_at

    def _save_batches(self, set_id: int, rows: Iterable[tuple], claim: IndexJobClaim | None) -> tuple[int, int]:
        chunk_count = source_count = 0
        for batch in iter_write_batches(rows):
            chunks = [row[1:] for row in batch if row[0] == "chunk"]
            sources = [row[1:] for row in batch if row[0] == "source"]
            with self.store.write() as db:
                owner_deadline = self._owner_deadline(db, claim)
                self._check_lease(db.execute(
                    """UPDATE chunk_sets SET lease_expires_at = min(unixepoch() + ?, ?) WHERE chunk_set_id = ?
                       AND status = 'writing' AND lease_expires_at > unixepoch()""",
                    (WRITE_LEASE_SECONDS, owner_deadline, set_id),
                ).rowcount)
                saved_chunks = db.executemany(INSERT_CHUNK, chunks).rowcount
                saved_sources = db.executemany(INSERT_SOURCE, sources).rowcount
                if (saved_chunks, saved_sources) != (len(chunks), len(sources)):
                    raise DocumentParseError("CHUNK_RESULT_INVALID", "A chunk batch was not saved in full.")
            chunk_count += saved_chunks
            source_count += saved_sources
        return chunk_count, source_count

    def _publish(self, set_id: int, chunk_count: int, source_count: int, claim: IndexJobClaim | None) -> ChunkSet:
        with self.store.write() as db:
            self._owner_deadline(db, claim)
            row = db.execute(
                f"""UPDATE chunk_sets SET status = 'ready', lease_expires_at = NULL, published_at = ?,
                       chunk_count = ?, source_span_count = ? WHERE chunk_set_id = ?
                       AND status = 'writing' AND lease_expires_at > unixepoch() RETURNING {SET_FIELDS}""",
                (utc_now(), chunk_count, source_count, set_id),
            ).fetchone()
            self._check_lease(int(row is not None))
        return ChunkSet(*row)

    def cleanup_interrupted(self, *, max_batches: int = 1) -> int:
        """Purge expired staging rows a bounded batch at a time."""
        for batch in range(max_batches):
            with self.store.connect() as db:
                if db.execute(EXPIRED_SET).fetchone() is None:
                    return batch
            with self.store.write(wait=False) as db:
                row = db.execute(EXPIRED_SET).fetchone()
                if row is None:
                    return batch
                self._purge_batch(db, row[0])
        return max_batches

    @staticmethod
    def _purge_batch(db: sqlite3.Connection, set_id: int) -> None:
        db.execute("UPDATE chunk_sets SET status = 'abandoned' WHERE chunk_set_id = ? AND status = 'writing'", (set_id,))
        for table, key in (("chunk_sources", "chunk_index, source_index"), ("chunks", "chunk_index")):
            removed = db.execute(
                f"""DELETE FROM {table} WHERE (chunk_set_id, {key}) IN (
                    SELECT chunk_set_id, {key} FROM {table} WHERE chunk_set_id = ? ORDER BY {key} LIMIT ?)""",
                (set_id, BATCH_ROWS),
            ).rowcount
            if removed:
                return
        db.execute("DELETE FROM chunk_sets WHERE chunk_set_id = ?", (set_id,))
