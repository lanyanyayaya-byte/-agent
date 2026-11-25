import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import suppress
from dataclasses import asdict, dataclass
from threading import Event
from typing import Literal

from app.parsing.models import (
    ContentBlock, DocumentParseError, DocxSourceLocation, MarkdownSourceLocation, ParsedDocument,
    ParseWarning, PdfSourceLocation, Section, SourceLocation,
)
from app.parsing.worker import check_cancelled
from app.storage.business import utc_now
from app.storage.library import DocumentFormat, KnowledgeStore

BATCH_ROWS = 512
BATCH_TEXT_BYTES = 2 * 1024 * 1024
WRITE_LEASE_SECONDS = 300
MAX_PAGE_SIZE = 1000

LOCATION_TYPES = {"pdf": PdfSourceLocation, "docx": DocxSourceLocation, "md": MarkdownSourceLocation}
LOCATION_KINDS = {location: kind for kind, location in LOCATION_TYPES.items()}

READY_PARSE = """SELECT p.parsed_document_id, p.document_version_id, v.document_id, d.format,
    p.parser_version, p.structure_version, p.config_key, p.parsed_at, p.published_at, p.warnings
    FROM parsed_documents AS p JOIN document_versions AS v USING (document_version_id)
    JOIN documents AS d USING (document_id) WHERE p.status = 'ready'"""
PARSE_BY_VERSION = READY_PARSE + " AND p.document_version_id = ?"
EXACT_SPEC = " AND p.parser_version = ? AND p.structure_version = ? AND p.config_key = ?"
LATEST_READY = " ORDER BY p.published_at DESC, p.parsed_document_id DESC LIMIT 1"
READY_VERSION = "SELECT document_version_id FROM parsed_documents WHERE parsed_document_id = ? AND status = 'ready'"
BLOCK_COLUMNS = "position, block_id, kind, text, section_heading_block_id, section_path, code_language"
EXPIRED_WRITER = """SELECT parsed_document_id FROM parsed_documents
    WHERE status = 'writing' AND lease_expires_at <= unixepoch()
    ORDER BY lease_expires_at, parsed_document_id LIMIT 1"""


@dataclass(frozen=True)
class ParseSpec:
    parser_version: str
    structure_version: int
    config_key: str


@dataclass(frozen=True)
class StoredParse:
    parsed_document_id: int
    document_version_id: str
    document_id: str
    format: DocumentFormat
    spec: ParseSpec
    parsed_at: str
    published_at: str
    warnings: tuple[ParseWarning, ...]


@dataclass(frozen=True)
class ParseAvailability:
    state: Literal["missing", "stale", "ready"]
    result: StoredParse | None


@dataclass(frozen=True)
class BlockPage:
    blocks: tuple[ContentBlock, ...]
    next_after_position: int | None


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def iter_write_batches(rows: Iterable[tuple]) -> Iterator[list[tuple]]:
    batch = []
    text_bytes = 0
    for row in rows:
        # Bound the UTF-8 size without encoding every string a second time.
        size = sum(4 * len(value) for value in row if isinstance(value, str))
        if batch and (len(batch) >= BATCH_ROWS or text_bytes + size > BATCH_TEXT_BYTES):
            yield batch
            batch = []
            text_bytes = 0
        batch.append(row)
        text_bytes += size
    if batch:
        yield batch


def _block_rows(parsed_id: int, parsed: ParsedDocument) -> Iterator[tuple]:
    for position, block in enumerate(parsed.blocks):
        if block.document_version_id != parsed.document_version_id:
            raise DocumentParseError("PARSED_VERSION_MISMATCH", "A block belongs to another document version.")
        yield (parsed_id, position, block.block_id, block.kind, block.text, block.section_heading_block_id,
               _json(block.section_path), block.code_language)


def _location_rows(parsed_id: int, parsed: ParsedDocument) -> Iterator[tuple]:
    for block_position, block in enumerate(parsed.blocks):
        for position, location in enumerate(block.source_locations):
            yield parsed_id, block_position, position, LOCATION_KINDS[type(location)], _json(asdict(location))


def _stored_parse(row: tuple) -> StoredParse:
    parsed_id, version_id, document_id, format, parser, structure, config, parsed_at, published_at, warnings = row
    return StoredParse(parsed_id, version_id, document_id, format, ParseSpec(parser, structure, config),
                       parsed_at, published_at, tuple(ParseWarning(**item) for item in json.loads(warnings)))


def read_ready_parse(db: sqlite3.Connection, parsed_id: int) -> StoredParse | None:
    row = db.execute(READY_PARSE + " AND p.parsed_document_id = ?", (parsed_id,)).fetchone()
    return _stored_parse(row) if row is not None else None


def collect_locations(rows: Iterable[tuple]) -> dict[int, list[SourceLocation]]:
    locations: dict[int, list[SourceLocation]] = {}
    for block_position, kind, details in rows:
        fields = json.loads(details)
        if kind == "pdf" and fields["page_bbox"] is not None:
            fields["page_bbox"] = tuple(fields["page_bbox"])
        locations.setdefault(block_position, []).append(LOCATION_TYPES[kind](**fields))
    return locations


def _locations(db: sqlite3.Connection, parsed_id: int, first: int, last: int) -> dict[int, list[SourceLocation]]:
    return collect_locations(db.execute(
        """SELECT block_position, kind, details FROM source_locations
           WHERE parsed_document_id = ? AND block_position BETWEEN ? AND ? ORDER BY block_position, position""",
        (parsed_id, first, last),
    ))


def _read_block(row: tuple, version_id: str, locations: Iterable[SourceLocation]) -> ContentBlock:
    _, block_id, kind, text, section, path, language = row
    return ContentBlock(block_id, version_id, kind, text, tuple(locations), section, tuple(json.loads(path)), language)


class ParsedDocumentStore:
    """Publishes complete parses and reads immutable results by their generated ID."""

    def __init__(self, store: KnowledgeStore):
        self.store = store

    @staticmethod
    def _find_exact(db: sqlite3.Connection, version_id: str, spec: ParseSpec) -> tuple | None:
        return db.execute(PARSE_BY_VERSION + EXACT_SPEC,
                          (version_id, spec.parser_version, spec.structure_version, spec.config_key)).fetchone()

    def get_by_id(self, parsed_id: int) -> StoredParse | None:
        with self.store.connect() as db:
            return read_ready_parse(db, parsed_id)

    def get_reusable(self, version_id: str, spec: ParseSpec) -> StoredParse | None:
        with self.store.connect() as db:
            row = self._find_exact(db, version_id, spec)
        return _stored_parse(row) if row is not None else None

    def get_availability(self, version_id: str, spec: ParseSpec) -> ParseAvailability:
        with self.store.connect() as db, db:
            db.execute("BEGIN")
            row = self._find_exact(db, version_id, spec)
            if row is not None:
                return ParseAvailability("ready", _stored_parse(row))
            row = db.execute(PARSE_BY_VERSION + LATEST_READY, (version_id,)).fetchone()
            if row is not None:
                return ParseAvailability("stale", _stored_parse(row))
        return ParseAvailability("missing", None)

    def get_sections(self, parsed_id: int) -> tuple[Section, ...] | None:
        with self.store.connect() as db, db:
            db.execute("BEGIN")
            if db.execute(READY_VERSION, (parsed_id,)).fetchone() is None:
                return None
            return tuple(Section(*row) for row in db.execute(
                """SELECT title, level, heading_block_id, parent_heading_block_id FROM parsed_sections
                   WHERE parsed_document_id = ? ORDER BY position""", (parsed_id,),
            ))

    def list_blocks(self, parsed_id: int, *, limit: int = 100, after_position: int = -1) -> BlockPage | None:
        """Use the previous page's next_after_position with the same parsed ID."""
        if not 1 <= limit <= MAX_PAGE_SIZE or after_position < -1:
            raise ValueError("Invalid block pagination parameters")
        with self.store.connect() as db, db:
            db.execute("BEGIN")
            version = db.execute(READY_VERSION, (parsed_id,)).fetchone()
            if version is None:
                return None
            rows = db.execute(
                f"""SELECT {BLOCK_COLUMNS} FROM parsed_blocks WHERE parsed_document_id = ? AND position > ?
                    ORDER BY position LIMIT ?""", (parsed_id, after_position, limit),
            ).fetchall()
            if not rows:
                return BlockPage((), None)
            last = rows[-1][0]
            has_more = len(rows) == limit and db.execute(
                "SELECT 1 FROM parsed_blocks WHERE parsed_document_id = ? AND position > ? LIMIT 1", (parsed_id, last),
            ).fetchone() is not None
            locations = _locations(db, parsed_id, rows[0][0], last)
            blocks = tuple(_read_block(row, version[0], locations.get(row[0], ())) for row in rows)
            return BlockPage(blocks, last if has_more else None)

    def get_block(self, parsed_id: int, block_id: str) -> ContentBlock | None:
        with self.store.connect() as db, db:
            db.execute("BEGIN")
            version = db.execute(READY_VERSION, (parsed_id,)).fetchone()
            if version is None:
                return None
            row = db.execute(
                f"SELECT {BLOCK_COLUMNS} FROM parsed_blocks WHERE parsed_document_id = ? AND block_id = ?",
                (parsed_id, block_id),
            ).fetchone()
            if row is None:
                return None
            position = row[0]
            return _read_block(row, version[0], _locations(db, parsed_id, position, position).get(position, ()))

    @staticmethod
    def _renew(db: sqlite3.Connection, parsed_id: int) -> None:
        renewed = db.execute(
            """UPDATE parsed_documents SET lease_expires_at = unixepoch() + ? WHERE parsed_document_id = ?
               AND status = 'writing' AND lease_expires_at > unixepoch()""", (WRITE_LEASE_SECONDS, parsed_id),
        ).rowcount
        if renewed != 1:
            raise DocumentParseError("PARSED_WRITE_INTERRUPTED", "The staged parse is no longer available for writing.")

    def _insert_batches(self, parsed_id: int, sql: str, rows: Iterable[tuple], cancelled: Event | None) -> None:
        for batch in iter_write_batches(rows):
            check_cancelled(cancelled)
            with self.store.write() as db:
                self._renew(db, parsed_id)
                db.executemany(sql, batch)

    def save(
        self, parsed: ParsedDocument, spec: ParseSpec, *, cancelled: Event | None = None,
    ) -> tuple[StoredParse, bool]:
        """Return the ready result and whether this call published it."""
        version_id = parsed.document_version_id
        warnings = _json([asdict(warning) for warning in parsed.warnings])
        with self.store.write() as db:
            ready = self._find_exact(db, version_id, spec)
            if ready is not None:
                return _stored_parse(ready), False
            version = db.execute(
                """SELECT v.document_id, d.format FROM document_versions AS v
                   JOIN documents AS d USING (document_id) WHERE v.document_version_id = ?""", (version_id,),
            ).fetchone()
            if version is None:
                raise DocumentParseError("DOCUMENT_VERSION_NOT_FOUND", "The requested document version does not exist.")
            if version != (parsed.document_id, parsed.format):
                raise DocumentParseError("PARSED_VERSION_MISMATCH", "The parse does not match the document version.")
            parsed_id, = db.execute(
                """INSERT INTO parsed_documents (document_version_id, parser_version, structure_version, config_key,
                   parsed_at, status, lease_expires_at, warnings) VALUES (?, ?, ?, ?, ?, 'writing', unixepoch() + ?, ?)
                   RETURNING parsed_document_id""",
                (version_id, spec.parser_version, spec.structure_version, spec.config_key,
                 utc_now(), WRITE_LEASE_SECONDS, warnings),
            ).fetchone()
        try:
            self._insert_batches(parsed_id, """INSERT INTO parsed_sections (parsed_document_id, position,
                heading_block_id, title, level, parent_heading_block_id) VALUES (?, ?, ?, ?, ?, ?)""", (
                (parsed_id, position, section.heading_block_id, section.title, section.level,
                 section.parent_heading_block_id) for position, section in enumerate(parsed.sections)
            ), cancelled)
            self._insert_batches(parsed_id, """INSERT INTO parsed_blocks (parsed_document_id, position, block_id,
                kind, text, section_heading_block_id, section_path, code_language) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                _block_rows(parsed_id, parsed), cancelled)
            self._insert_batches(parsed_id, """INSERT INTO source_locations (parsed_document_id, block_position,
                position, kind, details) VALUES (?, ?, ?, ?, ?)""", _location_rows(parsed_id, parsed), cancelled)
            check_cancelled(cancelled)
            return self._publish(parsed_id, version_id, spec)
        except Exception:
            # A hard exit skips this path. The last committed lease still expires.
            with suppress(sqlite3.Error), self.store.write(wait=False) as db:
                db.execute("UPDATE parsed_documents SET lease_expires_at = 0 WHERE parsed_document_id = ? "
                           "AND status = 'writing'", (parsed_id,))
            raise

    def _publish(self, parsed_id: int, version_id: str, spec: ParseSpec) -> tuple[StoredParse, bool]:
        # Sections and their heading blocks are staged separately. Check the reverse link
        # in a read snapshot before publication, while the writer lease fences cleanup.
        with self.store.connect() as db:
            missing_heading = db.execute(
                """SELECT 1 FROM parsed_sections AS s WHERE s.parsed_document_id = ? AND NOT EXISTS (
                   SELECT 1 FROM parsed_blocks AS b WHERE b.parsed_document_id = s.parsed_document_id
                     AND b.block_id = s.heading_block_id AND b.kind = 'heading') LIMIT 1""", (parsed_id,),
            ).fetchone()
        if missing_heading:
            raise DocumentParseError("PARSED_HEADING_MISSING", "A section has no matching heading block.")
        with self.store.write() as db:
            self._renew(db, parsed_id)
            winner = self._find_exact(db, version_id, spec)
            if winner is not None:
                db.execute("UPDATE parsed_documents SET lease_expires_at = 0 WHERE parsed_document_id = ?", (parsed_id,))
                return _stored_parse(winner), False
            db.execute("""UPDATE parsed_documents SET status = 'ready', lease_expires_at = NULL, published_at = ?
                          WHERE parsed_document_id = ?""", (utc_now(), parsed_id))
            row = self._find_exact(db, version_id, spec)
            assert row is not None
        return _stored_parse(row), True

    def cleanup_interrupted(self, *, max_batches: int = 1) -> int:
        """Remove expired, unpublished work in bounded transactions."""
        for batch in range(max_batches):
            with self.store.connect() as db:
                if db.execute(EXPIRED_WRITER).fetchone() is None:
                    return batch
            with self.store.write(wait=False) as db:
                row = db.execute(EXPIRED_WRITER).fetchone()
                if row is None:
                    return batch
                self._purge_batch(db, row[0])
        return max_batches

    @staticmethod
    def _purge_batch(db: sqlite3.Connection, parsed_id: int) -> None:
        for table, key, order in (
            ("source_locations", "block_position, position", "block_position, position"),
            ("parsed_blocks", "position", "position"),
            # Sections arrive in preorder. Remove children before their parents.
            ("parsed_sections", "position", "position DESC"),
        ):
            removed = db.execute(
                f"""DELETE FROM {table} WHERE (parsed_document_id, {key}) IN (
                    SELECT parsed_document_id, {key} FROM {table} WHERE parsed_document_id = ?
                    ORDER BY {order} LIMIT ?)""", (parsed_id, BATCH_ROWS),
            ).rowcount
            if removed:
                return
        db.execute("DELETE FROM parsed_documents WHERE parsed_document_id = ?", (parsed_id,))
