"""Internal local keyword search over published, fully indexed Chunk Sets."""

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from app.storage.chunks import PUBLISHED_SETS
from app.storage.keyword_terms import fts_phrase, keyword_terms
from app.storage.keywords import COMPLETE_KEYWORD_ROWS, KeywordIndex, KeywordIndexError
from app.storage.library import KnowledgeStore
from app.storage.retrieval_scope import CURRENT_VERSION, LATEST_PER_DOCUMENT, scope_filters

MAX_QUERY_CHARS = 256
MAX_TOP_K = 50
MAX_CANDIDATES = 200
QUERY_STEP_INTERVAL = 1_000
MAX_QUERY_VM_STEPS = 5_000_000

SELECT_SETS = f"""SELECT v.document_id, v.document_version_id, s.chunk_set_id,
    s.parsed_document_id, v.content_sha256, s.chunk_count, v.rowid AS version_order
    {PUBLISHED_SETS}
    JOIN document_versions AS v ON v.document_version_id = s.document_version_id
    JOIN documents AS d ON d.document_id = v.document_id
    JOIN keyword_index_sets AS i USING (chunk_set_id)
    WHERE s.status = 'ready' AND s.chunk_count > 0 AND i.indexed_at IS NOT NULL AND {COMPLETE_KEYWORD_ROWS}"""

# Read each column's postings once within the finalist rowid range. A correlated
# MATCH for every finalist can repeatedly traverse the same frequent postings.
RANKED_SEARCH = """ranked AS MATERIALIZED (
    SELECT r.row_id, r.chunk_set_id, r.chunk_index, s.document_id,
           s.document_version_id, s.parsed_document_id, keyword_fts.rank AS cost
    FROM keyword_fts
    CROSS JOIN keyword_index_rows AS r ON r.row_id = keyword_fts.rowid
    CROSS JOIN scope AS s ON s.chunk_set_id = r.chunk_set_id
    WHERE keyword_fts MATCH ? AND keyword_fts.rank MATCH 'bm25(1.0, 0.35)'
    ORDER BY cost, r.chunk_set_id, r.chunk_index LIMIT ?
),
hits AS MATERIALIZED (
    SELECT * FROM ranked ORDER BY cost, chunk_set_id, chunk_index LIMIT ?
),
body_hits AS MATERIALIZED (
    SELECT f.rowid AS row_id FROM keyword_fts AS f CROSS JOIN hits AS h ON h.row_id = f.rowid
    WHERE keyword_fts MATCH ?
      AND f.rowid BETWEEN (SELECT min(row_id) FROM hits) AND (SELECT max(row_id) FROM hits)
),
title_hits AS MATERIALIZED (
    SELECT f.rowid AS row_id FROM keyword_fts AS f CROSS JOIN hits AS h ON h.row_id = f.rowid
    WHERE keyword_fts MATCH ?
      AND f.rowid BETWEEN (SELECT min(row_id) FROM hits) AND (SELECT max(row_id) FROM hits)
)
SELECT c.chunk_id, h.document_id, h.document_version_id, h.chunk_set_id, h.parsed_document_id, h.chunk_index,
       c.text, -h.cost, c.section_heading_block_id, c.section_path, c.title_context,
       b.row_id IS NOT NULL, t.row_id IS NOT NULL
FROM hits AS h JOIN chunks AS c USING (chunk_set_id, chunk_index)
LEFT JOIN body_hits AS b ON b.row_id = h.row_id
LEFT JOIN title_hits AS t ON t.row_id = h.row_id
ORDER BY h.cost, h.chunk_set_id, h.chunk_index"""


@dataclass(frozen=True, slots=True)
class KeywordHit:
    chunk_id: str
    document_id: str
    document_version_id: str
    chunk_set_id: int
    parsed_document_id: int
    chunk_index: int
    body_text: str
    score: float
    section_heading_block_id: str | None
    section_path: tuple[str, ...]
    title_context: tuple[str, ...]
    matched_in: tuple[Literal["body", "title"], ...]

    @classmethod
    def from_row(cls, row: tuple) -> "KeywordHit":
        *fields, section, title, in_body, in_title = row
        matched_in = tuple(field for field, matched in (("body", in_body), ("title", in_title)) if matched)
        return cls(*fields, tuple(json.loads(section)), tuple(json.loads(title)), matched_in)


def prepare_keyword_query(query: str) -> str | None:
    if not isinstance(query, str) or len(query) > MAX_QUERY_CHARS or any(
        ord(char) == 0 or 0xD800 <= ord(char) <= 0xDFFF for char in query
    ):
        raise ValueError("Invalid keyword query.")
    terms = keyword_terms(query)
    return fts_phrase(terms) if any(char.isalnum() for char in terms) else None


class KeywordSearchService:
    def __init__(self, store: KnowledgeStore):
        self.store = store
        self.index = KeywordIndex(store)

    def search(self, query: str, top_k: int = 10, *, document_ids: Sequence[str] | None = None,
               chunk_set_ids: Sequence[int] | None = None) -> tuple[KeywordHit, ...]:
        """Search one complete set per selected textbook in a single read snapshot.

        An explicit set filter pins versions. Otherwise search only the latest
        imported version of each textbook, using its highest fully indexed set ID.
        """
        phrase = prepare_keyword_query(query)
        if type(top_k) is not int or not 1 <= top_k <= MAX_TOP_K:
            raise ValueError("Invalid keyword result limit.")
        filter_sql, values = scope_filters(document_ids, chunk_set_ids, label="keyword")
        if phrase is None:
            return ()
        if chunk_set_ids is None:
            filter_sql += CURRENT_VERSION
        eligible = f"{SELECT_SETS}{filter_sql}"
        scope = LATEST_PER_DOCUMENT if chunk_set_ids is None else "SELECT * FROM eligible"

        with self.store.connect() as db, db:
            db.execute("BEGIN")
            return self._search_scope(db, phrase, top_k, eligible, values, scope,
                                      check_unique=chunk_set_ids is not None)

    def search_indexed(self, db: sqlite3.Connection, phrase: str | None, limit: int,
                       set_ids: Sequence[int]) -> tuple[KeywordHit, ...]:
        """Search exact, already selected sets within the caller's read snapshot."""
        if phrase is None or not set_ids:
            return ()
        if type(limit) is not int or not 1 <= limit <= MAX_CANDIDATES:
            raise ValueError("Invalid keyword candidate limit.")
        eligible = f"{SELECT_SETS} AND s.chunk_set_id IN (SELECT value FROM json_each(?))"
        return self._search_scope(db, phrase, limit, eligible, (json.dumps(set_ids),), "SELECT * FROM eligible")

    @staticmethod
    def _search_scope(db, phrase, limit, eligible, values, scope, *, check_unique=False):
        intervals = 0

        def check_budget() -> int:
            nonlocal intervals
            intervals += 1
            return int(intervals * QUERY_STEP_INTERVAL >= MAX_QUERY_VM_STEPS)

        db.set_progress_handler(check_budget, QUERY_STEP_INTERVAL)
        try:
            cte = f"WITH eligible AS ({eligible})"
            if check_unique and db.execute(f"""{cte}
                SELECT 1 FROM eligible GROUP BY document_id HAVING count(*) > 1 LIMIT 1""", values).fetchone():
                raise ValueError("Keyword search requires one indexed Chunk Set per textbook.")
            rows = db.execute(
                f"{cte}, scope AS MATERIALIZED ({scope}), {RANKED_SEARCH}",
                (*values, phrase, MAX_CANDIDATES, limit, f"body : {phrase}", f"title : {phrase}"),
            ).fetchall()
        except sqlite3.DatabaseError as error:
            if error.sqlite_errorcode == sqlite3.SQLITE_INTERRUPT:
                raise KeywordIndexError("KEYWORD_QUERY_BUDGET_EXCEEDED") from error
            raise KeywordIndexError("KEYWORD_INDEX_QUERY_FAILED") from error
        finally:
            db.set_progress_handler(None, 0)
        return tuple(KeywordHit.from_row(row) for row in rows)
