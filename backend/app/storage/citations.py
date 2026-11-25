"""Bounded reads for a citation resolution snapshot."""

import json
import sqlite3
from collections.abc import Collection

from app.storage.retrieval_scope import CURRENT_VERSION

MAX_BODY_CHARS = 8_192
MAX_METADATA_BYTES = 256 * 1024
MAX_SOURCE_SPANS = 4_096
MAX_LOCATIONS = 8_192
MAX_LOCATION_BYTES = 2_048

# Use outer joins so missing lineage is visible to the resolver and cannot be
# mistaken for an unknown chunk. Scope predicates use the retrieval aliases.
_CHUNKS = f"""WITH requested AS (SELECT value AS chunk_id FROM json_each(?1))
SELECT c.chunk_id, c.chunk_set_id, c.chunk_index, c.text_length,
       CASE WHEN length(CAST(c.text AS BLOB)) <= ?2 * 4 AND c.text_length <= ?2
            THEN c.text END AS body_text,
       CASE WHEN length(CAST(c.section_path AS BLOB)) <= ?3 THEN c.section_path END AS section_path,
       c.section_heading_block_id,
       s.parsed_document_id, s.document_version_id, s.chunker_version,
       CASE WHEN length(CAST(s.chunker_config_key AS BLOB)) <= ?3
            THEN s.chunker_config_key END AS chunker_config_key,
       s.status AS set_status, s.chunk_count,
       p.status AS parse_status, p.document_version_id AS parse_version_id,
       p.parser_version, p.structure_version,
       CASE WHEN length(CAST(p.config_key AS BLOB)) <= ?3 THEN p.config_key END AS parse_config_key,
       CASE WHEN length(CAST(p.warnings AS BLOB)) <= ?3 THEN p.warnings END AS warnings,
       v.document_id, v.content_sha256, v.size_bytes, d.format,
       CASE WHEN length(CAST(d.name AS BLOB)) <= ?3 THEN d.name END AS name,
       d.document_id IS NOT NULL AS document_exists, blob.sha256 IS NOT NULL AS blob_exists,
       (1 = 1 {CURRENT_VERSION}) AS current_version,
       (1 = 1 {{scope_sql}}) AS in_scope
FROM requested AS r CROSS JOIN chunks AS c ON c.chunk_id = r.chunk_id
LEFT JOIN chunk_sets AS s ON s.chunk_set_id = c.chunk_set_id
LEFT JOIN parsed_documents AS p ON p.parsed_document_id = s.parsed_document_id
LEFT JOIN document_versions AS v ON v.document_version_id = s.document_version_id
LEFT JOIN documents AS d ON d.document_id = v.document_id
LEFT JOIN content_blobs AS blob ON blob.sha256 = v.content_sha256 AND blob.size_bytes = v.size_bytes
LIMIT ?"""

_SELECTED = """WITH selected AS (
    SELECT json_extract(value, '$[0]') AS first, json_extract(value, '$[1]') AS second FROM json_each(?)
)"""


def read_chunks(db: sqlite3.Connection, chunk_ids: Collection[str], scope_sql: str,
                scope_values: tuple) -> list[sqlite3.Row]:
    # The normal case is exactly one row per ID. Leave room to diagnose
    # duplicate identities while keeping the result bounded.
    params = (json.dumps(list(chunk_ids)), MAX_BODY_CHARS, MAX_METADATA_BYTES, *scope_values, 2 * len(chunk_ids) + 1)
    return db.execute(_CHUNKS.format(scope_sql=scope_sql), params).fetchall()


def read_spans(db: sqlite3.Connection, keys: Collection[tuple[int, int]]) -> list[sqlite3.Row]:
    return db.execute(_SELECTED + """SELECT x.chunk_set_id, x.chunk_index, x.source_index,
        x.parsed_document_id, x.source_block_id, x.block_start, x.block_end,
        x.chunk_start, x.chunk_end, x.is_overlap, b.position AS block_position, b.kind,
        b.section_path = c.section_path AND b.section_heading_block_id IS c.section_heading_block_id
            AS section_matches,
        CASE WHEN x.block_start >= 0 AND x.chunk_start >= 0 AND x.chunk_end <= c.text_length
                  AND x.block_end - x.block_start BETWEEN 1 AND ?
             THEN substr(b.text, x.block_start + 1, x.block_end - x.block_start)
                = substr(c.text, x.chunk_start + 1, x.chunk_end - x.chunk_start) END AS text_matches
        FROM selected AS r
        CROSS JOIN chunks AS c ON c.chunk_set_id = r.first AND c.chunk_index = r.second
        CROSS JOIN chunk_sources AS x ON x.chunk_set_id = c.chunk_set_id AND x.chunk_index = c.chunk_index
        LEFT JOIN parsed_blocks AS b ON b.parsed_document_id = x.parsed_document_id AND b.block_id = x.source_block_id
        LIMIT ?""", (json.dumps(list(keys)), MAX_BODY_CHARS, MAX_SOURCE_SPANS + 1)).fetchall()


def read_locations(db: sqlite3.Connection, blocks: Collection[tuple[int, int]]) -> list[sqlite3.Row]:
    return db.execute(_SELECTED + """SELECT l.parsed_document_id, l.block_position, l.position, l.kind,
         CASE WHEN length(CAST(l.details AS BLOB)) <= ? THEN l.details END AS details
         FROM selected AS r CROSS JOIN source_locations AS l
           ON l.parsed_document_id = r.first AND l.block_position = r.second
         LIMIT ?""", (json.dumps(list(blocks)), MAX_LOCATION_BYTES, MAX_LOCATIONS + 1)).fetchall()


def removed_documents(db: sqlite3.Connection, document_ids: Collection[str]) -> set[str]:
    if not document_ids:
        return set()
    return {doc_id for doc_id, in db.execute(
        """SELECT DISTINCT i.document_id FROM knowledge_imports AS i
        WHERE i.document_id IN (SELECT value FROM json_each(?))
          AND NOT EXISTS (SELECT 1 FROM documents AS d WHERE d.document_id = i.document_id)""",
        (json.dumps(list(document_ids)),),
    )}
