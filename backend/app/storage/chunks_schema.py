SCHEMA_V4 = (
    # The composite FK also prevents attaching a parse to another textbook version.
    "CREATE UNIQUE INDEX parse_version_identity ON parsed_documents (parsed_document_id, document_version_id)",
    """CREATE TABLE chunk_sets (
        chunk_set_id INTEGER PRIMARY KEY AUTOINCREMENT,
        parsed_document_id INTEGER NOT NULL,
        document_version_id TEXT NOT NULL,
        chunker_version TEXT NOT NULL,
        chunker_config_key TEXT NOT NULL,
        created_at TEXT NOT NULL,
        published_at TEXT,
        status TEXT NOT NULL CHECK (status IN ('writing', 'abandoned', 'ready')),
        lease_expires_at INTEGER,
        chunk_count INTEGER NOT NULL DEFAULT 0 CHECK (chunk_count >= 0),
        source_span_count INTEGER NOT NULL DEFAULT 0 CHECK (source_span_count >= chunk_count),
        UNIQUE (chunk_set_id, parsed_document_id),
        FOREIGN KEY (parsed_document_id, document_version_id)
            REFERENCES parsed_documents(parsed_document_id, document_version_id) ON DELETE CASCADE,
        CHECK ((status = 'ready' AND published_at IS NOT NULL AND lease_expires_at IS NULL)
            OR (status IN ('writing', 'abandoned') AND published_at IS NULL AND lease_expires_at IS NOT NULL
                AND chunk_count = 0 AND source_span_count = 0))
    )""",
    "CREATE INDEX chunk_sets_by_parse ON chunk_sets (parsed_document_id, document_version_id)",
    """CREATE UNIQUE INDEX active_chunk_set_identity ON chunk_sets
       (parsed_document_id, chunker_version, chunker_config_key) WHERE status IN ('writing', 'ready')""",
    """CREATE INDEX chunk_sets_by_expiry ON chunk_sets (lease_expires_at, chunk_set_id)
       WHERE status IN ('writing', 'abandoned')""",
    """CREATE TABLE chunks (
        chunk_set_id INTEGER NOT NULL REFERENCES chunk_sets(chunk_set_id) ON DELETE CASCADE,
        chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
        chunk_id TEXT NOT NULL,
        text TEXT NOT NULL,
        text_length INTEGER NOT NULL CHECK (text_length > 0),
        section_heading_block_id TEXT,
        section_path TEXT NOT NULL CHECK (json_valid(section_path) AND json_type(section_path) = 'array'),
        title_context TEXT NOT NULL CHECK (json_valid(title_context) AND json_type(title_context) = 'array'),
        PRIMARY KEY (chunk_set_id, chunk_index),
        UNIQUE (chunk_id, chunk_set_id)
    ) WITHOUT ROWID""",
    """CREATE TABLE chunk_sources (
        chunk_set_id INTEGER NOT NULL,
        chunk_index INTEGER NOT NULL,
        source_index INTEGER NOT NULL CHECK (source_index >= 0),
        parsed_document_id INTEGER NOT NULL,
        source_block_id TEXT NOT NULL,
        block_start INTEGER NOT NULL CHECK (block_start >= 0),
        block_end INTEGER NOT NULL CHECK (block_end > block_start),
        chunk_start INTEGER NOT NULL CHECK (chunk_start >= 0),
        chunk_end INTEGER NOT NULL CHECK (chunk_end > chunk_start),
        is_overlap INTEGER NOT NULL CHECK (is_overlap IN (0, 1)),
        PRIMARY KEY (chunk_set_id, chunk_index, source_index),
        FOREIGN KEY (chunk_set_id, chunk_index) REFERENCES chunks(chunk_set_id, chunk_index) ON DELETE CASCADE,
        FOREIGN KEY (chunk_set_id, parsed_document_id)
            REFERENCES chunk_sets(chunk_set_id, parsed_document_id) ON DELETE CASCADE,
        FOREIGN KEY (parsed_document_id, source_block_id) REFERENCES parsed_blocks(parsed_document_id, block_id),
        CHECK (block_end - block_start = chunk_end - chunk_start)
    ) WITHOUT ROWID""",
    "CREATE INDEX chunk_sources_by_set ON chunk_sources (chunk_set_id, parsed_document_id)",
    "CREATE INDEX chunk_sources_by_block ON chunk_sources (parsed_document_id, source_block_id)",
)
