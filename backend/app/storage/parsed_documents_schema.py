SCHEMA_V3 = (
    """CREATE TABLE parsed_documents (
        parsed_document_id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_version_id TEXT NOT NULL REFERENCES document_versions(document_version_id) ON DELETE CASCADE,
        parser_version TEXT NOT NULL,
        structure_version INTEGER NOT NULL CHECK (structure_version > 0),
        config_key TEXT NOT NULL,
        parsed_at TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('writing', 'ready')),
        lease_expires_at INTEGER,
        published_at TEXT,
        warnings TEXT NOT NULL CHECK (json_valid(warnings) AND json_type(warnings) = 'array'),
        CHECK ((status = 'writing' AND lease_expires_at IS NOT NULL AND published_at IS NULL)
            OR (status = 'ready' AND lease_expires_at IS NULL AND published_at IS NOT NULL))
    )""",
    "CREATE INDEX parses_by_version ON parsed_documents (document_version_id)",
    """CREATE UNIQUE INDEX ready_parse_identity
       ON parsed_documents (document_version_id, parser_version, structure_version, config_key)
       WHERE status = 'ready'""",
    """CREATE INDEX parses_by_publication
       ON parsed_documents (document_version_id, published_at DESC, parsed_document_id DESC)
       WHERE status = 'ready'""",
    """CREATE INDEX parses_by_expiry
       ON parsed_documents (lease_expires_at, parsed_document_id) WHERE status = 'writing'""",
    """CREATE TABLE parsed_sections (
        parsed_document_id INTEGER NOT NULL REFERENCES parsed_documents(parsed_document_id) ON DELETE CASCADE,
        position INTEGER NOT NULL CHECK (position >= 0),
        heading_block_id TEXT NOT NULL,
        title TEXT NOT NULL,
        level INTEGER NOT NULL CHECK (level > 0),
        parent_heading_block_id TEXT,
        PRIMARY KEY (parsed_document_id, position),
        UNIQUE (parsed_document_id, heading_block_id),
        FOREIGN KEY (parsed_document_id, parent_heading_block_id)
            REFERENCES parsed_sections(parsed_document_id, heading_block_id)
    ) WITHOUT ROWID""",
    "CREATE INDEX sections_by_parent ON parsed_sections (parsed_document_id, parent_heading_block_id)",
    """CREATE TABLE parsed_blocks (
        parsed_document_id INTEGER NOT NULL REFERENCES parsed_documents(parsed_document_id) ON DELETE CASCADE,
        position INTEGER NOT NULL CHECK (position >= 0),
        block_id TEXT NOT NULL,
        kind TEXT NOT NULL CHECK (kind IN ('heading', 'paragraph', 'list', 'code', 'table', 'blockquote',
                                         'thematic_break', 'html')),
        text TEXT NOT NULL,
        section_heading_block_id TEXT,
        section_path TEXT NOT NULL CHECK (json_valid(section_path) AND json_type(section_path) = 'array'),
        code_language TEXT,
        PRIMARY KEY (parsed_document_id, position),
        UNIQUE (parsed_document_id, block_id),
        FOREIGN KEY (parsed_document_id, section_heading_block_id)
            REFERENCES parsed_sections(parsed_document_id, heading_block_id)
    ) WITHOUT ROWID""",
    "CREATE INDEX blocks_by_section ON parsed_blocks (parsed_document_id, section_heading_block_id)",
    """CREATE TABLE source_locations (
        parsed_document_id INTEGER NOT NULL,
        block_position INTEGER NOT NULL,
        position INTEGER NOT NULL CHECK (position >= 0),
        kind TEXT NOT NULL CHECK (kind IN ('pdf', 'docx', 'md')),
        details TEXT NOT NULL CHECK (json_valid(details) AND json_type(details) = 'object'),
        PRIMARY KEY (parsed_document_id, block_position, position),
        FOREIGN KEY (parsed_document_id, block_position)
            REFERENCES parsed_blocks(parsed_document_id, position) ON DELETE CASCADE
    ) WITHOUT ROWID""",
)
