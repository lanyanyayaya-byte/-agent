SCHEMA_V5 = (
    """CREATE TABLE embedding_profiles (
        profile_key TEXT PRIMARY KEY NOT NULL,
        key_version INTEGER NOT NULL,
        provider TEXT NOT NULL,
        service_id TEXT NOT NULL,
        model_id TEXT NOT NULL,
        service_revision TEXT,
        compatibility_epoch TEXT NOT NULL CHECK (length(compatibility_epoch) > 0),
        dimensions INTEGER NOT NULL CHECK (typeof(dimensions) = 'integer' AND dimensions > 0),
        input_strategy_version TEXT NOT NULL,
        vector_format TEXT NOT NULL CHECK (vector_format = 'float32-le'),
        invalidated_at TEXT,
        UNIQUE (profile_key, dimensions)
    )""",
    """CREATE TABLE embedding_sets (
        chunk_set_id INTEGER NOT NULL REFERENCES chunk_sets(chunk_set_id) ON DELETE CASCADE,
        profile_key TEXT NOT NULL REFERENCES embedding_profiles(profile_key),
        created_at TEXT NOT NULL,
        published_at TEXT,
        PRIMARY KEY (chunk_set_id, profile_key)
    ) WITHOUT ROWID""",
    """CREATE TABLE embeddings (
        cache_key TEXT PRIMARY KEY NOT NULL,
        profile_key TEXT NOT NULL,
        dimensions INTEGER NOT NULL,
        input_sha256 TEXT NOT NULL,
        vector BLOB,
        created_at TEXT,
        lease_token TEXT,
        lease_expires_at INTEGER,
        retry_not_before INTEGER NOT NULL DEFAULT 0,
        UNIQUE (profile_key, cache_key),
        UNIQUE (profile_key, input_sha256),
        FOREIGN KEY (profile_key, dimensions) REFERENCES embedding_profiles(profile_key, dimensions),
        CHECK ((lease_token IS NULL) = (lease_expires_at IS NULL)),
        CHECK ((vector IS NULL AND created_at IS NULL)
            OR (typeof(vector) = 'blob' AND length(vector) = 4 * dimensions
                AND created_at IS NOT NULL AND lease_token IS NULL AND retry_not_before = 0))
    )""",
    "CREATE INDEX embeddings_by_lease ON embeddings (lease_token) WHERE lease_token IS NOT NULL",
    """CREATE TABLE chunk_embeddings (
        chunk_set_id INTEGER NOT NULL,
        chunk_index INTEGER NOT NULL,
        profile_key TEXT NOT NULL,
        cache_key TEXT NOT NULL,
        PRIMARY KEY (chunk_set_id, profile_key, chunk_index),
        FOREIGN KEY (chunk_set_id, chunk_index) REFERENCES chunks(chunk_set_id, chunk_index) ON DELETE CASCADE,
        FOREIGN KEY (chunk_set_id, profile_key)
            REFERENCES embedding_sets(chunk_set_id, profile_key) ON DELETE CASCADE,
        FOREIGN KEY (profile_key, cache_key) REFERENCES embeddings(profile_key, cache_key)
    ) WITHOUT ROWID""",
    "CREATE INDEX chunk_embeddings_by_chunk ON chunk_embeddings (chunk_set_id, chunk_index)",
    "CREATE INDEX chunk_embeddings_by_key ON chunk_embeddings (profile_key, cache_key)",
    """CREATE TRIGGER prune_unreferenced_embedding AFTER DELETE ON chunk_embeddings
       BEGIN
           DELETE FROM embeddings WHERE cache_key = OLD.cache_key AND NOT EXISTS (
               SELECT 1 FROM chunk_embeddings WHERE profile_key = OLD.profile_key AND cache_key = OLD.cache_key
           );
       END""",
)
