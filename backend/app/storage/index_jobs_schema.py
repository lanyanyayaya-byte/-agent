SCHEMA_V7 = (
    """CREATE UNIQUE INDEX document_version_identity
       ON document_versions (document_version_id, document_id)""",
    """CREATE TABLE index_jobs (
        job_id TEXT PRIMARY KEY NOT NULL,
        document_id TEXT NOT NULL,
        document_version_id TEXT NOT NULL,
        job_type TEXT NOT NULL,
        config_key TEXT NOT NULL,
        previous_job_id TEXT UNIQUE REFERENCES index_jobs(job_id),
        stage TEXT NOT NULL DEFAULT 'parsing' CHECK (
            stage IN ('parsing', 'chunking', 'embedding', 'keyword_indexing', 'publishing')
        ),
        status TEXT NOT NULL DEFAULT 'queued' CHECK (
            status IN ('queued', 'running', 'retry_wait', 'succeeded', 'failed', 'cancelled')
        ),
        completed_work INTEGER NOT NULL DEFAULT 0 CHECK (completed_work >= 0),
        total_work INTEGER CHECK (total_work >= completed_work),
        created_at INTEGER NOT NULL,
        started_at INTEGER,
        updated_at INTEGER NOT NULL,
        finished_at INTEGER,
        error_code TEXT,
        error_http_status INTEGER CHECK (error_http_status BETWEEN 100 AND 599),
        retry_count INTEGER NOT NULL DEFAULT 0,
        max_retries INTEGER NOT NULL CHECK (max_retries BETWEEN 0 AND 10),
        next_retry_at INTEGER,
        instance_id TEXT,
        lease_token TEXT,
        heartbeat_at INTEGER,
        lease_expires_at INTEGER,
        cancel_requested_at INTEGER,
        FOREIGN KEY (document_version_id, document_id)
            REFERENCES document_versions(document_version_id, document_id) ON DELETE CASCADE,
        CHECK (retry_count BETWEEN 0 AND max_retries),
        CHECK ((status = 'retry_wait') = (next_retry_at IS NOT NULL)),
        CHECK ((status IN ('succeeded', 'failed', 'cancelled')) = (finished_at IS NOT NULL)),
        CHECK (status <> 'succeeded' OR (stage = 'publishing' AND (total_work IS NULL OR total_work = completed_work))),
        CHECK (status NOT IN ('failed', 'retry_wait') OR error_code IS NOT NULL),
        CHECK (error_http_status IS NULL OR error_code IS NOT NULL),
        CHECK (cancel_requested_at IS NULL OR status IN ('running', 'cancelled')),
        CHECK (status <> 'cancelled' OR cancel_requested_at IS NOT NULL),
        CHECK (
            (status = 'running' AND instance_id IS NOT NULL AND lease_token IS NOT NULL
                AND heartbeat_at IS NOT NULL AND lease_expires_at > heartbeat_at AND started_at IS NOT NULL
                AND lease_expires_at IS NOT NULL)
            OR (status <> 'running' AND instance_id IS NULL AND lease_token IS NULL
                AND heartbeat_at IS NULL AND lease_expires_at IS NULL)
        )
    )""",
    """CREATE UNIQUE INDEX root_index_job ON index_jobs (document_version_id, config_key)
       WHERE previous_job_id IS NULL""",
    """CREATE UNIQUE INDEX active_index_job ON index_jobs (document_version_id, config_key)
       WHERE status IN ('queued', 'running', 'retry_wait')""",
    "CREATE INDEX index_jobs_by_version ON index_jobs (document_version_id, document_id)",
    """CREATE INDEX runnable_index_jobs ON index_jobs (created_at, job_id)
       WHERE status IN ('queued', 'retry_wait')""",
    """CREATE INDEX index_jobs_by_expiry ON index_jobs (lease_expires_at, job_id)
       WHERE status = 'running'""",
    """CREATE TRIGGER keep_index_job_identity
       BEFORE UPDATE OF job_id, document_id, document_version_id, job_type, config_key, previous_job_id ON index_jobs
       BEGIN SELECT RAISE(ABORT, 'index job identity is immutable'); END""",
    # Version advancement and fencing are committed by the same import transaction.
    """CREATE TRIGGER supersede_index_jobs AFTER INSERT ON document_versions
    BEGIN
        UPDATE index_jobs SET
            status = CASE WHEN cancel_requested_at IS NULL THEN 'failed' ELSE 'cancelled' END,
            error_code = CASE WHEN cancel_requested_at IS NULL THEN 'INDEX_VERSION_SUPERSEDED' END,
            error_http_status = NULL, updated_at = unixepoch(), finished_at = unixepoch(),
            next_retry_at = NULL, instance_id = NULL, lease_token = NULL,
            heartbeat_at = NULL, lease_expires_at = NULL
        WHERE document_version_id IN (
            SELECT document_version_id FROM document_versions
            WHERE document_id = NEW.document_id AND document_version_id <> NEW.document_version_id
        ) AND status IN ('queued', 'running', 'retry_wait');
    END""",
)
