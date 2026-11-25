"""Persistent scheduling metadata for version-bound index jobs."""

import math
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Literal, get_args
from uuid import UUID, uuid4

from app.storage.library import KnowledgeStore
from app.storage.retrieval_scope import CURRENT_VERSION

IndexJobStage = Literal["parsing", "chunking", "embedding", "keyword_indexing", "publishing"]
IndexJobStatus = Literal["queued", "running", "retry_wait", "succeeded", "failed", "cancelled"]
STAGES = get_args(IndexJobStage)
PIPELINE_JOB_TYPE = "index_pipeline"
PIPELINE_STAGES: tuple[IndexJobStage, ...] = ("parsing", "chunking", "keyword_indexing", "embedding", "publishing")
PIPELINE_MANAGED_SET = f"""EXISTS (SELECT 1 FROM index_jobs AS j
    WHERE j.document_version_id = s.document_version_id AND j.job_type = '{PIPELINE_JOB_TYPE}')"""
TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
JOB_FIELDS = """job_id, document_id, document_version_id, job_type, config_key, previous_job_id,
    stage, status, completed_work, total_work, created_at, started_at, updated_at, finished_at, error_code,
    error_http_status, retry_count, max_retries, next_retry_at, instance_id, heartbeat_at, lease_expires_at,
    cancel_requested_at"""
LATEST_ATTEMPT = f"""SELECT {JOB_FIELDS} FROM index_jobs
    WHERE document_id = ? AND document_version_id = ? AND config_key = ?
    ORDER BY rowid DESC LIMIT 1"""

DEFAULT_LEASE_SECONDS = 120
LEASE_EXPIRED = "INDEX_LEASE_EXPIRED"
TRANSIENT_ERRORS = {
    "EMBEDDING_RATE_LIMITED", "EMBEDDING_TIMEOUT", "EMBEDDING_NETWORK_ERROR",
    "EMBEDDING_BUILD_BUSY", "CHUNK_BUILD_BUSY", "INDEX_PIPELINE_STORAGE_BUSY",
}
TRANSIENT_HTTP_STATUSES = {408, 429, *range(500, 600)}
DIAGNOSTIC_CODE = re.compile(r"[A-Z][A-Z0-9_]{0,79}")


class IndexJobError(RuntimeError):
    """A queue operation was rejected with a diagnostic code."""


@dataclass(frozen=True, slots=True)
class IndexJob:
    job_id: str
    document_id: str
    document_version_id: str
    job_type: str
    config_key: str
    previous_job_id: str | None
    stage: IndexJobStage
    status: IndexJobStatus
    completed_work: int
    total_work: int | None
    created_at: int
    started_at: int | None
    updated_at: int
    finished_at: int | None
    error_code: str | None
    error_http_status: int | None
    retry_count: int
    max_retries: int
    next_retry_at: int | None
    instance_id: str | None
    heartbeat_at: int | None
    lease_expires_at: int | None
    cancel_requested_at: int | None


@dataclass(frozen=True, slots=True)
class IndexJobClaim:
    job: IndexJob
    lease_token: str = field(repr=False)


def _read_job(db: sqlite3.Connection, job_id: str) -> IndexJob | None:
    row = db.execute(f"SELECT {JOB_FIELDS} FROM index_jobs WHERE job_id = ?", (job_id,)).fetchone()
    return IndexJob(*row) if row is not None else None


def _require_job(db: sqlite3.Connection, job_id: str) -> IndexJob:
    job = _read_job(db, job_id)
    if job is None:
        raise IndexJobError("INDEX_JOB_NOT_FOUND")
    return job


def _require_current_version(db: sqlite3.Connection, document_id: str, version_id: str) -> None:
    if db.execute(f"""SELECT 1 FROM document_versions AS v
        WHERE v.document_id = ? AND v.document_version_id = ? {CURRENT_VERSION}""",
        (document_id, version_id),
    ).fetchone() is None:
        raise IndexJobError("INDEX_VERSION_NOT_CURRENT")


def _end_attempt(
    db: sqlite3.Connection, job: IndexJob, status: IndexJobStatus, *,
    error_code: str | None = None, http_status: int | None = None, retry_delay: int | None = None,
    count_retry: bool = True,
) -> IndexJob:
    row = db.execute(f"""UPDATE index_jobs SET status = ?, error_code = ?, error_http_status = ?,
        retry_count = retry_count + ?, next_retry_at = unixepoch() + ?, updated_at = unixepoch(),
        finished_at = CASE WHEN ? THEN unixepoch() END,
        instance_id = NULL, lease_token = NULL, heartbeat_at = NULL, lease_expires_at = NULL
        WHERE job_id = ? RETURNING {JOB_FIELDS}""",
        (status, error_code, http_status, int(count_retry and status in ("queued", "retry_wait")), retry_delay,
         status in TERMINAL_STATUSES, job.job_id),
    ).fetchone()
    return IndexJob(*row)


def _lease_duration(seconds: int) -> None:
    if type(seconds) is not int or not 1 <= seconds <= 3600:
        raise ValueError("Invalid index job lease duration.")


class IndexJobStore:
    def __init__(self, store: KnowledgeStore):
        self.store = store

    def create_or_get(
        self, document_id: str, document_version_id: str, config_key: str, *,
        job_type: str = "index", max_retries: int = 3, restart_after: str | None = None, current_only: bool = False,
    ) -> IndexJob:
        """Resolve an initial submission or the single successor of a terminal job."""
        if any(not isinstance(value, str) or not 1 <= len(value) <= 128 or "\0" in value
               for value in (config_key, job_type)):
            raise ValueError("Invalid index job type or configuration key.")
        if type(max_retries) is not int or not 0 <= max_retries <= 10:
            raise ValueError("Invalid index job retry limit.")
        with self.store.write() as db:
            if current_only:
                _require_current_version(db, document_id, document_version_id)
            if restart_after is not None:
                previous = _require_job(db, restart_after)
                if (previous.document_id, previous.document_version_id, previous.config_key, previous.job_type) != (
                    document_id, document_version_id, config_key, job_type,
                ):
                    raise IndexJobError("INDEX_JOB_IDENTITY_MISMATCH")
                if previous.status not in TERMINAL_STATUSES:
                    raise IndexJobError("INDEX_JOB_NOT_TERMINAL")
                if current_only and previous.status not in ("failed", "cancelled"):
                    raise IndexJobError("INDEX_JOB_NOT_RETRYABLE")
            if current_only and restart_after is None:
                row = db.execute(LATEST_ATTEMPT, (document_id, document_version_id, config_key)).fetchone()
            else:
                row = db.execute(f"""SELECT {JOB_FIELDS} FROM index_jobs
                    WHERE document_version_id = ? AND config_key = ? AND previous_job_id IS ?""",
                    (document_version_id, config_key, restart_after),
                ).fetchone()
            if row is not None:
                existing = IndexJob(*row)
                if (existing.document_id, existing.job_type) != (document_id, job_type):
                    raise IndexJobError("INDEX_JOB_IDENTITY_MISMATCH")
                return existing
            if not current_only:
                _require_current_version(db, document_id, document_version_id)
            row = db.execute(f"""INSERT INTO index_jobs
                (job_id, document_id, document_version_id, job_type, config_key, previous_job_id,
                 created_at, updated_at, max_retries)
                VALUES (?, ?, ?, ?, ?, ?, unixepoch(), unixepoch(), ?) RETURNING {JOB_FIELDS}""",
                (uuid4().hex, document_id, document_version_id, job_type, config_key, restart_after, max_retries),
            ).fetchone()
            return IndexJob(*row)

    def get_job(self, job_id: str) -> IndexJob | None:
        with self.store.connect() as db:
            return _read_job(db, job_id)

    def latest_attempt(self, document_id: str, version_id: str, config_key: str) -> IndexJob | None:
        with self.store.connect() as db:
            row = db.execute(LATEST_ATTEMPT, (document_id, version_id, config_key)).fetchone()
        return IndexJob(*row) if row is not None else None

    def list_jobs(
        self, *, document_id: str | None = None, status: IndexJobStatus | None = None,
        job_type: str | None = None, limit: int = 100, offset: int = 0,
    ) -> list[IndexJob]:
        if (type(limit) is not int or not 1 <= limit <= 1000
                or type(offset) is not int or not 0 <= offset <= 2**63 - 1):
            raise ValueError("Invalid index job page.")
        if status is not None and status not in get_args(IndexJobStatus):
            raise ValueError("Invalid index job status.")
        with self.store.connect() as db:
            return [IndexJob(*row) for row in db.execute(f"""SELECT {JOB_FIELDS} FROM index_jobs
                WHERE (? IS NULL OR document_id = ?) AND (? IS NULL OR status = ?)
                    AND (? IS NULL OR job_type = ?)
                ORDER BY rowid DESC LIMIT ? OFFSET ?""",
                (document_id, document_id, status, status, job_type, job_type, limit, offset))]

    def claim_next(
        self, instance_id: str | UUID, *, lease_seconds: int = DEFAULT_LEASE_SECONDS, job_type: str | None = None,
        job_id: str | None = None, config_keys: tuple[str, ...] | None = None, wait: bool = True,
    ) -> IndexJobClaim | None:
        _lease_duration(lease_seconds)
        if not isinstance(instance_id, (str, UUID)) or not 1 <= len(str(instance_id).strip()) <= 128:
            raise ValueError("Invalid index job execution instance.")
        config_filter = "" if config_keys is None else " AND config_key IN (" + ", ".join("?" for _ in config_keys) + ")"
        candidate = f"""SELECT job_id FROM index_jobs
            WHERE status IN ('queued', 'retry_wait') AND (next_retry_at IS NULL OR next_retry_at <= unixepoch())
                AND (? IS NULL OR job_type = ?) AND (? IS NULL OR job_id = ?) {config_filter}
            ORDER BY created_at, job_id LIMIT 1"""
        params = (job_type, job_type, job_id, job_id, *(config_keys or ()))
        # Poll the WAL snapshot while idle. Re-select under the writer reservation when work may be available.
        with self.store.connect() as db:
            expired = db.execute("""SELECT 1 FROM index_jobs
                WHERE status = 'running' AND lease_expires_at <= unixepoch() LIMIT 1""").fetchone()
            if expired is None and db.execute(candidate, params).fetchone() is None:
                return None
        with self.store.write(wait=wait) as db:
            self._recover_expired(db, limit=128)
            token = uuid4().hex
            row = db.execute(f"""UPDATE index_jobs SET status = 'running', instance_id = ?, lease_token = ?,
                heartbeat_at = unixepoch(), lease_expires_at = unixepoch() + ?, next_retry_at = NULL,
                started_at = coalesce(started_at, unixepoch()), updated_at = unixepoch()
                WHERE job_id = ({candidate}) RETURNING {JOB_FIELDS}""",
                (str(instance_id), token, lease_seconds, *params),
            ).fetchone()
            return IndexJobClaim(IndexJob(*row), token) if row is not None else None

    @staticmethod
    def _require_lease(db: sqlite3.Connection, claim: IndexJobClaim) -> IndexJob:
        if not db.in_transaction:
            raise ValueError("Index job lease checks require a transaction.")
        row = db.execute(f"""SELECT {JOB_FIELDS} FROM index_jobs
            WHERE job_id = ? AND instance_id = ? AND lease_token = ? AND status = 'running'
              AND lease_expires_at > unixepoch()""",
            (claim.job.job_id, claim.job.instance_id, claim.lease_token),
        ).fetchone()
        if row is None:
            raise IndexJobError("INDEX_JOB_LEASE_LOST")
        return IndexJob(*row)

    def require_active_claim(self, db: sqlite3.Connection, claim: IndexJobClaim) -> IndexJob:
        """Fence a short mutation inside the same KnowledgeStore.write transaction."""
        job = self._require_lease(db, claim)
        if job.cancel_requested_at is not None:
            raise IndexJobError("INDEX_JOB_CANCEL_REQUESTED")
        return job

    def check_claim(self, claim: IndexJobClaim) -> IndexJob:
        with self.store.connect() as db, db:
            db.execute("BEGIN")
            return self.require_active_claim(db, claim)

    def require_index_publication(
        self, db: sqlite3.Connection, set_id: int, claim: IndexJobClaim | None,
    ) -> None:
        """Authorize a publication in the transaction that will make it visible."""
        if claim is None:
            if db.execute(f"""SELECT 1 FROM chunk_sets AS s
                WHERE s.chunk_set_id = ? AND {PIPELINE_MANAGED_SET}""", (set_id,)).fetchone():
                raise IndexJobError("INDEX_PUBLICATION_REQUIRES_CLAIM")
            return
        job = self.require_active_claim(db, claim)
        if job.job_type != PIPELINE_JOB_TYPE or db.execute(f"""SELECT 1
            FROM chunk_sets AS s JOIN document_versions AS v USING (document_version_id)
            WHERE s.chunk_set_id = ? AND v.document_version_id = ?
              AND v.document_id = ? {CURRENT_VERSION}""",
            (set_id, job.document_version_id, job.document_id),
        ).fetchone() is None:
            raise IndexJobError("INDEX_PUBLICATION_SCOPE_MISMATCH")

    def update_progress(
        self, claim: IndexJobClaim, stage: IndexJobStage, *, completed_work: int, total_work: int | None = None,
    ) -> IndexJob:
        if (stage not in STAGES or type(completed_work) is not int or not 0 <= completed_work <= 2**63 - 1
                or (total_work is not None and
                    (type(total_work) is not int or not completed_work <= total_work <= 2**63 - 1))):
            raise ValueError("Invalid index job progress.")
        with self.store.write() as db:
            job = self.require_active_claim(db, claim)
            stages = PIPELINE_STAGES if job.job_type == PIPELINE_JOB_TYPE else STAGES
            step = stages.index(stage) - stages.index(job.stage)
            if step not in (0, 1):
                raise IndexJobError("INDEX_JOB_INVALID_STAGE_TRANSITION")
            if step == 0:
                if total_work is None:
                    total_work = job.total_work
                if (completed_work < job.completed_work or (job.total_work is not None and total_work != job.total_work)
                        or (total_work is not None and completed_work > total_work)):
                    raise IndexJobError("INDEX_JOB_INVALID_PROGRESS")
            else:
                self._require_stage_done(job)
            row = db.execute(f"""UPDATE index_jobs SET stage = ?, completed_work = ?, total_work = ?,
                updated_at = unixepoch() WHERE job_id = ? RETURNING {JOB_FIELDS}""",
                (stage, completed_work, total_work, job.job_id),
            ).fetchone()
            return IndexJob(*row)

    @staticmethod
    def _require_stage_done(job: IndexJob) -> None:
        if job.total_work is not None and job.completed_work != job.total_work:
            raise IndexJobError("INDEX_JOB_STAGE_INCOMPLETE")

    def renew_lease(self, claim: IndexJobClaim, *, lease_seconds: int = DEFAULT_LEASE_SECONDS) -> IndexJob:
        _lease_duration(lease_seconds)
        with self.store.write() as db:
            job = self.require_active_claim(db, claim)
            row = db.execute(f"""UPDATE index_jobs SET heartbeat_at = unixepoch(), updated_at = unixepoch(),
                lease_expires_at = max(lease_expires_at, unixepoch() + ?)
                WHERE job_id = ? RETURNING {JOB_FIELDS}""", (lease_seconds, job.job_id),
            ).fetchone()
            return IndexJob(*row)

    def request_cancel(self, job_id: str) -> IndexJob:
        with self.store.write() as db:
            job = _require_job(db, job_id)
            if job.status in TERMINAL_STATUSES or job.cancel_requested_at is not None:
                return job
            if job.status == "running":
                row = db.execute(f"""UPDATE index_jobs SET cancel_requested_at = unixepoch(), updated_at = unixepoch()
                    WHERE job_id = ? RETURNING {JOB_FIELDS}""", (job_id,),
                ).fetchone()
            else:
                row = db.execute(f"""UPDATE index_jobs SET cancel_requested_at = unixepoch(), status = 'cancelled',
                    updated_at = unixepoch(), finished_at = unixepoch(), next_retry_at = NULL,
                    error_code = NULL, error_http_status = NULL WHERE job_id = ? RETURNING {JOB_FIELDS}""",
                    (job_id,),
                ).fetchone()
            return IndexJob(*row)

    def acknowledge_cancel(self, claim: IndexJobClaim) -> IndexJob:
        with self.store.write() as db:
            job = self._require_lease(db, claim)
            if job.cancel_requested_at is None:
                raise IndexJobError("INDEX_JOB_CANCEL_NOT_REQUESTED")
            return _end_attempt(db, job, "cancelled")

    def interrupt(self, claim: IndexJobClaim) -> IndexJob:
        """Return a drained attempt to the queue when its Sidecar shuts down."""
        with self.store.write() as db:
            job = self._require_lease(db, claim)
            if job.cancel_requested_at is not None:
                return _end_attempt(db, job, "cancelled")
            return _end_attempt(db, job, "queued", count_retry=False)

    def complete(self, claim: IndexJobClaim) -> IndexJob:
        with self.store.write() as db:
            job = self.require_active_claim(db, claim)
            if job.stage != "publishing":
                raise IndexJobError("INDEX_JOB_INVALID_STAGE_TRANSITION")
            self._require_stage_done(job)
            return _end_attempt(db, job, "succeeded")

    def fail(
        self, claim: IndexJobClaim, error_code: str, *, http_status: int | None = None, retry_after: float = 0,
    ) -> IndexJob:
        """Persist a safe diagnostic code and retry only recognized temporary failures."""
        if not isinstance(error_code, str) or not DIAGNOSTIC_CODE.fullmatch(error_code) or error_code == LEASE_EXPIRED:
            raise ValueError("Invalid index job error code.")
        if http_status is not None and (type(http_status) is not int or not 100 <= http_status <= 599):
            raise ValueError("Invalid index job HTTP status.")
        if type(retry_after) not in (int, float) or not 0 <= retry_after < math.inf:
            raise ValueError("Invalid index job retry delay.")
        transient = (
            error_code in TRANSIENT_ERRORS and (http_status is None or http_status in TRANSIENT_HTTP_STATUSES)
        ) or (
            error_code == "EMBEDDING_SERVICE_ERROR" and http_status in TRANSIENT_HTTP_STATUSES
        )
        with self.store.write() as db:
            job = self.require_active_claim(db, claim)
            if transient and job.retry_count < job.max_retries:
                delay = max(min(300, 5 * 2**job.retry_count), math.ceil(min(3600, retry_after)))
                return _end_attempt(db, job, "retry_wait", error_code=error_code, http_status=http_status,
                                    retry_delay=delay)
            return _end_attempt(db, job, "failed", error_code=error_code, http_status=http_status)

    def recover_expired(self, *, limit: int = 128) -> list[IndexJob]:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("Invalid index job recovery limit.")
        with self.store.write() as db:
            return self._recover_expired(db, limit=limit)

    @staticmethod
    def _recover_expired(db: sqlite3.Connection, *, limit: int) -> list[IndexJob]:
        rows = db.execute(f"""SELECT {JOB_FIELDS} FROM index_jobs
            WHERE status = 'running' AND lease_expires_at <= unixepoch()
            ORDER BY lease_expires_at, job_id LIMIT ?""", (limit,),
        ).fetchall()
        recovered = []
        for row in rows:
            job = IndexJob(*row)
            if job.cancel_requested_at is not None:
                recovered.append(_end_attempt(db, job, "cancelled"))
            else:
                status = "queued" if job.retry_count < job.max_retries else "failed"
                recovered.append(_end_attempt(db, job, status, error_code=LEASE_EXPIRED))
        return recovered
