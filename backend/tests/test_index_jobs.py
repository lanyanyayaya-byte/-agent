import asyncio
import hashlib
import os
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier
from uuid import uuid4

import pytest

from app.main import SidecarControl, create_app
from app.models.embeddings import EmbeddingProfile, encode_vector
from app.services.index_readiness import get_index_readiness
from app.services.keyword_search import KeywordSearchService
from app.storage import library as library_storage
from app.storage.chunks import ChunkStore
from app.storage.embeddings import EmbeddingStore
from app.storage.index_jobs import STAGES, IndexJobError, IndexJobStore
from app.storage.library import KnowledgeStore
from app.storage.parsed_documents import ParsedDocumentStore, ParseSpec
from test_library import add_blob, add_document, add_version, anyio_backend

CONFIG = "fixture-index-config"
PARSE_SPEC = ParseSpec("fixture-parser", 1, "fixture-parse-config")
PROFILE = EmbeddingProfile("fixture", "local", "deterministic", 2, "fixture-v1")
STAMP = "2026-11-24T00:00:00.000Z"


def add_book(store, content=b"first"):
    with store.write() as db:
        document_id = add_document(db, "教材")
        version_id = add_version(db, document_id, add_blob(db, content))
    return document_id, version_id


@pytest.fixture
def queue(tmp_path):
    jobs = IndexJobStore(KnowledgeStore(tmp_path))
    return jobs, add_book(jobs.store)


def seed_parse(store, version_id):
    with store.write() as db:
        return db.execute("""INSERT INTO parsed_documents
            (document_version_id, parser_version, structure_version, config_key, parsed_at, status,
             published_at, warnings)
            VALUES (?, ?, ?, ?, ?, 'ready', ?, '[]') RETURNING parsed_document_id""",
            (version_id, PARSE_SPEC.parser_version, PARSE_SPEC.structure_version, PARSE_SPEC.config_key, STAMP, STAMP),
        ).fetchone()[0]


def seed_chunks(store, version_id, parsed_id, *texts):
    with store.write() as db:
        set_id, = db.execute("""INSERT INTO chunk_sets
            (parsed_document_id, document_version_id, chunker_version, chunker_config_key, created_at,
             published_at, status, chunk_count, source_span_count)
            VALUES (?, ?, 'fixture-chunker', 'fixture-chunk-config', ?, ?, 'ready', ?, ?) RETURNING chunk_set_id""",
            (parsed_id, version_id, STAMP, STAMP, len(texts), len(texts)),
        ).fetchone()
        db.executemany("""INSERT INTO chunks (chunk_set_id, chunk_index, chunk_id, text, text_length,
            section_path, title_context) VALUES (?, ?, ?, ?, ?, '[]', '[]')""",
            ((set_id, index, f"fixture-{index}", text, len(text)) for index, text in enumerate(texts)),
        )
        for index, text in enumerate(texts):
            block_id = f"block-{index}"
            db.execute("""INSERT INTO parsed_blocks (parsed_document_id, position, block_id, kind, text, section_path)
                VALUES (?, ?, ?, 'paragraph', ?, '[]')""", (parsed_id, index, block_id, text))
            db.execute("""INSERT INTO chunk_sources
                (chunk_set_id, chunk_index, source_index, parsed_document_id, source_block_id,
                 block_start, block_end, chunk_start, chunk_end, is_overlap)
                VALUES (?, ?, 0, ?, ?, 0, ?, 0, ?, 0)""",
                (set_id, index, parsed_id, block_id, len(text), len(text)),
            )
    return set_id


def seed_keyword_index(store, set_id, *, complete=True):
    with store.write() as db:
        db.execute("INSERT INTO keyword_index_sets (chunk_set_id, indexed_at) VALUES (?, ?)",
                   (set_id, STAMP if complete else None))
        db.execute("""INSERT INTO keyword_index_rows (chunk_set_id, chunk_index)
            SELECT chunk_set_id, chunk_index FROM chunks WHERE chunk_set_id = ?""", (set_id,))


def seed_vector(store, set_id, index, text):
    digest = hashlib.sha256(text.encode()).hexdigest()
    with store.write() as db:
        key = PROFILE.cache_key(digest)
        db.execute("""INSERT INTO embeddings (cache_key, profile_key, dimensions, input_sha256, vector, created_at)
            VALUES (?, ?, ?, ?, ?, ?)""", (key, PROFILE.key, PROFILE.dimensions, digest, encode_vector([1, 0], 2), STAMP))
        db.execute("""INSERT INTO chunk_embeddings (chunk_set_id, chunk_index, profile_key, cache_key)
            VALUES (?, ?, ?, ?)""", (set_id, index, PROFILE.key, key))


def advance_to_publishing(jobs, claim):
    for stage in STAGES[1:]:
        jobs.update_progress(claim, stage, completed_work=0)


def expire(store, job_id):
    with store.write() as db:
        db.execute("""UPDATE index_jobs SET heartbeat_at = unixepoch() - 120, lease_expires_at = unixepoch()
            WHERE job_id = ?""", (job_id,))


def test_create_get_and_restart_are_durable_and_idempotent(queue, tmp_path):
    jobs, book = queue
    job = jobs.create_or_get(*book, CONFIG)
    assert (job.document_id, job.document_version_id) == book
    assert (job.status, job.stage, job.completed_work, job.total_work) == ("queued", "parsing", 0, None)
    assert job.started_at is None and job.created_at == job.updated_at
    reopened = IndexJobStore(KnowledgeStore(tmp_path))
    assert reopened.create_or_get(*book, CONFIG) == job
    assert reopened.list_jobs(document_id=book[0], status="queued") == [job]
    with pytest.raises(IndexJobError, match="INDEX_JOB_IDENTITY_MISMATCH"):
        jobs.create_or_get(*book, CONFIG, job_type="rebuild")
    with pytest.raises(IndexJobError, match="INDEX_JOB_NOT_TERMINAL"):
        jobs.create_or_get(*book, CONFIG, restart_after=job.job_id)
    cancelled = jobs.request_cancel(job.job_id)
    assert cancelled.status == "cancelled" and cancelled.cancel_requested_at is not None
    assert jobs.claim_next("worker") is None
    assert reopened.create_or_get(*book, CONFIG) == cancelled
    successor = reopened.create_or_get(*book, CONFIG, restart_after=job.job_id)
    assert successor.previous_job_id == job.job_id and successor.job_id != job.job_id
    assert jobs.create_or_get(*book, CONFIG, restart_after=job.job_id) == successor
    assert jobs.create_or_get(*book, "other-config").job_id != successor.job_id
    with jobs.store.write() as db, pytest.raises(sqlite3.IntegrityError, match="index job identity is immutable"):
        db.execute("UPDATE index_jobs SET document_version_id = 'different' WHERE job_id = ?", (job.job_id,))
    with jobs.store.write() as db, pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        db.execute("""INSERT INTO index_jobs (job_id, document_id, document_version_id, job_type,
            config_key, previous_job_id, created_at, updated_at, max_retries)
            VALUES ('conflict', ?, ?, 'other-type', ?, ?, unixepoch(), unixepoch(), 3)""",
            (*book, CONFIG, successor.job_id))


def test_competing_submissions_and_claims_have_one_winner(queue):
    jobs, book = queue
    submit_barrier, claim_barrier = Barrier(8, timeout=5), Barrier(8, timeout=5)

    def worker(index):
        submit_barrier.wait()
        job = jobs.create_or_get(*book, CONFIG)
        claim_barrier.wait()
        return job.job_id, jobs.claim_next(f"worker-{index}")

    with ThreadPoolExecutor(max_workers=8) as workers:
        results = list(workers.map(worker, range(8)))
    assert len({job_id for job_id, _ in results}) == 1
    winner, = (claim for _, claim in results if claim is not None)
    assert jobs.get_job(winner.job.job_id) == winner.job
    assert winner.job.status == "running" and winner.job.started_at is not None


def test_state_machine_rejects_illegal_transitions(queue):
    jobs, book = queue
    job = jobs.create_or_get(*book, CONFIG)
    claim = jobs.claim_next("worker")
    with pytest.raises(IndexJobError, match="INDEX_JOB_CANCEL_NOT_REQUESTED"):
        jobs.acknowledge_cancel(claim)
    with pytest.raises(IndexJobError, match="INDEX_JOB_INVALID_STAGE_TRANSITION"):
        jobs.complete(claim)
    with pytest.raises(IndexJobError, match="INDEX_JOB_INVALID_STAGE_TRANSITION"):
        jobs.update_progress(claim, "embedding", completed_work=1)
    jobs.update_progress(claim, "parsing", completed_work=1, total_work=2)
    with pytest.raises(IndexJobError, match="INDEX_JOB_INVALID_PROGRESS"):
        jobs.update_progress(claim, "parsing", completed_work=0)
    with pytest.raises(IndexJobError, match="INDEX_JOB_STAGE_INCOMPLETE"):
        jobs.update_progress(claim, "chunking", completed_work=0)
    jobs.update_progress(claim, "parsing", completed_work=2)
    advance_to_publishing(jobs, claim)
    with pytest.raises(IndexJobError, match="INDEX_JOB_INVALID_STAGE_TRANSITION"):
        jobs.update_progress(claim, "parsing", completed_work=0)
    done = jobs.complete(claim)
    assert done.status == "succeeded" and done.finished_at is not None and done.instance_id is None
    assert jobs.request_cancel(done.job_id) == done
    assert jobs.create_or_get(*book, CONFIG) == done
    assert get_index_readiness(jobs.store, book[1], job_id=job.job_id) is None
    with pytest.raises(IndexJobError, match="INDEX_JOB_LEASE_LOST"):
        jobs.fail(claim, "PDF_INVALID_FILE")


def test_heartbeat_extends_lease_and_rejects_wrong_instance(queue):
    jobs, book = queue
    job = jobs.create_or_get(*book, CONFIG)
    claim = jobs.claim_next("worker", lease_seconds=10)
    with jobs.store.write() as db:
        db.execute("UPDATE index_jobs SET heartbeat_at = unixepoch() - 5 WHERE job_id = ?", (job.job_id,))
    before = jobs.get_job(job.job_id)
    heartbeat = jobs.renew_lease(claim, lease_seconds=60)
    assert heartbeat.heartbeat_at > before.heartbeat_at
    assert heartbeat.lease_expires_at == heartbeat.heartbeat_at + 60
    assert jobs.renew_lease(claim, lease_seconds=1).lease_expires_at >= heartbeat.lease_expires_at
    assert jobs.recover_expired() == [] and jobs.claim_next("second") is None
    wrong_instance = replace(claim, job=replace(claim.job, instance_id="second"))
    with pytest.raises(IndexJobError, match="INDEX_JOB_LEASE_LOST"):
        jobs.renew_lease(wrong_instance)
    expire(jobs.store, job.job_id)
    with pytest.raises(IndexJobError, match="INDEX_JOB_LEASE_LOST"):
        jobs.renew_lease(claim)


def test_hard_exit_reclaims_same_job_with_progress_and_reusable_artifacts(queue, tmp_path):
    jobs, book = queue
    job = jobs.create_or_get(*book, CONFIG)
    parsed_id = seed_parse(jobs.store, book[1])
    set_id = seed_chunks(jobs.store, book[1], parsed_id, "first body", "second body")
    EmbeddingStore(jobs.store).stage(set_id, PROFILE)
    seed_vector(jobs.store, set_id, 0, "first body")

    child = subprocess.run([sys.executable, "-c", """
import os
import sys
from pathlib import Path
from app.storage.index_jobs import IndexJobStore
from app.storage.library import KnowledgeStore
jobs = IndexJobStore(KnowledgeStore(Path(sys.argv[1])))
claim = jobs.claim_next('dead-sidecar')
jobs.update_progress(claim, 'chunking', completed_work=1, total_work=1)
jobs.update_progress(claim, 'embedding', completed_work=1, total_work=2)
os._exit(37)
""", os.fspath(tmp_path)], capture_output=True, timeout=10)
    assert child.returncode == 37, child.stderr.decode()
    reopened = IndexJobStore(KnowledgeStore(tmp_path))
    interrupted = reopened.get_job(job.job_id)
    assert interrupted.status == "running" and interrupted.instance_id == "dead-sidecar"
    assert reopened.claim_next("new-sidecar") is None
    expire(reopened.store, job.job_id)
    claim = reopened.claim_next("new-sidecar")
    assert claim.job.job_id == job.job_id
    assert (claim.job.stage, claim.job.completed_work, claim.job.total_work) == ("embedding", 1, 2)
    assert claim.job.error_code == "INDEX_LEASE_EXPIRED" and claim.job.retry_count == 1
    assert claim.job.started_at == interrupted.started_at
    parsed = ParsedDocumentStore(reopened.store).get_reusable(book[1], PARSE_SPEC)
    assert parsed.parsed_document_id == parsed_id
    chunks = ChunkStore(reopened.store).get_chunk_set(parsed_id, "fixture-chunker", "fixture-chunk-config")
    assert chunks.chunk_set_id == set_id
    assert EmbeddingStore(reopened.store).missing_chunks(set_id, PROFILE, -1, 20) == [(1, "second body", ())]


def test_old_claim_cannot_update_after_expiry_or_reclaim_by_same_instance(queue):
    jobs, book = queue
    job = jobs.create_or_get(*book, CONFIG)
    stale = jobs.claim_next("same-identity")
    advance_to_publishing(jobs, stale)
    expire(jobs.store, job.job_id)
    with pytest.raises(IndexJobError, match="INDEX_JOB_LEASE_LOST"):
        jobs.complete(stale)
    current = jobs.claim_next("same-identity")
    assert current.lease_token != stale.lease_token
    for operation in (
        lambda: jobs.update_progress(stale, "publishing", completed_work=1),
        lambda: jobs.renew_lease(stale),
        lambda: jobs.fail(stale, "EMBEDDING_TIMEOUT"),
        lambda: jobs.complete(stale),
    ):
        with pytest.raises(IndexJobError, match="INDEX_JOB_LEASE_LOST"):
            operation()
    with jobs.store.write() as db, pytest.raises(IndexJobError, match="INDEX_JOB_LEASE_LOST"):
        jobs.require_active_claim(db, stale)
    assert jobs.complete(current).status == "succeeded"


@pytest.mark.parametrize("crashed", [False, True])
def test_running_cancel_fences_completion_and_does_not_requeue(queue, crashed):
    jobs, book = queue
    job = jobs.create_or_get(*book, CONFIG)
    claim = jobs.claim_next("worker")
    advance_to_publishing(jobs, claim)
    requested = jobs.request_cancel(job.job_id)
    assert requested.cancel_requested_at is not None and requested.status == "running"
    assert jobs.request_cancel(job.job_id) == requested
    with pytest.raises(IndexJobError, match="INDEX_JOB_CANCEL_REQUESTED"):
        jobs.complete(claim)
    with pytest.raises(IndexJobError, match="INDEX_JOB_CANCEL_REQUESTED"):
        jobs.renew_lease(claim)
    if crashed:
        expire(jobs.store, job.job_id)
        cancelled, = jobs.recover_expired()
    else:
        cancelled = jobs.acknowledge_cancel(claim)
    assert cancelled.status == "cancelled" and cancelled.error_code is None and cancelled.retry_count == 0
    assert jobs.claim_next("other") is None


@pytest.mark.parametrize(("code", "http_status", "delay"), [
    ("EMBEDDING_RATE_LIMITED", 429, 9),
    ("EMBEDDING_TIMEOUT", None, 5),
    ("EMBEDDING_NETWORK_ERROR", None, 5),
    ("EMBEDDING_SERVICE_ERROR", 503, 5),
])
def test_transient_failure_backs_off_and_exhausts_budget(queue, code, http_status, delay):
    jobs, book = queue
    job = jobs.create_or_get(*book, CONFIG, max_retries=2)
    for attempt in range(3):
        claim = jobs.claim_next("worker")
        result = jobs.fail(claim, code, http_status=http_status, retry_after=8.1 if delay == 9 else 0)
        assert result.error_code == code and result.error_http_status == http_status
        assert result.retry_count == min(attempt + 1, 2)
        assert jobs.claim_next("worker") is None
        if attempt < 2:
            assert result.status == "retry_wait"
            assert result.next_retry_at - result.updated_at == max(5 * 2**attempt, delay)
            with jobs.store.write() as db:
                db.execute("UPDATE index_jobs SET next_retry_at = unixepoch() WHERE job_id = ?", (job.job_id,))
    assert result.status == "failed" and result.next_retry_at is None


@pytest.mark.parametrize(("code", "http_status"), [
    ("EMBEDDING_AUTH_FAILED", 401), ("EMBEDDING_NOT_CONFIGURED", None),
    ("EMBEDDING_FORBIDDEN", 403), ("PDF_INVALID_FILE", None), ("DOCX_INVALID_PACKAGE", None),
    ("EMBEDDING_SERVICE_ERROR", 400),
])
def test_permanent_failure_records_cause_without_retry(queue, code, http_status):
    jobs, book = queue
    job = jobs.create_or_get(*book, CONFIG)
    failed = jobs.fail(jobs.claim_next("worker"), code, http_status=http_status)
    assert (failed.status, failed.error_code, failed.error_http_status) == ("failed", code, http_status)
    assert failed.retry_count == 0 and failed.next_retry_at is None
    assert jobs.claim_next("worker") is None
    assert get_index_readiness(jobs.store, book[1], job_id=job.job_id) == "failed"


def test_recovery_is_bounded_and_retry_wait_can_be_cancelled(queue):
    jobs, book = queue
    first = jobs.create_or_get(*book, CONFIG, max_retries=0)
    jobs.claim_next("worker")
    expire(jobs.store, first.job_id)
    interrupted, = jobs.recover_expired()
    assert interrupted.status == "failed" and interrupted.error_code == "INDEX_LEASE_EXPIRED"
    assert interrupted.cancel_requested_at is None and jobs.claim_next("worker") is None
    next_job = jobs.create_or_get(*book, CONFIG, restart_after=first.job_id)
    result = jobs.fail(jobs.claim_next("other"), "EMBEDDING_TIMEOUT")
    assert result.status == "retry_wait"
    cancelled = jobs.request_cancel(next_job.job_id)
    assert cancelled.status == "cancelled" and cancelled.next_retry_at is None
    assert jobs.claim_next("worker") is None


def test_readiness_uses_exact_published_sets(queue):
    jobs, book = queue
    store, version = jobs.store, book[1]
    assert get_index_readiness(store, version) is None
    parsed_id = seed_parse(store, version)
    assert get_index_readiness(store, version) is None
    set_id = seed_chunks(store, version, parsed_id, "first body", "second body")
    assert get_index_readiness(store, version, chunk_set_id=set_id) is None
    seed_keyword_index(store, set_id, complete=False)
    assert get_index_readiness(store, version, chunk_set_id=set_id) is None
    with store.write() as db:
        db.execute("UPDATE keyword_index_sets SET indexed_at = ? WHERE chunk_set_id = ?", (STAMP, set_id))
    embeddings = EmbeddingStore(store)
    embeddings.stage(set_id, PROFILE)
    seed_vector(store, set_id, 0, "first body")
    assert get_index_readiness(store, version, chunk_set_id=set_id, profile=PROFILE) == "ready_keyword"
    seed_vector(store, set_id, 1, "second body")
    with store.write() as db:
        db.execute("UPDATE embedding_sets SET published_at = ? WHERE chunk_set_id = ?", (STAMP, set_id))
    assert get_index_readiness(store, version, chunk_set_id=set_id, profile=PROFILE) == "ready_hybrid"
    other_profile = replace(PROFILE, compatibility_epoch="other")
    assert get_index_readiness(store, version, chunk_set_id=set_id, profile=other_profile) == "ready_keyword"
    embeddings.invalidate_profile(PROFILE)
    assert get_index_readiness(store, version, chunk_set_id=set_id, profile=PROFILE) == "ready_keyword"
    with store.write() as db:
        db.execute("DELETE FROM keyword_index_rows WHERE chunk_set_id = ? AND chunk_index = 0", (set_id,))
    assert get_index_readiness(store, version, chunk_set_id=set_id, profile=PROFILE) is None
    _, empty_version = add_book(store, b"empty")
    empty_id = seed_chunks(store, empty_version, seed_parse(store, empty_version))
    assert get_index_readiness(store, empty_version, chunk_set_id=empty_id) == "unsearchable"


def test_new_revision_invalidates_claim_and_preserves_old_published_index(queue):
    jobs, book = queue
    store, (document, old_version) = jobs.store, book
    old_set = seed_chunks(store, old_version, seed_parse(store, old_version), "previous textbook")
    seed_keyword_index(store, old_set)
    before, = KeywordSearchService(store).search("previous", chunk_set_ids=[old_set])
    old_job = jobs.create_or_get(*book, CONFIG)
    stale = jobs.claim_next("worker")
    advance_to_publishing(jobs, stale)
    with store.write() as db:
        version = add_version(db, document, add_blob(db, b"new version"))
    assert jobs.get_job(old_job.job_id).error_code == "INDEX_VERSION_SUPERSEDED"
    with pytest.raises(IndexJobError, match="INDEX_VERSION_NOT_CURRENT"):
        jobs.create_or_get(*book, "another-config")
    with pytest.raises(IndexJobError, match="INDEX_JOB_LEASE_LOST"):
        jobs.complete(stale)
    new_job = jobs.create_or_get(document, version, CONFIG)
    jobs.fail(jobs.claim_next("worker"), "PDF_INVALID_FILE")
    assert get_index_readiness(store, version, chunk_set_id=old_set, job_id=new_job.job_id) == "failed"
    assert get_index_readiness(store, old_version, chunk_set_id=old_set) == "ready_keyword"
    assert KeywordSearchService(store).search("previous", chunk_set_ids=[old_set]) == (before,)
    assert KeywordSearchService(store).search("previous") == ()
    current = jobs.create_or_get(document, version, CONFIG, restart_after=new_job.job_id)
    claim = jobs.claim_next("worker")
    advance_to_publishing(jobs, claim)
    store.detach_document(document)
    assert jobs.get_job(current.job_id) is None and jobs.list_jobs(document_id=document) == []
    with pytest.raises(IndexJobError, match="INDEX_JOB_LEASE_LOST"):
        jobs.complete(claim)
    assert get_index_readiness(store, old_version, chunk_set_id=old_set) is None


def test_v6_migration_rolls_back_then_preserves_existing_publications(tmp_path, monkeypatch):
    migrations = library_storage.MIGRATIONS
    with monkeypatch.context() as patch:
        patch.setattr(library_storage, "MIGRATIONS", migrations[:6])
        patch.setattr(library_storage, "SCHEMA_VERSION", 6)
        old = KnowledgeStore(tmp_path)
    book = add_book(old)
    set_id = seed_chunks(old, book[1], seed_parse(old, book[1]), "published textbook")
    seed_keyword_index(old, set_id)
    before = KeywordSearchService(old).search("published")
    with old.connect() as db:
        schema = db.execute("SELECT * FROM sqlite_schema ORDER BY name").fetchall()
    with monkeypatch.context() as patch:
        patch.setattr(library_storage, "MIGRATIONS", (*migrations[:6], (*migrations[6], "INVALID SQL")))
        with pytest.raises(sqlite3.OperationalError, match="syntax error"):
            KnowledgeStore(tmp_path)
    with old.connect() as db:
        assert db.execute("PRAGMA user_version").fetchone() == (6,)
        assert db.execute("SELECT * FROM sqlite_schema ORDER BY name").fetchall() == schema
    upgraded = KnowledgeStore(tmp_path)
    assert KeywordSearchService(upgraded).search("published") == before
    assert IndexJobStore(upgraded).list_jobs() == []
    with upgraded.connect() as db:
        assert db.execute("PRAGMA user_version").fetchone() == (7,)
        assert db.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.anyio
async def test_sidecar_maintenance_recovers_jobs_without_claiming(queue, tmp_path, monkeypatch):
    jobs, book = queue
    job = jobs.create_or_get(*book, CONFIG)
    jobs.claim_next("old-sidecar")
    expire(jobs.store, job.job_id)
    loop = asyncio.get_running_loop()
    recovered = asyncio.Event()
    real_recover = IndexJobStore.recover_expired

    def observe_recovery(store):
        results = real_recover(store)
        loop.call_soon_threadsafe(recovered.set)
        return results

    monkeypatch.setattr(IndexJobStore, "recover_expired", observe_recovery)
    control = SidecarControl(uuid4(), 12345, asyncio.Event())
    app = create_app(uuid4().hex, uuid4().hex, database_path=tmp_path / "business.sqlite3", control=control)
    async with app.router.lifespan_context(app):
        await asyncio.wait_for(recovered.wait(), timeout=5)
        scheduled = app.state.index_jobs.get_job(job.job_id)
        assert scheduled.status == "queued" and scheduled.error_code == "INDEX_LEASE_EXPIRED"
        assert scheduled.instance_id is None
        with jobs.store.connect() as db:
            for table in ("parsed_documents", "chunk_sets", "embeddings", "keyword_index_sets"):
                assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone() == (0,)
    assert app.state.index_jobs is None
