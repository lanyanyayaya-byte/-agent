import asyncio
import json
import signal
import sqlite3
import subprocess
import sys
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from time import monotonic, sleep
from uuid import uuid4

import httpx
import pytest
from docx import Document

from app.models.siliconflow import SILICONFLOW_BASE_URL, SiliconFlowEmbedding
from app.services.document_embeddings import DocumentEmbeddingService
from app.services.index_pipeline import IndexPipeline
from app.services.index_worker import INDEX_EMBEDDING_PROFILE, IndexWorker
from app.services.keyword_search import KeywordSearchService
from app.storage.business import DATABASE_NAME, BusinessStore
from app.storage.index_jobs import PIPELINE_JOB_TYPE, IndexJobClaim, IndexJobError
from test_chat import ProviderStream, payload as chat_payload, run_path
from test_docx_parsing import serialize
from test_embedding_storage import counts, response, wire_input
from test_library_management import import_file, library
from test_pdf_parsing import make_pdf, page, text
from test_sidecar import COMMAND, READY, SHUTDOWN, assert_clean_exit, headers, running_sidecar

JOBS = "/api/v1/_host/knowledge/index-jobs"
DOCUMENTS = "/api/v1/knowledge/documents"
CREDENTIALS = "/api/v1/_host/model-credentials"
RESTORED = CREDENTIALS + "/restore-complete"
TEST_KEY = "test-key"


def make_pipeline(store, *, batch_size=32):
    return IndexPipeline(
        store, DocumentEmbeddingService(store, SiliconFlowEmbedding(), batch_size=batch_size), INDEX_EMBEDDING_PROFILE,
    )


def submit(pipeline, book, **options):
    return pipeline.submit(book.document_id, book.document_version_id, **options)


def until(probe, *, timeout=8):
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        result = probe()
        if result:
            return result
        sleep(0.025)
    pytest.fail(f"Condition did not become true within {timeout} seconds")


def wait_for_job(pipeline, job_id, status):
    def probe():
        job = pipeline.jobs.get_job(job_id)
        return job if job.status == status else None

    return until(probe)


def requests_at(directory):
    path = directory / "embedding-requests.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _pause_parser(directory, format):
    # Keep the real parser entry points and process owner pipe. Hold only the layout reader.
    from importlib import import_module
    from app.parsing.worker import open_worker

    module = import_module(f"app.parsing.{format}")

    @contextmanager
    def slow_worker(command, *, cwd, cancelled):
        code = """
import importlib
import os
import sys
import time
from pathlib import Path

directory, format, *args = sys.argv[1:]
layout = importlib.import_module(f"app.parsing.{format}_layout")

def pause(*_args):
    Path(directory, "parser-pid").write_text(str(os.getpid()))
    time.sleep(60)

setattr(layout, f"read_{format}", pause)
entry = importlib.import_module(f"app.parsing.{format}")
raise SystemExit(getattr(entry, f"{format}_worker_main")(args))
"""
        delayed = [sys.executable, "-E", "-s", "-B", "-c", code, str(directory), format, *command[-2:]]
        with open_worker(delayed, cwd=cwd, cancelled=cancelled) as worker:
            yield worker

    module.open_worker = slow_worker


def _fixture_sidecar(directory, mode):
    import app.main
    import app.services.index_worker
    from app.models.deepseek import DeepSeekAdapter, DeepSeekSettings
    from app.services.models import ModelService
    from app.sidecar import main

    app.services.index_worker.POLL_SECONDS = 0.1
    app.main.IndexWorker = partial(IndexWorker, lease_seconds=3)
    if mode in {"pdf", "docx"}:
        _pause_parser(directory, mode)
    if mode == "chunking":
        from app.services import document_chunking

        def pause_chunking(*_args, **_kwargs):
            (directory / "chunking-started").touch()
            sleep(60)
            raise AssertionError("Sidecar should have been killed during chunking")

        document_chunking.iter_stored_chunks = pause_chunking
    if mode == "raise":
        execute = IndexPipeline.execute

        async def execute_with_failure(self, claim, **kwargs):
            if not (directory / "worker-raised").exists():
                (directory / "worker-raised").touch()
                raise RuntimeError("private error details")
            return await execute(self, claim, **kwargs)

        IndexPipeline.execute = execute_with_failure

    async def embed(request):
        items = wire_input(request)
        with (directory / "embedding-requests.jsonl").open("a") as log:
            log.write(json.dumps(items) + "\n")
        request_count = len(requests_at(directory))
        if mode == "401":
            return httpx.Response(401)
        if mode == "network" and request_count == 1:
            raise httpx.ConnectError("fixture")
        if mode == "hold" or (mode == "hold-second" and request_count == 2):
            try:
                while not (directory / "release").exists():
                    await asyncio.sleep(0.025)
            except asyncio.CancelledError:
                (directory / "request-cancelled").touch()
                raise
        return response(len(items) if isinstance(items, list) else 1)

    class ChatGate:
        async def wait(self):
            while not (directory / "chat-release").exists():
                await asyncio.sleep(0.025)

    def factory(*args, **kwargs):
        models = ModelService(DeepSeekAdapter(
            DeepSeekSettings(),
            transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=ProviderStream(ChatGate()))),
        ))
        embedding = SiliconFlowEmbedding(transport=httpx.MockTransport(embed))
        return app.main.create_app(*args, models=models, embeddings=embedding, **kwargs)

    return main(app_factory=factory)


@contextmanager
def local_sidecar(directory, mode="normal", *, restore=True):
    command = [
        *COMMAND[:4], "-c", "import sys; from pathlib import Path; sys.path.insert(0, 'tests'); "
        "from test_index_worker import _fixture_sidecar; "
        "raise SystemExit(_fixture_sidecar(Path(sys.argv[1]), sys.argv[2]))",
        str(directory), mode,
    ]
    with running_sidecar(directory, command, restore_credentials=restore) as (child, ready, tokens):
        with httpx.Client(base_url=f"http://127.0.0.1:{ready['port']}", trust_env=False, timeout=3,
                          headers=headers(tokens["ui_token"])) as client:
            yield child, client, headers(tokens["host_token"]), tokens


def configure(client, host, *, key=True):
    if key:
        result = client.put(CREDENTIALS + "/siliconflow", headers=host,
                            json={"api_key": TEST_KEY, "base_url": SILICONFLOW_BASE_URL})
        assert result.status_code == 200
    assert client.post(RESTORED, headers=host).status_code == 204


def assert_chat_and_library_respond(client, host, directory, *, books):
    release = directory / "chat-release"
    release.unlink(missing_ok=True)
    model = client.put(CREDENTIALS + "/deepseek", headers=host, json={"api_key": "test-deepseek"})
    assert model.status_code == 200
    session = client.post("/api/v1/sessions", json={})
    assert session.status_code == 201
    started = monotonic()
    with client.stream("POST", run_path(session.json()["session_id"]), json=chat_payload()) as stream:
        assert stream.status_code == 200
        events = (json.loads(line[6:]) for line in stream.iter_lines() if line.startswith("data: "))
        assert next(events)["type"] == "run_started"
        assert next(events)["type"] == "message_delta"
        assert len(client.get(DOCUMENTS).json()) == books
        release.touch()
        terminal = list(events)[-1]
        assert terminal["type"] == "run_completed"
    return monotonic() - started, session.json()["session_id"]


def test_ready_before_credentials_then_serial_queue_keeps_chat_and_sse_responsive(library, tmp_path):
    pipeline = make_pipeline(library.store)
    books = [import_file(library, value) for value in (
        b"Octagon geometry", b"Basalt geology", b"Incompatible profile", b"Unqueued zoology",
    )]
    job_a, job_b = (submit(pipeline, book) for book in books[:2])
    unsupported = pipeline.jobs.create_or_get(
        books[2].document_id, books[2].document_version_id, "other-profile", job_type=PIPELINE_JOB_TYPE,
    )

    with local_sidecar(tmp_path, "hold") as (child, client, host, tokens):
        assert client.get(READY, headers=host).status_code == 200
        assert client.get(JOBS).status_code == 403
        page = client.get(JOBS, headers=host).json()
        assert page["active_job_id"] is None
        assert {job["status"] for job in page["jobs"]} == {"queued"}
        assert [job["job_id"] for job in page["jobs"] if not job["worker_supported"]] == [unsupported.job_id]
        assert not requests_at(tmp_path)

        configure(client, host)
        until(lambda: requests_at(tmp_path))
        snapshot = client.get(JOBS, headers=host).json()
        running, = (job for job in snapshot["jobs"] if job["status"] == "running")
        other = job_b if running["job_id"] == job_a.job_id else job_a
        assert snapshot["active_job_id"] == running["job_id"]
        assert running["published_readiness"] == "ready_keyword"
        assert running["stage"] == "embedding" and running["completed_work"] == 0
        assert "lease_token" not in running and "instance_id" not in running
        elapsed, _ = assert_chat_and_library_respond(client, host, tmp_path, books=4)
        assert elapsed < 2
        assert len(requests_at(tmp_path)) == 1
        assert pipeline.jobs.get_job(other.job_id).status == "queued"

        (tmp_path / "release").touch()
        for job in (job_a, job_b):
            wait_for_job(pipeline, job.job_id, "succeeded")
            detail = client.get(f"{JOBS}/{job.job_id}", headers=host)
            assert detail.headers["cache-control"] == "no-store"
            assert detail.json()["published_readiness"] == "ready_hybrid"
        assert pipeline.jobs.get_job(unsupported.job_id).status == "queued"
        assert pipeline.jobs.list_jobs(document_id=books[3].document_id) == []
        assert sorted(requests_at(tmp_path)) == ["Basalt geology", "Octagon geometry"]
        assert client.post(SHUTDOWN, headers=host).status_code == 204
        assert_clean_exit(child, tokens)


@pytest.mark.parametrize("ending", ["cancel", "shutdown"])
def test_cancel_or_shutdown_interrupts_inflight_request(library, tmp_path, ending):
    pipeline = make_pipeline(library.store)
    book = import_file(library, b"Geometry of an octagon.")
    job = submit(pipeline, book, max_retries=0)
    with local_sidecar(tmp_path, "hold") as (child, client, host, tokens):
        configure(client, host)
        until(lambda: requests_at(tmp_path))
        if ending == "cancel":
            pipeline.jobs.request_cancel(job.job_id)
            assert wait_for_job(pipeline, job.job_id, "cancelled").retry_count == 0
            assert client.get(f"{JOBS}/{job.job_id}", headers=host).json()["published_readiness"] == "ready_keyword"
        with library.store.write():
            assert client.post(SHUTDOWN, headers=host).status_code == 204
            if ending == "shutdown":
                until((tmp_path / "request-cancelled").exists)
        assert_clean_exit(child, tokens)
    assert (tmp_path / "request-cancelled").exists()
    assert counts(library.store)[::2] == (0, 0)
    assert KeywordSearchService(library.store).search("octagon")
    final = pipeline.jobs.get_job(job.job_id)
    assert final.status == ("cancelled" if ending == "cancel" else "queued")
    assert final.retry_count == 0 and final.error_code is None and final.instance_id is None

    if ending == "shutdown":
        (tmp_path / "release").touch()
        with local_sidecar(tmp_path) as (child, client, host, tokens):
            configure(client, host)
            assert wait_for_job(pipeline, job.job_id, "succeeded").retry_count == 0
            assert client.post(SHUTDOWN, headers=host).status_code == 204
            assert_clean_exit(child, tokens)


def test_sigkill_recovers_expired_job_and_committed_batch(library, tmp_path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGEAGENT_INDEX_EMBEDDING_BATCH_SIZE", "1")
    pipeline = make_pipeline(library.store, batch_size=1)
    book = import_file(library, b"```text\nfirst chunk\n```\n\n```text\nsecond chunk\n```")
    job = submit(pipeline, book)
    with local_sidecar(tmp_path, "hold-second") as (child, client, host, _):
        configure(client, host)
        until(lambda: len(requests_at(tmp_path)) == 2)
        assert counts(library.store) == (1, 2, 0)
        old = pipeline.jobs.get_job(job.job_id)
        with library.store.connect() as db:
            lease, = db.execute("SELECT lease_token FROM index_jobs WHERE job_id = ?", (job.job_id,)).fetchone()
            pending_deadline, = db.execute("SELECT lease_expires_at FROM embeddings WHERE vector IS NULL").fetchone()
        assert pending_deadline > old.lease_expires_at
        old_claim = IndexJobClaim(old, lease)
        parsed = pipeline.parsing.get_current(book.document_version_id)
        chunk_set = pipeline.chunking.get_chunk_set(parsed.parsed_document_id)
        child.kill()
        assert child.wait(timeout=5) == -signal.SIGKILL
    assert pipeline.jobs.get_job(job.job_id).status == "running"

    with local_sidecar(tmp_path, "hold") as (child, client, host, tokens):
        configure(client, host)
        until(lambda: len(requests_at(tmp_path)) == 3)
        successor = pipeline.jobs.get_job(job.job_id)
        assert successor.instance_id != old.instance_id and successor.retry_count == 1
        with pytest.raises(IndexJobError, match="INDEX_JOB_LEASE_LOST"):
            pipeline.jobs.complete(old_claim)
        with pytest.raises(IndexJobError, match="INDEX_JOB_LEASE_LOST"):
            pipeline.embeddings.results.stage(chunk_set.chunk_set_id, pipeline.profile, claim=old_claim)
        (tmp_path / "release").touch()
        wait_for_job(pipeline, job.job_id, "succeeded")
        assert client.get(f"{JOBS}/{job.job_id}", headers=host).json()["published_readiness"] == "ready_hybrid"
        assert client.post(SHUTDOWN, headers=host).status_code == 204
        assert_clean_exit(child, tokens)
    assert counts(library.store) == (2, 2, 1)
    assert requests_at(tmp_path) == ["first chunk\n", "second chunk\n", "second chunk\n"]
    assert pipeline.parsing.get_current(book.document_version_id) == parsed
    assert pipeline.chunking.get_chunk_set(parsed.parsed_document_id) == chunk_set


def test_sigkill_reclaims_interrupted_chunk_build_when_job_lease_expires(library, tmp_path):
    pipeline = make_pipeline(library.store)
    book = import_file(library, b"Geometry of an octagon.")
    job = submit(pipeline, book, max_retries=1)
    with local_sidecar(tmp_path, "chunking") as (child, client, host, _):
        configure(client, host)
        until((tmp_path / "chunking-started").exists)
        elapsed, _ = assert_chat_and_library_respond(client, host, tmp_path, books=1)
        assert elapsed < 2
        old = pipeline.jobs.get_job(job.job_id)
        with library.store.connect() as db:
            deadline, = db.execute("SELECT lease_expires_at FROM chunk_sets WHERE status = 'writing'").fetchone()
        assert deadline <= old.lease_expires_at
        parsed = pipeline.parsing.get_current(book.document_version_id)
        child.kill()
        assert child.wait(timeout=5) == -signal.SIGKILL

    with local_sidecar(tmp_path) as (child, client, host, tokens):
        configure(client, host)
        assert wait_for_job(pipeline, job.job_id, "succeeded").retry_count == 1
        assert client.post(SHUTDOWN, headers=host).status_code == 204
        assert_clean_exit(child, tokens)
    assert pipeline.parsing.get_current(book.document_version_id) == parsed
    assert requests_at(tmp_path) == ["Geometry of an octagon."]


def test_knowledge_write_contention_keeps_chat_and_library_responsive(library, tmp_path):
    pipeline = make_pipeline(library.store)
    job = submit(pipeline, import_file(library, b"Geometry of an octagon."))
    with local_sidecar(tmp_path) as (child, client, host, tokens):
        with library.store.write():
            configure(client, host)
            elapsed, _ = assert_chat_and_library_respond(client, host, tmp_path, books=1)
            assert elapsed < 2
            assert pipeline.jobs.get_job(job.job_id).status == "queued"
            assert not requests_at(tmp_path)
        wait_for_job(pipeline, job.job_id, "succeeded")
        assert client.post(SHUTDOWN, headers=host).status_code == 204
        assert_clean_exit(child, tokens)


def parser_is_gone(pid):
    status = subprocess.run(["/bin/ps", "-o", "stat=", "-p", str(pid)], capture_output=True, timeout=2).stdout.strip()
    # Some init processes defer reaping dead orphans. A zombie cannot execute parser code.
    return not status or status.startswith(b"Z")


@pytest.mark.parametrize("format", ["pdf", "docx"])
@pytest.mark.parametrize("ending", ["cancel", "shutdown", "kill"])
def test_parser_subprocess_stops_with_sidecar(library, tmp_path, format, ending):
    pipeline = make_pipeline(library.store)
    if format == "pdf":
        source = make_pdf([page(text(72, 720, "Geometry of an octagon."))])
    else:
        doc = Document()
        doc.add_paragraph("Geometry of an octagon.")
        source = serialize(doc)
    book = import_file(library, source, f"geometry.{format}")
    job = submit(pipeline, book)
    with local_sidecar(tmp_path, format, restore=False) as (child, client, host, tokens):
        marker = tmp_path / "parser-pid"
        parser_pid = int(until(lambda: marker.read_text() if marker.exists() else None))
        assert not parser_is_gone(parser_pid)
        if ending == "cancel":
            pipeline.jobs.request_cancel(job.job_id)
            wait_for_job(pipeline, job.job_id, "cancelled")
            until(lambda: parser_is_gone(parser_pid), timeout=3)
            assert client.post(SHUTDOWN, headers=host).status_code == 204
            assert_clean_exit(child, tokens)
        elif ending == "shutdown":
            elapsed, _ = assert_chat_and_library_respond(client, host, tmp_path, books=1)
            assert elapsed < 2
            assert client.post(SHUTDOWN, headers=host).status_code == 204
            assert_clean_exit(child, tokens)
        else:
            child.kill()
            assert child.wait(timeout=5) == -signal.SIGKILL
        until(lambda: parser_is_gone(parser_pid), timeout=3)
    assert pipeline.parsing.get_current(book.document_version_id) is None
    if ending == "cancel":
        assert pipeline.jobs.get_job(job.job_id).status == "cancelled"
        return
    assert pipeline.jobs.get_job(job.job_id).status == ("queued" if ending == "shutdown" else "running")
    with local_sidecar(tmp_path) as (child, client, host, tokens):
        configure(client, host)
        result = wait_for_job(pipeline, job.job_id, "succeeded")
        assert result.retry_count == (0 if ending == "shutdown" else 1)
        assert client.post(SHUTDOWN, headers=host).status_code == 204
        assert_clean_exit(child, tokens)


@pytest.mark.parametrize("mode,expected", [
    ("401", "EMBEDDING_AUTH_FAILED"), ("missing", "EMBEDDING_NOT_CONFIGURED"), ("network", "EMBEDDING_NETWORK_ERROR"),
])
def test_embedding_failure_leaves_keyword_publication_and_only_retries_temporary_errors(library, tmp_path, mode, expected):
    pipeline = make_pipeline(library.store)
    job = submit(pipeline, import_file(library, b"Geometry of an octagon."))
    with local_sidecar(tmp_path, mode) as (child, client, host, tokens):
        configure(client, host, key=mode != "missing")
        transient = mode == "network"
        result = wait_for_job(pipeline, job.job_id, "retry_wait" if transient else "failed")
        detail = client.get(f"{JOBS}/{job.job_id}", headers=host).json()
        assert detail["error_code"] == result.error_code == expected
        assert detail["next_retry_at"] == result.next_retry_at
        assert detail["published_readiness"] == "ready_keyword"
        assert len(requests_at(tmp_path)) == (0 if mode == "missing" else 1)
        assert KeywordSearchService(library.store).search("octagon")
        if transient:
            assert wait_for_job(pipeline, job.job_id, "succeeded").retry_count == 1
            assert len(requests_at(tmp_path)) == 2
        else:
            assert result.retry_count == 0 and result.next_retry_at is None
        assert client.post(SHUTDOWN, headers=host).status_code == 204
        assert_clean_exit(child, tokens)


def test_worker_exception_does_not_end_dispatch_or_chat(library, tmp_path):
    pipeline = make_pipeline(library.store)
    for body in (b"Basalt geology", b"Octagon geometry"):
        submit(pipeline, import_file(library, body))
    with local_sidecar(tmp_path, "raise") as (child, client, host, tokens):
        configure(client, host)
        until(lambda: pipeline.jobs.list_jobs(status="succeeded"))
        failed, = pipeline.jobs.list_jobs(status="failed")
        assert failed.error_code == "INDEX_WORKER_ERROR"
        assert_chat_and_library_respond(client, host, tmp_path, books=2)
        assert client.post(SHUTDOWN, headers=host).status_code == 204
        assert_clean_exit(child, tokens)
    assert "private error details" not in (tmp_path / "logs/sidecar.jsonl").read_text()


def test_fts_fault_is_isolated_from_chat_database(library, tmp_path):
    pipeline = make_pipeline(library.store)
    job = submit(pipeline, import_file(library, b"Geometry of an octagon."))
    with library.store.write() as db:
        db.execute("DROP TABLE keyword_fts")
    with local_sidecar(tmp_path) as (child, client, host, tokens):
        configure(client, host)
        failed = wait_for_job(pipeline, job.job_id, "failed")
        assert failed.error_code == "INDEX_PIPELINE_STORAGE_ERROR"
        detail = client.get(f"{JOBS}/{job.job_id}", headers=host).json()
        assert detail["published_readiness"] == "failed"
        elapsed, session_id = assert_chat_and_library_respond(client, host, tmp_path, books=1)
        assert elapsed < 2
        assert client.post(SHUTDOWN, headers=host).status_code == 204
        assert_clean_exit(child, tokens)
    assert not requests_at(tmp_path)
    store = BusinessStore(tmp_path / DATABASE_NAME)
    with sqlite3.connect(store.path) as db:
        assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert db.execute("SELECT status FROM runs WHERE session_id = ?", (session_id,)).fetchone() == ("completed",)
