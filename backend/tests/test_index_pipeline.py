import asyncio
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from docx import Document
from pydantic import SecretStr

from app.models import siliconflow
from app.services import document_chunking, document_parsing
from app.services.hybrid_retrieval import HybridRetrievalService
from app.services.index_pipeline import IndexPipeline
from app.services.index_readiness import get_index_readiness
from app.services.keyword_search import KeywordSearchService
from app.services.vector_retrieval import VectorRetrievalService
from app.storage.index_jobs import PIPELINE_JOB_TYPE, IndexJobError
from app.storage.library import KnowledgeStore
from test_docx_parsing import serialize
from test_embedding_storage import counts, make_service, response, unexpected, wire_input
from test_index_jobs import expire
from test_library_management import anyio_backend, docx, import_file, library
from test_pdf_parsing import FIXTURES, make_pdf, page, text

pytestmark = pytest.mark.anyio
DEEP_MARKDOWN = ("- " * 40 + "item\n").encode()


def pipeline_for(store, handler=unexpected, **options):
    embeddings, profile = make_service(store, handler, **options)
    return IndexPipeline(store, embeddings, profile)


def claim(pipeline, *, lease_seconds=120):
    result = pipeline.jobs.claim_next("pipeline-test", job_type=PIPELINE_JOB_TYPE, lease_seconds=lease_seconds)
    assert result is not None
    return result


def submit(pipeline, book, **options):
    return pipeline.submit(book.document_id, book.document_version_id, **options)


def chunk_set(pipeline, book):
    parsed = pipeline.parsing.get_current(book.document_version_id)
    return pipeline.chunking.get_chunk_set(parsed.parsed_document_id, pipeline.chunk_config)


def readiness(pipeline, book, job):
    ready = chunk_set(pipeline, book)
    return get_index_readiness(pipeline.store, book.document_version_id, chunk_set_id=ready.chunk_set_id,
                               profile=pipeline.profile, job_id=job.job_id)


def hybrid_search(pipeline):
    return HybridRetrievalService(KeywordSearchService(pipeline.store),
                                  VectorRetrievalService(pipeline.store, pipeline.embeddings.provider))


@pytest.mark.parametrize("format", ["pdf", "docx", "md"])
async def test_formats_publish_independent_channels(library, format):
    sentence = "A triangle has three sides."
    if format == "pdf":
        data = make_pdf([page(text(72, 720, sentence))])
    elif format == "docx":
        document = Document()
        document.add_paragraph(sentence)
        data = serialize(document)
    else:
        data = sentence.encode()
    book = import_file(library, data, f"geometry.{format}")
    calls = []

    async def provider(request):
        calls.append(wire_input(request))
        if len(calls) == 1:
            current = pipeline.jobs.get_job(job.job_id)
            assert (current.stage, current.completed_work, current.total_work) == ("embedding", 0, 1)
            assert readiness(pipeline, book, current) == "ready_keyword"
            result = await hybrid_search(pipeline).search("triangle", profile=pipeline.profile)
            assert result.channel_status == "keyword_only" and result.hits
        return response(1)

    pipeline = pipeline_for(library.store, provider)
    job = submit(pipeline, book)
    assert submit(pipeline, book) == job
    assert KeywordSearchService(library.store).search("triangle") == ()
    finished = await pipeline.execute(claim(pipeline))
    assert (finished.status, finished.stage, finished.completed_work) == ("succeeded", "publishing", 1)
    assert readiness(pipeline, book, finished) == "ready_hybrid"
    result = await hybrid_search(pipeline).search("triangle", profile=pipeline.profile)
    assert result.channel_status == "hybrid" and len(result.hits) == 1
    assert result.hits[0].document_version_id == book.document_version_id
    assert [rank.channel for rank in result.hits[0].channel_ranks] == ["keyword", "vector"]
    assert len(calls) == 2


@pytest.mark.parametrize(("filename", "data", "code"), [
    ("locked.pdf", (FIXTURES / "password.pdf").read_bytes(), "PDF_ENCRYPTED"),
    ("broken.docx", docx(), "DOCX_INVALID"),
    ("nested.md", DEEP_MARKDOWN, "MARKDOWN_NESTING_LIMIT"),
])
async def test_parse_failures_are_persisted(library, filename, data, code):
    book = import_file(library, data, filename)
    pipeline = pipeline_for(library.store)
    job = submit(pipeline, book)
    finished = await pipeline.execute(claim(pipeline))
    assert (finished.status, finished.stage, finished.error_code) == ("failed", "parsing", code)
    assert get_index_readiness(library.store, book.document_version_id, job_id=job.job_id) == "failed"


@pytest.mark.parametrize(("filename", "data", "has_chunks"), [
    ("scan.pdf", make_pdf([page(b"q 600 0 0 800 0 0 cm /Im0 Do Q")]), False),
    ("symbols.md", b"!!!", True),
])
async def test_no_searchable_text_skips_both_channels(library, filename, data, has_chunks):
    pipeline = pipeline_for(library.store)
    book = import_file(library, data, filename)
    submit(pipeline, book)
    finished = await pipeline.execute(claim(pipeline))
    assert finished.status == "succeeded"
    assert bool(chunk_set(pipeline, book).chunk_count) == has_chunks
    assert readiness(pipeline, book, finished) == "unsearchable"
    assert counts(library.store) == (0, 0, 0)
    with library.store.connect() as db:
        assert db.execute("SELECT count(*) FROM keyword_index_sets").fetchone() == (0,)
    result = await hybrid_search(pipeline).search("anything", profile=pipeline.profile)
    assert result.channel_status == "unavailable"


@pytest.mark.parametrize("http_status", [None, 401])
async def test_keyword_survives_auth_failure_and_upgrades_after_credentials(library, http_status):
    book = import_file(library, b"A triangle has three sides.")
    calls = []

    def provider(request):
        calls.append(wire_input(request))
        return httpx.Response(401) if len(calls) == 1 and http_status else response(1)

    pipeline = pipeline_for(library.store, provider, configured=http_status is not None)
    job = submit(pipeline, book)
    failed = await pipeline.execute(claim(pipeline))
    code = "EMBEDDING_AUTH_FAILED" if http_status else "EMBEDDING_NOT_CONFIGURED"
    assert (failed.status, failed.stage, failed.error_code, failed.error_http_status) == (
        "failed", "embedding", code, http_status,
    )
    assert readiness(pipeline, book, failed) == "ready_keyword"
    indexed, = KeywordSearchService(library.store).search("triangle")
    result = await hybrid_search(pipeline).search("triangle", profile=pipeline.profile)
    assert result.channel_status == "keyword_only" and len(calls) == bool(http_status)
    assert submit(pipeline, book).job_id == job.job_id

    restarted = pipeline_for(KnowledgeStore(library.store.path.parent.parent), lambda _: response(1), configured=False)
    restarted.embeddings.provider.configure(SecretStr("test-key"), siliconflow.SILICONFLOW_BASE_URL)
    successor = submit(restarted, book, restart_after=failed.job_id)
    completed = await restarted.execute(claim(restarted))
    assert completed.job_id == successor.job_id and completed.status == "succeeded"
    assert readiness(restarted, book, completed) == "ready_hybrid"
    assert KeywordSearchService(restarted.store).search("triangle") == (indexed,)
    assert restarted.jobs.get_job(job.job_id) == failed


@pytest.mark.parametrize(("failure", "code", "attempts"), [
    (429, "EMBEDDING_RATE_LIMITED", 1),
    (503, "EMBEDDING_SERVICE_ERROR", 3),
    (None, "EMBEDDING_TIMEOUT", 3),
])
async def test_provider_and_job_retry_budgets(library, monkeypatch, failure, code, attempts):
    monkeypatch.setattr(siliconflow, "uniform", lambda *_: 0)
    calls = 0

    def provider(request):
        nonlocal calls
        calls += 1
        if failure is None:
            raise httpx.ReadTimeout("temporary timeout", request=request)
        return httpx.Response(failure, headers={"Retry-After": "120"} if failure == 429 else {})

    pipeline = pipeline_for(library.store, provider)
    book = import_file(library, b"A triangle has three sides.")
    submit(pipeline, book, max_retries=1)
    waiting = await pipeline.execute(claim(pipeline))
    assert (waiting.status, waiting.error_code, waiting.error_http_status, waiting.retry_count) == (
        "retry_wait", code, failure, 1,
    )
    assert calls == attempts and readiness(pipeline, book, waiting) == "ready_keyword"
    assert pipeline.jobs.claim_next("too-early", job_type=PIPELINE_JOB_TYPE) is None
    with library.store.write() as db:
        db.execute("UPDATE index_jobs SET next_retry_at = unixepoch() WHERE job_id = ?", (waiting.job_id,))
        db.execute("UPDATE embeddings SET retry_not_before = 0")
    pipeline.embeddings.provider._cooldown_until = 0
    finished = await pipeline.execute(claim(pipeline))
    assert (finished.status, finished.error_code, finished.retry_count) == ("failed", code, 1)
    assert calls == 2 * attempts and readiness(pipeline, book, finished) == "ready_keyword"


@pytest.mark.parametrize("interrupt", ["cancel", "expire", "update", "delete"])
async def test_interruption_during_network_cannot_publish_vectors(library, interrupt):
    book = import_file(library, b"Octagon geometry")
    started, respond = asyncio.Event(), asyncio.Event()

    async def provider(_request):
        started.set()
        await respond.wait()
        return response(1)

    pipeline = pipeline_for(library.store, provider)
    job = submit(pipeline, book)
    current_claim = claim(pipeline)
    task = asyncio.create_task(pipeline.execute(current_claim))
    await asyncio.wait_for(started.wait(), 5)
    ready = chunk_set(pipeline, book)
    keyword = KeywordSearchService(library.store)
    assert keyword.search("Octagon") and readiness(pipeline, book, job) == "ready_keyword"
    # All four writes can commit while the embedding request is waiting.
    if interrupt == "cancel":
        pipeline.jobs.request_cancel(job.job_id)
    elif interrupt == "expire":
        expire(library.store, job.job_id)
        successor = claim(pipeline)
        assert successor.job.job_id == job.job_id and successor.lease_token != current_claim.lease_token
    elif interrupt == "update":
        updated = import_file(library, DEEP_MARKDOWN, document_id=book.document_id)
    else:
        library.delete_document(book.document_id)
    respond.set()
    if interrupt == "cancel":
        assert (await task).status == "cancelled"
    else:
        with pytest.raises(IndexJobError, match="INDEX_JOB_LEASE_LOST"):
            await task
    assert counts(library.store)[::2] == (0, 0)
    assert bool(keyword.search("Octagon")) == (interrupt in ("cancel", "expire"))
    assert bool(keyword.search("Octagon", chunk_set_ids=[ready.chunk_set_id])) == (interrupt != "delete")
    if interrupt == "update":
        submit(pipeline, updated)
        assert (await pipeline.execute(claim(pipeline))).status == "failed"
        result = await hybrid_search(pipeline).search("Octagon", profile=pipeline.profile)
        assert not result.hits and result.documents[0].document_version_id == updated.document_version_id
        history = await hybrid_search(pipeline).search(
            "Octagon", profile=pipeline.profile, chunk_set_ids=[ready.chunk_set_id],
        )
        assert history.channel_status == "keyword_only" and history.hits[0].document_version_id == book.document_version_id


async def test_latest_and_pinned_hybrid_results_use_separate_versions(library):
    pipeline = pipeline_for(library.store, lambda _: response(1))
    original = import_file(library, b"Geometry of triangles.")
    submit(pipeline, original)
    assert (await pipeline.execute(claim(pipeline))).status == "succeeded"
    old_set = chunk_set(pipeline, original)
    revised = import_file(library, b"Geometry of octagons.", document_id=original.document_id)
    search = hybrid_search(pipeline)
    before = await search.search("Geometry", profile=pipeline.profile)
    assert before.hits == () and before.documents[0].document_version_id == revised.document_version_id
    submit(pipeline, revised)
    assert (await pipeline.execute(claim(pipeline))).status == "succeeded"
    current = await search.search("Geometry", profile=pipeline.profile)
    pinned = await search.search("Geometry", profile=pipeline.profile, chunk_set_ids=[old_set.chunk_set_id])
    assert {hit.document_version_id for hit in current.hits} == {revised.document_version_id}
    assert {hit.document_version_id for hit in pinned.hits} == {original.document_version_id}
    assert current.channel_status == pinned.channel_status == "hybrid"
    library.delete_document(original.document_id)
    assert (await search.search("Geometry", profile=pipeline.profile)).hits == ()
    assert (await search.search("Geometry", profile=pipeline.profile, chunk_set_ids=[old_set.chunk_set_id])).hits == ()


@pytest.mark.parametrize("interrupt", ["cancel", "expire"])
async def test_keyword_publication_cannot_be_bypassed_by_maintenance(library, monkeypatch, interrupt):
    pipeline = pipeline_for(library.store)
    book = import_file(library, b"Octagon geometry")
    job = submit(pipeline, book)
    index = pipeline.keywords
    index_batch = index._index_batch

    def interrupt_before_write(set_id, *, claim):
        if interrupt == "cancel":
            pipeline.jobs.request_cancel(job.job_id)
        else:
            expire(library.store, job.job_id)
        return index_batch(set_id, claim=claim)

    monkeypatch.setattr(index, "_index_batch", interrupt_before_write)
    if interrupt == "cancel":
        assert (await pipeline.execute(claim(pipeline))).status == "cancelled"
    else:
        with pytest.raises(IndexJobError, match="INDEX_JOB_LEASE_LOST"):
            await pipeline.execute(claim(pipeline))
    ready = chunk_set(pipeline, book)
    assert index.index_pending() == ()
    index.rebuild()
    with pytest.raises(IndexJobError, match="INDEX_PUBLICATION_REQUIRES_CLAIM"):
        index.ensure_indexed(ready.chunk_set_id)
    assert KeywordSearchService(library.store).search("Octagon", chunk_set_ids=[ready.chunk_set_id]) == ()


async def test_cancellation_between_vector_commit_and_publication(library, monkeypatch):
    book = import_file(library, b"Octagon geometry")
    pipeline = pipeline_for(library.store, lambda _: response(1))
    job = submit(pipeline, book)
    publish = pipeline.embeddings.results.publish

    def cancel_then_publish(set_id, profile, *, claim):
        pipeline.jobs.request_cancel(job.job_id)
        return publish(set_id, profile, claim=claim)

    monkeypatch.setattr(pipeline.embeddings.results, "publish", cancel_then_publish)
    finished = await pipeline.execute(claim(pipeline))
    assert finished.status == "cancelled" and readiness(pipeline, book, finished) == "ready_keyword"
    assert counts(library.store) == (1, 1, 0)
    ready = chunk_set(pipeline, book)
    with pytest.raises(IndexJobError, match="INDEX_PUBLICATION_REQUIRES_CLAIM"):
        publish(ready.chunk_set_id, pipeline.profile)

    restarted = pipeline_for(library.store)
    submit(restarted, book, restart_after=job.job_id)
    completed = await restarted.execute(claim(restarted))
    assert completed.status == "succeeded" and readiness(restarted, book, completed) == "ready_hybrid"


async def test_claim_cannot_be_executed_with_a_different_profile(library):
    book = import_file(library, b"Octagon geometry")
    pipeline = pipeline_for(library.store)
    original = submit(pipeline, book)
    changed = pipeline_for(library.store, epoch="new-embedding-profile")
    for_changed_profile = submit(changed, book)
    original_claim = pipeline.jobs.claim_next("targeted-test", job_type=PIPELINE_JOB_TYPE, job_id=original.job_id)
    assert original_claim.job.job_id == original.job_id
    finished = await changed.execute(original_claim)
    assert (finished.status, finished.error_code) == ("failed", "INDEX_PIPELINE_CONFIG_MISMATCH")
    assert changed.jobs.get_job(for_changed_profile.job_id).status == "queued"
    assert pipeline.parsing.get_current(book.document_version_id) is None


def _crash_after_committed_batch(app_data):
    pipeline = pipeline_for(KnowledgeStore(app_data), lambda _: response(1), batch_size=1)
    commit = pipeline.embeddings.results.commit

    def commit_then_exit(*args, **kwargs):
        commit(*args, **kwargs)
        os._exit(73)

    pipeline.embeddings.results.commit = commit_then_exit
    asyncio.run(pipeline.execute(claim(pipeline)))


async def test_process_death_reuses_artifacts_and_only_embeds_missing_chunks(library, tmp_path, monkeypatch):
    pipeline = pipeline_for(library.store)
    book = import_file(library, b"```text\nfirst chunk\n```\n\n```text\nsecond chunk\n```")
    job = submit(pipeline, book)
    child = await asyncio.create_subprocess_exec(
        sys.executable, "-c",
        "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
        "from test_index_pipeline import _crash_after_committed_batch; _crash_after_committed_batch(Path(sys.argv[2]))",
        str(Path(__file__).parent), str(tmp_path), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(child.communicate(), 15)
    assert child.returncode == 73, (stdout, stderr)
    assert counts(library.store) == (1, 1, 0)
    stranded = pipeline.jobs.get_job(job.job_id)
    assert stranded.stage == "embedding" and stranded.status == "running"
    original_set = chunk_set(pipeline, book)
    assert readiness(pipeline, book, stranded) == "ready_keyword"
    expire(library.store, job.job_id)

    def no_rebuild(*_args, **_kwargs):
        pytest.fail("a completed artifact was rebuilt")

    monkeypatch.setattr(document_parsing, "parse_markdown_version", no_rebuild)
    monkeypatch.setattr(document_chunking, "iter_stored_chunks", no_rebuild)
    calls = []

    def provider(request):
        calls.append(wire_input(request))
        return response(1)

    restarted = pipeline_for(KnowledgeStore(tmp_path), provider, batch_size=1)
    monkeypatch.setattr(restarted.keywords, "_index_batch", no_rebuild)
    completed = await restarted.execute(claim(restarted))
    assert completed.job_id == job.job_id and completed.retry_count == 1
    assert readiness(restarted, book, completed) == "ready_hybrid"
    assert chunk_set(restarted, book) == original_set
    assert counts(restarted.store) == (2, 2, 1)
    assert calls == ["second chunk\n"]
    submit(restarted, book, restart_after=completed.job_id)
    assert (await restarted.execute(claim(restarted))).status == "succeeded"
    assert calls == ["second chunk\n"]


async def test_heartbeat_renews_during_slow_embedding(library):
    async def provider(_request):
        await asyncio.sleep(3.1)
        renewed = pipeline.jobs.get_job(job.job_id)
        assert renewed.lease_expires_at > initial.job.lease_expires_at
        return response(1)

    pipeline = pipeline_for(library.store, provider)
    book = import_file(library, b"A triangle has three sides.")
    job = submit(pipeline, book)
    initial = claim(pipeline, lease_seconds=3)
    assert (await pipeline.execute(initial, lease_seconds=3)).status == "succeeded"
