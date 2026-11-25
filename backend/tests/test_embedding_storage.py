import asyncio
import hashlib
import json
import sqlite3
import struct
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from pydantic import SecretStr

from app.chunking.models import ChunkerConfig
from app.models.embeddings import EmbeddingError, canonical_sha256
from app.main import create_app
from app.models.siliconflow import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, EMBEDDING_URL, SILICONFLOW_BASE_URL
from app.models import siliconflow
from app.services.document_chunking import DocumentChunkingService
from app.services.document_embeddings import DocumentEmbeddingService
from app.services.document_parsing import DocumentParsingService
from app.services.library import LibraryService
from app.storage import library as storage_library
from app.storage.library import KnowledgeStore
from test_library_management import anyio_backend, import_file, library

pytestmark = pytest.mark.anyio


def chunks(library, *texts, title=None, document_id=None, config=ChunkerConfig()):
    content = "\n\n".join(f"```text\n{text}\n```" for text in texts)
    if title is not None:
        content = f"# {title}\n\n" + content
    book = import_file(library, content.encode(), document_id=document_id)
    parsed = DocumentParsingService(library.store).ensure_parsed(book.document_version_id)
    return book, DocumentChunkingService(library.store).ensure_chunked(parsed.parsed_document_id, config).chunk_set_id


def response(items, *, revision=None):
    data = [{"index": i, "embedding": [i + 0.25] * EMBEDDING_DIMENSIONS} for i in reversed(range(items))]
    payload = {"object": "list", "model": EMBEDDING_MODEL, "data": data,
               "usage": {"prompt_tokens": 2, "total_tokens": 2}}
    if revision is not None:
        payload["model_version"] = revision
    return httpx.Response(200, json=payload)


def wire_input(request):
    assert request.url == EMBEDDING_URL
    assert request.headers["authorization"] == "Bearer test-key"
    payload = json.loads(request.content)
    assert set(payload) == {"model", "input", "encoding_format"}
    assert payload["model"] == EMBEDDING_MODEL and payload["encoding_format"] == "float"
    return payload["input"]


def make_service(store, handler, *, batch_size=32, epoch="test-epoch", strategy="title-body-v1", configured=True):
    provider = siliconflow.SiliconFlowEmbedding(transport=httpx.MockTransport(handler))
    if configured:
        provider.configure(SecretStr("test-key"), SILICONFLOW_BASE_URL)
    return (DocumentEmbeddingService(store, provider, batch_size=batch_size),
            provider.profile(compatibility_epoch=epoch, input_strategy=strategy))


def unexpected(_request):
    pytest.fail("unexpected SiliconFlow request")


def counts(store):
    with store.connect() as db:
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
        return db.execute("""SELECT count(vector), (SELECT count(*) FROM chunk_embeddings),
                            (SELECT count(published_at) FROM embedding_sets) FROM embeddings""").fetchone()


async def test_sort_wire_indices_and_reopen_little_endian_vectors(library, tmp_path):
    _, set_id = chunks(library, "线性代数", "e\u0301 🌍")

    def handler(request):
        assert wire_input(request) == ["线性代数\n", "e\u0301 🌍\n"]
        return response(2)

    service, profile = make_service(library.store, handler)
    assert service.read_vectors(set_id, profile) is None
    saved = await service.ensure_embedded(set_id, profile)
    assert counts(library.store) == (2, 2, 1)
    with library.store.connect() as db:
        rows = db.execute("SELECT input_sha256, vector FROM embeddings").fetchall()
        assert dict(rows)[hashlib.sha256("线性代数\n".encode()).hexdigest()] == struct.pack("<1024f", *[0.25] * 1024)
        assert all(len(vector) == 4096 for _, vector in rows)
        assert b"test-key" not in library.store.path.read_bytes()
    restarted, _ = make_service(KnowledgeStore(tmp_path), unexpected, configured=False)
    assert await restarted.ensure_embedded(set_id, profile) == saved
    vectors = restarted.read_vectors(set_id, profile)
    assert [(item.chunk_index, item.vector[0]) for item in vectors] == [(0, 0.25), (1, 1.25)]
    assert all(len(item.vector) == 1024 for item in vectors)


async def test_sidecar_embedding_service_uses_the_host_configured_adapter(tmp_path):
    def handler(request):
        assert wire_input(request) == "from published chunk\n"
        return response(1)

    provider = siliconflow.SiliconFlowEmbedding(transport=httpx.MockTransport(handler))
    ui, host = uuid4().hex, uuid4().hex
    business_path = tmp_path / "business.sqlite3"
    app = create_app(ui, host, database_path=business_path, embeddings=provider)
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1",
    ) as client:
        result = await client.put(
            "/api/v1/_host/model-credentials/siliconflow",
            headers={"Authorization": f"Bearer {host}"},
            json={"api_key": "test-key", "base_url": SILICONFLOW_BASE_URL},
        )
        assert result.status_code == 200
        service = app.state.document_embeddings
        _, set_id = chunks(app.state.library, "from published chunk")
        await service.ensure_embedded(set_id, provider.profile(compatibility_epoch="host-configured"))
        assert counts(app.state.knowledge_store) == (1, 1, 1)
        with sqlite3.connect(business_path) as db:
            assert db.execute("SELECT name FROM sqlite_schema WHERE name LIKE '%embedding%'").fetchall() == []
    assert app.state.document_embeddings is None


async def test_updates_deduplicate_same_batch_and_share_across_documents(library):
    book, set_id = chunks(library, "shared text", "old paragraph", "shared text")
    calls = []

    def handler(request):
        payload = wire_input(request)
        calls.append(payload)
        return response(1 if isinstance(payload, str) else len(payload))

    service, profile = make_service(library.store, handler)
    await service.ensure_embedded(set_id, profile)
    assert calls == [["shared text\n", "old paragraph\n"]]
    _, updated = chunks(library, "shared text", "new paragraph", document_id=book.document_id)
    await service.ensure_embedded(updated, profile)
    other_book, other_set = chunks(library, "shared text")
    await service.ensure_embedded(other_set, profile)
    assert calls[-1] == "new paragraph\n" and len(calls) == 2
    assert counts(library.store) == (3, 6, 3)
    before = service.read_vectors(other_set, profile)
    library.delete_document(book.document_id)
    assert counts(library.store) == (1, 1, 1)
    assert service.read_vectors(other_set, profile) == before
    with pytest.raises(EmbeddingError) as exc:
        service.read_vectors(updated, profile)
    assert exc.value.code == "EMBEDDING_CHUNK_SET_UNAVAILABLE"
    library.delete_document(other_book.document_id)
    assert counts(library.store) == (0, 0, 0)


async def test_title_changes_actual_input_and_profiles_separate_even_when_wire_text_matches(library):
    book, set_id = chunks(library, "identical body", title="旧章")
    _, updated = chunks(library, "identical body", title="新章", document_id=book.document_id)
    _, untitled = chunks(library, "identical body")
    calls = []

    def handler(request):
        payload = wire_input(request)
        calls.append(payload)
        return response(1 if isinstance(payload, str) else len(payload))

    service, profile = make_service(library.store, handler)
    for target in (set_id, updated, untitled):
        await service.ensure_embedded(target, profile)
    assert "旧章\n\nidentical body\n" in calls[0]
    assert "新章\n\nidentical body\n" in calls[1]
    assert calls[2] == "identical body\n"
    body_only = replace(profile, input_strategy_version="body-v1")
    assert service.get_embedding_set(untitled, body_only) is None
    await service.ensure_embedded(untitled, body_only)
    assert calls[2:] == ["identical body\n"] * 2
    digest = hashlib.sha256(b"identical body\n").hexdigest()
    for different in (body_only, replace(profile, model_id="another-model"), replace(profile, provider="other"),
                      replace(profile, service_id="other"), replace(profile, dimensions=128),
                      replace(profile, service_revision="observed"), replace(profile, compatibility_epoch="next")):
        assert different.cache_key(digest) != profile.cache_key(digest)
    other_model = replace(profile, model_id="another-model")
    assert service.read_vectors(untitled, other_model) is None
    with pytest.raises(EmbeddingError) as exc:
        await service.ensure_embedded(untitled, other_model)
    assert exc.value.code == "EMBEDDING_PROFILE_UNSUPPORTED"
    entries = service._build_inputs(service.results.missing_chunks(untitled, other_model, -1, 1), other_model)
    lease = uuid4().hex
    service.results.stage(untitled, other_model)
    assert service.results.claim(untitled, other_model, entries, lease) == entries
    service.results.commit(untitled, other_model, lease, entries, [[0.75] * EMBEDDING_DIMENSIONS])
    service.results.publish(untitled, other_model)
    assert service.read_vectors(untitled, other_model)[0].vector[0] == 0.75
    assert service.read_vectors(untitled, profile)[0].vector[0] == 0.25


async def test_revision_drift_fences_existing_sets_and_requires_new_epoch(library):
    _, old_set = chunks(library, "existing")
    _, new_set = chunks(library, "new")
    revisions = iter((None, "provider-2026-11", "provider-2026-11"))
    calls = []

    def handler(request):
        calls.append(wire_input(request))
        return response(1, revision=next(revisions))

    service, profile = make_service(library.store, handler)
    await service.ensure_embedded(old_set, profile)
    with pytest.raises(EmbeddingError) as exc:
        await service.ensure_embedded(new_set, profile)
    revision = canonical_sha256({"model_version": "provider-2026-11"})
    assert (exc.value.code, exc.value.observed_revision) == ("EMBEDDING_REVISION_CHANGED", revision)
    assert counts(library.store) == (1, 2, 1)
    with pytest.raises(EmbeddingError) as exc:
        service.read_vectors(old_set, profile)
    assert exc.value.code == "EMBEDDING_PROFILE_INVALIDATED"
    rebuilt = replace(profile, service_revision=revision, compatibility_epoch="new")
    assert service.read_vectors(old_set, rebuilt) is None
    await service.ensure_embedded(old_set, rebuilt)
    assert calls == ["existing\n", "new\n", "existing\n"]


async def test_bad_batch_does_not_publish_and_restart_sends_only_missing(library, tmp_path):
    _, set_id = chunks(library, "one", "two", "three", "four")
    calls = []

    def handler(request):
        calls.append(wire_input(request))
        return response(2 if len(calls) == 1 else 1)

    service, profile = make_service(library.store, handler, batch_size=2)
    with pytest.raises(EmbeddingError) as exc:
        await service.ensure_embedded(set_id, profile)
    assert exc.value.code == "EMBEDDING_INVALID_RESPONSE"
    assert service.get_embedding_set(set_id, profile) is None
    assert service.read_vectors(set_id, profile) is None
    assert counts(library.store) == (2, 4, 0)

    def healthy(request):
        calls.append(wire_input(request))
        return response(2)

    restarted, _ = make_service(KnowledgeStore(tmp_path), healthy, batch_size=2)
    assert (await restarted.ensure_embedded(set_id, profile)).chunk_count == 4
    assert calls == [["one\n", "two\n"], ["three\n", "four\n"], ["three\n", "four\n"]]
    assert counts(library.store) == (4, 4, 1)


@pytest.mark.parametrize(("failure", "code", "attempts"), [
    (401, "EMBEDDING_AUTH_FAILED", 1), (403, "EMBEDDING_FORBIDDEN", 1),
    (429, "EMBEDDING_RATE_LIMITED", 3), (503, "EMBEDDING_SERVICE_ERROR", 3), (504, "EMBEDDING_TIMEOUT", 3),
    ("read-timeout", "EMBEDDING_TIMEOUT", 3), ("server-length", "EMBEDDING_INPUT_TOO_LONG", 1),
    ("bad-json", "EMBEDDING_INVALID_RESPONSE", 1), ("duplicate-index", "EMBEDDING_INVALID_RESPONSE", 1),
    ("wrong-model", "EMBEDDING_INVALID_RESPONSE", 1), ("wrong-dimensions", "EMBEDDING_INVALID_RESPONSE", 1),
    ("non-finite", "EMBEDDING_INVALID_RESPONSE", 1), ("float32-overflow", "EMBEDDING_INVALID_VECTOR", 1),
])
async def test_sanitized_failures_and_bounded_retries(library, monkeypatch, caplog, failure, code, attempts):
    _, set_id = chunks(library, "first item", "second item")
    calls = 0
    monkeypatch.setattr(siliconflow, "uniform", lambda *_: 0)

    def handler(request):
        nonlocal calls
        calls += 1
        assert wire_input(request) == ["first item\n", "second item\n"]
        match failure:
            case "read-timeout":
                raise httpx.ReadTimeout("test-key", request=request)
            case "server-length":
                return httpx.Response(400, json={"code": 20015, "message": "test-key"})
            case "bad-json":
                return httpx.Response(200, text="test-key")
            case int():
                return httpx.Response(failure, text="test-key")
        payload = response(2).json()
        match failure:
            case "duplicate-index":
                payload["data"][1]["index"] = 1
            case "wrong-model":
                payload["model"] = "test-key"
            case "wrong-dimensions":
                payload["data"][0]["embedding"] = [0.25]
            case "non-finite":
                payload["data"][0]["embedding"][0] = float("nan")
            case "float32-overflow":
                payload["data"][0]["embedding"][0] = 1e40
        return httpx.Response(200, content=json.dumps(payload))

    service, profile = make_service(library.store, handler)
    with pytest.raises(EmbeddingError) as exc:
        await service.ensure_embedded(set_id, profile)
    assert exc.value.code == code and calls == attempts
    assert "test-key" not in str(exc.value) and "test-key" not in caplog.text
    assert counts(library.store) == (0, 2, 0)
    assert service.read_vectors(set_id, profile) is None


async def test_retry_after_outside_budget_survives_restart(library, tmp_path):
    _, set_id = chunks(library, "pending")
    calls = 0

    def limited(request):
        nonlocal calls
        calls += 1
        wire_input(request)
        return httpx.Response(429, headers={"Retry-After": "300"})

    service, profile = make_service(library.store, limited)
    with pytest.raises(EmbeddingError) as exc:
        await service.ensure_embedded(set_id, profile)
    assert exc.value.code == "EMBEDDING_RATE_LIMITED" and calls == 1
    restarted, _ = make_service(KnowledgeStore(tmp_path), unexpected)
    with pytest.raises(EmbeddingError) as exc:
        await restarted.ensure_embedded(set_id, profile)
    assert exc.value.code == "EMBEDDING_RATE_LIMITED" and exc.value.retry_after >= 290
    assert counts(library.store) == (0, 1, 0)
    with library.store.write() as db:
        db.execute("UPDATE embeddings SET retry_not_before = unixepoch() - 1")
    resumed, _ = make_service(KnowledgeStore(tmp_path), lambda _: response(1))
    assert (await resumed.ensure_embedded(set_id, profile)).chunk_count == 1


async def test_transient_backoff_uses_retry_after_then_commits(library, monkeypatch):
    _, set_id = chunks(library, "retry me")
    failures = iter((429, 503, 200))
    sleeps = []

    async def sleep(delay):
        sleeps.append(delay)

    def handler(request):
        assert wire_input(request) == "retry me\n"
        status = next(failures)
        headers = {"Retry-After": "2"} if status == 429 else {}
        return response(1) if status == 200 else httpx.Response(status, headers=headers)

    service, profile = make_service(library.store, handler)
    monkeypatch.setattr(siliconflow.asyncio, "sleep", sleep)
    monkeypatch.setattr(siliconflow, "uniform", lambda low, high: high)
    await service.ensure_embedded(set_id, profile)
    assert sleeps[0] == 2 and 0 < sleeps[1] <= 1
    assert counts(library.store) == (1, 1, 1)


async def test_overlong_effective_input_is_rejected_with_chunk_index(library):
    _, set_id = chunks(library, "x " * 8191 + "keep this suffix", config=ChunkerConfig(20000, 0))
    service, profile = make_service(library.store, unexpected)
    with pytest.raises(EmbeddingError, match="Chunk 0") as exc:
        await service.ensure_embedded(set_id, profile)
    assert exc.value.code == "EMBEDDING_INPUT_TOO_LONG"
    assert counts(library.store) == (0, 0, 0)
    with pytest.raises(EmbeddingError) as exc:
        siliconflow.validate_input("ﷺ" * 256)
    assert exc.value.code == "EMBEDDING_INPUT_TOO_LONG"
    with pytest.raises(EmbeddingError) as exc:
        siliconflow.validate_input("\ud800")
    assert exc.value.code == "EMBEDDING_INVALID_INPUT"


async def test_concurrent_workers_fence_shared_key_and_do_not_hold_sqlite_lock(library, tmp_path):
    _, first = chunks(library, "shared")
    _, second = chunks(library, "shared", "new for second")
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def slow(request):
        assert wire_input(request) == "shared\n"
        entered.set()
        await finish.wait()
        return response(1)

    worker, profile = make_service(library.store, slow)
    task = asyncio.create_task(worker.ensure_embedded(first, profile))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        with library.store.write(wait=False):
            pass
        contender, _ = make_service(KnowledgeStore(tmp_path), unexpected)
        for set_id in (first, second):
            with pytest.raises(EmbeddingError) as exc:
                await contender.ensure_embedded(set_id, profile)
            assert exc.value.code == "EMBEDDING_BUILD_BUSY"
            assert contender.read_vectors(set_id, profile) is None
    finally:
        finish.set()
        await task

    def missing_only(request):
        assert wire_input(request) == "new for second\n"
        return response(1)

    resumed, _ = make_service(KnowledgeStore(tmp_path), missing_only)
    await resumed.ensure_embedded(second, profile)
    assert counts(library.store) == (2, 3, 2)


@pytest.mark.parametrize("action", ["cancel", "delete", "invalidate", "lease-expired"])
async def test_inflight_batch_cannot_publish_after_interruption(library, tmp_path, action):
    book, set_id = chunks(library, "saved first", "inflight second")
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def slow(request):
        if wire_input(request) == "inflight second\n":
            entered.set()
            await finish.wait()
        return response(1)

    service, profile = make_service(library.store, slow, batch_size=1)
    task = asyncio.create_task(service.ensure_embedded(set_id, profile))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert counts(library.store) == (1, 2, 0)
        if action == "cancel":
            task.cancel()
        elif action == "delete":
            library.delete_document(book.document_id)
        elif action == "invalidate":
            service.results.invalidate_profile(profile)
        else:
            with library.store.write() as db:
                db.execute("UPDATE embeddings SET lease_expires_at = 0 WHERE lease_token IS NOT NULL")
            resumed, _ = make_service(KnowledgeStore(tmp_path), lambda _: response(1))
            await resumed.ensure_embedded(set_id, profile)
    finally:
        finish.set()
    if action == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await task
        assert service.read_vectors(set_id, profile) is None

        def healthy(request):
            assert wire_input(request) == "inflight second\n"
            return response(1)

        restarted, _ = make_service(KnowledgeStore(tmp_path), healthy)
        await restarted.ensure_embedded(set_id, profile)
        assert counts(library.store) == (2, 2, 1)
    else:
        with pytest.raises(EmbeddingError) as exc:
            await task
        if action == "delete":
            assert exc.value.code == "EMBEDDING_CHUNK_SET_UNAVAILABLE"
            assert counts(library.store) == (0, 0, 0)
        elif action == "invalidate":
            assert exc.value.code == "EMBEDDING_PROFILE_INVALIDATED"
            assert counts(library.store) == (1, 2, 0)
        else:
            assert exc.value.code == "EMBEDDING_WRITE_INTERRUPTED"
            assert counts(library.store) == (2, 2, 1)


async def test_hard_process_exit_recovers_after_lease_expiry(library, tmp_path):
    _, set_id = chunks(library, "before exit", "after exit")
    worker = """
import asyncio, json, os, sys
from pathlib import Path
import httpx
from pydantic import SecretStr
from app.models.siliconflow import EMBEDDING_MODEL, SILICONFLOW_BASE_URL, SiliconFlowEmbedding
from app.services.document_embeddings import DocumentEmbeddingService
from app.storage.library import KnowledgeStore

def respond(request):
    if json.loads(request.content)['input'] == 'after exit\\n':
        os._exit(19)
    return httpx.Response(200, json={'object': 'list', 'model': EMBEDDING_MODEL,
        'data': [{'index': 0, 'embedding': [0.25] * 1024}]})

provider = SiliconFlowEmbedding(transport=httpx.MockTransport(respond))
provider.configure(SecretStr('test-key'), SILICONFLOW_BASE_URL)
service = DocumentEmbeddingService(KnowledgeStore(Path(sys.argv[1])), provider, batch_size=1)
asyncio.run(service.ensure_embedded(int(sys.argv[2]), provider.profile(compatibility_epoch='test-epoch')))
"""
    process = subprocess.run([sys.executable, "-c", worker, str(tmp_path), str(set_id)],
                             cwd=Path(__file__).resolve().parents[1], timeout=10)
    assert process.returncode == 19
    assert counts(library.store) == (1, 2, 0)
    restart, profile = make_service(KnowledgeStore(tmp_path), unexpected)
    with pytest.raises(EmbeddingError) as exc:
        await restart.ensure_embedded(set_id, profile)
    assert exc.value.code == "EMBEDDING_BUILD_BUSY"
    with library.store.write() as db:
        db.execute("UPDATE embeddings SET lease_expires_at = 0 WHERE lease_token IS NOT NULL")

    def healthy(request):
        assert wire_input(request) == "after exit\n"
        return response(1)

    restart, _ = make_service(KnowledgeStore(tmp_path), healthy)
    await restart.ensure_embedded(set_id, profile)
    assert counts(library.store) == (2, 2, 1)


async def test_v5_migration_rolls_back_all_ddl_with_existing_v4_chunks(tmp_path, monkeypatch):
    migrations = storage_library.MIGRATIONS
    with monkeypatch.context() as patch:
        patch.setattr(storage_library, "MIGRATIONS", migrations[:4])
        patch.setattr(storage_library, "SCHEMA_VERSION", 4)
        old = KnowledgeStore(tmp_path)
        library = LibraryService(old, uuid4())
        try:
            _, set_id = chunks(library, "old published chunk")
        finally:
            library.close()
    snapshot = tuple(DocumentChunkingService(old).results.iter_chunks(set_id))
    with monkeypatch.context() as patch:
        patch.setattr(storage_library, "MIGRATIONS", (*migrations[:4], (*migrations[4], "INVALID SQL")))
        with pytest.raises(sqlite3.OperationalError, match="syntax error"):
            KnowledgeStore(tmp_path)
    with old.connect() as db:
        assert db.execute("PRAGMA user_version").fetchone() == (4,)
        assert db.execute("SELECT name FROM sqlite_schema WHERE name LIKE '%embedding%'").fetchall() == []
    upgraded = KnowledgeStore(tmp_path)
    assert tuple(DocumentChunkingService(upgraded).results.iter_chunks(set_id)) == snapshot
    with upgraded.connect() as db:
        assert db.execute("PRAGMA user_version").fetchone() == (storage_library.SCHEMA_VERSION,)
    service, profile = make_service(upgraded, lambda _: response(1))
    await service.ensure_embedded(set_id, profile)
    assert counts(upgraded) == (1, 1, 1)
