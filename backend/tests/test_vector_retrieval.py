import builtins
import math
from dataclasses import replace
from uuid import uuid4

import httpx
import numpy as np
import pytest
from pydantic import SecretStr

from app.main import create_app
from app.models import siliconflow
from app.models.embeddings import EmbeddingError
from app.models.siliconflow import SILICONFLOW_BASE_URL, SiliconFlowEmbedding
from app.services.document_embeddings import DocumentEmbeddingService
from app.services.vector_retrieval import VectorRetrievalError, VectorRetrievalService
from app.storage.chunks import ChunkStore
from app.storage.embeddings import EmbeddingStore
from app.storage.library import KnowledgeStore
from test_embedding_storage import chunks, response, unexpected, wire_input
from test_library_management import anyio_backend, library

pytestmark = pytest.mark.anyio
PROFILE = SiliconFlowEmbedding.profile(compatibility_epoch="vector-test")


def axis(index=0, value=1.0):
    vector = [0.0] * PROFILE.dimensions
    vector[index] = value
    return vector


def save_vectors(store, set_id, vectors, profile=PROFILE, *, publish=True):
    embeddings = EmbeddingStore(store)
    embeddings.stage(set_id, profile)
    inputs = DocumentEmbeddingService._build_inputs(
        embeddings.missing_chunks(set_id, profile, -1, len(vectors)), profile,
    )
    lease = uuid4().hex
    assert embeddings.claim(set_id, profile, inputs, lease) == inputs
    embeddings.commit(set_id, profile, lease, inputs, vectors)
    if publish:
        embeddings.publish(set_id, profile)


def forbid_requests(store):
    return VectorRetrievalService(store, SiliconFlowEmbedding(transport=httpx.MockTransport(unexpected)))


def test_exact_cosine_top_k_across_batches_and_source_round_trip(library):
    _, set_id = chunks(library, *(f"sample {i}" for i in range(550)), title="线性代数")
    matrix = np.random.default_rng(742).normal(size=(551, 1024)).astype("<f4")
    matrix *= np.arange(1, 552, dtype="<f4")[:, None]
    query = matrix[300].copy()
    matrix[10] = matrix[300]
    save_vectors(library.store, set_id, matrix.tolist())
    query_values = query.tolist()
    expected = [
        (math.fsum(a * b for a, b in zip(vector, query_values, strict=True)) /
         (math.hypot(*vector) * math.hypot(*query_values)), index)
        for index, vector in enumerate(matrix.tolist())
    ]
    expected.sort(key=lambda item: (-item[0], item[1]))

    hits = forbid_requests(library.store).search_vector(query, top_k=20, profile=PROFILE)
    assert [hit.chunk_index for hit in hits] == [index for _, index in expected[:20]]
    assert hits[0].chunk_index == 10 and hits[1].chunk_index == 300
    assert [hit.similarity for hit in hits] == pytest.approx([score for score, _ in expected[:20]], abs=2e-14)
    hit = hits[0]
    assert hit.body_text == "sample 9\n"
    assert hit.title_context == ("线性代数",)
    original = ChunkStore(library.store).get_chunk(hit.chunk_id)
    assert (hit.document_id, hit.document_version_id, hit.parsed_document_id, hit.content_sha256) == (
        original.source.document_id, original.document_version_id, original.parsed_document_id,
        original.source.content_sha256,
    )
    assert ChunkStore(library.store).get_chunk_sources(hit.chunk_id) == original.source_spans


async def test_filter_before_top_k_and_read_saved_vectors_without_provider(library, tmp_path):
    strong, strong_set = chunks(library, "first", "second")
    weak, weak_set = chunks(library, "third")
    save_vectors(library.store, strong_set, [axis(), axis(value=8.0)])
    save_vectors(library.store, weak_set, [axis(value=-1.0)])
    service = forbid_requests(KnowledgeStore(tmp_path))
    top = await service.search(axis(), top_k=1, profile=PROFILE)
    assert [(hit.document_id, hit.chunk_index, hit.similarity) for hit in top] == [(strong.document_id, 0, 1.0)]
    scoped = service.search_vector(axis(), 1, profile=PROFILE, document_ids=[weak.document_id])
    assert [(hit.document_id, hit.similarity) for hit in scoped] == [(weak.document_id, -1.0)]
    assert service.search_vector(axis(), profile=PROFILE, document_ids=[weak.document_id],
                                 chunk_set_ids=[strong_set]) == ()
    assert service.search_vector(axis(), profile=PROFILE, chunk_set_ids=[weak_set, weak_set]) == scoped
    assert service.search_vector(axis(), profile=PROFILE, document_ids=[]) == ()


def test_version_visibility_incomplete_publication_and_deletion(library):
    book, old_set = chunks(library, "old")
    save_vectors(library.store, old_set, [axis()])
    service = forbid_requests(library.store)
    assert service.search_vector(axis(), profile=PROFILE)[0].chunk_set_id == old_set

    newer, new_set = chunks(library, "new", "another new", document_id=book.document_id)
    assert service.search_vector(axis(), profile=PROFILE) == ()
    save_vectors(library.store, new_set, [axis(1), axis(2)], publish=False)
    assert service.search_vector(axis(), profile=PROFILE, chunk_set_ids=[new_set]) == ()
    pinned = service.search_vector(axis(), profile=PROFILE, chunk_set_ids=[old_set], document_ids=[book.document_id])
    assert pinned[0].document_version_id == book.document_version_id

    EmbeddingStore(library.store).publish(new_set, PROFILE)
    assert {hit.document_version_id for hit in service.search_vector(axis(), profile=PROFILE)} == {
        newer.document_version_id,
    }
    with pytest.raises(ValueError, match="one published Chunk Set"):
        service.search_vector(axis(), profile=PROFILE, chunk_set_ids=[old_set, new_set])
    with library.store.write() as db:
        db.execute("DELETE FROM chunk_embeddings WHERE chunk_set_id = ? AND chunk_index = 1", (new_set,))
    assert service.search_vector(axis(), profile=PROFILE) == ()
    assert service.search_vector(axis(), profile=PROFILE, chunk_set_ids=[new_set]) == ()

    library.delete_document(book.document_id)
    assert service.search_vector(axis(), profile=PROFILE, chunk_set_ids=[old_set, new_set]) == ()


def test_profiles_isolate_model_dimension_epoch_revision_and_input_strategy(library):
    _, set_id = chunks(library, "model spaces")
    save_vectors(library.store, set_id, [axis()])
    service = forbid_requests(library.store)
    for profile in (
        replace(PROFILE, model_id="another-model"),
        replace(PROFILE, compatibility_epoch="new-epoch"),
        replace(PROFILE, service_revision="new-revision"),
        replace(PROFILE, input_strategy_version="body-v1"),
    ):
        assert service.search_vector(axis(), profile=profile) == ()
        save_vectors(library.store, set_id, [axis(value=-1.0)], profile)
        assert service.search_vector(axis(), profile=profile)[0].similarity == -1.0
    smaller = replace(PROFILE, dimensions=2)
    save_vectors(library.store, set_id, [[-4.0, 0.0]], smaller)
    assert service.search_vector((1, 0), profile=smaller)[0].similarity == -1.0
    with pytest.raises(EmbeddingError, match="维度"):
        service.search_vector(axis(), profile=smaller)
    assert service.search_vector(axis(), profile=PROFILE)[0].similarity == 1.0


@pytest.mark.parametrize("top_k", [0, -1, 51, True, 2.0])
def test_invalid_result_limit(library, top_k):
    with pytest.raises(ValueError, match="result limit"):
        forbid_requests(library.store).search_vector(axis(), top_k, profile=PROFILE)


@pytest.mark.parametrize("vector", [
    [1.0], np.ones((1, 1024)), axis(value=math.nan), axis(value=math.inf),
    axis(value=0.0), axis(value=True), axis(value=1e100), axis(value=1e-100),
])
def test_invalid_query_vector(library, vector):
    with pytest.raises(EmbeddingError) as error:
        forbid_requests(library.store).search_vector(vector, profile=PROFILE)
    assert error.value.code == "EMBEDDING_INVALID_VECTOR"


def test_float32_extremes_and_corrupt_stored_vector(library):
    _, set_id = chunks(library, "tiny", "large")
    save_vectors(library.store, set_id, [axis(value=1e-44), axis(value=3e38)])
    service = forbid_requests(library.store)
    assert [hit.similarity for hit in service.search_vector(axis(value=1e-44), profile=PROFILE)] == [1.0, 1.0]
    with library.store.write() as db:
        db.execute("UPDATE embeddings SET vector = zeroblob(4096)")
    with pytest.raises(EmbeddingError) as error:
        service.search_vector(axis(), profile=PROFILE)
    assert error.value.code == "EMBEDDING_INVALID_VECTOR"


@pytest.mark.parametrize(("query", "code"), [
    ("  \t ", "EMBEDDING_EMPTY_INPUT"),
    ("中" * 2731, "EMBEDDING_INPUT_TOO_LONG"),
    ("\ud800", "EMBEDDING_INVALID_INPUT"),
])
async def test_invalid_text_does_not_call_api(library, query, code):
    with pytest.raises(EmbeddingError) as error:
        await forbid_requests(library.store).search(query, profile=PROFILE)
    assert error.value.code == code


async def test_sidecar_host_credentials_retry_network_failure_and_fts(tmp_path, monkeypatch):
    events = []

    def handler(request):
        assert wire_input(request) == "概率定义"
        events.append("query")
        if len(events) == 1:
            return httpx.Response(503)
        if len(events) == 2:
            return response(1)
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(siliconflow, "uniform", lambda *_: 0.0)
    provider = SiliconFlowEmbedding(transport=httpx.MockTransport(handler))
    ui, host = uuid4().hex, uuid4().hex
    app = create_app(ui, host, database_path=tmp_path / "business.sqlite3", embeddings=provider)
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1",
    ) as client:
        _, set_id = chunks(app.state.library, "概率定义解释")
        save_vectors(app.state.knowledge_store, set_id, [axis()])
        fts = app.state.keyword_search
        fts.index.ensure_indexed(set_id)
        vector = app.state.vector_retrieval
        with pytest.raises(EmbeddingError) as error:
            await vector.search("概率定义", profile=PROFILE)
        assert error.value.code == "EMBEDDING_NOT_CONFIGURED"
        assert len(fts.search("概率")) == 1

        result = await client.put(
            "/api/v1/_host/model-credentials/siliconflow",
            headers={"Authorization": f"Bearer {host}"},
            json={"api_key": "test-key", "base_url": SILICONFLOW_BASE_URL},
        )
        assert result.status_code == 200
        hits = await vector.search("概率定义", profile=PROFILE)
        assert hits[0].similarity == pytest.approx(1 / 32)
        assert len(events) == 2
        with pytest.raises(EmbeddingError) as error:
            await vector.search("概率定义", profile=PROFILE)
        assert error.value.code == "EMBEDDING_NETWORK_ERROR"
        assert len(events) == 3
        assert (await vector.search(axis(), profile=PROFILE))[0].similarity == 1.0
        assert len(events) == 3 and len(fts.search("概率")) == 1
    assert app.state.vector_retrieval is None


async def test_model_revision_change_invalidates_direct_and_text_queries(library):
    _, set_id = chunks(library, "published in old model space")
    save_vectors(library.store, set_id, [axis()])
    provider = SiliconFlowEmbedding(transport=httpx.MockTransport(lambda _: response(1, revision="new")))
    provider.configure(SecretStr("test-key"), SILICONFLOW_BASE_URL)
    service = VectorRetrievalService(library.store, provider)
    with pytest.raises(EmbeddingError) as error:
        await service.search("test query", profile=PROFILE)
    assert error.value.code == "EMBEDDING_REVISION_CHANGED"
    with pytest.raises(EmbeddingError) as error:
        service.search_vector(axis(), profile=PROFILE)
    assert error.value.code == "EMBEDDING_PROFILE_INVALIDATED"


async def test_unavailable_native_engine_preserves_sidecar_and_fts(tmp_path, monkeypatch):
    real_import = builtins.__import__

    def import_without_native_engine(name, *args, **kwargs):
        if name in ("numpy", "sqlite_vec"):
            raise ImportError("native binary unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_native_engine)
    app = create_app(uuid4().hex, uuid4().hex, database_path=tmp_path / "business.sqlite3")
    async with app.router.lifespan_context(app):
        _, set_id = chunks(app.state.library, "关键词检索")
        app.state.keyword_search.index.ensure_indexed(set_id)
        with pytest.raises(VectorRetrievalError) as error:
            app.state.vector_retrieval.search_vector(axis(), profile=PROFILE)
        assert error.value.code == "VECTOR_ENGINE_UNAVAILABLE"
        assert len(app.state.keyword_search.search("检索")) == 1
