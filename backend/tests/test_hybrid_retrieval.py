from dataclasses import replace

import httpx
import pytest
from pydantic import SecretStr

from app.main import create_app
from app.models import siliconflow
from app.models.embeddings import EmbeddingError
from app.models.siliconflow import SILICONFLOW_BASE_URL, SiliconFlowEmbedding
from app.services.document_chunking import ChunkerConfig, DocumentChunkingService
from app.services.document_parsing import DocumentParsingService
from app.services.hybrid_retrieval import HybridRetrievalConfig, HybridRetrievalService
from app.services.keyword_search import KeywordSearchService
from app.services.vector_retrieval import VectorRetrievalService
from app.storage.chunks import ChunkStore
from app.storage.embeddings import EmbeddingStore
from test_embedding_storage import response, unexpected
from test_keyword_search import publish_chunks
from test_library_management import anyio_backend, import_file, library
from test_vector_retrieval import PROFILE, axis, save_vectors

pytestmark = pytest.mark.anyio


def services(store, provider=None, **config):
    if provider is None:
        provider = SiliconFlowEmbedding(transport=httpx.MockTransport(unexpected))
    return HybridRetrievalService(KeywordSearchService(store), VectorRetrievalService(store, provider),
                                  config=HybridRetrievalConfig(**config))


def index(service, ready):
    service.keyword.index.ensure_indexed(ready.chunk_set_id)


async def offline(service, query, **options):
    return await service.search(query, profile=PROFILE, query_vector=axis(), **options)


async def test_rrf_ranks_and_scores_and_source_round_trip(library):
    book, ready = publish_chunks(library, "精确术语 MLE " * 12, "精确术语 MLE", "语义相关解释",
                                 title="统计推断")
    service = services(library.store, rrf_k=10)
    index(service, ready)
    chunks = list(ChunkStore(library.store).iter_chunks(ready.chunk_set_id))
    exact = next(chunk for chunk in chunks if chunk.text == "精确术语 MLE\n")
    semantic = next(chunk for chunk in chunks if "语义相关解释" in chunk.text)
    save_vectors(library.store, ready.chunk_set_id, [
        axis() if chunk in (exact, semantic) else axis(1) for chunk in chunks
    ])
    query = "精确术语 MLE"
    keyword = service.keyword.search(query)
    vector = service.vector.search_vector(axis(), profile=PROFILE)
    result = await offline(service, query, document_ids=[book.document_id])
    assert result.channel_status == "hybrid"
    assert len(result.hits) == 3
    assert [item.chunk_id for item in result.hits] == [exact.chunk_id, keyword[0].chunk_id, semantic.chunk_id]
    hit = result.hits[0]
    assert [(entry.channel, entry.rank, entry.score) for entry in hit.channel_ranks] == [
        ("keyword", 2, keyword[1].score), ("vector", 1, vector[0].similarity),
    ]
    assert hit.fusion_score == pytest.approx(1/12 + 1/11)
    assert hit.keyword_matched_in == ("body",)
    original = ChunkStore(library.store).get_chunk(hit.chunk_id)
    assert hit.source_spans == original.source_spans
    assert (hit.body_text, hit.title_context, hit.content_sha256) == (
        original.text, ("统计推断",), original.source.content_sha256,
    )
    assert hit.source_spans[0].source_locations[0].start_line > 1

    title = await service.search("统计推断", retrieval_mode="keyword")
    title_only = next(item for item in title.hits if item.keyword_matched_in == ("title",))
    assert "统计推断" not in title_only.body_text
    assert title_only.title_context == ("统计推断",)
    again = await offline(service, query, document_ids=[book.document_id])
    assert (again.hits, again.documents) == (result.hits, result.documents)


async def test_per_document_recall_quota_threshold_and_top_k(library):
    strong, large = publish_chunks(library, *[f"共同术语 " * 10 + str(i) for i in range(12)])
    weak, small = publish_chunks(library, "共同术语 简短定义", "共同术语 例解", filename="小.md")
    unrelated, negative = publish_chunks(library, "距地日冕观测", filename="天文.md")
    service = services(library.store, candidate_k=3, per_document_quota=2)
    for ready, vectors in ((large, [axis()] * 12), (small, [[0.7, 0.7, *axis()[2:]]] * 2),
                           (negative, [[0.2, 0.98, *axis()[2:]]])):
        index(service, ready)
        save_vectors(library.store, ready.chunk_set_id, vectors)
    global_result = await offline(service, "共同术语", top_k=3)
    assert {hit.document_id for hit in global_result.hits} == {strong.document_id}
    compared = await offline(service, "共同术语", top_k=4, coverage_mode="per_document")
    docs = {doc.document_id: doc for doc in compared.documents}
    assert all(docs[book.document_id].returned_count == 2 for book in (strong, weak))
    assert docs[unrelated.document_id].evidence_status == "no_evidence"
    assert docs[unrelated.document_id].searched_channels == ("keyword", "vector")
    assert not [hit for hit in compared.hits if hit.document_id == unrelated.document_id]
    short = await offline(service, "共同术语", top_k=1, coverage_mode="per_document")
    assert sum(doc.evidence_status == "not_returned" for doc in short.documents) == 1
    for mode in ("keyword", "vector"):
        filtered = await service.search(
            "共同术语", retrieval_mode=mode, document_ids=[weak.document_id],
            profile=PROFILE, query_vector=axis() if mode == "vector" else None,
        )
        assert len(filtered.hits) == 2 and {hit.document_id for hit in filtered.hits} == {weak.document_id}


async def test_default_version_pins_incomplete_embeddings_and_deletion(library):
    book, old = publish_chunks(library, "共同术语 旧版")
    service = services(library.store)
    index(service, old)
    save_vectors(library.store, old.chunk_set_id, [axis()])
    updated, current = publish_chunks(library, "共同术语 新版", document_id=book.document_id)
    unavailable = await offline(service, "共同术语", coverage_mode="per_document")
    assert unavailable.hits == ()
    assert unavailable.documents[0].document_version_id == updated.document_version_id
    assert unavailable.documents[0].evidence_status == "unavailable"

    index(service, current)
    save_vectors(library.store, current.chunk_set_id, [axis()], publish=False)
    only = await service.search("共同术语", profile=PROFILE)
    assert only.channel_status == "keyword_only" and only.timings.query_embedding_ms == 0
    assert not only.documents[0].vector_indexed
    assert {hit.chunk_set_id for hit in only.hits} == {current.chunk_set_id}
    historical = await offline(service, "共同术语", chunk_set_ids=[old.chunk_set_id, old.chunk_set_id])
    assert historical.channel_status == "hybrid" and historical.hits[0].chunk_set_id == old.chunk_set_id
    with pytest.raises(ValueError, match="one pinned"):
        await offline(service, "共同术语", chunk_set_ids=[old.chunk_set_id, current.chunk_set_id])

    EmbeddingStore(library.store).publish(current.chunk_set_id, PROFILE)
    both = await offline(service, "共同术语")
    assert both.channel_status == "hybrid" and both.hits[0].chunk_set_id == current.chunk_set_id
    other, other_set = publish_chunks(library, "共同术语 补充", filename="另一本.md")
    index(service, other_set)
    partial = await offline(service, "共同术语")
    assert partial.channel_status == "partial"
    assert {doc.document_id: doc.searched_channels for doc in partial.documents}[other.document_id] == ("keyword",)
    assert (await offline(service, "共同术语", document_ids=[other.document_id],
                          chunk_set_ids=[old.chunk_set_id])).hits == ()

    with library.store.write() as db:
        db.execute("DELETE FROM chunk_embeddings WHERE chunk_set_id = ?", (current.chunk_set_id,))
    damaged = await offline(service, "共同术语", document_ids=[book.document_id])
    assert damaged.channel_status == "keyword_only" and not damaged.documents[0].vector_indexed
    library.delete_document(book.document_id)
    deleted = await offline(service, "共同术语", document_ids=[book.document_id], coverage_mode="per_document")
    assert deleted.hits == () and deleted.documents[0].evidence_status == "unavailable"
    assert (await offline(service, "共同术语", chunk_set_ids=[old.chunk_set_id])).hits == ()


async def test_empty_published_indexes_are_unavailable_without_embedding(library):
    book, empty = publish_chunks(library, filename="空.md")
    service = services(library.store)
    index(service, empty)
    embeddings = EmbeddingStore(library.store)
    embeddings.stage(empty.chunk_set_id, PROFILE)
    embeddings.publish(empty.chunk_set_id, PROFILE)

    for mode in ("keyword", "vector", "hybrid"):
        for scope in ({"document_ids": [book.document_id]}, {"chunk_set_ids": [empty.chunk_set_id]}):
            result = await service.search("查找正文", retrieval_mode=mode, profile=PROFILE, **scope)
            assert result.channel_status == "unavailable"
            assert result.hits == () and result.diagnostic is None
            assert result.timings.query_embedding_ms == 0
            document, = result.documents
            assert document.document_version_id == book.document_version_id
            assert document.searched_channels == () and document.evidence_status == "unavailable"
            assert not document.keyword_indexed
            assert document.vector_indexed is (None if mode == "keyword" else False)


async def test_same_version_configurations_share_one_set_and_allow_vector_only(library):
    book, keyword_set = publish_chunks(library, "共同术语 Markdown")
    service = services(library.store)
    index(service, keyword_set)
    semantic_set = DocumentChunkingService(library.store).ensure_chunked(
        keyword_set.parsed_document_id, ChunkerConfig(chunk_size=200),
    )
    save_vectors(library.store, semantic_set.chunk_set_id, [axis()])
    default = await offline(service, "共同术语")
    assert default.channel_status == "keyword_only"
    assert default.hits[0].chunk_set_id == keyword_set.chunk_set_id
    pinned = await offline(service, "共同术语", chunk_set_ids=[semantic_set.chunk_set_id])
    assert pinned.channel_status == "vector_only"
    assert pinned.hits[0].chunk_set_id == semantic_set.chunk_set_id
    vector = await offline(service, "改写", retrieval_mode="vector", document_ids=[book.document_id])
    assert vector.hits[0].chunk_set_id == semantic_set.chunk_set_id
    wrong_profile = replace(PROFILE, compatibility_epoch="different")
    mismatch = await service.search("共同术语", profile=wrong_profile, query_vector=axis())
    assert mismatch.channel_status == "keyword_only"


async def test_overlap_uses_adjacent_source_intervals(library):
    original = ("重叠" * 85 + "。" + "另一段描述。" * 8 + "\n").encode()
    book = import_file(library, original, "窗口.md")
    parsed = DocumentParsingService(library.store).ensure_parsed(book.document_version_id)
    ready = DocumentChunkingService(library.store).ensure_chunked(
        parsed.parsed_document_id, ChunkerConfig(chunk_size=100, chunk_overlap=70,
                                                 long_paragraph_strategy="character"),
    )
    service = services(library.store)
    index(service, ready)
    unfiltered = service.keyword.search("重叠")
    assert len(unfiltered) >= 3
    result = await service.search("重叠", retrieval_mode="keyword")
    chunks = ChunkStore(library.store)
    kept = {hit.chunk_index for hit in result.hits}
    assert 0 < len(kept) < len(unfiltered)
    def shared_fraction(a, b):
        starts = (a.block_start, b.block_start)
        ends = (a.block_end, b.block_end)
        return max(0, min(ends) - max(starts)) / min(a.block_end - a.block_start, b.block_end - b.block_start)

    adjacent = [(a, b) for a, b in zip(sorted(kept), sorted(kept)[1:]) if b == a + 1]
    by_index = {hit.chunk_index: hit.source_spans for hit in result.hits}
    assert all(shared_fraction(by_index[a][0], by_index[b][0]) < 0.65 for a, b in adjacent)
    assert all(hit.source_spans == chunks.get_chunk_sources(hit.chunk_id) for hit in result.hits)


@pytest.mark.parametrize(("fault", "code", "attempts"), [
    (401, "EMBEDDING_AUTH_FAILED", 1),
    (429, "EMBEDDING_RATE_LIMITED", 1),
    (503, "EMBEDDING_SERVICE_ERROR", 3),
    ("timeout", "EMBEDDING_TIMEOUT", 3),
    ("network", "EMBEDDING_NETWORK_ERROR", 1),
    ("malformed", "EMBEDDING_INVALID_RESPONSE", 1),
    ("revision", "EMBEDDING_REVISION_CHANGED", 1),
])
async def test_embedding_failures_fall_back_with_diagnostics(library, monkeypatch, fault, code, attempts):
    _, ready = publish_chunks(library, "概率 probability")
    requests = []

    def handler(request):
        requests.append(request.url.path)
        if fault == 429:
            return httpx.Response(429, headers={"Retry-After": "90"})
        if fault == "timeout":
            raise httpx.ReadTimeout("network timeout", request=request)
        if fault == "network":
            raise httpx.ConnectError("connection failed", request=request)
        if fault == "malformed":
            return httpx.Response(200, json={"not_embedding_data": True})
        if fault == "revision":
            return response(1, revision="new")
        return httpx.Response(fault)

    monkeypatch.setattr(siliconflow, "uniform", lambda *_: 0)
    provider = SiliconFlowEmbedding(transport=httpx.MockTransport(handler))
    provider.configure(SecretStr("test-credential"), SILICONFLOW_BASE_URL)
    service = services(library.store, provider)
    index(service, ready)
    save_vectors(library.store, ready.chunk_set_id, [axis()])

    degraded = await service.search("概率 probability", profile=PROFILE)
    assert len(requests) == attempts
    assert degraded.channel_status == "keyword_only" and degraded.diagnostic.code == code
    assert degraded.timings.query_embedding_ms > 0 and degraded.query_prompt_tokens is None
    assert degraded.diagnostic.http_status == (fault if isinstance(fault, int) else None)
    if fault == 429:
        assert degraded.diagnostic.retry_after >= 90
    if fault == "revision":
        assert degraded.diagnostic.observed_revision and not degraded.documents[0].vector_indexed
    keyword = await service.search("概率 probability", retrieval_mode="keyword")
    assert keyword.hits == degraded.hits and len(requests) == attempts
    with pytest.raises(EmbeddingError) as error:
        await service.search("概率 probability", retrieval_mode="vector", profile=PROFILE)
    assert error.value.code == ("EMBEDDING_PROFILE_INVALIDATED" if fault == "revision" else code)


async def test_one_query_embedding_and_fresh_snapshot_after_network(library):
    first, first_set = publish_chunks(library, "精确词 共同文本")
    obsolete, obsolete_set = publish_chunks(library, "精确词 另一教材", filename="即将删除.md")
    remaining, remaining_set = publish_chunks(library, "精确词 第三教材", filename="保留.md")
    calls = []

    def handler(request):
        calls.append(request.url.path)
        library.delete_document(obsolete.document_id)
        return response(1)

    provider = SiliconFlowEmbedding(transport=httpx.MockTransport(handler))
    provider.configure(SecretStr("test-credential"), SILICONFLOW_BASE_URL)
    service = services(library.store, provider)
    for ready in (first_set, obsolete_set, remaining_set):
        index(service, ready)
        save_vectors(library.store, ready.chunk_set_id, [axis()])
    result = await service.search("精确词", profile=PROFILE, coverage_mode="per_document",
                                  document_ids=[first.document_id, obsolete.document_id, remaining.document_id])
    assert len(calls) == 1 and result.channel_status == "hybrid"
    assert result.query_prompt_tokens is not None and result.timings.query_embedding_ms > 0
    assert {hit.document_id for hit in result.hits} == {first.document_id, remaining.document_id}
    assert next(doc for doc in result.documents if doc.document_id == obsolete.document_id).evidence_status == "unavailable"


async def test_missing_credentials_and_corrupt_vector_keep_keyword_available(library):
    _, ready = publish_chunks(library, "概率 probability")
    service = services(library.store)
    index(service, ready)
    save_vectors(library.store, ready.chunk_set_id, [axis()])
    degraded = await service.search("概率 probability", profile=PROFILE)
    assert degraded.channel_status == "keyword_only"
    assert degraded.diagnostic.code == "EMBEDDING_NOT_CONFIGURED"

    with library.store.write() as db:
        db.execute("UPDATE embeddings SET vector = zeroblob(length(vector))")
    corrupt = await offline(service, "概率 probability")
    assert corrupt.channel_status == "keyword_only" and corrupt.hits == degraded.hits
    assert corrupt.diagnostic.code == "EMBEDDING_INVALID_VECTOR"


async def test_channels_and_sources_share_a_snapshot(library, monkeypatch):
    book, ready = publish_chunks(library, "共同术语")
    service = services(library.store)
    index(service, ready)
    save_vectors(library.store, ready.chunk_set_id, [axis()])
    search_indexed = service.keyword.search_indexed

    def read_then_delete(*args):
        hits = search_indexed(*args)
        if hits:
            library.delete_document(book.document_id)
        return hits

    monkeypatch.setattr(service.keyword, "search_indexed", read_then_delete)
    first = await offline(service, "共同术语")
    assert first.hits[0].source_spans
    assert [rank.channel for rank in first.hits[0].channel_ranks] == ["keyword", "vector"]
    assert (await offline(service, "共同术语")).hits == ()


async def test_empty_invalid_and_comparison_limit(library):
    service = services(library.store)
    for index_number in range(9):
        publish_chunks(library, f"暂无索引 {index_number}", filename=f"等候-{index_number}.md")
    assert (await service.search("无匹配", retrieval_mode="keyword", document_ids=[])).hits == ()
    assert (await service.search("???", retrieval_mode="keyword")).hits == ()
    for options in (
        {"retrieval_mode": "invalid"}, {"coverage_mode": "invalid"}, {"top_k": True}, {"top_k": 51},
        {"document_ids": "invalid"}, {"chunk_set_ids": [0]}, {"chunk_set_ids": [2**63]},
        {"coverage_mode": "per_document"}, {"coverage_mode": "per_document", "document_ids": list("abcdefghi")},
        {"query_vector": axis()},
    ):
        with pytest.raises(ValueError):
            await service.search("无匹配", **({"retrieval_mode": "keyword"} | options))
    for query in ("", " ", "\x00", "\ud800", "a" * 257, None):
        with pytest.raises(ValueError):
            await service.search(query, retrieval_mode="keyword")
    with pytest.raises(EmbeddingError, match="查询向量"):
        await service.search("无匹配", profile=PROFILE, query_vector=[0.0])
    for options in ({"candidate_k": 201}, {"rrf_k": 0}, {"min_vector_similarity": float("nan")},
                    {"overlap_threshold": 2}, {"per_document_quota": 0}):
        with pytest.raises(ValueError):
            HybridRetrievalConfig(**options)


async def test_sidecar_wires_shared_services_and_keyword_is_network_free(tmp_path):
    provider = SiliconFlowEmbedding(transport=httpx.MockTransport(unexpected))
    provider.configure(SecretStr("test-credential"), SILICONFLOW_BASE_URL)
    app = create_app("d" * 32, "h" * 32, database_path=tmp_path / "business.db", embeddings=provider)
    assert app.state.hybrid_retrieval is None
    async with app.router.lifespan_context(app):
        service = app.state.hybrid_retrieval
        assert service.keyword is app.state.keyword_search and service.vector is app.state.vector_retrieval
        _, ready = publish_chunks(app.state.library, "概率 probability")
        index(service, ready)
        result = await service.search("概率 probability", retrieval_mode="keyword")
        assert result.hits[0].keyword_matched_in == ("body",)
        assert result.timings.query_embedding_ms == 0
    assert app.state.hybrid_retrieval is None
