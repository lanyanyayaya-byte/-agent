"""Comprehensive P10.1 & P10.2 (RAG 2.0) unit and integration tests."""

import json
import os
import sqlite3
import time
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.main import create_app


@pytest.fixture
def anyio_backend():
    return "asyncio"
from app.models.embeddings import EmbeddingProfile
from app.models.reranker import SiliconFlowReranker
from app.models.siliconflow import SILICONFLOW_BASE_URL, SiliconFlowEmbedding
from app.services.document_chunking import DocumentChunkingService
from app.services.document_embeddings import DocumentEmbeddingService
from app.services.document_parsing import DocumentParsingService
from app.services.hybrid_retrieval import HybridRetrievalService
from app.services.index_worker import INDEX_EMBEDDING_PROFILE
from app.services.keyword_search import KeywordSearchService
from app.services.knowledge_qa import KnowledgeQAService, resolve_strategy
from app.services.library import LibraryService
from app.services.vector_retrieval import VectorRetrievalService
from app.storage.keyword_terms import build_bm25_or_query, extract_bm25_terms, is_explicit_exact_query
from app.storage.keywords import KeywordIndex
from app.storage.knowledge_settings import KnowledgeSettingsStore
from app.storage.library import KnowledgeStore

UI_TOKEN = "u" * 32
HOST_TOKEN = "h" * 32
TEST_PROFILE = INDEX_EMBEDDING_PROFILE


def _deterministic_vec(text: str, dims: int = 1024) -> list[float]:
    vec = [0.01] * dims
    low = text.casefold()
    # Semantic clusters for mesh vs probability vs linear algebra
    if any(k in low for k in ("delaunay", "网格", "mesh", "advancing front", "前沿推进", "looping")):
        vec[0] = 1.0
        vec[1] = 0.8
    if any(k in low for k in ("mle", "最大似然", "likelihood", "概率", "bayes", "贝叶斯")):
        vec[2] = 1.0
        vec[3] = 0.8
    if any(k in low for k in ("矩阵", "eigenvalue", "特征值", "svd", "奇异值")):
        vec[4] = 1.0
        vec[5] = 0.8
    return vec


def _make_siliconflow_embedding(call_counter: list[int] | None = None, dims: int = 1024) -> SiliconFlowEmbedding:
    def handler(request: httpx.Request) -> httpx.Response:
        if call_counter is not None:
            call_counter.append(1)
        payload = json.loads(request.content.decode("utf-8"))
        inputs = payload["input"]
        if isinstance(inputs, str):
            inputs = [inputs]
        data = [{"index": i, "embedding": _deterministic_vec(str(t), dims)} for i, t in enumerate(inputs)]
        return httpx.Response(200, json={
            "object": "list",
            "model": payload.get("model", "BAAI/bge-m3"),
            "data": data,
            "usage": {"prompt_tokens": len(inputs) * 8, "total_tokens": len(inputs) * 8},
        })

    provider = SiliconFlowEmbedding(transport=httpx.MockTransport(handler))
    provider.configure(SecretStr("sf-test-key"), SILICONFLOW_BASE_URL)
    return provider


def _make_reranker(
    embedding_provider: SiliconFlowEmbedding,
    scores_by_substr: dict[str, float] | None = None,
    fail_status: int | None = None,
) -> SiliconFlowReranker:
    def handler(request: httpx.Request) -> httpx.Response:
        if fail_status is not None:
            return httpx.Response(fail_status, json={"message": "upstream unavailable"})
        payload = json.loads(request.content.decode("utf-8"))
        docs: list[str] = payload["documents"]
        top_n: int = payload.get("top_n", len(docs))
        scored = []
        for idx, doc in enumerate(docs):
            score = 0.1
            for substr, s in (scores_by_substr or {}).items():
                if substr.casefold() in doc.casefold():
                    score = max(score, s)
            scored.append({"index": idx, "relevance_score": score})
        scored.sort(key=lambda x: -x["relevance_score"])
        return httpx.Response(200, json={"results": scored[:top_n]})

    return SiliconFlowReranker(embedding_provider, transport=httpx.MockTransport(handler))


def _stage(directory: Path, content: bytes) -> str:
    import_id = str(uuid4())
    fd = os.open(directory / import_id, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(content)
    return import_id


def _import_md_api(client: TestClient, app, filename: str, text: str) -> dict:
    import_id = _stage(app.state.library.staging_dir, text.encode("utf-8"))
    resp = client.post(
        "/_host/knowledge/imports",
        headers={"Authorization": f"Bearer {HOST_TOKEN}"},
        json={"import_id": import_id, "filename": filename},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _import_md_direct(lib: LibraryService, filename: str, text: str):
    import_id = _stage(lib.staging_dir, text.encode("utf-8"))
    return lib.import_textbook(import_id, filename)


def _wait_idle(client: TestClient, auth: dict, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        ov = client.get("/api/v1/knowledge/index-status", headers=auth).json()
        if ov["in_progress"] == 0 and ov["active_job_id"] is None:
            return ov
        if time.monotonic() >= deadline:
            return ov
        time.sleep(0.05)


# ---------- 1. Migration V8 & Settings Persistence ----------


def _mk_store(root: Path) -> KnowledgeStore:
    root.mkdir(parents=True, exist_ok=True)
    return KnowledgeStore(root)


def test_schema_v8_settings_persistence_and_legacy_auto_inherit(tmp_path):
    store = _mk_store(tmp_path / "fresh")
    settings_store = KnowledgeSettingsStore(store)

    # Fresh database defaults: auto_index ON, cloud_embedding OFF until one-time consent
    initial = settings_store.get()
    assert initial.auto_index_on_import is True
    assert initial.cloud_embedding_enabled is False
    assert initial.cloud_rerank_enabled is True
    assert initial.enable_bilingual_rewrite is True

    # Persist one-time consent and verify reload
    updated = settings_store.update(cloud_embedding_enabled=True, cloud_rerank_enabled=False)
    assert updated.cloud_embedding_enabled is True
    assert updated.cloud_rerank_enabled is False
    reloaded = KnowledgeSettingsStore(_mk_store(tmp_path / "fresh")).get()
    assert reloaded.cloud_embedding_enabled is True
    assert reloaded.cloud_rerank_enabled is False

    # Simulate an upgraded legacy DB that already had an index_job row before settings row was written
    store2 = _mk_store(tmp_path / "legacy_upgrade")
    lib2 = LibraryService(store2, uuid4())
    imp2 = _import_md_direct(lib2, "legacy.md", "# Legacy\n\nSome content.\n")
    with sqlite3.connect(store2.path) as conn:
        conn.execute("DELETE FROM knowledge_settings")
        conn.execute(
            "INSERT INTO index_jobs(job_id, job_type, document_id, document_version_id, config_key, status, stage, "
            "completed_work, total_work, retry_count, max_retries, created_at, updated_at, finished_at) "
            "VALUES ('job1', 'index_pipeline', ?, ?, 'cfg', 'succeeded', 'publishing', 1, 1, 0, 3, 1, 1, 1)",
            (imp2.document_id, imp2.document_version_id),
        )
        conn.commit()
    inherited = KnowledgeSettingsStore(store2).get()
    assert inherited.cloud_embedding_enabled is True


# ---------- 2. Auto-Index on Import & One-Click Complete Index ----------


def test_auto_index_on_import_local_and_hybrid_and_no_reembed_on_complete(tmp_path):
    embed_calls: list[int] = []
    provider = _make_siliconflow_embedding(embed_calls, dims=1024)
    app = create_app(UI_TOKEN, HOST_TOKEN, database_path=tmp_path / "business.db", embeddings=provider)
    auth = {"Authorization": f"Bearer {UI_TOKEN}"}

    with TestClient(app, base_url="http://127.0.0.1") as client:
        # 1) Default setting has cloud_embedding_enabled=False -> auto-index runs local BM25 only (0 embedding calls)
        res1 = _import_md_api(client, app, "mesh_notes.md", "# 非结构网格\n\nDelaunay 三角化与前沿推进法对比分析。\n")
        assert res1["outcome"] == "imported"
        overview = _wait_idle(client, auth)
        assert len(embed_calls) == 0

        assert overview["total_documents"] == 1
        assert overview["ready_keyword"] == 1
        assert overview["ready_hybrid"] == 0
        doc1 = overview["documents"][0]
        assert doc1["parse_status"] == "ready"
        assert doc1["chunk_count"] >= 1
        assert doc1["keyword_status"] == "ready"
        assert doc1["embedded_count"] == 0
        assert doc1["embedding_status"] == "disabled"
        orig_chunk_count = doc1["chunk_count"]

        # Duplicate import does NOT enqueue a new job
        dup = _import_md_api(client, app, "mesh_notes.md", "# 非结构网格\n\nDelaunay 三角化与前沿推进法对比分析。\n")
        assert dup["outcome"] == "duplicate"
        ov_dup = _wait_idle(client, auth)
        assert ov_dup["total_documents"] == 1
        assert len(embed_calls) == 0

        # 2) Enable global cloud embedding consent -> one-click index-all completes BM25-only book
        # reusing existing chunks and calling SiliconFlow only for missing vectors
        s_resp = client.post("/api/v1/knowledge/settings", headers=auth, json={"cloud_embedding_enabled": True})
        assert s_resp.status_code == 200 and s_resp.json()["cloud_embedding_enabled"] is True

        all_resp = client.post("/api/v1/knowledge/index-all", headers=auth)
        assert all_resp.status_code == 200
        body = all_resp.json()
        assert body["queued_count"] == 1
        assert body["skipped_complete"] == 0
        assert body["mode"] == "hybrid"

        overview2 = _wait_idle(client, auth)
        assert len(embed_calls) >= 1
        calls_after_first = len(embed_calls)

        assert overview2["ready_hybrid"] == 1
        assert overview2["missing_for_completion"] == 0
        doc1_after = overview2["documents"][0]
        assert doc1_after["chunk_count"] == orig_chunk_count
        assert doc1_after["embedded_count"] == orig_chunk_count
        assert doc1_after["vector_coverage_pct"] == 100.0
        assert doc1_after["published_readiness"] == "ready_hybrid"

        # 3) Running index-all again skips ready_hybrid completely with 0 embedding API calls!
        all_resp2 = client.post("/api/v1/knowledge/index-all", headers=auth)
        assert all_resp2.status_code == 200
        assert all_resp2.json()["queued_count"] == 0
        assert all_resp2.json()["skipped_complete"] == 1
        _wait_idle(client, auth)
        assert len(embed_calls) == calls_after_first

        # 4) Importing a second book while cloud_embedding_enabled=True auto-builds full hybrid index
        res2 = _import_md_api(client, app, "stats_notes.md", "# 概率论\n\n最大似然估计 MLE 的渐近性质与贝叶斯推断。\n")
        assert res2["outcome"] == "imported"
        overview3 = _wait_idle(client, auth)
        assert overview3["total_documents"] == 2
        assert overview3["ready_hybrid"] == 2


# ---------- 3. BM25 Natural-Language Recall & Exact-Phrase Preservation ----------


def test_bm25_term_extraction_and_natural_language_recall(tmp_path):
    assert is_explicit_exact_query('"最大似然估计"') is True
    assert is_explicit_exact_query("L(\\theta) = 0") is True
    assert is_explicit_exact_query("什么是最大似然估计") is False

    terms = extract_bm25_terms("请比较 Delaunay 三角化与前沿推进法的优缺点")
    assert any("delaunay" in t.lower() for t in terms)
    assert any("三角化" in t or "前沿推进" in t for t in terms)
    or_expr = build_bm25_or_query("请比较 Delaunay 三角化与前沿推进法的优缺点")
    assert or_expr is not None and " OR " in or_expr

    # End-to-end BM25 natural-language recall on a book where the exact question sentence does not appear verbatim
    store = _mk_store(tmp_path / "k_bm25")
    lib = LibraryService(store, uuid4())
    imported = _import_md_direct(
        lib, "mesh.md", "# 网格生成\n\nDelaunay 三角化满足空外接圆准则，而前沿推进法从边界逐步向内部推进生成单元。\n"
    )
    parsed = DocumentParsingService(store).ensure_parsed(imported.document_version_id)
    cs = DocumentChunkingService(store).ensure_chunked(parsed.parsed_document_id)
    KeywordIndex(store).ensure_indexed(cs.chunk_set_id)

    kw = KeywordSearchService(store)
    # Exact phrase mode alone returns 0 because the exact question doesn't appear verbatim
    exact_hits = kw.search("请比较 Delaunay 三角化与前沿推进法的区别", exact_phrase=True)
    assert len(exact_hits) == 0
    # Default P10.2 search automatically falls back to OR-expanded BM25 and recalls the chunk!
    auto_hits = kw.search("请比较 Delaunay 三角化与前沿推进法的区别")
    assert len(auto_hits) == 1
    assert "空外接圆准则" in auto_hits[0].body_text


# ---------- 4. Cross-Textbook 12-Book Comparison, Strategy Preservation & Reranker ----------


@pytest.mark.anyio
async def test_compare_12_books_unified_ranking_and_reranker_and_degradation(tmp_path):
    store = _mk_store(tmp_path / "k12")
    lib = LibraryService(store, uuid4())
    parser = DocumentParsingService(store)
    chunker = DocumentChunkingService(store)
    kw_index = KeywordIndex(store)
    provider = _make_siliconflow_embedding(dims=1024)
    embedder = DocumentEmbeddingService(store, provider)

    # Create 2 mesh textbooks and 10 unrelated statistics/algebra textbooks (12 total > old 8 limit!)
    books = [
        ("非结构网格生成技术调研.md", "# 非结构网格\n\n本文系统综述了 Delaunay 三角化、前沿推进法（Advancing Front）与 Looping Algorithm 算法原理。\n"),
        ("Mesh_Generation_Review.md", "# Unstructured Mesh\n\nDelaunay triangulation and Advancing Front Method are core unstructured mesh generation algorithms.\n"),
    ] + [
        (f"统计教材_{i}.md", f"# 第{i}章 概率统计\n\n最大似然估计 MLE 与贝叶斯估计在样本量增大时具有一致性与渐近正态性（教材 {i}）。\n")
        for i in range(10)
    ]
    id_to_name: dict[str, str] = {}
    for name, body in books:
        imp = _import_md_direct(lib, name, body)
        id_to_name[imp.document_id] = name
        pd = parser.ensure_parsed(imp.document_version_id)
        cs = chunker.ensure_chunked(pd.parsed_document_id)
        kw_index.ensure_indexed(cs.chunk_set_id)
        await embedder.ensure_embedded(cs.chunk_set_id, profile=TEST_PROFILE)

    kw = KeywordSearchService(store)
    vec = VectorRetrievalService(store, provider)

    # 1) With Reranker enabled: 12-book per_document compare succeeds, mesh books rank #1 and #2,
    # and reranker puts the Looping Algorithm Chinese survey at #1!
    reranker_ok = _make_reranker(provider, {"Looping Algorithm": 0.96, "Advancing Front Method": 0.88, "最大似然": 0.01})
    hybrid = HybridRetrievalService(kw, vec, reranker=reranker_ok)
    res = await hybrid.search(
        "对比 Delaunay、Advancing Front 和 Looping Algorithm",
        top_k=6,
        retrieval_mode="hybrid",
        coverage_mode="per_document",
        profile=TEST_PROFILE,
        use_rerank=True,
    )
    assert len(res.documents) == 12
    assert res.rerank_used is True
    assert res.rerank_diagnostic is None
    assert len(res.hits) >= 2
    assert id_to_name[res.hits[0].document_id] == "非结构网格生成技术调研.md"
    assert res.hits[0].rerank_score == pytest.approx(0.96)
    assert id_to_name[res.hits[1].document_id] == "Mesh_Generation_Review.md"
    # Irrelevant statistics books must NOT displace the 2 mesh books
    top2_names = {id_to_name[h.document_id] for h in res.hits[:2]}
    assert top2_names == {"非结构网格生成技术调研.md", "Mesh_Generation_Review.md"}

    # 2) Simulated Reranker failure -> explicit degradation diagnostic, retrieval still succeeds via RRF!
    reranker_fail = _make_reranker(provider, fail_status=503)
    hybrid_fail = HybridRetrievalService(kw, vec, reranker=reranker_fail)
    res_deg = await hybrid_fail.search(
        "对比 Delaunay、Advancing Front 和 Looping Algorithm",
        top_k=6,
        retrieval_mode="hybrid",
        coverage_mode="per_document",
        profile=TEST_PROFILE,
        use_rerank=True,
    )
    assert res_deg.rerank_used is False
    assert res_deg.rerank_diagnostic is not None
    assert res_deg.rerank_diagnostic.code == "RERANK_SERVICE_ERROR"
    assert len(res_deg.hits) >= 2
    assert {"非结构网格生成技术调研.md", "Mesh_Generation_Review.md"}.issubset({id_to_name[h.document_id] for h in res_deg.hits[:2]})


# ---------- 5. Explicit Strategy Preservation & QueryPlan ----------


def test_resolve_strategy_never_overrides_explicit_choice():
    # Explicit global with "对比" keyword MUST stay global!
    assert resolve_strategy("对比两本教材的贝叶斯定义", requested_strategy="global", coverage_mode="global", document_ids=None)[1:] == ("global", "global")
    assert resolve_strategy("对比两本教材的贝叶斯定义", requested_strategy="scoped", coverage_mode="global", document_ids=["doc1"])[1:] == ("scoped", "global")
    assert resolve_strategy("什么是最大似然估计", requested_strategy="compare", coverage_mode="global", document_ids=None)[1:] == ("compare", "per_document")
    # Only requested_strategy="auto" routes based on intent/document_ids
    assert resolve_strategy("对比 Delaunay 和 Advancing Front", requested_strategy="auto", coverage_mode="global", document_ids=None)[1:] == ("compare", "per_document")
    assert resolve_strategy("什么是最大似然估计", requested_strategy="auto", coverage_mode="global", document_ids=["doc1"])[1:] == ("scoped", "global")
    assert resolve_strategy("什么是最大似然估计", requested_strategy="auto", coverage_mode="global", document_ids=None)[1:] == ("global", "global")


@pytest.mark.anyio
async def test_bilingual_query_plan_preserves_original_and_falls_back_cleanly(tmp_path):
    store = _mk_store(tmp_path / "k_qp")
    lib = LibraryService(store, uuid4())
    imp = _import_md_direct(
        lib,
        "Mesh_Generation_Handbook.md",
        "# Unstructured Mesh Generation\n\nDelaunay triangulation and advancing front methods generate high quality unstructured meshes for finite element simulations.\n",
    )
    pd = DocumentParsingService(store).ensure_parsed(imp.document_version_id)
    DocumentChunkingService(store).ensure_chunked(pd.parsed_document_id)
    provider = _make_siliconflow_embedding(dims=1024)
    kw = KeywordSearchService(store)
    vec = VectorRetrievalService(store, provider)
    hybrid = HybridRetrievalService(kw, vec)

    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    class _FakeModels:
        configured = True

        def __init__(self, payload: str | Exception):
            self.payload = payload

        @asynccontextmanager
        async def open_model(self):
            payload = self.payload

            class _M:
                async def ainvoke(self, _prompt, **_kwargs):
                    if isinstance(payload, Exception):
                        raise payload
                    return SimpleNamespace(content=payload)

            yield _M()

    qa_ok = KnowledgeQAService(
        store,
        hybrid,
        profile=TEST_PROFILE,
        models=_FakeModels(json.dumps({
            "standalone_query": "非结构网格生成算法有哪些",
            "zh_terms": ["非结构网格", "前沿推进法"],
            "en_terms": ["Delaunay triangulation", "Advancing Front"],
            "subqueries": ["unstructured mesh generation Delaunay triangulation"],
        })),
    )
    plan = await qa_ok.plan_query("非结构网格生成算法有哪些", rewrite_mode="llm")
    assert plan.original_query == "非结构网格生成算法有哪些"
    assert plan.rewrite_mode == "llm"
    assert plan.query_variants[0] == "非结构网格生成算法有哪些"
    assert len(plan.query_variants) <= 3
    assert any("Delaunay" in v for v in plan.query_variants)

    # LLM failure gracefully falls back to original query without crashing
    qa_fail = KnowledgeQAService(store, hybrid, profile=TEST_PROFILE, models=_FakeModels(RuntimeError("timeout")))
    plan_fb = await qa_fail.plan_query("非结构网格生成算法有哪些", rewrite_mode="llm")
    assert plan_fb.original_query == "非结构网格生成算法有哪些"
    assert plan_fb.query_variants == ("非结构网格生成算法有哪些",)
    assert plan_fb.rewrite_error is not None
