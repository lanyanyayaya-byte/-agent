from __future__ import annotations

import asyncio
import os
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient

from app.chunking.hybrid import CHUNKER_VERSION, ChunkerConfig, iter_chunks
from app.chunking.models import ChunkSource
from app.main import create_app
from app.parsing.models import ContentBlock, PdfSourceLocation
from app.parsing.pdf_layout import (
    Line,
    _detect_running_headers,
    _ExtractedPage,
    _join_paragraph_lines,
)
from app.services.document_parsing import PARSER_SPECS
from app.services.hybrid_retrieval import _is_reference_chunk, _query_wants_references
from app.services.knowledge_qa import KnowledgeQAService
from app.services.retrieval_trace import RetrievalTrace, make_failed_trace
from app.storage.business import BusinessStore

UI_TOKEN = "u" * 32
HOST_TOKEN = "h" * 32

_TEST_SOURCE = ChunkSource(
    document_id="doc1",
    document_version_id="ver1",
    format="pdf",
    content_sha256="a" * 64,
    parsed_document_id=1,
    parse_spec=PARSER_SPECS["pdf"],
)


def _make_block(
    block_id: int,
    text: str,
    *,
    kind: str = "paragraph",
    section_path: tuple[str, ...] = ("Chapter 1",),
    page: int = 1,
) -> ContentBlock:
    return ContentBlock(
        block_id=f"b{block_id}",
        document_version_id="ver1",
        kind=kind,
        text=text,
        source_locations=(PdfSourceLocation(physical_page=page, page_label=str(page)),),
        section_heading_block_id="h1",
        section_path=section_path,
    )


def _stage_and_import_md(client: TestClient, app, filename: str, text: str) -> dict:
    staging_dir = app.state.library.staging_dir
    staging_dir.mkdir(parents=True, exist_ok=True)
    import_id = str(uuid4())
    fd = os.open(staging_dir / import_id, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(text.encode("utf-8"))
    resp = client.post(
        "/_host/knowledge/imports",
        headers={"Authorization": f"Bearer {HOST_TOKEN}"},
        json={"import_id": import_id, "filename": filename},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _wait_idle(client: TestClient, headers: dict, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        ov = client.get("/api/v1/knowledge/index-status", headers=headers).json()
        if ov["in_progress"] == 0 and ov["active_job_id"] is None:
            return ov
        if time.monotonic() >= deadline:
            return ov
        time.sleep(0.05)


def test_p10_4_pdf_dehyphenation_and_running_headers():
    def _mk_line(text: str, top: float) -> Line:
        return Line(
            text=text,
            box=(50.0, top, 400.0, top + 12.0),
            baseline=top + 10.0,
            size=10.0,
            bold=False,
            leading_glyph_size=10.0,
        )

    joined_ordinary = _join_paragraph_lines(
        [_mk_line("triangu-", 100.0), _mk_line("lation method", 114.0)],
        is_heading=False,
    )
    assert joined_ordinary == "triangulation method"

    joined_compound = _join_paragraph_lines(
        [_mk_line("non-", 100.0), _mk_line("linear system", 114.0)],
        is_heading=False,
    )
    assert "non-" in joined_compound and "linear system" in joined_compound and "nonlinear" not in joined_compound

    pages = [
        _ExtractedPage(
            physical_page=p,
            label=str(p),
            lines=[
                _mk_line("Numerical Analysis 5th Ed", 20.0),
                _mk_line(f"Body text for page {p}", 200.0),
                _mk_line(f"{p}", 770.0),
            ],
            body_size=10.0,
            width=600.0,
            height=800.0,
            rotation=0,
            analysis_rotation=0,
            analysis_width=600.0,
            analysis_height=800.0,
            unit=1.0,
            warnings=[],
            chars_count=80,
        )
        for p in range(1, 6)
    ]
    running = _detect_running_headers(pages)
    assert any("numerical analysis" in r for r in running)


def test_p10_4_chunking_v2_no_midword_cuts_and_tiny_merge():
    assert CHUNKER_VERSION == "2"
    long_en = " ".join(
        f"unstructured triangulation algorithm step{i} computes Delaunay simplexes accurately."
        for i in range(40)
    )
    blocks = [_make_block(1, long_en, section_path=("Section 2", "2.1 Delaunay"), page=5)]
    chunks = list(iter_chunks(blocks, source=_TEST_SOURCE, config=ChunkerConfig(chunk_size=360, chunk_overlap=80)))
    assert len(chunks) >= 2
    for ch in chunks[1:]:
        first_token = ch.text.lstrip().split()[0]
        assert first_token in {
            "unstructured",
            "triangulation",
            "algorithm",
            "computes",
            "Delaunay",
            "simplexes",
            "accurately.",
        } or first_token.startswith("step"), f"Mid-word overlap cut detected: {first_token!r}"

    tiny_blocks = [
        _make_block(1, "Short intro sentence one.", section_path=("Ch 1",), page=1),
        _make_block(2, "Short intro sentence two.", section_path=("Ch 1",), page=1),
        _make_block(3, "A sufficiently long body paragraph " + ("details " * 40), section_path=("Ch 1",), page=1),
    ]
    merged = list(iter_chunks(tiny_blocks, source=_TEST_SOURCE, config=ChunkerConfig(chunk_size=900, chunk_overlap=100)))
    assert len(merged) == 1
    assert "Short intro sentence one." in merged[0].text
    assert "Short intro sentence two." in merged[0].text

    thm_blocks = [
        _make_block(
            1,
            "Theorem 2.1 (Convergence). Let x_k be generated by Newton's method. " + ("condition " * 22),
            section_path=("Ch 2",),
            page=10,
        ),
        _make_block(
            2,
            "Proof. By Taylor expansion around the root x*, we obtain quadratic convergence. " + ("step " * 22),
            section_path=("Ch 2",),
            page=10,
        ),
    ]
    thm_chunks = list(iter_chunks(thm_blocks, source=_TEST_SOURCE, config=ChunkerConfig(chunk_size=450, chunk_overlap=60)))
    assert any("Theorem 2.1" in c.text and "Proof." in c.text for c in thm_chunks)


def test_p10_3_bilingual_query_variant_priority():
    qa = KnowledgeQAService(
        store=SimpleNamespace(),
        hybrid_retrieval=SimpleNamespace(),
        citation_resolver=SimpleNamespace(),
    )
    qa._corpus_language_flags = lambda _doc_ids=None: (True, True)  # type: ignore[method-assign]

    async def _fake_expand(query: str):
        if "ADI" in query:
            return (("交替方向隐式格式", "ADI格式"), ("ADI scheme", "alternating direction implicit"), (), 12.0, None)
        return (("最大似然估计", "极大似然估计"), ("maximum likelihood estimation", "MLE"), (), 12.0, None)

    qa._llm_bilingual_expand = _fake_expand  # type: ignore[method-assign]

    # English query must place Chinese terms right after the original query (index 1).
    plan_en = asyncio.run(qa.plan_query("What is the ADI scheme for parabolic PDEs?", rewrite_mode="llm"))
    assert plan_en.query_variants[0] == "What is the ADI scheme for parabolic PDEs?"
    assert plan_en.chinese_terms == ("交替方向隐式格式", "ADI格式")
    assert "交替方向隐式格式" in plan_en.query_variants[1], plan_en.query_variants

    # Chinese query must place English terms right after the original query (index 1).
    plan_zh = asyncio.run(qa.plan_query("讲解一下最大似然估计的渐近性质", rewrite_mode="llm"))
    assert plan_zh.query_variants[0] == "讲解一下最大似然估计的渐近性质"
    assert plan_zh.english_terms == ("maximum likelihood estimation", "MLE")
    assert "maximum likelihood estimation" in plan_zh.query_variants[1], plan_zh.query_variants

    # rewrite_mode='off' keeps only original query, zero LLM calls.
    plan_off = asyncio.run(qa.plan_query("What is the ADI scheme?", rewrite_mode="off"))
    assert plan_off.query_variants == ("What is the ADI scheme?",)
    assert plan_off.rewrite_mode == "off"


def test_p10_3_references_soft_demotion_detection():
    assert _is_reference_chunk(["References"], "[1] Bowyer A. Computing Dirichlet tessellations. 1981.")
    assert _is_reference_chunk(["参考文献"], "[12] 张三. 非结构网格综述. 2020.")
    assert _is_reference_chunk([], "[1] A. Smith (2018).\n[2] B. Jones (2019).\n[3] C. Wang (2020).")
    assert not _is_reference_chunk(["第 2 章", "2.1 Delaunay"], "Delaunay 三角剖分满足空外接圆准则，是常用的非结构网格生成算法。")

    assert _query_wants_references("这篇论文引用了哪些参考文献？")
    assert _query_wants_references("List the references and bibliography")
    assert not _query_wants_references("什么是 Delaunay 三角剖分的空外接圆性质？")


def test_p10_3_business_store_v8_trace_persistence_and_cleanup(tmp_path: Path):
    store = BusinessStore(tmp_path / "business.sqlite3")
    session = store.create_session("RAG Trace Test")
    run = store.start_run(session.session_id, "req-1", "测试最大似然估计", accept_new_run=True)

    trace = RetrievalTrace(
        trace_id="tr_persist_1",
        original_query="测试最大似然估计",
        resolved_query="测试最大似然估计",
        query_variants=["测试最大似然估计", "maximum likelihood estimation"],
        requested_strategy="global",
        actual_strategy="global",
        requested_mode="hybrid",
        effective_channels=["keyword", "vector"],
        in_scope_documents=3,
        searchable_documents=3,
        hit_documents=2,
        cited_documents=2,
        prompt_evidence_count=4,
        bm25_used=True,
        vector_used=True,
        rerank_used=True,
        total_elapsed_ms=88.5,
    )
    store.save_retrieval_trace(
        session.session_id, run.run_id, summary=trace.to_summary_dict(), trace=trace.to_compact_dict()
    )

    loaded = store.get_retrieval_trace(run.run_id)
    assert loaded is not None
    assert loaded["trace_id"] == "tr_persist_1"
    assert loaded["rerank_used"] is True

    runs = store.list_runs(session.session_id, limit=10, before=None)
    assert len(runs) == 1
    assert runs[0].retrieval_trace_summary is not None
    assert runs[0].retrieval_trace_summary["hit_documents"] == 2

    store.complete_run(run, "ok", str(uuid4()))
    run2 = store.start_run(session.session_id, "req-2", "无效范围", accept_new_run=True)
    failed = make_failed_trace(
        query="无效范围",
        requested_strategy="scoped",
        requested_mode="hybrid",
        error_code="KNOWLEDGE_INVALID_SCOPE",
        error_message="指定的教材不存在",
    )
    store.save_retrieval_trace(
        session.session_id, run2.run_id, summary=failed.to_summary_dict(), trace=failed.to_compact_dict()
    )
    loaded_failed = store.get_retrieval_trace(run2.run_id)
    assert loaded_failed is not None
    assert loaded_failed["error_code"] == "KNOWLEDGE_INVALID_SCOPE"

    store.fail_run(run2, "KNOWLEDGE_INVALID_SCOPE", "指定的教材不存在")
    store.delete_session(session.session_id)
    assert store.get_retrieval_trace(run.run_id) is None
    assert store.get_retrieval_trace(run2.run_id) is None


def test_p10_3_4_api_debug_search_chunk_browser_book_summary_and_stale_rebuild(tmp_path: Path):
    app = create_app(UI_TOKEN, HOST_TOKEN, database_path=tmp_path / "business.db")
    headers = {"Authorization": f"Bearer {UI_TOKEN}"}

    with TestClient(app, base_url="http://127.0.0.1") as client:
        md_content = (
            "# 1 引言\n\n"
            + ("非结构网格生成是计算流体力学与有限元分析的核心基础步骤。" * 8)
            + "\n\n# 2 Delaunay 三角剖分\n\n"
            + ("Delaunay 三角剖分满足空外接圆准则（empty circumcircle property），Bowyer-Watson 算法支持逐点插入。" * 8)
            + "\n\n# 3 前沿推进法\n\n"
            + ("前沿推进法（Advancing Front Method, AFM）从边界离散出发逐步向内部推进生成单元。" * 8)
            + "\n\n# References\n\n"
            + "[1] Bowyer A. Computing Dirichlet tessellations. The Computer Journal, 1981.\n"
        )

        imp = _stage_and_import_md(client, app, "mesh_survey.md", md_content)
        doc_id = imp["document_id"]
        ov0 = _wait_idle(client, headers)
        assert ov0["ready_keyword"] == 1

        # 1. Debug Search (keyword mode, rewrite=off -> 0 LLM calls)
        dbg = client.post(
            "/api/v1/knowledge/debug-search",
            headers=headers,
            json={
                "query": "Delaunay 空外接圆",
                "mode": "keyword",
                "strategy": "global",
                "rerank": False,
                "rewrite": "off",
                "top_k": 5,
                "expand_neighbors": True,
            },
        )
        assert dbg.status_code == 200, dbg.text
        body = dbg.json()
        assert body["hits"], body
        assert body["trace"]["rewrite_mode"] == "off"
        assert body["trace"]["query_variants"] == ["Delaunay 空外接圆"]
        assert len(body["trace"]["stages"]) >= 2
        assert len(body["trace"]["candidates"]) >= 1

        # 2. Debug Search with book_summary strategy (no LLM generation)
        bs_resp = client.post(
            "/api/v1/knowledge/debug-search",
            headers=headers,
            json={
                "query": "请完整总结这篇文献中的非结构网格生成方法",
                "document_ids": [doc_id],
                "strategy": "book_summary",
                "mode": "keyword",
                "rerank": False,
                "rewrite": "off",
            },
        )
        assert bs_resp.status_code == 200, bs_resp.text
        bs_data = bs_resp.json()
        assert bs_data["actual_strategy"] == "book_summary"
        cov = bs_data["trace"]["summary_coverage"]
        assert cov is not None
        assert cov["summary_mode"] == "full_read"
        assert any("Delaunay" in s for s in cov["covered_sections"])
        assert any("References" in s["section"] for s in cov["skipped_sections"])

        # 3. Chunk Browser API + Neighbors API
        cb_resp = client.get(
            f"/api/v1/knowledge/documents/{doc_id}/chunks?q=Delaunay",
            headers=headers,
        )
        assert cb_resp.status_code == 200, cb_resp.text
        cb = cb_resp.json()
        assert cb["total_chunks"] >= 2
        assert cb["filtered_total"] >= 1
        assert cb["chunker_current"] is True
        first_idx = cb["chunks"][0]["chunk_index"]

        nb_resp = client.get(
            f"/api/v1/knowledge/documents/{doc_id}/chunks/{first_idx}/neighbors",
            headers=headers,
        )
        assert nb_resp.status_code == 200, nb_resp.text
        nb = nb_resp.json()
        assert nb["current"]["chunk_index"] == first_idx

        # 4. Simulate stale chunker version -> status reports needs_rebuild=True, index still queried, & rebuild-stale queues it
        with sqlite3.connect(app.state.knowledge_store.path) as conn:
            conn.execute("UPDATE chunk_sets SET chunker_version = '1'")
            conn.commit()

        stale_dbg = client.post(
            "/api/v1/knowledge/debug-search",
            headers=headers,
            json={"query": "Delaunay", "mode": "keyword", "strategy": "global", "rerank": False, "rewrite": "off"},
        )
        assert stale_dbg.status_code == 200 and len(stale_dbg.json()["trace"]["candidates"]) >= 1

        ov_resp = client.get("/api/v1/knowledge/index-status", headers=headers)
        assert ov_resp.status_code == 200
        ov = ov_resp.json()
        assert ov["stale_count"] == 1
        assert ov["documents"][0]["needs_rebuild"] is True

        rb_resp = client.post("/api/v1/knowledge/rebuild-stale", headers=headers)
        assert rb_resp.status_code == 200
        assert rb_resp.json()["queued_count"] == 1
        _wait_idle(client, headers)
