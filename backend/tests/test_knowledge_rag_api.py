import json
import sqlite3
from contextlib import closing
from uuid import uuid4

import httpx
import pytest

from app.chunking.models import ChunkerConfig
from app.services.document_chunking import DocumentChunkingService
from app.services.document_parsing import DocumentParsingService
from app.storage.business import MIGRATIONS, BusinessStore
from app.storage.checkpoints import CHECKPOINT_DATABASE_NAME
from app.storage.keywords import KeywordIndex
from test_chat import ProviderStream, chunk, configure, frames, live_api, new_session, payload, rows, run_path
from test_library_management import import_file
from test_pdf_parsing import make_pdf, page, text

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def index_book_keyword(store, book, warnings=None):
    parsed = DocumentParsingService(store).ensure_parsed(book.document_version_id)
    if warnings:
        with store.write() as db:
            db.execute(
                "UPDATE parsed_documents SET warnings = ? WHERE parsed_document_id = ?",
                (json.dumps(warnings), parsed.parsed_document_id),
            )
    chunking = DocumentChunkingService(store)
    chunk_set = chunking.ensure_chunked(parsed.parsed_document_id, ChunkerConfig(chunk_size=300, chunk_overlap=30))
    KeywordIndex(store).ensure_indexed(chunk_set.chunk_set_id)
    return chunk_set.chunk_set_id


async def test_knowledge_search_and_citation_validation_api(tmp_path):
    business_path = tmp_path / "business.sqlite3"
    async with live_api(business_path, lambda _: httpx.Response(200, stream=ProviderStream())) as (client, host):
        library = client._transport.app.state.library if hasattr(client._transport, "app") else None
        # Obtain library from the running app via store initialization in the same tmp_path
        from app.storage.library import KnowledgeStore
        from app.services.library import LibraryService

        kstore = KnowledgeStore(tmp_path)
        lib_service = LibraryService(kstore, uuid4())

        md_book = import_file(
            lib_service,
            "# 统计推断\n\n最大似然估计（MLE）通过最大化似然函数估计参数。\n\n## 贝叶斯方法\n\n贝叶斯估计结合先验分布与样本似然。\n".encode("utf-8"),
            "数理统计讲义.md",
        )
        pdf_bytes = make_pdf(
            [page(text(72, 720, "Maximum likelihood estimation chooses theta to maximize L(theta)."))],
            labels=b"/PageLabels << /Nums [0 << /S /D /St 46 >>] >>",
        )
        pdf_book = import_file(lib_service, pdf_bytes, "线性代数与统计.pdf")
        md_set_id = index_book_keyword(kstore, md_book)
        pdf_set_id = index_book_keyword(
            kstore,
            pdf_book,
            warnings=[{"code": "PDF_UNMAPPED_GLYPHS", "message": "Unmapped math glyphs omitted", "physical_page": 1}],
        )

        # Auth checks
        assert (await client.post("/api/v1/knowledge/search", headers={"Authorization": ""}, json={"query": "似然"})).status_code == 401
        assert (await client.post("/api/v1/knowledge/search", headers=host, json={"query": "似然"})).status_code == 403

        # Local keyword/hybrid search (without SiliconFlow configured, hybrid degrades gracefully to keyword_only)
        res = await client.post("/api/v1/knowledge/search", json={
            "query": "最大似然估计",
            "retrieval_mode": "hybrid",
            "coverage_mode": "global",
            "top_k": 5,
        })
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["channel_status"] == "keyword_only"
        assert len(data["hits"]) >= 1
        first_hit = data["hits"][0]
        assert first_hit["document_id"] == md_book.document_id
        assert first_hit["textbook_name"] == "数理统计讲义.md"
        assert first_hit["format"] == "md"
        assert first_hit["section_path"] == ["统计推断"]
        assert "行" in first_hit["location_summary"]
        assert first_hit["evidence_id"].startswith("ev1:")

        # PDF search with warning and page label
        pdf_res = await client.post("/api/v1/knowledge/search", json={
            "query": "Maximum likelihood",
            "document_ids": [pdf_book.document_id],
            "retrieval_mode": "keyword",
        })
        assert pdf_res.status_code == 200
        pdf_hit = pdf_res.json()["hits"][0]
        assert pdf_hit["document_id"] == pdf_book.document_id
        assert pdf_hit["physical_pages"] == [1]
        assert pdf_hit["page_labels"] == ["46"]
        assert "第 1 页" in pdf_hit["location_summary"] and "46" in pdf_hit["location_summary"]
        assert pdf_hit["warnings"][0]["code"] == "PDF_UNMAPPED_GLYPHS"

        # Validate citation endpoint
        val_res = await client.post("/api/v1/knowledge/citations/validate", json={
            "items": [
                {"evidence_id": first_hit["evidence_id"], "chunk_set_id": md_set_id},
                {"evidence_id": pdf_hit["evidence_id"], "chunk_set_id": pdf_set_id},
            ],
        })
        assert val_res.status_code == 200
        validated = val_res.json()
        assert [item["status"] for item in validated] == ["resolved", "resolved"]


async def test_chat_rag_qa_followup_rewrite_citation_persistence_and_lifecycle(tmp_path):
    business_path = tmp_path / "business.sqlite3"
    checkpoint_path = tmp_path / CHECKPOINT_DATABASE_NAME
    captured_requests = []

    def provider(request):
        body = json.loads(request.content)
        captured_requests.append(body)
        reply_text = "根据《数理统计讲义.md》[引用 1]，最大似然估计通过最大化似然函数求解参数。"
        stream = httpx.ByteStream(
            chunk({"role": "assistant", "content": reply_text[:15]})
            + chunk({"content": reply_text[15:]})
            + chunk({}, "stop")
            + b"data: [DONE]\n\n"
        )
        return httpx.Response(200, stream=stream, headers={"Content-Type": "text/event-stream"})

    async with live_api(business_path, provider) as (client, host):
        await configure(client, host)
        from app.storage.library import KnowledgeStore
        from app.services.library import LibraryService

        kstore = KnowledgeStore(tmp_path)
        lib_service = LibraryService(kstore, uuid4())
        book_a = import_file(
            lib_service,
            "# 第一章 统计估计\n\n最大似然估计（MLE）是寻找使观测样本出现概率最大的参数值。Bayesian estimation 贝叶斯估计则引入先验分布。\n".encode("utf-8"),
            "数理统计A.md",
        )
        book_b = import_file(
            lib_service,
            make_pdf([page(text(72, 720, "MLE and Bayesian estimation converge asymptotically, check formula b = 0."))]),
            "高等统计B.pdf",
        )
        index_book_keyword(kstore, book_a)
        index_book_keyword(
            kstore,
            book_b,
            warnings=[{"code": "PDF_UNMAPPED_GLYPHS", "message": "Unmapped math glyphs omitted", "physical_page": 1}],
        )

        session = await new_session(client)

        # Turn 1: Ask about MLE
        turn1_body = {
            **payload("MLE"),
            "knowledge": {
                "enabled": True,
                "document_ids": [book_a.document_id, book_b.document_id],
                "retrieval_mode": "hybrid",
                "coverage_mode": "global",
                "top_k": 4,
            },
        }
        res1 = await client.post(run_path(session), json=turn1_body)
        assert res1.status_code == 200
        events1 = [item async for item in frames(res1) if isinstance(item, dict)]
        event_types1 = [item["type"] for item in events1]
        assert "tool_started" in event_types1
        assert "tool_completed" in event_types1
        assert "run_citations" in event_types1
        assert events1[-1]["type"] == "run_completed"

        citation_event = next(item for item in events1 if item["type"] == "run_citations")
        assert len(citation_event["data"]["citations"]) == 2
        assert citation_event["data"]["channel_status"] == "keyword_only"

        # Verify system prompt contained evidence and PDF math glyph warning
        sys_msg = captured_requests[-1]["messages"][0]["content"]
        assert "【教材知识库问答模式】" in sys_msg
        assert "[引用 1]" in sys_msg
        assert "PDF_UNMAPPED_GLYPHS" in sys_msg

        # Verify raw textbook chunks are NOT persisted in checkpoints.sqlite3
        assert rows(checkpoint_path, "SELECT count(*) FROM writes WHERE instr(value, ?) > 0",
                    ("寻找使观测样本出现概率最大".encode("utf-8"),)) == [(0,)]

        # Turn 2: Follow-up with coreference & comparison intent ("它与 Bayesian estimation 在不同教材里怎么对比？")
        turn2_body = {
            **payload("它与 Bayesian estimation 在不同教材里怎么对比？"),
            "knowledge": {
                "enabled": True,
                "document_ids": [book_a.document_id, book_b.document_id],
                "retrieval_mode": "keyword",
                "coverage_mode": "global",
                "top_k": 4,
            },
        }
        res2 = await client.post(run_path(session), json=turn2_body)
        events2 = [item async for item in frames(res2) if isinstance(item, dict)]
        cit_ev2 = next(item for item in events2 if item["type"] == "run_citations")
        # Followup rewrite should have prepended "MLE" and upgraded coverage_mode to "per_document"
        assert "MLE" in cit_ev2["data"]["retrieval_query"]
        assert cit_ev2["data"]["coverage_mode"] == "per_document"

        # Check GET /sessions/{session}/runs returns persisted tool_calls and citations
        runs_page = (await client.get(f"/api/v1/sessions/{session}/runs")).json()
        latest_run = runs_page["items"][0]
        assert len(latest_run["tool_calls"]) == 1
        assert latest_run["tool_calls"][0]["tool_name"] == "search_knowledge"
        assert len(latest_run["citations"]) >= 1
        assert all(c["status"] == "resolved" for c in latest_run["citations"])

        # Delete book_b and verify historical citations mark book_b as source_removed while keeping book_a resolved
        lib_service.delete_document(book_b.document_id)
        runs_after_delete = (await client.get(f"/api/v1/sessions/{session}/runs")).json()["items"][0]
        statuses_by_doc = {c["document_id"]: c["status"] for c in runs_after_delete["citations"]}
        assert statuses_by_doc[book_a.document_id] == "resolved"
        assert statuses_by_doc[book_b.document_id] == "source_removed"

        # Test session rename and delete endpoints
        renamed = await client.post(f"/api/v1/sessions/{session}/rename", json={"title": "数理统计专题复习"})
        assert renamed.status_code == 200 and renamed.json()["title"] == "数理统计专题复习"
        deleted = await client.post(f"/api/v1/sessions/{session}/delete")
        assert deleted.status_code == 204
        assert (await client.get(f"/api/v1/sessions/{session}/messages")).status_code == 404


def test_v4_to_v5_migration_preserves_rows(tmp_path):
    path = tmp_path / "business.sqlite3"
    session_id, run_id, req_id, msg_id = (str(uuid4()) for _ in range(4))
    with closing(sqlite3.connect(path)) as db, db:
        for migration in MIGRATIONS[:4]:
            for stmt in migration:
                db.execute(stmt)
        db.execute("PRAGMA user_version=4")
        db.execute("INSERT INTO sessions VALUES (?, 'v4会话', NULL, 't', 't')", (session_id,))
        db.execute("INSERT INTO runs (run_id, session_id, client_request_id, status, started_at, finished_at, final_checkpoint_id) VALUES (?, ?, ?, 'completed', 't', 't', 'cp1')",
                   (run_id, session_id, req_id))
        db.execute("INSERT INTO messages VALUES (?, ?, ?, 'user', '问题', 't')", (msg_id, session_id, run_id))
    store = BusinessStore(path)
    assert rows(path, "PRAGMA user_version") == [(5,)]
    assert store.list_runs(session_id, 10, None)[0].citations == ()
