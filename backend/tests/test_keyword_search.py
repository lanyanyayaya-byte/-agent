import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import httpx
import pytest
from pydantic import SecretStr

from app.chunking.models import ChunkerConfig
from app.main import create_app
from app.models.siliconflow import SILICONFLOW_BASE_URL, SiliconFlowEmbedding
from app.services import keyword_search
from app.services.document_chunking import DocumentChunkingService
from app.services.document_parsing import DocumentParsingService
from app.services.keyword_search import KeywordSearchService
from app.services.library import LibraryService
from app.storage import keywords, library as library_storage
from app.storage.keywords import KeywordIndexError
from app.storage.library import KnowledgeStore
from test_library_management import anyio_backend, import_file, library


def publish_chunks(library, *texts, title=None, document_id=None, filename="教材.md"):
    content = "\n\n".join(f"```text\n{text}\n```" for text in texts)
    if title:
        content = f"# {title}\n\n{content}"
    result = import_file(library, (content or "\n").encode(), filename, document_id=document_id)
    parsed = DocumentParsingService(library.store).ensure_parsed(result.document_version_id)
    return result, DocumentChunkingService(library.store).ensure_chunked(parsed.parsed_document_id)


def index_state(store, set_id):
    with store.connect() as db:
        meta = db.execute("SELECT last_chunk_index, indexed_at FROM keyword_index_sets WHERE chunk_set_id = ?",
                          (set_id,)).fetchone()
        rows = db.execute("SELECT row_id, chunk_index FROM keyword_index_rows WHERE chunk_set_id = ? ORDER BY chunk_index",
                          (set_id,)).fetchall()
    return meta, rows


def test_chinese_english_mixed_math_and_source_coordinates(library):
    body = "实验：最大似然估计（MLE）与贝叶斯估计。用Python计算概率。x^2 + θ ≠ 0；采用 4-bit。"
    book, ready = publish_chunks(library, body, "参考：probability", title="统计推断")
    service = KeywordSearchService(library.store)
    service.index.ensure_indexed(ready.chunk_set_id)
    for term in (
        "最大似然估计", "概率", "似然", "贝叶斯估计", "最大似然估计 MLE", "Python计算", "python",
        "x^2 + θ ≠ 0", "x ^ 2+θ≠0", "４‑ＢＩＴ",
    ):
        hit, = service.search(term)
        assert (hit.document_id, hit.document_version_id, hit.chunk_set_id, hit.parsed_document_id) == (
            book.document_id, book.document_version_id, ready.chunk_set_id, ready.parsed_document_id
        )
        assert hit.matched_in == ("body",)
        chunks = DocumentChunkingService(library.store).results
        assert hit.body_text == chunks.get_chunk(hit.chunk_id).text == body + "\n"
        assert hit.section_path == hit.title_context == ("统计推断",)
        assert hit.section_heading_block_id is not None
        span, = chunks.get_chunk_sources(hit.chunk_id)
        assert span.source_locations[0].start_line > 1
        assert hit.score > 0
    assert service.search("x^2 − θ ≠ 0") == ()
    assert service.search("x^2 + θ = 0") == ()
    assert service.search("4 bit") == ()
    assert service.search("probabil") == ()


def test_exact_phrase_ranking_and_title_only_match(library):
    exact = "Maximum Likelihood Estimation"
    _, ready = publish_chunks(library, exact + " " + exact, exact + " with extra detail",
                              "Maximum plausible Likelihood Estimation", title="机器学习专题")
    service = KeywordSearchService(library.store)
    service.index.ensure_indexed(ready.chunk_set_id)
    hits = service.search(exact)
    assert len(hits) == 2 and hits[0].score > hits[1].score
    assert hits[0].body_text.count(exact) == 2
    assert service.search(exact, 1) == hits[:1]
    title_hit = next(hit for hit in service.search("机器学习") if hit.matched_in == ("title",))
    assert "机器学习" not in title_hit.body_text
    assert title_hit.title_context == ("机器学习专题",)


def test_batch_column_matches_respect_phrase_and_scope(library):
    term = "对数似然"
    _, body = publish_chunks(library, term)
    selected, both = publish_chunks(library, term, "无关键词的正文", title=term, filename="专题.md")
    _, split = publish_chunks(library, "似然", title="对数", filename="拆分.md")
    service = KeywordSearchService(library.store)
    for ready in (body, both, split):
        service.index.ensure_indexed(ready.chunk_set_id)
    hits = service.search(term)
    assert {(hit.chunk_set_id, hit.matched_in) for hit in hits} == {
        (body.chunk_set_id, ("body",)),
        (both.chunk_set_id, ("body", "title")),
        (both.chunk_set_id, ("title",)),
    }
    assert service.search(term, document_ids=[selected.document_id]) == tuple(
        hit for hit in hits if hit.document_id == selected.document_id
    )


def test_filters_and_one_indexed_set_per_textbook(library):
    first, v1 = publish_chunks(library, "概率旧版 MLE")
    second, other = publish_chunks(library, "概率第二教材 MLE", filename="第二教材.md")
    newest, v2 = publish_chunks(library, "概率新版 MLE", document_id=first.document_id)
    service = KeywordSearchService(library.store)
    for ready in (v1, other):
        service.index.ensure_indexed(ready.chunk_set_id)
    assert {h.document_id for h in service.search("概率")} == {second.document_id}
    assert service.search("概率", document_ids=[first.document_id]) == ()
    assert service.search("概率", chunk_set_ids=[v1.chunk_set_id])[0].document_version_id == first.document_version_id
    assert service.search("概率", chunk_set_ids=[v2.chunk_set_id, 99]) == ()
    assert service.search("概率", document_ids=[]) == ()
    assert service.search("概率", chunk_set_ids=[]) == ()
    service.index.ensure_indexed(v2.chunk_set_id)
    assert {h.document_version_id for h in service.search("概率")} == {
        newest.document_version_id, second.document_version_id
    }
    assert {h.chunk_set_id for h in service.search("概率", chunk_set_ids=[v1.chunk_set_id, other.chunk_set_id])} == {
        v1.chunk_set_id, other.chunk_set_id
    }
    assert service.search("概率", document_ids=[second.document_id], chunk_set_ids=[v1.chunk_set_id]) == ()
    with pytest.raises(ValueError, match="one indexed Chunk Set"):
        service.search("概率", chunk_set_ids=[v1.chunk_set_id, v2.chunk_set_id])
    same_version = DocumentChunkingService(library.store).ensure_chunked(
        v2.parsed_document_id, ChunkerConfig(chunk_size=200)
    )
    service.index.ensure_indexed(same_version.chunk_set_id)
    assert {h.chunk_set_id for h in service.search("概率", document_ids=[first.document_id])} == {
        same_version.chunk_set_id
    }
    assert service.search("概率", chunk_set_ids=[v2.chunk_set_id])[0].chunk_set_id == v2.chunk_set_id


def test_reverse_completion_and_empty_set(library):
    first, old = publish_chunks(library, "概率旧版")
    _, new = publish_chunks(library, "概率新版", document_id=first.document_id)
    _, empty = publish_chunks(library, filename="空.md")
    service = KeywordSearchService(library.store)
    service.index.ensure_indexed(new.chunk_set_id)
    assert service.index.index_pending(limit=1) == (old.chunk_set_id,)
    assert service.index.index_pending() == (empty.chunk_set_id,)
    assert index_state(library.store, empty.chunk_set_id)[0][1] is not None
    assert {hit.chunk_set_id for hit in service.search("概率")} == {new.chunk_set_id}
    service.index.check_integrity()


def test_unpublished_sets_are_not_indexed(library):
    _, ready = publish_chunks(library, "公开概率")
    with library.store.write() as db:
        writing, = db.execute("""INSERT INTO chunk_sets
            (parsed_document_id, document_version_id, chunker_version, chunker_config_key,
             created_at, status, lease_expires_at)
            SELECT parsed_document_id, document_version_id, chunker_version, 'new',
                   created_at, 'writing', unixepoch() + 300 FROM chunk_sets WHERE chunk_set_id = ?
            RETURNING chunk_set_id""", (ready.chunk_set_id,)).fetchone()
        db.execute("""INSERT INTO chunks
            SELECT ?, chunk_index, chunk_id, '秘密概率', 4, section_heading_block_id, section_path, title_context
            FROM chunks WHERE chunk_set_id = ?""", (writing, ready.chunk_set_id))
    service = KeywordSearchService(library.store)
    with pytest.raises(KeywordIndexError, match="KEYWORD_CHUNK_SET_UNAVAILABLE"):
        service.index.ensure_indexed(writing)
    assert service.index.index_pending() == (ready.chunk_set_id,)
    assert service.search("概率", chunk_set_ids=[writing]) == ()
    service.index.rebuild()
    assert index_state(library.store, writing) == (None, [])
    assert service.search("秘密概率") == ()


def test_idempotence_in_place_updates_cascade_delete_and_restart(library, tmp_path, monkeypatch):
    book, ready = publish_chunks(library, "似然旧正文", "要删掉的 Python 段落")
    _, other = publish_chunks(library, "独立教材的概率", filename="独立.md")
    service = KeywordSearchService(library.store)
    assert service.index.index_pending() == (ready.chunk_set_id, other.chunk_set_id)
    original = index_state(library.store, ready.chunk_set_id)
    service.index.ensure_indexed(ready.chunk_set_id)
    assert original == index_state(library.store, ready.chunk_set_id)

    unchanged_identity = service.search("似然")[0].chunk_id
    first, delete = original[1]
    with library.store.write() as db:
        db.execute("""UPDATE chunks SET text = '概率新正文', text_length = 5, title_context = '["贝叶斯估计"]'
            WHERE chunk_set_id = ? AND chunk_index = ?""", (ready.chunk_set_id, first[1]))
        db.execute("DELETE FROM chunks WHERE chunk_set_id = ? AND chunk_index = ?", (ready.chunk_set_id, delete[1]))
    assert service.search("似然") == service.search("Python") == ()
    assert service.search("贝叶斯估计")[0].matched_in == ("title",)
    assert service.search("新正文")[0].chunk_id == unchanged_identity
    service.index.check_integrity()

    restarted = KeywordSearchService(KnowledgeStore(tmp_path))
    monkeypatch.setattr(restarted.index, "_index_batch", lambda _: pytest.fail("complete index was rebuilt"))
    restarted.index.ensure_indexed(ready.chunk_set_id)
    assert restarted.search("概率") == service.search("概率")
    library.delete_document(book.document_id)
    assert {hit.chunk_set_id for hit in restarted.search("概率")} == {other.chunk_set_id}
    assert restarted.search("贝叶斯估计") == ()
    assert index_state(library.store, ready.chunk_set_id) == (None, [])
    restarted.index.check_integrity()
    with library.store.connect() as db:
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []


def test_index_interruption_hides_old_version_by_default_and_resumes_batches(library, tmp_path, monkeypatch):
    first, old = publish_chunks(library, "MLE 旧版本")
    _, new = publish_chunks(library, *(f"MLE 新版本编号 {i}" for i in range(7)), document_id=first.document_id)
    service = KeywordSearchService(library.store)
    service.index.ensure_indexed(old.chunk_set_id)
    program = """
from contextlib import contextmanager
import os, sys
from pathlib import Path
from app.storage import keywords
from app.storage.library import KnowledgeStore

keywords.INDEX_BATCH_SIZE = 2

class InterruptedStore(KnowledgeStore):
    writes = 0

    @contextmanager
    def write(self):
        with super().write() as db:
            yield db
            self.writes += 1
            if self.writes == 2:
                os._exit(73)

keywords.KeywordIndex(InterruptedStore(Path(sys.argv[1]))).ensure_indexed(int(sys.argv[2]))
"""
    crashed = subprocess.run([sys.executable, "-c", program, str(tmp_path), str(new.chunk_set_id)], check=False)
    assert crashed.returncode == 73
    meta, rows = index_state(library.store, new.chunk_set_id)
    assert meta == (1, None) and [index for _, index in rows] == [0, 1]
    assert service.search("MLE") == ()
    assert {hit.chunk_set_id for hit in service.search("MLE", chunk_set_ids=[old.chunk_set_id])} == {old.chunk_set_id}
    assert service.search("MLE", chunk_set_ids=[new.chunk_set_id]) == ()
    service.index.check_integrity()

    monkeypatch.setattr(keywords, "INDEX_BATCH_SIZE", 2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        def resume(_):
            KeywordSearchService(KnowledgeStore(tmp_path)).index.ensure_indexed(new.chunk_set_id)
        list(pool.map(resume, range(2)))
    meta, completed = index_state(library.store, new.chunk_set_id)
    assert meta[1] is not None and completed[:2] == rows and len(completed) == 7
    assert {hit.chunk_set_id for hit in service.search("MLE", 7)} == {new.chunk_set_id}
    service.index.check_integrity()


@pytest.mark.parametrize("damage", [
    "DROP TABLE keyword_fts",
    "DROP TABLE keyword_fts_data",
    "DROP TABLE keyword_fts_config",
    "UPDATE keyword_fts_config SET v = -1 WHERE k = 'version'",
    "DELETE FROM keyword_index_rows",
    """INSERT INTO keyword_fts(keyword_fts, rowid, body, title)
       SELECT 'delete', row_id, body, title FROM keyword_index_content""",
])
def test_detect_and_rebuild_missing_or_corrupt_index(library, damage):
    _, ready = publish_chunks(library, "最大似然估计")
    service = KeywordSearchService(library.store)
    service.index.ensure_indexed(ready.chunk_set_id)
    expected = service.search("似然")
    with library.store.write() as db:
        db.execute(damage)
    with pytest.raises(KeywordIndexError, match=r"KEYWORD_INDEX_(MISSING|INCONSISTENT|CHECK_FAILED)"):
        service.index.check_integrity()
    service.index.rebuild()
    assert service.search("似然") == expected
    service.index.check_integrity()


def test_rebuild_failure_rolls_back_derived_objects(library, monkeypatch):
    _, ready = publish_chunks(library, "概率")
    service = KeywordSearchService(library.store)
    service.index.ensure_indexed(ready.chunk_set_id)
    before = index_state(library.store, ready.chunk_set_id)
    with monkeypatch.context() as patch:
        patch.setattr(keywords, "CREATE_TRIGGERS", (*keywords.CREATE_TRIGGERS, "INVALID SQL"))
        with pytest.raises(sqlite3.OperationalError, match="syntax error"):
            service.index.rebuild()
    assert index_state(library.store, ready.chunk_set_id) == before
    assert service.search("概率")
    service.index.check_integrity()


def test_query_limits_syntax_and_index_work_budget(library, monkeypatch):
    _, ready = publish_chunks(library, *("最大似然估计 MLE" for _ in range(70)))
    service = KeywordSearchService(library.store)
    service.index.ensure_indexed(ready.chunk_set_id)
    for query in ('概率" OR MLE --', 'NEAR(概率, MLE)', "'; DROP TABLE chunks; --", '""*', ""):
        assert service.search(query) == ()
    for params in ({"query": "x" * 257}, {"query": "x\x00y"}, {"query": "\ud800"}, {"top_k": 51},
                   {"top_k": True}, {"top_k": 0}, {"chunk_set_ids": [-1]}, {"document_ids": ["x"] * 129}):
        with pytest.raises(ValueError):
            service.search(**({"query": "MLE"} | params))
    assert len(service.search("MLE", 50)) == 50
    assert service.search("MLE", document_ids=["' OR 1=1 --"]) == ()
    monkeypatch.setattr(keyword_search, "QUERY_STEP_INTERVAL", 1)
    monkeypatch.setattr(keyword_search, "MAX_QUERY_VM_STEPS", 40)
    with pytest.raises(KeywordIndexError, match="KEYWORD_QUERY_BUDGET_EXCEEDED"):
        service.search("MLE")
    assert index_state(library.store, ready.chunk_set_id)[0][1] is not None


def test_v5_migration_failure_and_retry_preserve_published_chunks(tmp_path, monkeypatch):
    migrations = library_storage.MIGRATIONS
    with monkeypatch.context() as patch:
        patch.setattr(library_storage, "MIGRATIONS", migrations[:5])
        patch.setattr(library_storage, "SCHEMA_VERSION", 5)
        store = KnowledgeStore(tmp_path)
        old_library = LibraryService(store, uuid4())
        try:
            _, ready = publish_chunks(old_library, "迁移样本最大似然估计")
        finally:
            old_library.close()
    expected, = DocumentChunkingService(store).results.iter_chunks(ready.chunk_set_id)
    with monkeypatch.context() as patch:
        patch.setattr(library_storage, "MIGRATIONS", (*migrations[:5], (*migrations[5], "INVALID SQL")))
        with pytest.raises(sqlite3.OperationalError, match="syntax error"):
            KnowledgeStore(tmp_path)
    with store.connect() as db:
        assert db.execute("PRAGMA user_version").fetchone() == (5,)
        assert db.execute("SELECT name FROM sqlite_schema WHERE name GLOB 'keyword_*'").fetchall() == []
    upgraded = KnowledgeStore(tmp_path)
    with upgraded.connect() as db:
        assert db.execute("PRAGMA user_version").fetchone() == (library_storage.SCHEMA_VERSION,)
    service = KeywordSearchService(upgraded)
    assert service.index.index_pending() == (ready.chunk_set_id,)
    assert service.search("似然")[0].chunk_id == expected.chunk_id
    assert DocumentChunkingService(upgraded).results.get_chunk(expected.chunk_id) == expected


@pytest.mark.anyio
async def test_sidecar_keyword_path_issues_no_siliconflow_requests(tmp_path):
    def unexpected(_):
        pytest.fail("keyword path contacted SiliconFlow")
    provider = SiliconFlowEmbedding(transport=httpx.MockTransport(unexpected))
    provider.configure(SecretStr("offline-test-key"), SILICONFLOW_BASE_URL)
    app = create_app(uuid4().hex, uuid4().hex, database_path=tmp_path / "business.sqlite3", embeddings=provider)
    async with app.router.lifespan_context(app):
        book, ready = publish_chunks(app.state.library, "Python 概率")
        service = app.state.keyword_search
        service.index.ensure_indexed(ready.chunk_set_id)
        assert service.search("Python 概率")[0].document_id == book.document_id
        assert app.state.library.list_documents()[0].management_status == "imported"
        with service.store.connect() as db:
            assert db.execute("SELECT count(*) FROM embeddings").fetchone() == (0,)
    assert app.state.keyword_search is None
