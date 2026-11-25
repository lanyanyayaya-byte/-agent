import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest

from app.chunking.models import ChunkerConfig
from app.parsing.models import DocumentParseError
from app.services import document_chunking, document_parsing
from app.services.document_chunking import DocumentChunkingService, iter_stored_chunks
from app.services.document_parsing import DocumentParsingService
from app.storage import chunks as storage
from app.storage import library as library_storage
from app.storage.library import KnowledgeStore
from test_chunking import assert_mapping, stored
from test_library_management import import_file, library, original
from test_pdf_parsing import make_pdf, page, text

CONFIG = ChunkerConfig(53, 10)
MARKDOWN = ("# 主章\n\n## 一节\n\n小段一。\n\n小段二。\n\n" + "中English x² + 𝛑 = π e\u0301 🌍。" * 32).encode()


def setup_book(library, data=MARKDOWN):
    book = import_file(library, data)
    parsed = DocumentParsingService(library.store).ensure_parsed(book.document_version_id)
    return book, parsed, DocumentChunkingService(library.store)


def test_round_trip_of_unicode_spans_and_reuse_after_restart(library, monkeypatch):
    _, parsed, service = setup_book(library)
    parsed_id = parsed.parsed_document_id
    blocks = service.parses.list_blocks(parsed_id).blocks
    expected = tuple(iter_stored_chunks(service.parses, parsed, config=CONFIG, page_size=1))
    chunk_set = service.ensure_chunked(parsed_id, CONFIG)
    assert (chunk_set.chunk_count, chunk_set.source_span_count) == (
        len(expected), sum(len(chunk.source_spans) for chunk in expected),
    )
    assert chunk_set.created_at and chunk_set.published_at
    assert service.get_chunk_set(parsed_id, CONFIG) == chunk_set
    assert service.get_parse_state(chunk_set) == "current"

    def unexpected(*args, **kwargs):
        pytest.fail("reused chunks called the chunker or opened the parse's blocks")

    monkeypatch.setattr(document_chunking, "iter_stored_chunks", unexpected)
    monkeypatch.setattr(document_chunking.ParsedDocumentStore, "list_blocks", unexpected)
    private_copy = original(library, MARKDOWN)
    private_copy.rename(library.store.path.parent / "missing-original")
    restarted = DocumentChunkingService(KnowledgeStore(library.store.path.parent.parent))
    assert restarted.ensure_chunked(parsed_id, CONFIG) == chunk_set
    results = restarted.results
    first = results.list_chunks(chunk_set.chunk_set_id, limit=2)
    assert first == storage.ChunkPage(expected[:2], 1)
    last = results.list_chunks(chunk_set.chunk_set_id, limit=2, after_index=len(expected) - 2)
    assert last == storage.ChunkPage(expected[-1:], None)
    assert results.list_chunks(chunk_set.chunk_set_id, after_index=len(expected)) == storage.ChunkPage((), None)
    for limit in (1, 3, 100):
        assert tuple(results.iter_chunks(chunk_set.chunk_set_id, page_size=limit)) == expected
    actual = tuple(results.iter_chunks(chunk_set.chunk_set_id))
    assert_mapping(blocks, actual, CONFIG)
    for chunk in actual:
        assert results.get_chunk(chunk.chunk_id) == chunk
        assert results.get_chunk_sources(chunk.chunk_id) == chunk.source_spans
    assert any(span.is_overlap for chunk in actual for span in chunk.source_spans)
    assert any(chunk.title_context == ("主章", "一节") and "一节" not in chunk.text for chunk in actual)
    with library.store.connect() as db:
        assert db.execute("SELECT text_length FROM chunks ORDER BY chunk_index").fetchall() == [
            (len(chunk.text),) for chunk in actual
        ]
        assert db.execute("SELECT count(*) FROM chunk_sets").fetchone() == (1,)

    worker = """
import sys
from pathlib import Path
from app.chunking.models import ChunkerConfig
from app.services import document_chunking
from app.storage.library import KnowledgeStore

def unexpected(*args, **kwargs):
    raise AssertionError('chunking ran during restart')

document_chunking.iter_stored_chunks = unexpected
service = document_chunking.DocumentChunkingService(KnowledgeStore(Path(sys.argv[1])))
ready = service.ensure_chunked(int(sys.argv[2]), ChunkerConfig(53, 10))
for chunk in service.results.iter_chunks(ready.chunk_set_id):
    print(chunk.chunk_id)
"""
    child = subprocess.run([sys.executable, "-E", "-s", "-B", "-c", worker,
                            str(library.store.path.parent.parent), str(parsed_id)],
                           cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=15)
    assert child.returncode == 0, child.stderr
    assert child.stdout.splitlines() == [chunk.chunk_id for chunk in expected]


def test_persists_ordered_multi_block_and_cross_page_pdf_positions(library):
    data = make_pdf([
        page(text(72, 72, "This paragraph continues")),
        page(text(72, 765, "on the second page.") + text(72, 752, "The next line stays here.")),
        page(b""),
    ], labels=b"/PageLabels << /Nums [0 << /S /r >>] >> ")
    parsing, parsed, blocks = stored(library, data, "book.pdf")
    service = DocumentChunkingService(library.store)
    chunk_set = service.ensure_chunked(parsed.parsed_document_id)
    chunk, = service.results.iter_chunks(chunk_set.chunk_set_id)
    assert tuple(iter_stored_chunks(parsing.results, parsed)) == (chunk,)
    assert_mapping(blocks, (chunk,), ChunkerConfig())
    assert len(chunk.source_block_ids) == 2
    assert [(location.physical_page, location.page_label) for location in chunk.source_locations] == [
        (1, "i"), (2, "ii"), (2, "ii"),
    ]
    assert all(location.page_bbox is not None for location in chunk.source_locations)
    assert any(warning.physical_page == 3 for warning in chunk.source.warnings)


def test_empty_parse_publishes_an_empty_set_and_missing_parse_fails(library):
    _, parsed, _ = stored(library, make_pdf([page(b"")]), "blank.pdf")
    service = DocumentChunkingService(library.store)
    chunk_set = service.ensure_chunked(parsed.parsed_document_id)
    assert chunk_set.chunk_count == chunk_set.source_span_count == 0
    assert service.ensure_chunked(parsed.parsed_document_id) == chunk_set
    assert service.results.list_chunks(chunk_set.chunk_set_id) == storage.ChunkPage((), None)
    assert tuple(service.results.iter_chunks(chunk_set.chunk_set_id)) == ()
    with pytest.raises(DocumentParseError, match="PARSED_DOCUMENT_UNAVAILABLE"):
        service.ensure_chunked(999_999)
    assert service.get_chunk_set(999_999) is None


def test_stale_parse_and_config_sets_keep_exact_source_identity(library, monkeypatch):
    book, first, service = setup_book(library)
    initial = service.ensure_chunked(first.parsed_document_id, CONFIG)
    other_config = replace(CONFIG, title_context_strategy="none")
    other = service.ensure_chunked(first.parsed_document_id, other_config)
    assert initial.chunk_set_id != other.chunk_set_id
    assert service.ensure_chunked(first.parsed_document_id, other_config) == other
    initial_chunks = tuple(service.results.iter_chunks(initial.chunk_set_id))
    other_chunks = tuple(service.results.iter_chunks(other.chunk_set_id))
    assert [chunk.text for chunk in initial_chunks] == [chunk.text for chunk in other_chunks]
    assert {chunk.chunk_id for chunk in initial_chunks}.isdisjoint(chunk.chunk_id for chunk in other_chunks)

    monkeypatch.setitem(document_parsing.PARSER_SPECS, "md", replace(first.spec, parser_version="fixture-update"))
    assert service.get_parse_state(initial) == "stale"
    second = DocumentParsingService(library.store).ensure_parsed(book.document_version_id)
    assert service.get_chunk_set(second.parsed_document_id, CONFIG) is None
    new = service.ensure_chunked(second.parsed_document_id, CONFIG)
    assert service.get_parse_state(new) == "current"
    assert service.get_parse_state(initial) == "stale"
    new_chunks = tuple(service.results.iter_chunks(new.chunk_set_id))
    assert all(chunk.source.parse_spec == second.spec and chunk.parsed_document_id == second.parsed_document_id
               for chunk in new_chunks)
    assert {chunk.chunk_id for chunk in new_chunks}.isdisjoint(chunk.chunk_id for chunk in initial_chunks)
    assert tuple(service.results.iter_chunks(initial.chunk_set_id)) == initial_chunks


def test_cascade_removes_only_deleted_textbook_and_never_reuses_old_reference(library):
    book, parsed, service = setup_book(library)
    first = service.ensure_chunked(parsed.parsed_document_id, CONFIG)
    expected = tuple(service.results.iter_chunks(first.chunk_set_id))
    updated = import_file(library, b"# Updated version\n\nUpdated.", document_id=book.document_id)
    other = import_file(library, b"# Independent textbook\n")
    shared = import_file(library, MARKDOWN, document_id=other.document_id)
    parsing = DocumentParsingService(library.store)
    updated_parse = parsing.ensure_parsed(updated.document_version_id)
    shared_parse = parsing.ensure_parsed(shared.document_version_id)
    updated_set = service.ensure_chunked(updated_parse.parsed_document_id, CONFIG)
    shared_set = service.ensure_chunked(shared_parse.parsed_document_id, CONFIG)
    shared_chunks = tuple(service.results.iter_chunks(shared_set.chunk_set_id))
    assert {chunk.chunk_id for chunk in expected}.isdisjoint(chunk.chunk_id for chunk in shared_chunks)
    assert tuple(service.results.iter_chunks(first.chunk_set_id)) == expected
    assert all(chunk.document_version_id == updated.document_version_id
               for chunk in service.results.iter_chunks(updated_set.chunk_set_id))
    with library.store.write() as db:
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            db.execute("UPDATE chunk_sets SET document_version_id = ? WHERE chunk_set_id = ?",
                       (updated.document_version_id, shared_set.chunk_set_id))
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            db.execute("UPDATE chunk_sources SET parsed_document_id = ?, source_block_id = ? WHERE chunk_set_id = ?",
                       (parsed.parsed_document_id, expected[0].source_block_ids[0], shared_set.chunk_set_id))

    library.delete_document(book.document_id)
    assert service.get_parse_state(first) == "unavailable"
    assert service.results.get_chunk(expected[0].chunk_id) is None
    assert service.results.get_chunk_sources(expected[0].chunk_id) is None
    assert service.results.list_chunks(first.chunk_set_id) is None
    assert service.results.list_chunks(updated_set.chunk_set_id) is None
    with pytest.raises(DocumentParseError, match="CHUNK_SET_UNAVAILABLE"):
        list(service.results.iter_chunks(first.chunk_set_id))
    assert tuple(service.results.iter_chunks(shared_set.chunk_set_id)) == shared_chunks
    assert original(library, MARKDOWN).read_bytes() == MARKDOWN
    with library.store.connect() as db:
        assert db.execute("SELECT DISTINCT chunk_set_id FROM chunk_sources").fetchall() == [(shared_set.chunk_set_id,)]
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []

    library.delete_document(other.document_id)
    reimported, reparse, _ = setup_book(library)
    assert reimported.document_id != book.document_id
    rebuilt = service.ensure_chunked(reparse.parsed_document_id, CONFIG)
    assert rebuilt.chunk_set_id > shared_set.chunk_set_id
    assert service.results.get_chunk(expected[0].chunk_id) is None


def test_concurrent_build_is_invisible_and_does_not_lock_library_during_generation(library, monkeypatch):
    _, parsed, service = setup_book(library)
    staged, release = Event(), Event()
    generate = document_chunking.iter_stored_chunks

    def pause(*args, **kwargs):
        staged.set()
        assert release.wait(5)
        yield from generate(*args, **kwargs)

    monkeypatch.setattr(document_chunking, "iter_stored_chunks", pause)
    parsed_id = parsed.parsed_document_id
    with ThreadPoolExecutor(max_workers=1) as worker:
        pending = worker.submit(service.ensure_chunked, parsed_id, CONFIG)
        try:
            assert staged.wait(5)
            with library.store.write(wait=False) as db:
                set_id, = db.execute("SELECT chunk_set_id FROM chunk_sets WHERE status='writing'").fetchone()
            assert service.get_chunk_set(parsed_id, CONFIG) is None
            assert service.results.list_chunks(set_id) is None
            with pytest.raises(DocumentParseError, match="CHUNK_BUILD_BUSY"):
                service.ensure_chunked(parsed_id, CONFIG)
            assert service.results.cleanup_interrupted() == 0
        finally:
            release.set()
        ready = pending.result(timeout=5)
    assert ready.chunk_set_id == set_id
    assert service.ensure_chunked(parsed_id, CONFIG) == ready
    with library.store.connect() as db:
        assert db.execute("SELECT status FROM chunk_sets").fetchall() == [("ready",)]


@pytest.mark.parametrize("failure", ["ABORT", "IGNORE"])
def test_failed_batch_keeps_previous_ready_result_and_retries(library, monkeypatch, failure):
    _, parsed, service = setup_book(library)
    parsed_id = parsed.parsed_document_id
    ready = service.ensure_chunked(parsed_id)
    expected = tuple(service.results.iter_chunks(ready.chunk_set_id))
    monkeypatch.setattr("app.storage.parsed_documents.BATCH_ROWS", 4)
    expression = "ABORT, 'fixture abort'" if failure == "ABORT" else "IGNORE"
    with library.store.write() as db:
        db.execute(f"""CREATE TRIGGER reject_span BEFORE INSERT ON chunk_sources WHEN NEW.chunk_index = 3
                       BEGIN SELECT RAISE({expression}); END""")
    error, message = ((sqlite3.IntegrityError, "fixture abort") if failure == "ABORT"
                      else (DocumentParseError, "CHUNK_RESULT_INVALID"))
    with pytest.raises(error, match=message):
        service.ensure_chunked(parsed_id, CONFIG)
    with library.store.write() as db:
        failed, = db.execute("SELECT chunk_set_id FROM chunk_sets WHERE status='abandoned'").fetchone()
        assert db.execute("SELECT count(*) FROM chunks WHERE chunk_set_id = ? AND chunk_index = 3",
                          (failed,)).fetchone() == (0,)
        assert db.execute("SELECT count(*) FROM chunks WHERE chunk_set_id = ?", (failed,)).fetchone()[0] > 0
        db.execute("DROP TRIGGER reject_span")
    assert service.get_chunk_set(parsed_id, CONFIG) is None
    assert service.results.list_chunks(failed) is None
    assert tuple(service.results.iter_chunks(ready.chunk_set_id)) == expected
    new = service.ensure_chunked(parsed_id, CONFIG)
    assert new.chunk_set_id != failed
    assert service.results.cleanup_interrupted(max_batches=10) > 0
    assert tuple(service.results.iter_chunks(new.chunk_set_id)) == tuple(iter_stored_chunks(service.parses, parsed, config=CONFIG))


def test_hard_exit_rolls_back_open_transaction_and_reclaims_exact_identity(library, tmp_path, monkeypatch):
    _, parsed, _ = setup_book(library)
    worker = """
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from app.chunking.models import ChunkerConfig
from app.services.document_chunking import DocumentChunkingService
from app.storage import parsed_documents
from app.storage.library import KnowledgeStore

store = KnowledgeStore(Path(sys.argv[1]))
parsed_documents.BATCH_ROWS = 3
original_write = store.write
writes = 0

@contextmanager
def crash_mid_write(*, wait=True):
    global writes
    writes += 1
    with original_write(wait=wait) as db:
        yield db
        if writes == 3:
            os._exit(75)

store.write = crash_mid_write
DocumentChunkingService(store).ensure_chunked(int(sys.argv[2]), ChunkerConfig(53, 10))
"""
    child = subprocess.run([sys.executable, "-E", "-s", "-B", "-c", worker, str(tmp_path),
                            str(parsed.parsed_document_id)], cwd=Path(__file__).resolve().parents[1],
                           capture_output=True, timeout=15)
    assert child.returncode == 75, child.stderr.decode()
    restarted = DocumentChunkingService(KnowledgeStore(tmp_path))
    with library.store.connect() as db:
        staging, = db.execute("SELECT chunk_set_id FROM chunk_sets WHERE status='writing'").fetchone()
        assert db.execute("SELECT count(*) FROM chunks").fetchone() == (2,)
        assert db.execute("SELECT count(*) FROM chunk_sources").fetchone() == (1,)
        first_id, = db.execute("SELECT chunk_id FROM chunks WHERE chunk_index=0").fetchone()
    assert restarted.results.get_chunk(first_id) is None
    assert restarted.results.get_chunk_sources(first_id) is None
    assert restarted.results.list_chunks(staging) is None
    assert restarted.results.cleanup_interrupted() == 0
    with library.store.write() as db:
        db.execute("UPDATE chunk_sets SET lease_expires_at = 0")
    recreated = restarted.ensure_chunked(parsed.parsed_document_id, CONFIG)
    assert recreated.chunk_set_id > staging
    assert restarted.results.get_chunk(first_id).chunk_index == 0
    assert restarted.results.list_chunks(staging) is None
    monkeypatch.setattr(storage, "BATCH_ROWS", 1)
    assert restarted.results.cleanup_interrupted() == 1
    with library.store.connect() as db:
        assert db.execute("SELECT count(*) FROM chunk_sources WHERE chunk_set_id = ?", (staging,)).fetchone() == (0,)
        assert db.execute("SELECT count(*) FROM chunks WHERE chunk_set_id = ?", (staging,)).fetchone() == (2,)
    assert restarted.results.cleanup_interrupted(max_batches=10) == 3
    with library.store.connect() as db:
        assert db.execute("SELECT chunk_set_id, status FROM chunk_sets").fetchall() == [(recreated.chunk_set_id, "ready")]
        assert db.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize("interruption", ["expired", "deleted"])
def test_lost_lease_cannot_publish(library, monkeypatch, interruption):
    book, parsed, service = setup_book(library)
    staged, resume = Event(), Event()
    publish = service.results._publish

    def pause_first(set_id, *args):
        if not staged.is_set():
            staged.set()
            assert resume.wait(5)
        return publish(set_id, *args)

    monkeypatch.setattr(service.results, "_publish", pause_first)
    with ThreadPoolExecutor(max_workers=1) as worker:
        pending = worker.submit(service.ensure_chunked, parsed.parsed_document_id, CONFIG)
        try:
            assert staged.wait(5)
            if interruption == "deleted":
                library.delete_document(book.document_id)
            else:
                with library.store.write() as db:
                    db.execute("UPDATE chunk_sets SET lease_expires_at = 0 WHERE status = 'writing'")
                winner = service.ensure_chunked(parsed.parsed_document_id, CONFIG)
        finally:
            resume.set()
        with pytest.raises(DocumentParseError, match="CHUNK_WRITE_INTERRUPTED"):
            pending.result(timeout=5)
    if interruption == "deleted":
        assert service.get_chunk_set(parsed.parsed_document_id, CONFIG) is None
    else:
        assert service.get_chunk_set(parsed.parsed_document_id, CONFIG) == winner
    with library.store.connect() as db:
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []


def test_v4_migration_rolls_back_to_v3_and_preserves_old_parse(tmp_path, monkeypatch):
    full_migrations = library_storage.MIGRATIONS
    with monkeypatch.context() as patch:
        patch.setattr(library_storage, "MIGRATIONS", full_migrations[:3])
        patch.setattr(library_storage, "SCHEMA_VERSION", 3)
        old = KnowledgeStore(tmp_path)
        with old.write() as db:
            db.execute("INSERT INTO documents (document_id, name, format) VALUES ('book', '旧教材', 'md')")
            db.execute("INSERT INTO content_blobs (sha256, size_bytes) VALUES (?, 1)", ("a" * 64,))
            db.execute("INSERT INTO document_versions (document_version_id, document_id, content_sha256, size_bytes) "
                       "VALUES ('version', 'book', ?, 1)", ("a" * 64,))
            db.execute("""INSERT INTO parsed_documents (document_version_id, parser_version, structure_version,
                           config_key, parsed_at, status, published_at, warnings)
                           VALUES ('version', '1', 1, 'commonmark-table', 'old', 'ready', 'old', '[]')""")
            db.execute("""INSERT INTO parsed_blocks (parsed_document_id, position, block_id, kind, text, section_path)
                           VALUES (1, 0, 'old-block', 'paragraph', '原文 𝛑', '[]')""")
            db.execute("""INSERT INTO source_locations VALUES
                           (1, 0, 0, 'md', '{"start_line":1,"end_line":1}')""")

    with monkeypatch.context() as patch:
        patch.setattr(library_storage, "MIGRATIONS", (*full_migrations[:3], (*full_migrations[3], "INVALID SQL")))
        with pytest.raises(sqlite3.OperationalError, match="syntax error"):
            KnowledgeStore(tmp_path)
    with old.connect() as db:
        assert db.execute("PRAGMA user_version").fetchone() == (3,)
        assert db.execute("SELECT name FROM sqlite_schema WHERE name LIKE 'chunk%' OR name='parse_version_identity'").fetchall() == []

    with monkeypatch.context() as patch:
        patch.setattr(library_storage, "MIGRATIONS", full_migrations[:4])
        patch.setattr(library_storage, "SCHEMA_VERSION", 4)
        store = KnowledgeStore(tmp_path)
    service = DocumentChunkingService(store)
    ready = service.ensure_chunked(1)
    chunk, = service.results.iter_chunks(ready.chunk_set_id)
    assert chunk.text == "原文 𝛑" and chunk.source_block_ids == ("old-block",)
    assert [(location.start_line, location.end_line) for location in chunk.source_locations] == [(1, 1)]
    with store.connect() as db:
        assert db.execute("PRAGMA user_version").fetchone() == (4,)
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
