import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from secrets import token_urlsafe
from threading import Barrier, Event

import httpx
import pytest
from anyio import to_thread

from app.main import create_app
from app.models.deepseek import DeepSeekAdapter, DeepSeekSettings
from app.parsing.models import DocumentParseError
from app.services import document_parsing as parsing
from app.services.document_parsing import DocumentParsingService
from app.services.models import ModelService
from app.storage import parsed_documents as storage
from app.storage.library import KnowledgeStore
from test_chat import ProviderStream, configure, frames, new_session, payload, run_path
from test_docx_parsing import repack, textbook
from test_library_management import PDF, anyio_backend, import_file, library, original
from test_pdf_parsing import make_pdf, page, text

MARKDOWN = "前言\n\n# 第一章\n\n## 一节\n\n```python\nprint('教材')\n```\n\n正文。\n".encode()


def assert_round_trip(service, result, expected):
    parsed_id = result.parsed_document_id
    assert (result.document_id, result.document_version_id, result.format, result.warnings) == (
        expected.document_id, expected.document_version_id, expected.format, expected.warnings,
    )
    assert service.results.get_sections(parsed_id) == expected.sections
    offset = -1
    for index in range(0, len(expected.blocks), 2):
        page = service.results.list_blocks(parsed_id, limit=2, after_position=offset)
        assert page.blocks == expected.blocks[index:index + 2]
        for block in page.blocks:
            assert service.results.get_block(parsed_id, block.block_id) == block
        offset = page.next_after_position
    assert offset is None
    assert service.results.get_block(parsed_id, "unknown-block") is None


def assert_restart(library, result, expected, data, monkeypatch):
    def unexpected_parser(*args, **kwargs):
        pytest.fail("reading a stored result called a parser")

    for format in ("markdown", "pdf", "docx"):
        monkeypatch.setattr(parsing, f"parse_{format}_version", unexpected_parser)
    private_copy = original(library, data)
    private_copy.rename(library.store.path.parent / "unavailable-source")
    restarted = DocumentParsingService(KnowledgeStore(library.store.path.parent.parent))
    version_id = result.document_version_id
    assert restarted.get_current(version_id) == restarted.ensure_parsed(version_id) == result
    assert_round_trip(restarted, result, expected)
    return restarted


def test_markdown_restart_reads_complete_result_without_original_or_parsing(library, monkeypatch):
    data = (
        "前言\r\n\r\n# 第一章\r\r\n## 一节\r\n\r\n"
        "```python\nprint('教材')\n```\n\n正文。\n\n- 列表\n\n| 列 |\n| --- |\n| 值 |\n\n正文。\n"
    ).encode()
    imported = import_file(library, data)
    version_id = imported.document_version_id
    service = DocumentParsingService(library.store)
    assert service.get_status(version_id).state == "missing"
    expected = parsing.parse_markdown_version(library.store, document_id=imported.document_id,
                                              document_version_id=version_id)
    result = service.ensure_parsed(version_id)
    assert result.parsed_at and result.published_at
    assert service.get_status(version_id).state == "ready"
    assert expected.blocks[3].section_path == ("第一章", "一节")
    assert expected.blocks[3].code_language == "python"
    assert (expected.blocks[3].source_locations[0].start_line, expected.blocks[3].source_locations[0].end_line) == (7, 9)
    assert [(block.kind, block.source_locations[0].start_line, block.source_locations[0].end_line)
            for block in expected.blocks[4:]] == [
        ("paragraph", 11, 11), ("list", 13, 13), ("table", 15, 17), ("paragraph", 19, 19),
    ]
    assert expected.blocks[4].text == expected.blocks[7].text == "正文。"
    assert expected.blocks[4].block_id != expected.blocks[7].block_id
    assert_round_trip(service, result, expected)
    restarted = assert_restart(library, result, expected, data, monkeypatch)
    assert restarted.results.list_blocks(result.parsed_document_id, after_position=100).blocks == ()
    with library.store.connect() as db:
        assert db.execute("SELECT status FROM parsed_documents").fetchall() == [("ready",)]


def test_pdf_round_trip_keeps_labels_boxes_locations_and_warnings(library, monkeypatch):
    pair = text(72, 720, "First line of a paragraph.") + text(72, 707, "Second line of the same block.")
    data = make_pdf(
        [page(pair + text(72, 72, "This paragraph continues")),
         page(text(72, 765, "on the second page.") + pair), page(b""), page(text(72, 720, "Appendix"))],
        labels=b"/PageLabels << /Nums [0 << /S /r >> 3 << /P (A-) /S /D /St 7 >>] >> ",
    )
    book = import_file(library, data, "pages.pdf")
    expected = parsing.parse_pdf_version(library.store, document_id=book.document_id,
                                         document_version_id=book.document_version_id)
    first, across_first, across_second, repeat, last = expected.blocks
    assert len(first.source_locations) == 2
    assert first.source_locations[0].page_label == "i"
    assert first.source_locations[0].page_bbox[0] == 72
    assert first.text == repeat.text and first.block_id != repeat.block_id
    assert [(block.text, block.source_locations[0].physical_page) for block in (across_first, across_second)] == [
        ("This paragraph continues", 1), ("on the second page.", 2),
    ]
    assert last.source_locations[0].page_label == "A-7"
    assert last.source_locations[0].physical_page == 4
    assert any(warning.physical_page == 3 for warning in expected.warnings)
    service = DocumentParsingService(library.store)
    result = service.ensure_parsed(book.document_version_id)
    assert_round_trip(service, result, expected)
    assert_restart(library, result, expected, data, monkeypatch)


def test_docx_round_trip_keeps_nested_sections_paragraphs_and_table_cells(library, textbook, monkeypatch):
    book = import_file(library, textbook, "chapters.docx")
    expected = parsing.parse_docx_version(library.store, document_id=book.document_id,
                                          document_version_id=book.document_version_id)
    table = next(block for block in expected.blocks if block.kind == "table")
    assert len(table.source_locations) == 5
    cell = table.source_locations[0]
    assert (cell.body_index, cell.table_row, cell.table_column, cell.row_span, cell.column_span) == (7, 1, 1, 2, 2)
    assert expected.blocks[2].source_locations[0].paragraph_number == 3
    assert expected.sections[1].parent_heading_block_id == expected.sections[0].heading_block_id
    service = DocumentParsingService(library.store)
    result = service.ensure_parsed(book.document_version_id)
    assert_round_trip(service, result, expected)
    assert_restart(library, result, expected, textbook, monkeypatch)


def test_pdf_with_warnings_and_no_blocks_is_a_complete_result(library):
    version_id = import_file(library, make_pdf([page(b"")]), "blank.pdf").document_version_id
    service = DocumentParsingService(library.store)
    result = service.ensure_parsed(version_id)
    assert service.get_current(version_id) == result
    assert service.get_status(version_id).state == "ready"
    assert [(warning.code, warning.physical_page) for warning in result.warnings] == [("PDF_NO_TEXT_LAYER", 1)]
    assert service.results.get_sections(result.parsed_document_id) == ()
    assert service.results.list_blocks(result.parsed_document_id) == storage.BlockPage((), None)


@pytest.mark.parametrize("changes", [{"parser_version": "2"}, {"structure_version": 2}, {"config_key": "alternate"}])
def test_changed_spec_is_stale_and_regeneration_retains_old_parse(library, monkeypatch, changes):
    version_id = import_file(library, MARKDOWN).document_version_id
    service = DocumentParsingService(library.store)
    first = service.ensure_parsed(version_id)
    updated_spec = replace(first.spec, **changes)
    monkeypatch.setitem(parsing.PARSER_SPECS, "md", updated_spec)
    assert service.get_status(version_id) == storage.ParseAvailability("stale", first)
    assert service.get_current(version_id) == first
    second = service.ensure_parsed(version_id)
    assert second.spec == updated_spec
    assert second.parsed_document_id != first.parsed_document_id
    assert service.get_status(version_id) == storage.ParseAvailability("ready", second)
    assert service.ensure_parsed(version_id) == second
    assert service.results.list_blocks(first.parsed_document_id) == service.results.list_blocks(second.parsed_document_id)
    monkeypatch.setitem(parsing.PARSER_SPECS, "md", first.spec)
    assert service.get_current(version_id) == first


def test_failed_reparse_preserves_previous_result_and_retry(library, monkeypatch):
    version_id = import_file(library, MARKDOWN).document_version_id
    service = DocumentParsingService(library.store)
    first = service.ensure_parsed(version_id)
    monkeypatch.setitem(parsing.PARSER_SPECS, "md", replace(first.spec, parser_version="2"))
    with library.store.write() as db:
        db.execute("""CREATE TRIGGER reject_location BEFORE INSERT ON source_locations
                    BEGIN SELECT RAISE(ABORT, 'fixture abort'); END""")
    with pytest.raises(sqlite3.IntegrityError, match="fixture abort"):
        service.ensure_parsed(version_id)
    assert service.get_status(version_id) == storage.ParseAvailability("stale", first)
    with library.store.write() as db:
        interrupted, = db.execute("SELECT parsed_document_id FROM parsed_documents WHERE lease_expires_at = 0").fetchone()
        db.execute("DROP TRIGGER reject_location")
    assert service.results.list_blocks(interrupted) is None
    assert service.results.list_blocks(first.parsed_document_id).blocks
    assert service.results.cleanup_interrupted(max_batches=10) > 0
    second = service.ensure_parsed(version_id)
    assert second.parsed_document_id > interrupted
    assert service.get_current(version_id) == second
    assert service.results.get_sections(first.parsed_document_id)


def test_hard_exit_keeps_staged_rows_invisible_and_cleanup_is_bounded(library, tmp_path, monkeypatch):
    version_id = import_file(library, MARKDOWN).document_version_id
    worker = '''
import os
import sys
from pathlib import Path
from app.services.document_parsing import DocumentParsingService
from app.storage.library import KnowledgeStore

service = DocumentParsingService(KnowledgeStore(Path(sys.argv[1])))
service.results._publish = lambda *args: os._exit(75)
service.ensure_parsed(sys.argv[2])
'''
    child = subprocess.run([sys.executable, "-E", "-s", "-B", "-c", worker, str(tmp_path), version_id],
                           cwd=Path(__file__).resolve().parents[1], capture_output=True, timeout=10)
    assert child.returncode == 75, child.stderr.decode()
    store = KnowledgeStore(tmp_path)
    service = DocumentParsingService(store)
    with store.connect() as db:
        parsed_id, = db.execute("SELECT parsed_document_id FROM parsed_documents WHERE status='writing'").fetchone()
        assert db.execute("SELECT count(*) FROM source_locations").fetchone() == (5,)
    assert service.get_status(version_id).state == "missing"
    assert service.get_current(version_id) is None
    assert service.results.get_sections(parsed_id) is None
    assert service.results.list_blocks(parsed_id) is None
    assert service.results.get_block(parsed_id, f"{version_id}:md:1:1") is None
    assert service.results.cleanup_interrupted() == 0

    with store.write() as db:
        db.execute("UPDATE parsed_documents SET lease_expires_at = 0")
    monkeypatch.setattr(storage, "BATCH_ROWS", 1)
    assert service.results.cleanup_interrupted() == 1
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM source_locations").fetchone() == (4,)
    assert service.results.cleanup_interrupted(max_batches=20) == 12
    with store.connect() as db:
        for table in ("parsed_documents", "parsed_sections", "parsed_blocks", "source_locations"):
            assert db.execute(f"SELECT count(*) FROM {table}").fetchone() == (0,)
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
    assert service.ensure_parsed(version_id).parsed_document_id != parsed_id


def test_simultaneous_saves_only_publish_one_result(library, monkeypatch):
    version_id = import_file(library, MARKDOWN).document_version_id
    service = DocumentParsingService(library.store)
    both_staged = Barrier(2, timeout=5)
    publish = service.results._publish

    def publish_together(*args):
        assert service.get_current(version_id) is None
        both_staged.wait()
        return publish(*args)

    monkeypatch.setattr(service.results, "_publish", publish_together)
    with ThreadPoolExecutor(max_workers=2) as threads:
        results = [threads.submit(service.ensure_parsed, version_id) for _ in range(2)]
        first, second = (future.result(timeout=8) for future in results)
    assert first == second
    assert service.results.cleanup_interrupted(max_batches=5) > 0
    with library.store.connect() as db:
        assert db.execute("SELECT parsed_document_id, status FROM parsed_documents").fetchall() == [
            (first.parsed_document_id, "ready"),
        ]


def test_parsing_holds_no_write_lock_and_deleted_version_cannot_publish(library, monkeypatch):
    book = import_file(library, MARKDOWN)
    service = DocumentParsingService(library.store)
    parsed = Event()
    resume = Event()
    parse = parsing.parse_markdown_version

    def wait_after_parse(*args, **kwargs):
        result = parse(*args, **kwargs)
        parsed.set()
        assert resume.wait(5)
        return result

    monkeypatch.setattr(parsing, "parse_markdown_version", wait_after_parse)
    with ThreadPoolExecutor(max_workers=1) as threads:
        future = threads.submit(service.ensure_parsed, book.document_version_id)
        try:
            assert parsed.wait(5)
            with library.store.write(wait=False):
                pass
            library.delete_document(book.document_id)
        finally:
            resume.set()
        with pytest.raises(DocumentParseError, match="DOCUMENT_VERSION_NOT_FOUND"):
            future.result(timeout=5)
    assert service.get_current(book.document_version_id) is None


def test_document_versions_and_shared_copies_have_independent_parses(library):
    first = import_file(library, MARKDOWN)
    service = DocumentParsingService(library.store)
    first_parse = service.ensure_parsed(first.document_version_id)
    update = import_file(library, b"# Updated\n", document_id=first.document_id)
    assert service.get_current(update.document_version_id) is None
    updated_parse = service.ensure_parsed(update.document_version_id)
    other = import_file(library, b"# Other textbook\n")
    shared = import_file(library, MARKDOWN, document_id=other.document_id)
    shared_parse = service.ensure_parsed(shared.document_version_id)
    assert len({first_parse.parsed_document_id, updated_parse.parsed_document_id, shared_parse.parsed_document_id}) == 3
    assert service.results.list_blocks(updated_parse.parsed_document_id).blocks[0].text == "Updated"
    assert service.get_current(first.document_version_id) == first_parse
    assert service.results.list_blocks(first_parse.parsed_document_id).blocks[0].text == "前言"
    historical_block = service.results.list_blocks(first_parse.parsed_document_id).blocks[0]
    assert service.results.get_block(updated_parse.parsed_document_id, historical_block.block_id) is None
    assert service.results.get_block(shared_parse.parsed_document_id, historical_block.block_id) is None

    library.delete_document(first.document_id)
    assert service.get_current(first.document_version_id) is None
    assert service.results.get_block(first_parse.parsed_document_id, historical_block.block_id) is None
    assert service.results.list_blocks(updated_parse.parsed_document_id) is None
    assert service.get_current(shared.document_version_id) == shared_parse
    assert original(library, MARKDOWN).read_bytes() == MARKDOWN
    with library.store.connect() as db:
        assert db.execute("SELECT document_version_id FROM parsed_documents").fetchall() == [(shared.document_version_id,)]
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
    library.delete_document(other.document_id)
    assert not original(library, MARKDOWN).exists()

    reimported = import_file(library, MARKDOWN)
    reimported_parse = service.ensure_parsed(reimported.document_version_id)
    assert reimported.document_id != first.document_id
    assert reimported.document_version_id != first.document_version_id
    assert reimported_parse.parsed_document_id > shared_parse.parsed_document_id
    assert service.results.get_block(reimported_parse.parsed_document_id, historical_block.block_id) is None


@pytest.mark.anyio
async def test_failed_ensure_parsed_in_all_formats_keeps_published_knowledge_and_chat(tmp_path, textbook):
    ui_token, host_token = token_urlsafe(32), token_urlsafe(32)
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, stream=ProviderStream()))
    models = ModelService(DeepSeekAdapter(DeepSeekSettings(model="fixture-model"), transport=transport))
    app = create_app(ui_token, host_token, database_path=tmp_path / "business.db", models=models)
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1",
        headers={"Authorization": f"Bearer {ui_token}"},
    ) as client:
        library = app.state.library
        service = DocumentParsingService(library.store)
        clean = import_file(library, MARKDOWN)
        ready = await to_thread.run_sync(service.ensure_parsed, clean.document_version_id)
        expected = parsing.parse_markdown_version(
            library.store, document_id=clean.document_id, document_version_id=clean.document_version_id,
        )
        await configure(client, {"Authorization": f"Bearer {host_token}"})
        session = await new_session(client)
        for filename, data, code in (
            ("damaged.pdf", PDF, "PDF_INVALID"),
            ("damaged.docx", repack(textbook, {"word/document.xml": b"<damaged>"}), "DOCX_INVALID"),
            ("deeply-nested.md", b"- " * 40 + b"item", "MARKDOWN_NESTING_LIMIT"),
        ):
            bad = import_file(library, data, filename)
            with pytest.raises(DocumentParseError, match=code):
                await to_thread.run_sync(service.ensure_parsed, bad.document_version_id)
            assert service.get_status(bad.document_version_id) == storage.ParseAvailability("missing", None)
            assert service.get_current(clean.document_version_id) == ready
            assert_round_trip(service, ready, expected)
        assert original(library, MARKDOWN).read_bytes() == MARKDOWN
        with library.store.connect() as db:
            assert db.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
            assert db.execute("PRAGMA foreign_key_check").fetchall() == []
            assert db.execute("SELECT status FROM parsed_documents").fetchall() == [("ready",)]
        response = await client.post(run_path(session), json=payload())
        assert response.status_code == 200
        events = [item async for item in frames(response) if isinstance(item, dict)]
        assert events[-1]["type"] == "run_completed"
