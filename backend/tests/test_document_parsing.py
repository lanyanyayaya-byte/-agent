import codecs
from secrets import token_urlsafe

import httpx
import pytest

from app.main import create_app
from app.models.deepseek import DeepSeekAdapter, DeepSeekSettings
from app.parsing import markdown
from app.parsing.models import (
    ContentBlock, DocumentParseError, DocxSourceLocation, MarkdownSourceLocation, PdfSourceLocation,
)
from app.services.document_parsing import parse_markdown_version
from app.services.models import ModelService
from test_chat import ProviderStream, anyio_backend, configure, frames, new_session, payload, run_path
from test_library_management import PDF, import_file, library, original


def parse_import(library, imported):
    return parse_markdown_version(
        library.store, document_id=imported.document_id, document_version_id=imported.document_version_id,
    )


def test_parse_private_copy_with_sections_and_exact_original_lines(library, tmp_path):
    source_lines = [
        "", "前言 Intro", "", "# 第一章 Course #", "", "混合 **Markdown** 正文。", "Second `line` in English.", "",
        "### 深入 Details", "", "```python extra", "# literal heading", 'print("你好 world")', "```", "",
        "- 中文 item", "  1. nested item", "", "| 名称 | Name |", "| --- | --- |", r"| 例子 | A \| B |", "",
        "练习 Exercises", "---------------", "", "> # Quote heading", "> 引用 quote", "", "---", "",
        "# 附录 Appendix", "Last line 甲\u2028乙",
    ]
    source = "\r\n".join(source_lines[:5]) + "\r" + "\n".join(source_lines[5:])
    content = codecs.BOM_UTF8 + source.encode()
    selected_file = tmp_path / "教材.md"
    selected_file.write_bytes(content)
    imported = import_file(library, selected_file.read_bytes(), selected_file.name)
    selected_file.rename(tmp_path / "moved.md")

    parsed = parse_import(library, imported)
    assert (parsed.document_id, parsed.document_version_id, parsed.format) == (
        imported.document_id, imported.document_version_id, "md",
    )
    root, detail, exercises, appendix = parsed.sections
    assert [(s.title, s.level, s.parent_heading_block_id) for s in parsed.sections] == [
        ("第一章 Course", 1, None),
        ("深入 Details", 3, root.heading_block_id),
        ("练习 Exercises", 2, root.heading_block_id),
        ("附录 Appendix", 1, None),
    ]
    assert [(b.kind, b.source_locations) for b in parsed.blocks] == [
        ("paragraph", (MarkdownSourceLocation(2, 2),)),
        ("heading", (MarkdownSourceLocation(4, 4, 1),)),
        ("paragraph", (MarkdownSourceLocation(6, 7),)),
        ("heading", (MarkdownSourceLocation(9, 9, 3),)),
        ("code", (MarkdownSourceLocation(11, 14),)),
        ("list", (MarkdownSourceLocation(16, 17),)),
        ("table", (MarkdownSourceLocation(19, 21),)),
        ("heading", (MarkdownSourceLocation(23, 24, 2),)),
        ("blockquote", (MarkdownSourceLocation(26, 27),)),
        ("thematic_break", (MarkdownSourceLocation(29, 29),)),
        ("heading", (MarkdownSourceLocation(31, 31, 1),)),
        ("paragraph", (MarkdownSourceLocation(32, 32),)),
    ]
    preamble, heading, paragraph, _, code, items, table, _, quote, _, _, last = parsed.blocks
    assert preamble.text == "前言 Intro" and preamble.section_path == ()
    assert preamble.section_heading_block_id is None
    assert heading.block_id == root.heading_block_id
    assert paragraph.text == "混合 **Markdown** 正文。\nSecond `line` in English."
    assert paragraph.section_path == ("第一章 Course",)
    assert code.text == '# literal heading\nprint("你好 world")\n' and code.code_language == "python"
    assert code.section_heading_block_id == detail.heading_block_id
    assert items.text == "\n".join(source_lines[15:17])
    assert table.text == "\n".join(source_lines[18:21])
    assert table.section_path == ("第一章 Course", "深入 Details")
    assert quote.text == "\n".join(source_lines[25:27])
    assert quote.section_heading_block_id == exercises.heading_block_id
    assert last.text == "Last line 甲\u2028乙" and last.section_path == ("附录 Appendix",)
    assert last.section_heading_block_id == appendix.heading_block_id
    assert (tmp_path / "moved.md").read_bytes() == original(library, content).read_bytes() == content


def test_block_ids_distinguish_repeated_text_sections_and_versions(library):
    content = "# 重名 Title\n\n重复 Same\n\n重复 Same\n\n# 重名 Title\n\n重复 Same\n".encode()
    first = import_file(library, content)
    second = import_file(library, b"# Edition 2\n", document_id=first.document_id)
    reverted = import_file(library, content, document_id=first.document_id)

    old = parse_import(library, first)
    new = parse_import(library, reverted)
    assert old == parse_import(library, first)
    assert parse_import(library, second).blocks[0].text == "Edition 2"
    assert [b.text for b in old.blocks] == [b.text for b in new.blocks]
    assert len({b.block_id for b in (*old.blocks, *new.blocks)}) == 10
    for parsed in (old, new):
        left, right = parsed.sections
        assert [b.document_version_id for b in parsed.blocks] == [parsed.document_version_id] * 5
        assert [b.section_heading_block_id for b in parsed.blocks] == (
            [left.heading_block_id] * 3 + [right.heading_block_id] * 2
        )
        assert [b.source_locations[0].start_line for b in parsed.blocks] == [1, 3, 5, 7, 9]


def test_indented_and_unclosed_fenced_code_keep_literal_text_and_line_ranges(library):
    imported = import_file(library, b"    # code\n\n1. ordered\n\n~~~\n# still code\n\n")
    parsed = parse_import(library, imported)
    assert parsed.sections == ()
    assert [(b.kind, b.text, b.source_locations[0]) for b in parsed.blocks] == [
        ("code", "# code\n", MarkdownSourceLocation(1, 1)),
        ("list", "1. ordered", MarkdownSourceLocation(3, 3)),
        ("code", "# still code\n\n", MarkdownSourceLocation(5, 7)),
    ]
    empty = parse_import(library, import_file(library, b"\n\t\n"))
    assert empty.sections == empty.blocks == ()

    nested = parse_import(library, import_file(library, b"- item\n  ```\n  code\n  \n"))
    assert nested.blocks[0].text == "- item\n  ```\n  code\n  "
    assert nested.blocks[0].source_locations == (MarkdownSourceLocation(1, 4),)


def test_unreliable_deep_nesting_fails_without_swallowing_remaining_sections(library):
    content = ("- " * 40 + "item\n\n# Following section\n").encode()
    imported = import_file(library, content)
    with pytest.raises(DocumentParseError, match="MARKDOWN_NESTING_LIMIT"):
        parse_import(library, imported)
    assert original(library, content).read_bytes() == content


def test_shared_models_allow_multiple_locations_and_unknown_fields():
    pdf_locations = (PdfSourceLocation(2, "ii", (10.0, 24.0, 80.0, 50.0)), PdfSourceLocation(3))
    paragraph = ContentBlock("pdf-block", "pdf-version", "paragraph", "Across pages", pdf_locations)
    assert paragraph.source_locations == pdf_locations
    assert pdf_locations[1].page_label is pdf_locations[1].page_bbox is None
    assert DocxSourceLocation(paragraph_number=5).heading_level is None
    assert DocxSourceLocation(heading_level=2).paragraph_number is None
    assert MarkdownSourceLocation().start_line is MarkdownSourceLocation().end_line is None


def test_missing_wrong_and_changed_sources_report_explicit_errors(library):
    content = b"# Original\n"
    imported = import_file(library, content)
    pdf = import_file(library, PDF, "教材.pdf")
    with pytest.raises(DocumentParseError, match="UNSUPPORTED_DOCUMENT_FORMAT"):
        parse_import(library, pdf)
    with pytest.raises(DocumentParseError, match="DOCUMENT_VERSION_NOT_FOUND"):
        parse_markdown_version(library.store, document_id=imported.document_id, document_version_id=pdf.document_version_id)

    private_copy = original(library, content)
    moved = private_copy.with_name("moved-test-copy")
    private_copy.rename(moved)
    with pytest.raises(DocumentParseError, match="MARKDOWN_SOURCE_MISSING"):
        parse_import(library, imported)
    assert not private_copy.exists()
    private_copy.symlink_to(moved)
    with pytest.raises(DocumentParseError, match="MARKDOWN_SOURCE_UNREADABLE"):
        parse_import(library, imported)
    moved.rename(private_copy)
    private_copy.chmod(0o600)
    changed = b"# Impostor\n"
    assert len(changed) == len(content)
    private_copy.write_bytes(changed)
    with pytest.raises(DocumentParseError, match="MARKDOWN_SOURCE_MISMATCH"):
        parse_import(library, imported)
    assert private_copy.read_bytes() == changed
    assert library.get_document(imported.document_id).versions[0].document_version_id == imported.document_version_id


@pytest.mark.anyio
async def test_parser_failure_leaves_the_same_apps_files_and_chat_usable(tmp_path, monkeypatch):
    ui_token, host_token = token_urlsafe(32), token_urlsafe(32)
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, stream=ProviderStream()))
    models = ModelService(DeepSeekAdapter(DeepSeekSettings(model="fixture-model"), transport=transport))
    app = create_app(ui_token, host_token, database_path=tmp_path / "business.db", models=models)

    def fail(*_args):
        raise RuntimeError("injected Markdown library failure")

    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1",
        headers={"Authorization": f"Bearer {ui_token}"},
    ) as client:
        content = b"# Valid Markdown\n\nBody\n"
        selected_file = tmp_path / "original.md"
        selected_file.write_bytes(content)
        library = app.state.library
        imported = import_file(library, selected_file.read_bytes())
        await configure(client, {"Authorization": f"Bearer {host_token}"})
        session = await new_session(client)

        with monkeypatch.context() as patch:
            patch.setattr(markdown.MarkdownIt, "parse", fail)
            with pytest.raises(DocumentParseError, match="MARKDOWN_PARSE_FAILED"):
                parse_import(library, imported)
            response = await client.post(run_path(session), json=payload())
            assert response.status_code == 200
            events = [item async for item in frames(response) if isinstance(item, dict)]
            assert events[-1]["type"] == "run_completed"

        assert selected_file.read_bytes() == original(library, content).read_bytes() == content
        assert parse_import(library, imported).blocks[-1].text == "Body"
