import csv
import hashlib
from collections import defaultdict
from dataclasses import replace
from io import StringIO

import pytest
from docx import Document

from app.chunking.hybrid import iter_chunks
from app.chunking.models import ChunkerConfig
from app.parsing.models import ContentBlock, DocumentParseError
from app.services.document_chunking import iter_stored_chunks
from app.services.document_parsing import PARSER_SPECS, DocumentParsingService
from test_docx_parsing import serialize
from test_library_management import import_file, library
from test_pdf_parsing import make_pdf, page, text


def stored(library, data, name):
    book = import_file(library, data, name)
    service = DocumentParsingService(library.store)
    parsed = service.ensure_parsed(book.document_version_id)
    blocks = service.results.list_blocks(parsed.parsed_document_id, limit=1000).blocks
    return service, parsed, blocks


def assert_mapping(blocks, chunks, config):
    by_id = {block.block_id: block for block in blocks}
    new_ranges = defaultdict(list)
    previous_spans = ()
    for index, chunk in enumerate(chunks):
        assert chunk.chunk_index == index and chunk.chunker_config_key == config.key
        assert len(chunk.text) <= config.chunk_size
        assert any(not span.is_overlap for span in chunk.source_spans)
        assert chunk.source_locations == tuple(
            location for block_id in chunk.source_block_ids for location in by_id[block_id].source_locations
        )
        seen_fresh = False
        end = 0
        for span in chunk.source_spans:
            block = by_id[span.block_id]
            assert (chunk.section_heading_block_id, chunk.section_path) == (
                block.section_heading_block_id, block.section_path,
            )
            assert span.chunk_start >= end
            assert chunk.text[end:span.chunk_start] in ("", "\n\n")
            assert chunk.text[span.chunk_start:span.chunk_end] == block.text[span.block_start:span.block_end]
            assert 0 <= span.block_start < span.block_end <= len(block.text)
            assert span.source_locations == block.source_locations and span.location_granularity == "block"
            if span.is_overlap:
                assert not seen_fresh and span.chunk_end <= config.chunk_overlap
                assert any(prior.block_id == span.block_id and not prior.is_overlap
                           and prior.block_start <= span.block_start < span.block_end <= prior.block_end
                           for prior in previous_spans)
            else:
                seen_fresh = True
                new_ranges[span.block_id].append((span.block_start, span.block_end))
            end = span.chunk_end
        assert end == len(chunk.text)
        previous_spans = chunk.source_spans

    assert list(new_ranges) == [block.block_id for block in blocks if block.text]
    for block_id, ranges in new_ranges.items():
        assert [start for start, _ in ranges] == [0, *(end for _, end in ranges[:-1])]
        assert ranges[-1][1] == len(by_id[block_id].text)


def test_markdown_sections_paragraphs_and_real_overlap(library):
    sentence = "中英混排 x² = π。"
    long_paragraph = sentence * 22
    data = f"# 主章\n\n## 同名\n\n小段一。\n\n小段二。\n\n{long_paragraph}\n\n## 同名\n\n小段一。\n".encode()
    service, parsed, blocks = stored(library, data, "book.md")
    config = ChunkerConfig(55, 8)
    chunks = list(iter_stored_chunks(service.results, parsed, config=config, page_size=1))
    assert chunks == list(iter_stored_chunks(service.results, parsed, config=config, page_size=7))
    assert chunks == list(iter_chunks(blocks, source=chunks[0].source, config=config))
    assert_mapping(blocks, chunks, config)
    assert all(block.block_id in chunks[1].source_block_ids for block in blocks[2:4])
    assert chunks[-1].section_path == ("主章", "同名")
    assert chunks[-1].section_heading_block_id != chunks[1].section_heading_block_id
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert chunks[0].source.content_sha256 == hashlib.sha256(data).hexdigest()
    long_spans = [span for chunk in chunks for span in chunk.source_spans if span.block_id == blocks[4].block_id]
    assert any(span.is_overlap for span in long_spans)
    assert all(long_paragraph[span.block_end - 1] == "。" for span in long_spans)
    body_chunk = next(chunk for chunk in chunks if chunk.source_block_ids == (blocks[4].block_id,))
    assert body_chunk.title_context == ("主章", "同名") and "同名" not in body_chunk.text

    for policy, context in [("leaf", ("同名",)), ("none", ())]:
        altered = list(iter_stored_chunks(service.results, parsed,
                                          config=replace(config, title_context_strategy=policy)))
        assert altered[body_chunk.chunk_index].title_context == context
        assert [chunk.text for chunk in altered] == [chunk.text for chunk in chunks]
        assert {chunk.chunk_id for chunk in altered}.isdisjoint(chunk.chunk_id for chunk in chunks)


def test_pdf_page_and_pagination_boundaries_are_not_chunk_boundaries(library):
    data = make_pdf([
        page(text(72, 72, "This paragraph continues")),
        page(text(72, 765, "on the second page.") + text(72, 752, "The next line stays here.")),
        page(b""),
    ], labels=b"/PageLabels << /Nums [0 << /S /r >>] >> ")
    service, parsed, blocks = stored(library, data, "book.pdf")
    config = ChunkerConfig(100, 12)
    chunks = list(iter_stored_chunks(service.results, parsed, config=config, page_size=1))
    assert_mapping(blocks, chunks, config)
    chunk, = chunks
    assert [(location.physical_page, location.page_label) for location in chunk.source_locations] == [
        (1, "i"), (2, "ii"), (2, "ii"),
    ]
    assert all(location.page_bbox is not None for location in chunk.source_locations)
    assert any(warning.physical_page == 3 for warning in chunk.source.warnings)


def test_markdown_table_and_code_blocks_split_only_when_needed(library):
    long_line = "x" * 60
    data = f"# Source\n\nParagraph.\n\n| A | B |\n|---|---|\n| C | D |\n\n```py\na = 1\n{long_line}\nz = 2\n```\n".encode()
    service, parsed, blocks = stored(library, data, "structured.md")
    config = ChunkerConfig(35, 6)
    chunks = list(iter_stored_chunks(service.results, parsed, config=config))
    assert_mapping(blocks, chunks, config)
    table, code = blocks[-2:]
    table_chunk, = [chunk for chunk in chunks if table.block_id in chunk.source_block_ids]
    assert table_chunk.text == table.text and table_chunk.source_block_ids == (table.block_id,)
    code_chunks = [chunk for chunk in chunks if code.block_id in chunk.source_block_ids]
    assert code_chunks[0].text == "a = 1\n"
    assert code_chunks[1].text == "x" * 35
    assert "".join(chunk.text for chunk in code_chunks) == code.text
    assert all(chunk.source_block_ids == (code.block_id,) and not chunk.source_spans[0].is_overlap
               for chunk in code_chunks)

    narrow = replace(config, chunk_size=12)
    narrow_chunks = list(iter_stored_chunks(service.results, parsed, config=narrow))
    assert_mapping(blocks, narrow_chunks, narrow)
    assert [chunk.text for chunk in narrow_chunks if table.block_id in chunk.source_block_ids] == (
        table.text.splitlines(keepends=True)
    )


def test_docx_table_keeps_multiline_quoted_rows_and_original_cell_locations(library):
    document = Document()
    document.add_heading("表格", level=1)
    table = document.add_table(rows=3, cols=2)
    values = [["name", "details"], ['含引号"here"\n第二行', "值"], ["尾行", "b" * 20]]
    for row, cells in zip(table.rows, values, strict=True):
        for cell, value in zip(row.cells, cells, strict=True):
            cell.text = value
    service, parsed, blocks = stored(library, serialize(document), "structured.docx")
    config = ChunkerConfig(25, 4)
    chunks = list(iter_stored_chunks(service.results, parsed, config=config))
    assert_mapping(blocks, chunks, config)
    table_chunks = [chunk for chunk in chunks if blocks[-1].block_id in chunk.source_block_ids]
    assert [row for chunk in table_chunks for row in csv.reader(StringIO(chunk.text), delimiter="\t", strict=True)] == values
    assert len(table_chunks) == 3 and all(len(chunk.source_locations) == 6 for chunk in table_chunks)


def test_same_text_ids_are_scoped_to_position_version_parse_and_config(library, monkeypatch):
    data = ("同一正文\n\n" * 2).encode()
    service, parsed, _ = stored(library, data, "first.md")
    other = import_file(library, b"other text", "other.md")
    shared = import_file(library, data, "other.md", other.document_id)
    shared_parse = service.ensure_parsed(shared.document_version_id)
    config = ChunkerConfig(4, 1)

    def read(parse, options=config):
        return list(iter_stored_chunks(service.results, parse, config=options))

    first = read(parsed)
    second = read(shared_parse)
    assert [chunk.text for chunk in first] == [chunk.text for chunk in second] == ["同一正文", "同一正文"]
    assert first[0].source.content_sha256 == second[0].source.content_sha256
    assert len({chunk.chunk_id for chunk in first + second}) == 4
    assert first[0].source_block_ids != first[1].source_block_ids
    assert all(a.chunk_id != b.chunk_id for a, b in zip(first, read(parsed, ChunkerConfig(5, 0)), strict=True))

    monkeypatch.setitem(PARSER_SPECS, "md", replace(parsed.spec, parser_version="fixture-new-profile"))
    assert service.get_status(parsed.document_version_id).state == "stale"
    current = service.ensure_parsed(parsed.document_version_id)
    assert read(parsed) == first
    assert [c.source_block_ids for c in read(current)] == [c.source_block_ids for c in first]
    assert {c.chunk_id for c in read(current)}.isdisjoint(c.chunk_id for c in first)
    assert first[0].source.parse_spec == parsed.spec
    assert first[0].parsed_document_id == parsed.parsed_document_id != current.parsed_document_id
    library.delete_document(parsed.document_id)
    with pytest.raises(DocumentParseError, match="PARSED_DOCUMENT_UNAVAILABLE"):
        read(parsed)
    assert read(shared_parse) == second


def test_oversized_without_boundaries_progresses_and_input_version_is_checked(library):
    service, parsed, blocks = stored(library, "∑λ🙂".encode(), "tiny.md")
    source = next(iter_stored_chunks(service.results, parsed)).source
    block = replace(blocks[0], text="∑λ🙂" * 100)
    config = ChunkerConfig(2, 1)
    chunks = list(iter_chunks([block], source=source, config=config))
    assert_mapping([block], chunks, config)
    assert len(chunks) == 299
    for config in (ChunkerConfig(8, 0, "character"), ChunkerConfig(8, 2, "line")):
        block = replace(block, text="abc\nABC\n" * 10)
        chunks = list(iter_chunks([block], source=source, config=config))
        assert_mapping([block], chunks, config)
        assert chunks[0].text.endswith("\n")
    with pytest.raises(DocumentParseError, match="PARSED_VERSION_MISMATCH"):
        next(iter_chunks([replace(block, document_version_id="other")], source=source))
    with pytest.raises(ValueError, match="chunk_overlap"):
        ChunkerConfig(5, 5)


def test_stream_consumes_only_blocks_needed_for_first_chunk(library):
    service, parsed, _ = stored(library, b"body", "stream.md")
    source = next(iter_stored_chunks(service.results, parsed)).source

    def blocks():
        for i in range(60_000):
            yield ContentBlock(str(i), source.document_version_id, "paragraph", "正文", ())

    consumed = 0

    def counted_blocks():
        nonlocal consumed
        for block in blocks():
            consumed += 1
            yield block

    config = ChunkerConfig(12, 0)
    chunks = iter_chunks(counted_blocks(), source=source, config=config)
    assert next(chunks).source_block_ids == ("0", "1", "2")
    assert consumed == 4
    assert sum(1 for _ in chunks) == 19_999 and consumed == 60_000
