import asyncio
import json
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict, replace

import pytest
from docx import Document

from app.chunking.models import ChunkerConfig
from app.services import citation_resolver as citations
from app.services.citation_resolver import CitationResolver, CitationScope
from app.services.document_chunking import DocumentChunkingService
from app.services.document_parsing import PARSER_SPECS, DocumentParsingService
from app.services.hybrid_retrieval import HybridRetrievalService
from app.services.keyword_search import KeywordSearchService
from app.services.vector_retrieval import VectorRetrievalService
from app.storage import citations as citation_storage
from app.storage.keywords import KeywordIndex
from test_docx_parsing import serialize
from test_embedding_storage import make_service, unexpected
from test_library_management import import_file, library
from test_pdf_parsing import make_pdf, page, text


def publish(store, book, config=ChunkerConfig()):
    parsed = DocumentParsingService(store).ensure_parsed(book.document_version_id)
    chunking = DocumentChunkingService(store)
    chunks = chunking.ensure_chunked(parsed.parsed_document_id, config)
    return chunks.chunk_set_id, tuple(chunking.results.iter_chunks(chunks.chunk_set_id))


def resolve(resolver, chunk, *, scope=CitationScope()):
    result = resolver.resolve_chunk(chunk.chunk_id, scope=scope)
    assert result.status == "resolved", result
    return result.evidence


def test_cross_page_pdf_preserves_boxes_and_warns_about_labels(library):
    data = make_pdf([page(text(72, 720, "The first physical page.")),
                     page(text(72, 710, "The second physical page.")), page(b"")],
                    labels=b"/PageLabels << /Nums [0 << /S /r >> 1 << /S /D >>] >>")
    book = import_file(library, data, "physics.pdf")
    set_id, (chunk,) = publish(library.store, book)
    resolver = CitationResolver(library.store)
    evidence = resolve(resolver, chunk)
    assert (evidence.source.document_id, evidence.source.document_version_id, evidence.chunk_set_id) == (
        book.document_id, book.document_version_id, set_id)
    assert evidence.source.parsed_document_id == chunk.source.parsed_document_id
    assert evidence.textbook_name == "physics.pdf" and evidence.source.format == "pdf"
    assert evidence.source_spans == chunk.source_spans
    first, second = (span.source_locations[0] for span in evidence.source_spans)
    assert [(loc.physical_page, loc.page_label) for loc in (first, second)] == [(1, "i"), (2, "1")]
    assert first.page_bbox[0] == second.page_bbox[0] == 72
    assert 0 < first.page_bbox[1] < second.page_bbox[1] < 800
    assert first.coordinate_system == "unrotated_mediabox_top_left_pt"
    assert (evidence.location_precision, evidence.location_status) == ("block", "available")
    assert evidence.source.warnings == ()  # The blank third page is outside this evidence.

    warnings = [dict(code="PDF_PAGE_LABELS_INVALID", message="Unreliable label", physical_page=2),
                dict(code="PDF_UNMAPPED_GLYPHS", message="Unmapped math glyphs omitted", physical_page=1)]
    with library.store.write() as db:
        db.execute("UPDATE parsed_documents SET warnings = json_insert(warnings, '$[#]', json(?), '$[#]', json(?))",
                   tuple(map(json.dumps, warnings)))
    new = resolve(resolver, chunk)
    assert new.body_text == chunk.text and new.text_fidelity == "parsed_text_unverified"
    assert new.source_spans[0].source_locations[0].page_label == "i"
    assert new.source_spans[1].source_locations[0].page_label is None
    assert [w.code for w in new.source.warnings] == [w["code"] for w in warnings]
    assert resolver.validate_evidence(evidence.evidence_id).status == "evidence_mismatch"

    with library.store.write() as db:
        db.execute("UPDATE source_locations SET details = json_set(details, '$.page_bbox', null)")
    assert resolve(resolver, chunk).location_precision == "page"
    with library.store.write() as db:
        db.execute("DELETE FROM source_locations WHERE block_position = 1")
    partial = resolve(resolver, chunk)
    assert (partial.location_precision, partial.location_status) == ("page", "partial")
    assert {w.physical_page for w in partial.source.warnings} == {1, 2, 3}
    with library.store.write() as db:
        db.execute("DELETE FROM source_locations")
    missing = resolve(resolver, chunk)
    assert (missing.location_precision, missing.location_status) == ("unavailable", "unavailable")


def test_docx_keeps_heading_paragraph_and_table_grid(library):
    doc = Document()
    doc.add_heading("Mechanics", level=1)
    doc.add_paragraph("")
    doc.add_paragraph("A force changes momentum.")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Mass"
    table.cell(0, 1).text = "Velocity"
    table.cell(1, 0).text = "2 kg"
    table.cell(1, 1).text = "3 m/s"
    book = import_file(library, serialize(doc), "mechanics.docx")
    _, (prose, grid) = publish(library.store, book)
    resolver = CitationResolver(library.store)
    evidence = resolve(resolver, prose)
    assert evidence.section_path == ("Mechanics",)
    heading, paragraph = (span.source_locations[0] for span in evidence.source_spans)
    assert (heading.body_index, heading.paragraph_number, heading.heading_level) == (1, 1, 1)
    assert (paragraph.body_index, paragraph.paragraph_number, paragraph.heading_level) == (3, 3, None)
    table_evidence = resolve(resolver, grid)
    assert table_evidence.location_precision == "structure"
    assert table_evidence.section_path == ("Mechanics",)
    cells = table_evidence.source_spans[0].source_locations
    assert [(c.body_index, c.table_row, c.table_column) for c in cells] == [
        (4, 1, 1), (4, 1, 2), (4, 2, 1), (4, 2, 2)]
    assert all("physical_page" not in asdict(cell) for cell in cells)


def test_markdown_overlap_title_only_search_and_unicode_ranges(library):
    passage = "Unicode 𝛑 e\u0301 🌍. " * 25
    original = f"# Spectral\n\n{passage}\n"
    book = import_file(library, original.encode(), "math.md")
    set_id, chunks = publish(library.store, book, ChunkerConfig(chunk_size=64, chunk_overlap=15))
    KeywordIndex(library.store).ensure_indexed(set_id)
    resolver = CitationResolver(library.store)
    embeddings, _ = make_service(library.store, unexpected, configured=False)
    search = HybridRetrievalService(KeywordSearchService(library.store),
                                    VectorRetrievalService(library.store, embeddings.provider))
    hits = asyncio.run(search.search("Spectral", retrieval_mode="keyword")).hits
    hit = next(hit for hit in hits if hit.keyword_matched_in == ("title",))
    forged_display = replace(hit, body_text="fake quote", title_context=("fake title",),
                             section_path=("fake",), source_spans=(), fusion_score=1.0)
    resolved, = resolver.resolve_results((forged_display,))
    evidence = resolved.evidence
    assert evidence.body_text == hit.body_text and "Spectral" not in evidence.body_text
    assert evidence.section_path == ("Spectral",)
    assert resolver.resolve_results((replace(hit, document_version_id="forged"),))[0].status == "inconsistent_source"
    assert resolver.resolve_results(asyncio.run(search.search("Spectral", retrieval_mode="keyword")))[0].evidence

    results = resolver.resolve_chunks(tuple(chunk.chunk_id for chunk in chunks))
    resolved_chunks = [result.evidence for result in results]
    first, second = resolved_chunks[1:3]
    assert all(item.location_precision == "line" for item in resolved_chunks)
    assert first.source_block_ids == second.source_block_ids
    span, tail = first.source_spans[-1], second.source_spans[0]
    assert tail.is_overlap and span.block_start < tail.block_start < tail.block_end == span.block_end
    assert tail.source_locations == span.source_locations
    assert [(loc.start_line, loc.end_line) for loc in tail.source_locations] == [(3, 3)]
    for item in (first, second):
        for src in item.source_spans:
            assert (item.body_text[src.chunk_start:src.chunk_end]
                    == passage.strip()[src.block_start:src.block_end])
    assert "title_context" not in asdict(evidence)
    json.dumps(asdict(evidence), allow_nan=False)


def test_restart_historical_parse_and_deleted_evidence(library, monkeypatch):
    doc = import_file(library, b"old source", "one.md")
    old_set, (old_chunk,) = publish(library.store, doc)
    resolver = CitationResolver(library.store)
    old = resolve(resolver, old_chunk)
    root = library.store.path.parent.parent
    run = subprocess.run(
        [sys.executable, "-c", """
import sys
from pathlib import Path
from app.services.citation_resolver import CitationResolver
from app.storage.library import KnowledgeStore
assert CitationResolver(KnowledgeStore(Path(sys.argv[1]))).validate_evidence(sys.argv[2]).status == 'resolved'
""", str(root), old.evidence_id], check=True, capture_output=True, text=True,
    )
    assert not run.stdout

    monkeypatch.setitem(PARSER_SPECS, "md", replace(PARSER_SPECS["md"], config_key="test-new-settings"))
    rebuilt_set, (rebuilt,) = publish(library.store, doc)
    assert old_set != rebuilt_set
    assert resolver.validate_evidence(old.evidence_id).status == "out_of_scope"
    old_scope = CitationScope(chunk_set_ids=(old_set,))
    pinned = resolver.validate_evidence(old.evidence_id, scope=old_scope).evidence
    assert pinned.historical and not pinned.parse_spec_current
    assert pinned.source.parse_spec == old.source.parse_spec
    assert resolve(resolver, rebuilt).evidence_id != old.evidence_id

    latest = import_file(library, b"new source", "one.md", document_id=doc.document_id)
    _, (latest_chunk,) = publish(library.store, latest)
    assert resolver.resolve_chunk(rebuilt.chunk_id).status == "out_of_scope"
    assert resolve(resolver, latest_chunk).body_text == "new source"
    assert resolver.validate_evidence(old.evidence_id, scope=old_scope).evidence.body_text == "old source"
    assert resolver.validate_evidence(old.evidence_id, scope=CitationScope(chunk_set_ids=(rebuilt_set,))).status == (
        "out_of_scope")
    other = import_file(library, b"separate source", "two.md")
    shared = import_file(library, b"old source", "two.md", document_id=other.document_id)
    _, (shared_chunk,) = publish(library.store, shared)
    library.delete_document(doc.document_id)
    assert resolver.validate_evidence(old.evidence_id, scope=old_scope).status == "source_removed"
    assert resolve(resolver, shared_chunk).source.document_id == shared.document_id
    assert resolver.resolve_chunk(old_chunk.chunk_id, scope=old_scope).status == "not_found"


def test_original_missing_changed_and_forged_evidence_ids(library):
    data = b"A test source."
    doc = import_file(library, data)
    _, (chunk,) = publish(library.store, doc)
    resolver = CitationResolver(library.store)
    evidence = resolve(resolver, chunk)
    assert resolver.resolve_chunk("invalid").status == "invalid_chunk_id"
    assert resolver.resolve_chunk("chunk:" + "0" * 64).status == "not_found"
    assert resolver.validate_evidence("1").status == "invalid_evidence_id"
    wrong = evidence.evidence_id[:-64] + "0" * 64
    assert resolver.validate_evidence(wrong).status == "evidence_mismatch"
    original = library.store.originals_dir / evidence.source.content_sha256
    original.chmod(0o600)
    original.write_bytes(b"X" + data[1:])
    assert resolver.validate_evidence(evidence.evidence_id).status == "original_mismatch"
    original.unlink()
    assert resolver.validate_evidence(evidence.evidence_id).status == "original_missing"
    assert resolver.resolve_chunk(chunk.chunk_id).evidence is None


@pytest.mark.parametrize("change", [
    "UPDATE chunks SET text = upper(text)",
    # Same text and heading in the same parse, but a different physical block.
    """UPDATE chunk_sources SET source_block_id =
        (SELECT block_id FROM parsed_blocks WHERE position = 1)
        WHERE source_index = 0""",
    "UPDATE source_locations SET kind = 'pdf', details = '{\"page_bbox\":null,\"physical_page\":1}'",
    "DELETE FROM chunk_sources WHERE source_index = 0",
])
def test_inconsistent_lineage_never_yields_evidence(library, change):
    doc = import_file(library, b"same\n\nsame")
    _, (chunk,) = publish(library.store, doc)
    resolver = CitationResolver(library.store)
    resolve(resolver, chunk)
    with library.store.write() as db:
        db.execute(change)
    result = resolver.resolve_chunk(chunk.chunk_id)
    assert result.status == "inconsistent_source" and result.evidence is None


def test_rejects_cross_parse_relationships_and_unpublished_sets(library):
    old = import_file(library, b"same text")
    other = import_file(library, b"another book")
    second = import_file(library, b"same text", document_id=other.document_id)
    set_id, (chunk,) = publish(library.store, old)
    second_set, (second_chunk,) = publish(library.store, second)
    resolver = CitationResolver(library.store)
    assert resolver.resolve_chunk(chunk.chunk_id, scope=CitationScope(document_ids=(other.document_id,))).status == (
        "out_of_scope")
    with library.store.connect() as db:
        db.execute("PRAGMA foreign_keys=OFF")
        db.execute("UPDATE chunk_sources SET parsed_document_id = ?, source_block_id = ? WHERE chunk_set_id = ?",
                   (second_chunk.parsed_document_id, second_chunk.source_block_ids[0], set_id))
        db.execute("""UPDATE chunk_sets SET status = 'writing', lease_expires_at = unixepoch() + 60,
                    published_at = NULL, chunk_count = 0, source_span_count = 0 WHERE chunk_set_id = ?""",
                   (second_set,))
    assert resolver.resolve_chunk(chunk.chunk_id).status == "inconsistent_source"
    assert resolver.resolve_chunk(second_chunk.chunk_id, scope=CitationScope(chunk_set_ids=(second_set,))).status == (
        "not_found")


def test_bounds_and_constant_sql_count(library, monkeypatch):
    doc = import_file(library, ("paragraph\n\n" * 50).encode())
    set_id, chunks = publish(library.store, doc, ChunkerConfig(chunk_size=22, chunk_overlap=0))
    resolver = CitationResolver(library.store)
    statements = []
    original_connect = library.store.connect

    @contextmanager
    def tracked():
        with original_connect() as db:
            db.set_trace_callback(statements.append)
            yield db

    monkeypatch.setattr(library.store, "connect", tracked)
    small = resolver.resolve_chunk(chunks[0].chunk_id)
    single_count = len(statements)
    statements.clear()
    batch = resolver.resolve_chunks(tuple(chunk.chunk_id for chunk in chunks))
    assert all(res.status == "resolved" for res in batch)
    assert small == batch[0] and len(statements) == single_count == 5  # BEGIN, three bounded reads, COMMIT.
    with pytest.raises(ValueError):
        resolver.resolve_chunks([chunks[0].chunk_id] * (citations.MAX_EVIDENCE_COUNT + 1))

    monkeypatch.setattr(citation_storage, "MAX_BODY_CHARS", 8)
    assert resolver.resolve_chunk(chunks[0].chunk_id).status == "limit_exceeded"
    monkeypatch.setattr(citation_storage, "MAX_BODY_CHARS", 8192)
    monkeypatch.setattr(citation_storage, "MAX_LOCATIONS", 2)
    duplicate = resolver.resolve_chunks((chunks[0].chunk_id, chunks[0].chunk_id))
    assert [item.status for item in duplicate] == ["resolved", "limit_exceeded"]
    mixed = resolver.resolve_chunks((chunks[0].chunk_id, chunks[-1].chunk_id),
                                    scope=CitationScope(chunk_set_ids=(set_id,)))
    assert [item.status for item in mixed] == ["limit_exceeded", "limit_exceeded"]


def test_original_read_budget_is_shared_by_identical_bytes(library, monkeypatch):
    first = b"First content."
    second = b"Second content."
    one = import_file(library, first)
    two = import_file(library, second)
    _, (chunk_one,) = publish(library.store, one)
    _, (chunk_two,) = publish(library.store, two)
    monkeypatch.setattr(citations, "MAX_ORIGINAL_BYTES_PER_BATCH", len(first))
    result = CitationResolver(library.store).resolve_chunks((chunk_one.chunk_id, chunk_one.chunk_id, chunk_two.chunk_id))
    assert [item.status for item in result] == ["resolved", "resolved", "limit_exceeded"]
