"""Resolve citations against an exact, published SQLite lineage."""

import hashlib
import json
import os
import re
import sqlite3
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from math import isfinite
from typing import Literal

from app.chunking.hybrid import BLOCK_SEPARATOR, CHUNKER_VERSION, chunk_id_from_spans, chunk_identity
from app.chunking.models import ChunkSource, ChunkSourceSpan
from app.parsing.models import DocxSourceLocation, MarkdownSourceLocation, ParseWarning, PdfSourceLocation, SourceLocation
from app.services.document_parsing import PARSER_SPECS
from app.services.hybrid_retrieval import HybridHit, HybridResult
from app.services.keyword_search import MAX_TOP_K, KeywordHit
from app.services.library import COPY_BLOCK_BYTES, HASH_NAME, MAX_FILE_BYTES, file_revision, private_directory, private_file
from app.services.vector_retrieval import VectorHit
from app.storage import citations
from app.storage.library import ORIGINALS_DIRECTORY, KnowledgeStore, KnowledgeStoreError
from app.storage.parsed_documents import ParseSpec, collect_locations
from app.storage.retrieval_scope import scope_filters

MAX_EVIDENCE_COUNT = MAX_TOP_K
MAX_ORIGINAL_BYTES_PER_BATCH = 2 * MAX_FILE_BYTES
_CHUNK_ID = re.compile(r"chunk:[0-9a-f]{64}")
_DOCUMENT_ID = re.compile(r"[0-9a-f]{32}")
_EVIDENCE_ID = re.compile(r"ev1:([0-9a-f]{32}):([1-9][0-9]{0,18}):(chunk:[0-9a-f]{64}):[0-9a-f]{64}")
_HIT_TYPES = (HybridHit, KeywordHit, VectorHit)

RetrievalHit = HybridHit | KeywordHit | VectorHit
LocationPrecision = Literal["block", "page", "structure", "line", "unavailable"]
CitationStatus = Literal[
    "resolved", "invalid_chunk_id", "invalid_evidence_id", "not_found", "source_removed",
    "out_of_scope", "inconsistent_source", "evidence_mismatch", "original_missing",
    "original_unreadable", "original_mismatch", "limit_exceeded", "storage_error",
]


@dataclass(frozen=True, slots=True)
class CitationScope:
    """A server-authorized document filter and optional exact Chunk Set allowlist."""

    document_ids: tuple[str, ...] | None = None
    chunk_set_ids: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class Evidence:
    evidence_id: str
    source: ChunkSource
    chunk_set_id: int
    chunk_id: str
    chunk_index: int
    textbook_name: str
    original_size_bytes: int
    section_path: tuple[str, ...]
    body_text: str
    source_block_ids: tuple[str, ...]
    source_spans: tuple[ChunkSourceSpan, ...]
    location_precision: LocationPrecision
    location_status: Literal["available", "partial", "unavailable"]
    historical: bool
    parse_spec_current: bool
    text_fidelity: Literal["parsed_text_unverified"] = "parsed_text_unverified"


@dataclass(frozen=True, slots=True)
class CitationResolution:
    chunk_id: str | None
    status: CitationStatus
    evidence: Evidence | None = None


@dataclass(frozen=True, slots=True)
class _Request:
    chunk_id: str
    hit: RetrievalHit | None = None
    expected_id: str | None = None
    document_id: str | None = None
    set_id: int | None = None


class _InvalidSource(ValueError):
    """The database no longer describes a coherent citation."""


class _LimitExceeded(ValueError):
    pass


def _check_batch(items: Sequence) -> None:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)) or len(items) > MAX_EVIDENCE_COUNT:
        raise ValueError(f"Expected a sequence of up to {MAX_EVIDENCE_COUNT} citations.")


def _positive(value: object) -> bool:
    return type(value) is int and value > 0


def _spec(row: sqlite3.Row) -> ParseSpec:
    return ParseSpec(row["parser_version"], row["structure_version"], row["parse_config_key"])


def _parse_current(row: sqlite3.Row) -> bool:
    return _spec(row) == PARSER_SPECS.get(row["format"])


def _current(row: sqlite3.Row) -> bool:
    return bool(row["current_version"] and _parse_current(row) and row["chunker_version"] == CHUNKER_VERSION)


def _select_row(req: _Request, rows: list[sqlite3.Row], scope: CitationScope,
                removed: set[str]) -> sqlite3.Row | CitationStatus:
    if req.set_id is not None and scope.chunk_set_ids is not None and req.set_id not in scope.chunk_set_ids:
        return "out_of_scope"
    if req.document_id is not None and scope.document_ids is not None and req.document_id not in scope.document_ids:
        return "out_of_scope"
    if not rows:
        return "source_removed" if req.document_id in removed else "not_found"
    if req.set_id is not None:
        rows = [row for row in rows if row["chunk_set_id"] == req.set_id]
        if not rows:
            return "evidence_mismatch" if req.expected_id else "inconsistent_source"

    ready = []
    outside_scope = False
    for row in rows:
        if (row["set_status"] is None or row["parse_status"] is None or not row["document_exists"]
                or not row["blob_exists"] or row["parse_version_id"] != row["document_version_id"]):
            return "inconsistent_source"
        if not row["in_scope"] or (scope.chunk_set_ids is None and not _current(row)):
            outside_scope = True
        elif row["set_status"] == "ready" and row["parse_status"] == "ready":
            ready.append(row)
    if len(ready) > 1:
        return "inconsistent_source"
    if not ready:
        return "out_of_scope" if outside_scope else "not_found"

    row = ready[0]
    if req.document_id is not None and row["document_id"] != req.document_id:
        return "evidence_mismatch" if req.expected_id else "inconsistent_source"
    if req.hit is not None and (
        (req.hit.document_version_id, req.hit.parsed_document_id, req.hit.chunk_index)
        != (row["document_version_id"], row["parsed_document_id"], row["chunk_index"])
        or (isinstance(req.hit, (HybridHit, VectorHit)) and req.hit.content_sha256 != row["content_sha256"])
    ):
        return "inconsistent_source"
    return row


def _locations(rows: list[sqlite3.Row]) -> tuple[SourceLocation, ...]:
    ordered = sorted(rows, key=lambda row: row["position"])
    if any(row["details"] is None for row in ordered):
        raise _LimitExceeded
    if any(row["position"] != index for index, row in enumerate(ordered)):
        raise _InvalidSource
    return tuple(collect_locations((0, row["kind"], row["details"]) for row in ordered).get(0, ()))


def _location_precision(location: SourceLocation, format: str) -> LocationPrecision:
    if isinstance(location, PdfSourceLocation) and format == "pdf":
        page, box = location.physical_page, location.page_bbox
        if location.coordinate_system != "unrotated_mediabox_top_left_pt" or (
            page is not None and not _positive(page)
        ):
            raise _InvalidSource
        if box is not None and (
            len(box) != 4 or any(type(value) not in (int, float) or not isfinite(value) for value in box)
            or not (0 <= box[0] < box[2] and 0 <= box[1] < box[3])
        ):
            raise _InvalidSource
        return "unavailable" if page is None else "page" if box is None else "block"
    if isinstance(location, DocxSourceLocation) and format == "docx":
        fields = (location.body_index, location.paragraph_number, location.heading_level, location.table_row,
                  location.table_column, location.row_span, location.column_span, location.numbering_id,
                  location.list_level)
        if any(value is not None and not _positive(value) for value in fields):
            raise _InvalidSource
        if ((location.table_row is None) != (location.table_column is None)
                or (location.list_format is not None and not isinstance(location.list_format, str))):
            raise _InvalidSource
        return "structure" if location.body_index or location.paragraph_number else "unavailable"
    if isinstance(location, MarkdownSourceLocation) and format == "md":
        start, end = location.start_line, location.end_line
        if any(value is not None and not _positive(value) for value in (start, end, location.heading_level)):
            raise _InvalidSource
        if start is None or end is None:
            return "unavailable"
        if end < start:
            raise _InvalidSource
        return "line"
    raise _InvalidSource


def _warnings(raw: str) -> tuple[ParseWarning, ...]:
    warnings = tuple(ParseWarning(**item) for item in json.loads(raw))
    if any(not isinstance(w.code, str) or not isinstance(w.message, str)
           or (w.physical_page is not None and not _positive(w.physical_page)) for w in warnings):
        raise _InvalidSource
    return warnings


def _without_unreliable_labels(spans: tuple[ChunkSourceSpan, ...],
                               warnings: tuple[ParseWarning, ...]) -> tuple[ChunkSourceSpan, ...]:
    invalid = {warning.physical_page for warning in warnings if warning.code == "PDF_PAGE_LABELS_INVALID"}
    sanitized = {}
    for span in spans:
        if span.block_id in sanitized:
            continue
        sanitized[span.block_id] = tuple(
            replace(location, page_label=None)
            if isinstance(location, PdfSourceLocation) and location.page_label is not None and (
                location.physical_page is None or None in invalid or location.physical_page in invalid
                or not isinstance(location.page_label, str)
                or not (0 < len(location.page_label) <= 64 and location.page_label.isprintable())
            ) else location
            for location in span.source_locations
        )
    return tuple(replace(span, source_locations=sanitized[span.block_id]) for span in spans)


def _verified_spans(row: sqlite3.Row, sources: list[sqlite3.Row]) -> tuple[ChunkSourceSpan, ...]:
    body = row["body_text"]
    spans = []
    previous_block = None
    previous_position = -1
    previous_end = 0
    has_new_text = False
    for index, saved in enumerate(sorted(sources, key=lambda item: item["source_index"])):
        block, position = saved["source_block_id"], saved["block_position"]
        start, end = saved["block_start"], saved["block_end"]
        left, right = saved["chunk_start"], saved["chunk_end"]
        overlap = saved["is_overlap"]
        if (saved["source_index"] != index or saved["parsed_document_id"] != row["parsed_document_id"]
                or not saved["section_matches"] or not saved["text_matches"]
                or type(position) is not int or position < previous_position
                or any(type(value) is not int for value in (start, end, left, right, overlap))
                or not (0 <= start < end and previous_end <= left < right <= len(body))
                or end - start != right - left or overlap not in (0, 1) or (overlap and has_new_text)):
            raise _InvalidSource
        expected_gap = BLOCK_SEPARATOR if previous_block is not None and previous_block != block else ""
        if body[previous_end:left] != expected_gap or (
            previous_block == block and spans[-1].block_end != start
        ):
            raise _InvalidSource
        spans.append(ChunkSourceSpan(block, saved["kind"], start, end, left, right, (), bool(overlap)))
        previous_block, previous_position, previous_end = block, position, right
        has_new_text |= not overlap
    if previous_end != len(body) or not has_new_text:
        raise _InvalidSource
    return tuple(spans)


def _evidence(row: sqlite3.Row, spans: tuple[ChunkSourceSpan, ...]) -> Evidence:
    warnings = _warnings(row["warnings"])
    format = row["format"]
    precision = {_location_precision(location, format) for span in spans for location in span.source_locations}
    if any(not span.source_locations for span in spans):
        precision.add("unavailable")
    known = precision - {"unavailable"}
    precision_floor = "page" if "page" in known else next(iter(known), "unavailable")
    location_status = "unavailable" if not known else "partial" if "unavailable" in precision else "available"

    pages = {location.physical_page for span in spans for location in span.source_locations
             if isinstance(location, PdfSourceLocation) and location.physical_page is not None}
    if format == "pdf" and pages and location_status == "available":
        warnings = tuple(w for w in warnings if w.physical_page is None or w.physical_page in pages)
    path = tuple(json.loads(row["section_path"]))
    if any(not isinstance(part, str) for part in path):
        raise _InvalidSource
    source = ChunkSource(row["document_id"], row["document_version_id"], row["content_sha256"],
                         row["parsed_document_id"], _spec(row), format, warnings)
    identity = chunk_identity(source, row["chunker_version"], row["chunker_config_key"])
    index, chunk_id = row["chunk_index"], row["chunk_id"]
    if chunk_id_from_spans(identity, index, spans) != chunk_id:
        raise _InvalidSource

    digest = hashlib.sha256(json.dumps(
        (asdict(source), row["chunk_set_id"], index, chunk_id, row["size_bytes"], path, row["body_text"],
         [asdict(span) for span in spans]),
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("ascii")).hexdigest()
    evidence_id = f"ev1:{source.document_id}:{row['chunk_set_id']}:{chunk_id}:{digest}"
    if format == "pdf":
        spans = _without_unreliable_labels(spans, warnings)
    return Evidence(
        evidence_id, source, row["chunk_set_id"], chunk_id, index, row["name"], row["size_bytes"], path,
        row["body_text"], tuple(dict.fromkeys(span.block_id for span in spans)), spans, precision_floor, location_status,
        not _current(row), _parse_current(row),
    )


def _verify_original(
    store: KnowledgeStore, sha256: str, size: int,
    verified_revisions: dict[tuple[str, int], tuple] | None = None,
) -> tuple[CitationStatus, bool]:
    """Check the same private file boundary as parsing, with revision cache and bounded streaming."""
    key = (sha256, size)
    try:
        with private_directory(store.path.parent) as root, private_directory(ORIGINALS_DIRECTORY, root) as originals:
            fd = os.open(sha256, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=originals)
            try:
                before = os.fstat(fd)
                if not private_file(before):
                    if verified_revisions is not None:
                        verified_revisions.pop(key, None)
                    return "original_unreadable", False
                if before.st_size != size:
                    if verified_revisions is not None:
                        verified_revisions.pop(key, None)
                    return "original_mismatch", False
                rev = file_revision(before)
                if verified_revisions is not None and verified_revisions.get(key) == rev:
                    return "resolved", False
                with os.fdopen(fd, "rb", closefd=False) as stream:
                    digest = hashlib.sha256()
                    remaining = size
                    while remaining:
                        data = stream.read(min(remaining, COPY_BLOCK_BYTES))
                        if not data:
                            if verified_revisions is not None:
                                verified_revisions.pop(key, None)
                            return "original_mismatch", True
                        digest.update(data)
                        remaining -= len(data)
                after = os.stat(sha256, dir_fd=originals, follow_symlinks=False)
                after_rev = file_revision(after)
                if rev != after_rev or digest.hexdigest() != sha256:
                    if verified_revisions is not None:
                        verified_revisions.pop(key, None)
                    return "original_mismatch", True
                if verified_revisions is not None:
                    verified_revisions[key] = after_rev
            finally:
                os.close(fd)
    except FileNotFoundError:
        if verified_revisions is not None:
            verified_revisions.pop(key, None)
        return "original_missing", False
    except (OSError, KnowledgeStoreError):
        if verified_revisions is not None:
            verified_revisions.pop(key, None)
        return "original_unreadable", False
    return "resolved", True


@dataclass(frozen=True, slots=True)
class CitationView:
    evidence_id: str
    chunk_id: str
    chunk_set_id: int
    chunk_index: int
    document_id: str
    document_version_id: str
    textbook_name: str
    format: Literal["pdf", "docx", "md"]
    section_path: tuple[str, ...]
    location_summary: str
    location_precision: LocationPrecision
    location_status: Literal["available", "partial", "unavailable"]
    physical_pages: tuple[int, ...]
    page_labels: tuple[str, ...]
    line_range: tuple[int, int] | None
    paragraph_numbers: tuple[int, ...]
    warnings: tuple[ParseWarning, ...]
    historical: bool
    parse_spec_current: bool
    text_fidelity: str
    snippet: str
    status: CitationStatus = "resolved"


def summarize_evidence_location(evidence: Evidence) -> tuple[
    str, tuple[int, ...], tuple[str, ...], tuple[int, int] | None, tuple[int, ...]
]:
    format = evidence.source.format
    locations = [loc for span in evidence.source_spans for loc in span.source_locations]
    if format == "pdf":
        pages = tuple(dict.fromkeys(
            loc.physical_page for loc in locations
            if isinstance(loc, PdfSourceLocation) and loc.physical_page is not None
        ))
        labels = tuple(dict.fromkeys(
            loc.page_label for loc in locations
            if isinstance(loc, PdfSourceLocation) and loc.page_label is not None
        ))
        if not pages:
            summary = "页码不可用"
        elif len(pages) == 1:
            summary = f"第 {pages[0]} 页"
        elif pages[-1] - pages[0] == len(pages) - 1:
            summary = f"第 {pages[0]}–{pages[-1]} 页"
        else:
            summary = f"第 {', '.join(map(str, pages[:4]))} 页"
        if labels and labels != tuple(map(str, pages)):
            summary += f"（页标签 {', '.join(labels[:4])}）"
        return summary, pages, labels, None, ()

    if format == "md":
        starts = [loc.start_line for loc in locations if isinstance(loc, MarkdownSourceLocation) and loc.start_line]
        ends = [loc.end_line for loc in locations if isinstance(loc, MarkdownSourceLocation) and loc.end_line]
        if starts and ends:
            start, end = min(starts), max(ends)
            summary = f"第 {start} 行" if start == end else f"第 {start}–{end} 行"
            return summary, (), (), (start, end), ()
        return "行号不可用", (), (), None, ()

    paragraphs = tuple(dict.fromkeys(
        loc.paragraph_number for loc in locations
        if isinstance(loc, DocxSourceLocation) and loc.paragraph_number is not None
    ))
    cells = tuple(dict.fromkeys(
        (loc.table_row, loc.table_column) for loc in locations
        if isinstance(loc, DocxSourceLocation) and loc.table_row is not None and loc.table_column is not None
    ))
    bodies = tuple(dict.fromkeys(
        loc.body_index for loc in locations
        if isinstance(loc, DocxSourceLocation) and loc.body_index is not None
    ))
    parts: list[str] = []
    if paragraphs:
        parts.append(f"第 {paragraphs[0]} 段" if len(paragraphs) == 1 else f"第 {paragraphs[0]}–{paragraphs[-1]} 段")
    if cells:
        parts.append(f"表格第 {cells[0][0]} 行第 {cells[0][1]} 列" if len(cells) == 1 else f"表格（{len(cells)} 个单元格）")
    if not parts and bodies:
        parts.append(f"结构项 {bodies[0]}")
    return (" · ".join(parts) if parts else "结构位置不可用"), (), (), None, paragraphs


def format_citation_view(evidence: Evidence, *, status: CitationStatus = "resolved") -> CitationView:
    summary, pages, labels, line_range, paragraphs = summarize_evidence_location(evidence)
    return CitationView(
        evidence_id=evidence.evidence_id,
        chunk_id=evidence.chunk_id,
        chunk_set_id=evidence.chunk_set_id,
        chunk_index=evidence.chunk_index,
        document_id=evidence.source.document_id,
        document_version_id=evidence.source.document_version_id,
        textbook_name=evidence.textbook_name,
        format=evidence.source.format,
        section_path=evidence.section_path,
        location_summary=summary,
        location_precision=evidence.location_precision,
        location_status=evidence.location_status,
        physical_pages=pages,
        page_labels=labels,
        line_range=line_range,
        paragraph_numbers=paragraphs,
        warnings=evidence.source.warnings,
        historical=evidence.historical,
        parse_spec_current=evidence.parse_spec_current,
        text_fidelity=evidence.text_fidelity,
        snippet=evidence.body_text,
        status=status,
    )


class CitationResolver:
    """Read current evidence, or use a Chunk Set allowlist supplied by trusted server code."""

    def __init__(self, store: KnowledgeStore):
        self.store = store
        self._verified_revisions: dict[tuple[str, int], tuple] = {}

    def resolve_chunk(self, chunk_id: str, *, scope: CitationScope = CitationScope()) -> CitationResolution:
        return self.resolve_chunks((chunk_id,), scope=scope)[0]

    def resolve_chunks(self, chunk_ids: Sequence[str], *,
                       scope: CitationScope = CitationScope()) -> tuple[CitationResolution, ...]:
        _check_batch(chunk_ids)
        return self._resolve(tuple(_Request(chunk_id) for chunk_id in chunk_ids), scope)

    def resolve_results(self, results: HybridResult | Sequence[RetrievalHit], *,
                        scope: CitationScope = CitationScope()) -> tuple[CitationResolution, ...]:
        hits = results.hits if isinstance(results, HybridResult) else results
        _check_batch(hits)
        if any(not isinstance(hit, _HIT_TYPES) for hit in hits):
            raise ValueError("Expected internal retrieval hits.")
        if any(not isinstance(hit.document_id, str) or not _DOCUMENT_ID.fullmatch(hit.document_id)
               or not _positive(hit.chunk_set_id) or not _positive(hit.parsed_document_id)
               or type(hit.chunk_index) is not int or hit.chunk_index < 0 for hit in hits):
            raise ValueError("Invalid retrieval hit identity.")
        return self._resolve(tuple(_Request(hit.chunk_id, hit, document_id=hit.document_id,
                                            set_id=hit.chunk_set_id) for hit in hits), scope)

    def validate_evidence(self, evidence_id: str, *, scope: CitationScope = CitationScope()) -> CitationResolution:
        match = _EVIDENCE_ID.fullmatch(evidence_id) if isinstance(evidence_id, str) else None
        if match is None or int(match[2]) > 2**63 - 1:
            return CitationResolution(None, "invalid_evidence_id")
        document_id, set_id, chunk_id = match.groups()
        return self._resolve((_Request(chunk_id, expected_id=evidence_id, document_id=document_id,
                                       set_id=int(set_id)),), scope)[0]

    def _resolve(self, requests: tuple[_Request, ...], scope: CitationScope) -> tuple[CitationResolution, ...]:
        scope_sql, params = scope_filters(scope.document_ids, scope.chunk_set_ids, label="citation")
        valid = {r.chunk_id for r in requests if isinstance(r.chunk_id, str) and _CHUNK_ID.fullmatch(r.chunk_id)}
        results = [CitationResolution(r.chunk_id, "not_found") if isinstance(r.chunk_id, str) and r.chunk_id in valid
                   else CitationResolution(None, "invalid_chunk_id") for r in requests]
        if not valid:
            return tuple(results)

        try:
            with self.store.connect() as db, db:
                db.row_factory = sqlite3.Row
                db.execute("BEGIN")
                candidates = citations.read_chunks(db, valid, scope_sql, params)
                grouped = defaultdict(list)
                for row in candidates:
                    grouped[row["chunk_id"]].append(row)
                removed = citations.removed_documents(db, {
                    request.document_id for request in requests
                    if request.document_id is not None and isinstance(request.chunk_id, str) and request.chunk_id in valid
                    and request.chunk_id not in grouped
                })
                selected = {}
                for index, request in enumerate(requests):
                    if results[index].chunk_id is None:
                        continue
                    row = (_select_row(request, grouped[request.chunk_id], scope, removed)
                           if len(candidates) < 2 * len(valid) + 1 else "limit_exceeded")
                    if isinstance(row, sqlite3.Row):
                        selected[index] = row
                    else:
                        results[index] = CitationResolution(request.chunk_id, row)
                self._read_evidence(db, selected, requests, results)
        except sqlite3.DatabaseError:
            return tuple(CitationResolution(r.chunk_id, "storage_error") if r.chunk_id is not None else r
                         for r in results)
        return self._check_originals(results)

    def _read_evidence(self, db, selected, requests, results) -> None:
        keys = set()
        for index, row in selected.items():
            if any(row[field] is None for field in ("body_text", "name", "section_path", "warnings",
                                                   "chunker_config_key", "parse_config_key")):
                results[index] = CitationResolution(row["chunk_id"], "limit_exceeded")
            elif (not isinstance(row["body_text"], str) or not _positive(row["text_length"])
                  or len(row["body_text"]) != row["text_length"]
                  or "\0" in row["body_text"]
                  or not _positive(row["chunk_count"]) or type(row["chunk_index"]) is not int
                  or not 0 <= row["chunk_index"] < row["chunk_count"] or row["format"] not in PARSER_SPECS
                  or not isinstance(row["document_id"], str) or not _DOCUMENT_ID.fullmatch(row["document_id"])
                  or not isinstance(row["content_sha256"], str) or not HASH_NAME.fullmatch(row["content_sha256"])
                  or not _positive(row["size_bytes"])
                  or row["size_bytes"] > MAX_FILE_BYTES):
                results[index] = CitationResolution(row["chunk_id"], "inconsistent_source")
            else:
                keys.add((row["chunk_set_id"], row["chunk_index"]))
        if not keys:
            return

        sources_by_chunk = defaultdict(list)
        locations_by_block = defaultdict(list)
        limited = False
        source_rows = citations.read_spans(db, keys)
        if len(source_rows) <= citations.MAX_SOURCE_SPANS:
            for saved in source_rows:
                sources_by_chunk[saved["chunk_set_id"], saved["chunk_index"]].append(saved)
            block_keys = {(saved["parsed_document_id"], saved["block_position"]) for saved in source_rows
                          if saved["block_position"] is not None}
            location_rows = citations.read_locations(db, block_keys)
            limited = len(location_rows) > citations.MAX_LOCATIONS
            for location in location_rows:
                locations_by_block[location["parsed_document_id"], location["block_position"]].append(location)
        else:
            limited = True

        cached = {}
        decoded = {}
        output_locations = 0
        for index, row in selected.items():
            key = (row["chunk_set_id"], row["chunk_index"])
            if results[index].status != "not_found":
                continue
            if limited:
                results[index] = CitationResolution(row["chunk_id"], "limit_exceeded")
                continue
            try:
                sources = sources_by_chunk[key]
                if key not in cached:
                    spans = _verified_spans(row, sources)
                    positions = {saved["source_block_id"]: saved["block_position"] for saved in sources}
                    locations = {}
                    for block_id, position in positions.items():
                        block_key = (row["parsed_document_id"], position)
                        if block_key not in decoded:
                            decoded[block_key] = _locations(locations_by_block[block_key])
                        locations[block_id] = decoded[block_key]
                    count = sum(len(locations[span.block_id]) for span in spans)
                    if output_locations + count > citations.MAX_LOCATIONS:
                        raise _LimitExceeded
                    spans = tuple(replace(span, source_locations=locations[span.block_id]) for span in spans)
                    cached[key] = _evidence(row, spans), count
                evidence, count = cached[key]
                if requests[index].expected_id is not None and requests[index].expected_id != evidence.evidence_id:
                    results[index] = CitationResolution(row["chunk_id"], "evidence_mismatch")
                elif output_locations + count > citations.MAX_LOCATIONS:
                    raise _LimitExceeded
                else:
                    output_locations += count
                    results[index] = CitationResolution(row["chunk_id"], "resolved", evidence)
            except _LimitExceeded:
                results[index] = CitationResolution(row["chunk_id"], "limit_exceeded")
            except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
                results[index] = CitationResolution(row["chunk_id"], "inconsistent_source")

    def _check_originals(self, results: list[CitationResolution]) -> tuple[CitationResolution, ...]:
        checked = {}
        read_budget = MAX_ORIGINAL_BYTES_PER_BATCH
        for index, result in enumerate(results):
            evidence = result.evidence
            if evidence is None:
                continue
            key = (evidence.source.content_sha256, evidence.original_size_bytes)
            if key not in checked:
                size = evidence.original_size_bytes
                if size > read_budget and key not in self._verified_revisions:
                    checked[key] = "limit_exceeded"
                else:
                    status, streamed = _verify_original(self.store, *key, self._verified_revisions)
                    if streamed:
                        if size > read_budget:
                            checked[key] = "limit_exceeded"
                        else:
                            read_budget -= size
                            checked[key] = status
                    else:
                        checked[key] = status
            if checked[key] != "resolved":
                results[index] = CitationResolution(result.chunk_id, checked[key])
        return tuple(results)
