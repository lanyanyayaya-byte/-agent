"""Read-only Chunk Browser service over published textbook parses and chunk sets."""

import json
import sqlite3
from dataclasses import dataclass

from app.chunking.hybrid import CHUNKER_VERSION
from app.models.embeddings import EmbeddingProfile
from app.parsing.models import ParseWarning
from app.services.document_parsing import PARSER_SPECS
from app.services.index_worker import INDEX_EMBEDDING_PROFILE
from app.storage.library import DocumentFormat, KnowledgeStore


@dataclass(frozen=True, slots=True)
class ChunkSectionSummary:
    section_path: tuple[str, ...]
    title: str
    chunk_count: int
    first_chunk_index: int
    page_start: int | None
    page_end: int | None


@dataclass(frozen=True, slots=True)
class ChunkBrowserItem:
    chunk_id: str
    chunk_index: int
    text: str
    char_count: int
    section_path: tuple[str, ...]
    title_context: tuple[str, ...]
    physical_pages: tuple[int, ...]
    page_labels: tuple[str, ...]
    location_summary: str
    embedded: bool
    warnings: tuple[ParseWarning, ...]
    has_prev: bool
    has_next: bool


@dataclass(frozen=True, slots=True)
class ChunkBrowserResponse:
    document_id: str
    textbook_name: str
    format: DocumentFormat
    document_version_id: str
    content_sha256: str
    parsed_document_id: int | None
    chunk_set_id: int | None
    parser_version: str | None
    chunker_version: str | None
    parse_spec_current: bool
    chunker_current: bool
    total_chunks: int
    filtered_total: int
    offset: int
    limit: int
    pages: tuple[int, ...]
    sections: tuple[ChunkSectionSummary, ...]
    chunks: tuple[ChunkBrowserItem, ...]


@dataclass(frozen=True, slots=True)
class ChunkNeighborsResponse:
    document_id: str
    chunk_set_id: int
    center_index: int
    previous: ChunkBrowserItem | None
    current: ChunkBrowserItem | None
    next: ChunkBrowserItem | None


def load_chunk_pages_map(
    db: sqlite3.Connection,
    chunk_set_id: int,
    chunk_indices: list[int] | None = None,
) -> dict[int, tuple[tuple[int, ...], tuple[str, ...]]]:
    """Load (physical_pages, page_labels) per chunk_index via chunk_sources -> parsed_blocks -> source_locations."""
    params: list[Any] = [chunk_set_id]
    where_extra = ""
    if chunk_indices is not None:
        if not chunk_indices:
            return {}
        placeholders = ",".join("?" for _ in chunk_indices)
        where_extra = f" AND cs.chunk_index IN ({placeholders})"
        params.extend(chunk_indices)
    rows = db.execute(
        f"""SELECT cs.chunk_index, sl.details
            FROM chunk_sources cs
            JOIN parsed_blocks pb
              ON pb.parsed_document_id = cs.parsed_document_id
             AND pb.block_id = cs.source_block_id
            JOIN source_locations sl
              ON sl.parsed_document_id = pb.parsed_document_id
             AND sl.block_position = pb.position
            WHERE cs.chunk_set_id = ?{where_extra}
            ORDER BY cs.chunk_index, cs.source_index, sl.position""",
        params,
    ).fetchall()
    raw_pages: dict[int, list[int]] = {}
    raw_labels: dict[int, list[str]] = {}
    seen_p: dict[int, set[int]] = {}
    seen_l: dict[int, set[str]] = {}
    for cidx, details_json in rows:
        try:
            loc = json.loads(details_json)
        except (json.JSONDecodeError, TypeError):
            continue
        p = loc.get("physical_page")
        if isinstance(p, int):
            sp = seen_p.setdefault(cidx, set())
            if p not in sp:
                sp.add(p)
                raw_pages.setdefault(cidx, []).append(p)
        lbl = loc.get("page_label")
        if isinstance(lbl, str) and lbl:
            sl = seen_l.setdefault(cidx, set())
            if lbl not in sl:
                sl.add(lbl)
                raw_labels.setdefault(cidx, []).append(lbl)
    result: dict[int, tuple[tuple[int, ...], tuple[str, ...]]] = {}
    all_keys = set(raw_pages) | set(raw_labels)
    for cidx in all_keys:
        ps = sorted(raw_pages.get(cidx, ()))
        ls = raw_labels.get(cidx, [])
        result[cidx] = (tuple(ps), tuple(ls))
    return result


def _parse_spans_pages(spans_json: str) -> tuple[tuple[int, ...], tuple[str, ...]]:
    try:
        spans = json.loads(spans_json)
    except json.JSONDecodeError:
        return (), ()
    pages: list[int] = []
    labels: list[str] = []
    seen_p: set[int] = set()
    seen_l: set[str] = set()
    for span in spans:
        for loc in span.get("source_locations", ()):
            p = loc.get("physical_page")
            if isinstance(p, int) and p not in seen_p:
                seen_p.add(p)
                pages.append(p)
            lbl = loc.get("page_label")
            if isinstance(lbl, str) and lbl and lbl not in seen_l:
                seen_l.add(lbl)
                labels.append(lbl)
    pages.sort()
    return tuple(pages), tuple(labels)


def _format_pages_summary(fmt: DocumentFormat, pages: tuple[int, ...], labels: tuple[str, ...]) -> str:
    if fmt == "pdf" and pages:
        if len(pages) == 1:
            base = f"第 {pages[0]} 页"
        else:
            base = f"第 {pages[0]}–{pages[-1]} 页"
        if labels and tuple(str(p) for p in pages) != labels:
            return f"{base}（页标签 {', '.join(labels[:3])}）"
        return base
    if fmt == "docx":
        return "Word 正文段落"
    if fmt == "md":
        return "Markdown 正文"
    return "正文"


class ChunkBrowserService:
    def __init__(self, store: KnowledgeStore, *, profile: EmbeddingProfile = INDEX_EMBEDDING_PROFILE):
        self.store = store
        self.profile = profile

    def _latest_chunk_set(self, db: sqlite3.Connection, document_id: str):
        doc_row = db.execute(
            """SELECT d.document_id, d.name, d.format, v.document_version_id, v.content_sha256
               FROM documents d
               JOIN document_versions v ON v.document_id = d.document_id
               WHERE d.document_id = ?
               ORDER BY v.rowid DESC LIMIT 1""",
            (document_id,),
        ).fetchone()
        if doc_row is None:
            return None
        doc_id, name, fmt, version_id, sha256 = doc_row
        cs_row = db.execute(
            """SELECT s.chunk_set_id, s.parsed_document_id, s.chunker_version, s.chunk_count,
                      p.parser_version, p.structure_version, p.config_key, p.warnings
               FROM chunk_sets s
               JOIN parsed_documents p ON p.parsed_document_id = s.parsed_document_id
               WHERE s.document_version_id = ? AND s.status = 'ready'
               ORDER BY s.chunk_set_id DESC LIMIT 1""",
            (version_id,),
        ).fetchone()
        return doc_id, name, fmt, version_id, sha256, cs_row

    def list_chunks(
        self,
        document_id: str,
        *,
        page: int | None = None,
        section: str | None = None,
        query: str | None = None,
        offset: int = 0,
        limit: int = 30,
    ) -> ChunkBrowserResponse | None:
        offset = max(0, offset)
        limit = max(1, min(100, limit))
        sec_filter = section.strip().casefold() if section and section.strip() else None
        q_filter = query.strip().casefold() if query and query.strip() else None

        with self.store.connect() as db:
            meta = self._latest_chunk_set(db, document_id)
            if meta is None:
                return None
            doc_id, name, fmt, version_id, sha256, cs_row = meta
            if cs_row is None:
                return ChunkBrowserResponse(
                    document_id=doc_id,
                    textbook_name=name,
                    format=fmt,
                    document_version_id=version_id,
                    content_sha256=sha256,
                    parsed_document_id=None,
                    chunk_set_id=None,
                    parser_version=None,
                    chunker_version=None,
                    parse_spec_current=False,
                    chunker_current=False,
                    total_chunks=0,
                    filtered_total=0,
                    offset=offset,
                    limit=limit,
                    pages=(),
                    sections=(),
                    chunks=(),
                )

            cs_id, parsed_id, chunker_ver, total_chunks, p_ver, p_struct, p_cfg, warnings_json = cs_row
            spec = PARSER_SPECS[fmt]
            parse_spec_current = (p_ver == spec.parser_version and p_struct == spec.structure_version and p_cfg == spec.config_key)
            chunker_current = chunker_ver == CHUNKER_VERSION

            all_warnings: list[ParseWarning] = []
            try:
                for w in json.loads(warnings_json or "[]"):
                    all_warnings.append(ParseWarning(w["code"], w["message"], w.get("physical_page")))
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
            warnings_by_page: dict[int, list[ParseWarning]] = {}
            for w in all_warnings:
                if w.physical_page is not None:
                    warnings_by_page.setdefault(w.physical_page, []).append(w)

            embedded_indices = {
                int(r[0])
                for r in db.execute(
                    """SELECT a.chunk_index
                       FROM chunk_embeddings a
                       JOIN embeddings v USING (profile_key, cache_key)
                       WHERE a.chunk_set_id = ? AND a.profile_key = ? AND v.vector IS NOT NULL""",
                    (cs_id, self.profile.key),
                ).fetchall()
            }

            pages_map = load_chunk_pages_map(db, cs_id)
            rows = db.execute(
                """SELECT chunk_id, chunk_index, text, section_path, title_context
                   FROM chunks
                   WHERE chunk_set_id = ?
                   ORDER BY chunk_index""",
                (cs_id,),
            ).fetchall()

        all_pages: set[int] = set()
        section_stats: dict[tuple[str, ...], list] = {}
        matched_items: list[ChunkBrowserItem] = []

        for cid, cidx, text, sec_json, title_json in rows:
            sec_path = tuple(json.loads(sec_json)) if sec_json else ()
            title_ctx = tuple(json.loads(title_json)) if title_json else ()
            pages, labels = pages_map.get(cidx, ((), ()))
            all_pages.update(pages)

            sec_key = sec_path
            if sec_key not in section_stats:
                section_stats[sec_key] = [
                    0, cidx,
                    pages[0] if pages else None,
                    pages[-1] if pages else None,
                ]
            st = section_stats[sec_key]
            st[0] += 1
            if pages:
                st[2] = pages[0] if st[2] is None else min(st[2], pages[0])
                st[3] = pages[-1] if st[3] is None else max(st[3], pages[-1])

            if page is not None and page not in pages:
                continue
            if sec_filter is not None:
                joined_sec = " / ".join(sec_path).casefold()
                if sec_filter not in joined_sec:
                    continue
            if q_filter is not None and q_filter not in text.casefold():
                continue

            chunk_warns: list[ParseWarning] = []
            seen_wc: set[tuple[str, int | None]] = set()
            for p in pages:
                for w in warnings_by_page.get(p, ()):
                    key = (w.code, w.physical_page)
                    if key not in seen_wc:
                        seen_wc.add(key)
                        chunk_warns.append(w)

            matched_items.append(ChunkBrowserItem(
                chunk_id=cid,
                chunk_index=cidx,
                text=text,
                char_count=len(text),
                section_path=sec_path,
                title_context=title_ctx,
                physical_pages=pages,
                page_labels=labels,
                location_summary=_format_pages_summary(fmt, pages, labels),
                embedded=cidx in embedded_indices,
                warnings=tuple(chunk_warns),
                has_prev=cidx > 0,
                has_next=cidx + 1 < total_chunks,
            ))

        sections = tuple(
            ChunkSectionSummary(
                section_path=sk,
                title=" / ".join(sk) if sk else "（未分节正文）",
                chunk_count=st[0],
                first_chunk_index=st[1],
                page_start=st[2],
                page_end=st[3],
            )
            for sk, st in sorted(section_stats.items(), key=lambda kv: kv[1][1])
        )
        return ChunkBrowserResponse(
            document_id=doc_id,
            textbook_name=name,
            format=fmt,
            document_version_id=version_id,
            content_sha256=sha256,
            parsed_document_id=parsed_id,
            chunk_set_id=cs_id,
            parser_version=p_ver,
            chunker_version=chunker_ver,
            parse_spec_current=parse_spec_current,
            chunker_current=chunker_current,
            total_chunks=total_chunks,
            filtered_total=len(matched_items),
            offset=offset,
            limit=limit,
            pages=tuple(sorted(all_pages)),
            sections=sections,
            chunks=tuple(matched_items[offset : offset + limit]),
        )

    def get_neighbors(self, document_id: str, chunk_index: int) -> ChunkNeighborsResponse | None:
        with self.store.connect() as db:
            meta = self._latest_chunk_set(db, document_id)
            if meta is None or meta[5] is None:
                return None
            doc_id, _name, fmt, _ver, _sha, cs_row = meta
            cs_id, _parsed_id, _cv, total_chunks, *_rest, warnings_json = cs_row
            indices = [idx for idx in (chunk_index - 1, chunk_index, chunk_index + 1) if 0 <= idx < total_chunks]
            if not indices:
                return ChunkNeighborsResponse(doc_id, cs_id, chunk_index, None, None, None)

            warnings_by_page: dict[int, list[ParseWarning]] = {}
            try:
                for w in json.loads(warnings_json or "[]"):
                    pw = ParseWarning(w["code"], w["message"], w.get("physical_page"))
                    if pw.physical_page is not None:
                        warnings_by_page.setdefault(pw.physical_page, []).append(pw)
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

            placeholders = ",".join("?" for _ in indices)
            embedded_indices = {
                int(r[0])
                for r in db.execute(
                    f"""SELECT a.chunk_index
                        FROM chunk_embeddings a
                        JOIN embeddings v USING (profile_key, cache_key)
                        WHERE a.chunk_set_id = ? AND a.profile_key = ?
                          AND a.chunk_index IN ({placeholders}) AND v.vector IS NOT NULL""",
                    (cs_id, self.profile.key, *indices),
                ).fetchall()
            }
            pages_map = load_chunk_pages_map(db, cs_id, indices)
            rows = db.execute(
                f"""SELECT chunk_id, chunk_index, text, section_path, title_context
                    FROM chunks
                    WHERE chunk_set_id = ? AND chunk_index IN ({placeholders})
                    ORDER BY chunk_index""",
                (cs_id, *indices),
            ).fetchall()

        by_idx: dict[int, ChunkBrowserItem] = {}
        for cid, cidx, text, sec_json, title_json in rows:
            sec_path = tuple(json.loads(sec_json)) if sec_json else ()
            title_ctx = tuple(json.loads(title_json)) if title_json else ()
            pages, labels = pages_map.get(cidx, ((), ()))
            chunk_warns: list[ParseWarning] = []
            for p in pages:
                chunk_warns.extend(warnings_by_page.get(p, ()))
            by_idx[cidx] = ChunkBrowserItem(
                chunk_id=cid,
                chunk_index=cidx,
                text=text,
                char_count=len(text),
                section_path=sec_path,
                title_context=title_ctx,
                physical_pages=pages,
                page_labels=labels,
                location_summary=_format_pages_summary(fmt, pages, labels),
                embedded=cidx in embedded_indices,
                warnings=tuple(chunk_warns),
                has_prev=cidx > 0,
                has_next=cidx + 1 < total_chunks,
            )
        return ChunkNeighborsResponse(
            document_id=doc_id,
            chunk_set_id=cs_id,
            center_index=chunk_index,
            previous=by_idx.get(chunk_index - 1),
            current=by_idx.get(chunk_index),
            next=by_idx.get(chunk_index + 1),
        )
