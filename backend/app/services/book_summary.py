"""P10.4 Full-book and chapter-level summary strategy (`book_summary`).

Builds a structure-aware section coverage plan from `parsed_sections` and published chunks
rather than relying on a small global Top-K vector slice.
"""

import asyncio
import json
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

from app.services.citation_resolver import CitationResolver, CitationScope, format_citation_view
from app.services.chunk_browser import load_chunk_pages_map
from app.storage.library import KnowledgeStore

SummaryMode = Literal["full_read", "section_representative", "hierarchical_summary"]

SKIP_SECTION_RE = re.compile(
    r"^(?:references|bibliography|acknowledg(?:e)?ments?|table of contents|contents|index|参考文献|目录|致谢|索引|前言|序言)$",
    re.I,
)
REFERENCES_QUERY_RE = re.compile(r"(?:参考文献|引用文献|bibliography|references|致谢|目录)", re.I)
CHAPTER_HINT_RE = re.compile(
    r"(?:第\s*([零〇一二三四五六七八九十百0-9]+)\s*[章篇节部分]|chapter\s+(\d+)|section\s+(\d+(?:\.\d+)?)|§\s*(\d+(?:\.\d+)?))",
    re.I,
)
CN_NUM_MAP = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def _cn_to_int(raw: str) -> int | None:
    if raw.isdigit():
        return int(raw)
    if raw == "十":
        return 10
    if "十" in raw:
        left, _, right = raw.partition("十")
        tens = CN_NUM_MAP.get(left, 1) if left else 1
        ones = CN_NUM_MAP.get(right, 0) if right else 0
        return tens * 10 + ones
    return CN_NUM_MAP.get(raw)


def _extract_chapter_filters(query: str) -> tuple[set[str], list[str]]:
    """Extract requested chapter numbers and explicit topic keywords from a summary query."""
    nums: set[str] = set()
    for m in CHAPTER_HINT_RE.finditer(query):
        token = next((g for g in m.groups() if g), None)
        if not token:
            continue
        nums.add(token.casefold())
        if token[0] in CN_NUM_MAP or token == "十":
            n = _cn_to_int(token)
            if n is not None:
                nums.add(str(n))
        elif token.isdigit():
            nums.add(token)
    # Remove boilerplate words to see if user named a specific chapter topic
    cleaned = re.sub(
        r"(?:请|帮我|完整|全面|系统|详细|按章节|整理|总结|概括|综述|归纳|梳理|这篇|本文|文献|教材|书中|书里|的|中|里|主要|内容|方法|原理|优缺点|适用场景|和|与|及|《[^》]+》|第[零〇一二三四五六七八九十百0-9]+[章篇节部分]|chapter\s+\d+|section\s+\d+(?:\.\d+)?)",
        " ",
        query,
        flags=re.I,
    )
    topic_terms = [t.strip() for t in re.split(r"[\s,，。、；;：:？?！!()（）]+", cleaned) if len(t.strip()) >= 2]
    return nums, topic_terms


@dataclass(frozen=True, slots=True)
class SummarySectionGroup:
    section_key: str
    title: str
    chunk_indices: tuple[int, ...]
    pages: tuple[int, ...]
    total_chars: int
    skipped_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SummarySelectedChunk:
    chunk_id: str
    chunk_index: int
    chunk_set_id: int
    document_id: str
    document_version_id: str
    section_key: str
    section_path: tuple[str, ...]
    text: str
    pages: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class BookSummaryPlan:
    document_id: str
    textbook_name: str
    document_version_id: str
    chunk_set_id: int
    summary_mode: SummaryMode
    total_chunks: int
    total_chars: int
    selected_chunks: tuple[SummarySelectedChunk, ...]
    covered_sections: tuple[str, ...]
    skipped_sections: tuple[dict[str, str], ...]
    covered_pages: tuple[int, ...]
    uncovered_pages: tuple[int, ...]
    all_sections: tuple[SummarySectionGroup, ...]
    elapsed_ms: float
    hierarchical_notes: tuple[str, ...] = ()

    def coverage_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "textbook_name": self.textbook_name,
            "summary_mode": self.summary_mode,
            "total_chunks": self.total_chunks,
            "selected_chunks": len(self.selected_chunks),
            "covered_sections": list(self.covered_sections),
            "skipped_sections": list(self.skipped_sections),
            "covered_pages_count": len(self.covered_pages),
            "uncovered_pages_count": len(self.uncovered_pages),
            "covered_page_range": (
                f"第 {self.covered_pages[0]}–{self.covered_pages[-1]} 页"
                if self.covered_pages else "—"
            ),
        }


class BookSummaryService:
    """Structure-aware textbook/paper summary planner with optional budgeted hierarchical synthesis."""

    SHORT_DOC_CHAR_BUDGET = 26_000
    PROMPT_CHAR_BUDGET = 34_000

    def __init__(self, store: KnowledgeStore, citation_resolver: CitationResolver):
        self.store = store
        self.citation_resolver = citation_resolver

    def _resolve_target_document(self, query: str, document_ids: Sequence[str] | None) -> tuple[str, str, str, int, str] | None:
        with self.store.connect() as db:
            if document_ids and len(document_ids) == 1:
                row = db.execute(
                    """SELECT d.document_id, d.name, v.document_version_id, s.chunk_set_id, d.format
                       FROM documents d
                       JOIN document_versions v ON v.document_id = d.document_id
                       JOIN chunk_sets s ON s.document_version_id = v.document_version_id AND s.status = 'ready'
                       WHERE d.document_id = ?
                       ORDER BY v.rowid DESC, s.chunk_set_id DESC LIMIT 1""",
                    (document_ids[0],),
                ).fetchone()
                return tuple(row) if row else None

            # Try matching explicit 《book name》 or filename in query
            rows = db.execute(
                """SELECT d.document_id, d.name, v.document_version_id, s.chunk_set_id, d.format
                   FROM documents d
                   JOIN document_versions v ON v.document_id = d.document_id
                   JOIN chunk_sets s ON s.document_version_id = v.document_version_id AND s.status = 'ready'
                   GROUP BY d.document_id
                   HAVING s.chunk_set_id = max(s.chunk_set_id)"""
            ).fetchall()
            scope_set = set(document_ids) if document_ids else None
            q_low = query.casefold()
            for row in rows:
                if scope_set is not None and row[0] not in scope_set:
                    continue
                name = str(row[1])
                stem = name.rsplit(".", 1)[0]
                if name.casefold() in q_low or (len(stem) >= 4 and stem.casefold() in q_low):
                    return tuple(row)
            if document_ids:
                for row in rows:
                    if row[0] == document_ids[0]:
                        return tuple(row)
        return None

    def build_plan(
        self,
        query: str,
        *,
        document_ids: Sequence[str] | None = None,
        max_chunks: int = 28,
        char_budget: int = PROMPT_CHAR_BUDGET,
    ) -> BookSummaryPlan | None:
        started = perf_counter()
        target = self._resolve_target_document(query, document_ids)
        if target is None:
            return None
        doc_id, book_name, ver_id, cs_id, _fmt = target

        with self.store.connect() as db:
            pages_map = load_chunk_pages_map(db, cs_id)
            rows = db.execute(
                """SELECT chunk_id, chunk_index, text, section_path
                   FROM chunks
                   WHERE chunk_set_id = ?
                   ORDER BY chunk_index""",
                (cs_id,),
            ).fetchall()

        if not rows:
            return None

        include_refs = bool(REFERENCES_QUERY_RE.search(query))
        chapter_nums, topic_terms = _extract_chapter_filters(query)

        # Parse chunks and group into structural sections (or contiguous page windows if headings are absent)
        parsed_rows = []
        all_pages: set[int] = set()
        non_empty_sections = 0
        for cid, cidx, text, sec_json in rows:
            sec_path = tuple(json.loads(sec_json)) if sec_json else ()
            pages, _labels = pages_map.get(cidx, ((), ()))
            all_pages.update(pages)
            if sec_path:
                non_empty_sections += 1
            parsed_rows.append((cid, cidx, text, sec_path, pages))

        use_page_windows = non_empty_sections < max(2, len(parsed_rows) // 10)
        groups_ordered: list[str] = []
        group_rows: dict[str, list] = defaultdict(list)

        for item in parsed_rows:
            cid, cidx, text, sec_path, pages = item
            if use_page_windows:
                if pages:
                    bucket = ((pages[0] - 1) // 10) * 10 + 1
                    gkey = f"第 {bucket}–{bucket + 9} 页"
                else:
                    bucket = (cidx // 15) + 1
                    gkey = f"正文分区 {bucket}"
            else:
                # Group by top-level or level-2 section so sub-methods stay visible!
                if len(sec_path) >= 2:
                    gkey = " / ".join(sec_path[:2])
                elif sec_path:
                    gkey = sec_path[0]
                else:
                    gkey = "引言与概述"
            if gkey not in group_rows:
                groups_ordered.append(gkey)
            group_rows[gkey].append(item)

        # Determine chapter-scoped vs full-book summary
        def _matches_chapter_filter(gkey: str, gitems: list) -> bool:
            if not chapter_nums:
                return True
            low = gkey.casefold()
            for num in chapter_nums:
                if (
                    f"第{num}章" in low
                    or f"第 {num} 章" in low
                    or f"第{num}节" in low
                    or f"chapter {num}" in low
                    or f"section {num}" in low
                    or f"§{num}" in low
                    or f"§ {num}" in low
                    or re.match(rf"^{re.escape(num)}(?:\.|\s|$)", low)
                ):
                    return True
            # Also check if a Chinese numeral matches digit or vice versa in heading
            for m in CHAPTER_HINT_RE.finditer(gkey):
                tok = next((g for g in m.groups() if g), None)
                if tok:
                    n = _cn_to_int(tok)
                    if (n is not None and str(n) in chapter_nums) or tok.casefold() in chapter_nums:
                        return True
            return False

        chapter_matched_keys = [k for k in groups_ordered if _matches_chapter_filter(k, group_rows[k])] if chapter_nums else []
        use_chapter_scope = bool(chapter_nums and chapter_matched_keys)

        # Build SummarySectionGroups with skip annotations
        all_sections: list[SummarySectionGroup] = []
        eligible_keys: list[str] = []
        skipped_sections: list[dict[str, str]] = []

        for gkey in groups_ordered:
            gitems = group_rows[gkey]
            gpages = sorted({p for item in gitems for p in item[4]})
            gchars = sum(len(item[2]) for item in gitems)
            leaf = gkey.split(" / ")[-1].strip()
            reason: str | None = None
            if not include_refs and SKIP_SECTION_RE.match(leaf):
                reason = "references_or_frontmatter_excluded"
            elif use_chapter_scope and gkey not in chapter_matched_keys:
                reason = "outside_requested_chapter"
            elif gchars < 80 and len(gitems) == 1:
                reason = "short_header_only"

            all_sections.append(SummarySectionGroup(
                section_key=gkey,
                title=gkey,
                chunk_indices=tuple(item[1] for item in gitems),
                pages=tuple(gpages),
                total_chars=gchars,
                skipped_reason=reason,
            ))
            if reason is None:
                eligible_keys.append(gkey)
            else:
                skipped_sections.append({"section": gkey, "reason": reason})

        if not eligible_keys:
            eligible_keys = list(groups_ordered)
            skipped_sections = []

        # If a non-chapter topic hint matches a subset of sections in a very long book (> 12 sections), boost them
        eligible_chars = sum(sum(len(it[2]) for it in group_rows[k]) for k in eligible_keys)
        total_chars = sum(len(it[2]) for it in parsed_rows)

        selected: list[SummarySelectedChunk] = []
        covered_sections: list[str] = []
        covered_pages: set[int] = set()

        if eligible_chars <= self.SHORT_DOC_CHAR_BUDGET and sum(len(group_rows[k]) for k in eligible_keys) <= max_chunks * 2:
            # Short paper / chapter: read almost complete eligible body across every section!
            summary_mode: SummaryMode = "full_read"
            used_chars = 0
            for gkey in eligible_keys:
                gitems = group_rows[gkey]
                added_here = False
                for cid, cidx, text, sec_path, pages in gitems:
                    if used_chars + len(text) > char_budget and selected:
                        break
                    selected.append(SummarySelectedChunk(cid, cidx, cs_id, doc_id, ver_id, gkey, sec_path, text, pages))
                    used_chars += len(text)
                    covered_pages.update(pages)
                    added_here = True
                if added_here:
                    covered_sections.append(gkey)
                else:
                    skipped_sections.append({"section": gkey, "reason": "context_budget_limit"})
        else:
            # Medium/Long textbook: round-robin representative coverage across ALL major sections!
            summary_mode = "section_representative"
            # Pick representative indices per section: first substantive chunk, middle chunk, and topic-matched chunks
            section_queues: list[tuple[str, list]] = []
            for gkey in eligible_keys:
                gitems = group_rows[gkey]
                if not gitems:
                    continue
                scored_order: list = []
                seen_idx: set[int] = set()
                # Always include opening substantive chunk of the section
                first = next((it for it in gitems if len(it[2].strip()) >= 140), gitems[0])
                scored_order.append(first)
                seen_idx.add(first[1])
                # Include topic-matching chunks if query mentioned specific terms
                if topic_terms:
                    for it in gitems:
                        if it[1] in seen_idx:
                            continue
                        low_t = it[2].casefold()
                        if any(term.casefold() in low_t for term in topic_terms):
                            scored_order.append(it)
                            seen_idx.add(it[1])
                # Include evenly spaced chunks across the section (middle & closing)
                if len(gitems) >= 3:
                    for pos in (len(gitems) // 3, (2 * len(gitems)) // 3, len(gitems) - 1):
                        it = gitems[pos]
                        if it[1] not in seen_idx:
                            scored_order.append(it)
                            seen_idx.add(it[1])
                for it in gitems:
                    if it[1] not in seen_idx:
                        scored_order.append(it)
                        seen_idx.add(it[1])
                section_queues.append((gkey, scored_order))

            used_chars = 0
            covered_set: set[str] = set()
            # Round-robin pass so EVERY major section gets at least 1-2 chunks before any section takes more!
            max_rounds = 4
            for rnd in range(max_rounds):
                for gkey, qlist in section_queues:
                    if len(selected) >= max_chunks or used_chars >= char_budget:
                        break
                    if rnd >= len(qlist):
                        continue
                    cid, cidx, text, sec_path, pages = qlist[rnd]
                    if used_chars + len(text) > char_budget and selected:
                        continue
                    selected.append(SummarySelectedChunk(cid, cidx, cs_id, doc_id, ver_id, gkey, sec_path, text, pages))
                    used_chars += len(text)
                    covered_pages.update(pages)
                    if gkey not in covered_set:
                        covered_set.add(gkey)
                        covered_sections.append(gkey)
                if len(selected) >= max_chunks or used_chars >= char_budget:
                    break

            for gkey in eligible_keys:
                if gkey not in covered_set:
                    skipped_sections.append({"section": gkey, "reason": "context_budget_limit"})

        # Keep selected chunks in natural reading order
        selected.sort(key=lambda c: c.chunk_index)
        uncovered = sorted(all_pages - covered_pages)
        elapsed_ms = round((perf_counter() - started) * 1000, 1)

        return BookSummaryPlan(
            document_id=doc_id,
            textbook_name=book_name,
            document_version_id=ver_id,
            chunk_set_id=cs_id,
            summary_mode=summary_mode,
            total_chunks=len(parsed_rows),
            total_chars=total_chars,
            selected_chunks=tuple(selected),
            covered_sections=tuple(covered_sections),
            skipped_sections=tuple(skipped_sections),
            covered_pages=tuple(sorted(covered_pages)),
            uncovered_pages=tuple(uncovered),
            all_sections=tuple(all_sections),
            elapsed_ms=elapsed_ms,
        )

    async def maybe_hierarchical_condense(
        self,
        plan: BookSummaryPlan,
        query: str,
        models: Any,
        *,
        max_batches: int = 3,
    ) -> BookSummaryPlan:
        """For long books whose representative chunks exceed single-turn comfort, run budgeted section condensation."""
        total_sel_chars = sum(len(c.text) for c in plan.selected_chunks)
        if models is None or not getattr(models, "configured", False) or total_sel_chars <= 22_000:
            return plan

        # Group selected chunks into up to `max_batches` sequential section batches
        chunks = list(plan.selected_chunks)
        batch_size = max(1, (len(chunks) + max_batches - 1) // max_batches)
        batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)][:max_batches]

        notes: list[str] = []
        for b_idx, batch in enumerate(batches, 1):
            sec_names = list(dict.fromkeys(c.section_key for c in batch))
            blocks = []
            for c in batch:
                p_str = f"第 {c.pages[0]}–{c.pages[-1]} 页" if len(c.pages) > 1 else (f"第 {c.pages[0]} 页" if c.pages else "")
                blocks.append(f"[Chunk #{c.chunk_index} | 章节: {c.section_key} | {p_str}]\n{c.text[:900]}")
            prompt = (
                f"你是学术教材摘要助手。请基于以下《{plan.textbook_name}》第 {b_idx} 批章节原文，"
                f"紧扣用户问题「{query}」，按章节提炼核心概念、算法原理、优缺点与适用场景，"
                "并在每条要点末尾保留对应的页码和 Chunk 编号，不得编造原文没有的内容（控制在 450 字以内）：\n\n"
                + "\n\n".join(blocks)
            )
            try:
                async with asyncio.timeout(25.0):
                    async with models.open_model() as model:
                        resp = await model.ainvoke(prompt, max_tokens=550, temperature=0)
                content = resp.content if isinstance(resp.content, str) else str(resp.content)
                notes.append(f"### 分段摘要 {b_idx}（覆盖章节：{'、'.join(sec_names)}）\n{content.strip()}")
            except Exception:
                # Fall back cleanly to direct representative chunks
                return plan

        return BookSummaryPlan(
            document_id=plan.document_id,
            textbook_name=plan.textbook_name,
            document_version_id=plan.document_version_id,
            chunk_set_id=plan.chunk_set_id,
            summary_mode="hierarchical_summary",
            total_chunks=plan.total_chunks,
            total_chars=plan.total_chars,
            selected_chunks=plan.selected_chunks,
            covered_sections=plan.covered_sections,
            skipped_sections=plan.skipped_sections,
            covered_pages=plan.covered_pages,
            uncovered_pages=plan.uncovered_pages,
            all_sections=plan.all_sections,
            elapsed_ms=plan.elapsed_ms,
            hierarchical_notes=tuple(notes),
        )
