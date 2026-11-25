"""Version-consistent keyword, semantic, unified RRF, and optional Reranker retrieval."""

import asyncio
import math
import re
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Literal

from app.chunking.models import ChunkSourceSpan
from app.models import ModelError
from app.models.embeddings import EmbeddingError, EmbeddingProfile
from app.models.reranker import RerankError, SiliconFlowReranker
from app.services.keyword_search import (
    MAX_CANDIDATES, MAX_TOP_K, SELECT_SETS, KeywordHit, KeywordSearchService, prepare_keyword_query,
)
from app.services.vector_retrieval import (
    VECTOR_SCOPE_FIELDS, VectorHit, VectorRetrievalError, VectorRetrievalService, VectorScope,
    check_vector_profile_active, complete_vector_scope,
)
from app.storage.chunks import read_chunk_source_spans
from app.storage.retrieval_scope import CURRENT_VERSION, scope_filters

RetrievalMode = Literal["keyword", "vector", "hybrid"]
CoverageMode = Literal["global", "per_document"]
Channel = Literal["keyword", "vector"]
ChannelStatus = Literal["hybrid", "keyword_only", "vector_only", "partial", "unavailable"]
EvidenceStatus = Literal["evidence", "no_evidence", "not_returned", "not_assessed", "unavailable"]

# Soft auto-selection count when comparing across the entire library (no longer a hard crash limit).
MAX_COMPARISON_DOCUMENTS = 8
COMPARE_AUTO_TOP_BOOKS = 5
RERANK_LOW_RELEVANCE_FLOOR = 0.02
RERANK_STRONG_SIGNAL_THRESHOLD = 0.20
REFERENCE_SECTION_RE = re.compile(r"^(?:references|bibliography|works cited|参考文献|引用文献)$", re.I)
REFERENCE_QUERY_RE = re.compile(r"(?:参考文献|引用文献|bibliography|references|citation list|论文列表|引文)", re.I)
CITATION_LINE_RE = re.compile(r"(?:^|\n)\s*(?:\[\d{1,3}\]|\d{1,3}\.\s+[A-Z][a-z]+,)")


def _query_wants_references(query: str) -> bool:
    return bool(REFERENCE_QUERY_RE.search(query))


def _is_reference_chunk(section_path: tuple[str, ...], body_text: str) -> bool:
    if section_path and REFERENCE_SECTION_RE.match(section_path[-1].strip()):
        return True
    if len(CITATION_LINE_RE.findall(body_text)) >= 3:
        return True
    return False


class HybridRetrievalError(ModelError):
    pass


@dataclass(frozen=True, slots=True)
class HybridRetrievalConfig:
    candidate_k: int = 60
    rrf_k: int = 60
    per_document_quota: int = 2
    min_vector_similarity: float = 0.45
    overlap_threshold: float = 0.65
    rerank_candidates: int = 30
    compare_auto_books: int = COMPARE_AUTO_TOP_BOOKS
    bm25_or_rrf_weight: float = 0.72
    reference_demotion_factor: float = 0.55

    def __post_init__(self):
        if (type(self.candidate_k) is not int or not 1 <= self.candidate_k <= MAX_CANDIDATES
                or type(self.rrf_k) is not int or not 1 <= self.rrf_k <= 10_000
                or type(self.per_document_quota) is not int or not 1 <= self.per_document_quota <= MAX_TOP_K
                or type(self.min_vector_similarity) not in (int, float) or not 0 < self.min_vector_similarity <= 1
                or type(self.overlap_threshold) not in (int, float) or not 0 < self.overlap_threshold <= 1
                or type(self.rerank_candidates) is not int or not 1 <= self.rerank_candidates <= 100
                or type(self.compare_auto_books) is not int or not 1 <= self.compare_auto_books <= 20):
            raise ValueError("Invalid hybrid retrieval configuration.")


@dataclass(frozen=True, slots=True)
class ChannelRank:
    channel: Channel
    rank: int
    score: float


@dataclass(frozen=True, slots=True)
class HybridHit:
    chunk_id: str
    document_id: str
    document_version_id: str
    chunk_set_id: int
    parsed_document_id: int
    chunk_index: int
    content_sha256: str
    body_text: str
    section_heading_block_id: str | None
    section_path: tuple[str, ...]
    title_context: tuple[str, ...]
    keyword_matched_in: tuple[Literal["body", "title"], ...]
    channel_ranks: tuple[ChannelRank, ...]
    fusion_score: float
    source_spans: tuple[ChunkSourceSpan, ...]
    rerank_score: float | None = None
    rerank_rank: int | None = None


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    chunk_id: str
    document_id: str
    chunk_index: int
    section_path: tuple[str, ...]
    bm25_raw: float | None
    bm25_rank: int | None
    cosine: float | None
    vector_rank: int | None
    rrf_score: float
    rrf_rank: int
    rerank_score: float | None = None
    rerank_rank: int | None = None
    selected: bool = False
    exclusion_reason: str | None = None
    is_reference_chunk: bool = False
    snippet: str | None = None


@dataclass(frozen=True, slots=True)
class DocumentCoverage:
    document_id: str
    document_version_id: str | None
    chunk_set_id: int | None
    keyword_indexed: bool
    vector_indexed: bool | None
    searched_channels: tuple[Channel, ...]
    evidence_status: EvidenceStatus
    returned_count: int


@dataclass(frozen=True, slots=True)
class RetrievalDiagnostic:
    code: str
    message: str
    retry_after: float = 0
    observed_revision: str | None = None
    http_status: int | None = None

    @classmethod
    def from_error(cls, error: ModelError):
        if isinstance(error, EmbeddingError):
            return cls(error.code, str(error), error.retry_after, error.observed_revision, error.http_status)
        if isinstance(error, RerankError):
            return cls(error.code, str(error), http_status=error.http_status)
        return cls(error.code, str(error))


@dataclass(frozen=True, slots=True)
class RetrievalTimings:
    scope_ms: float = 0
    fts_ms: float = 0
    vector_ms: float = 0
    query_embedding_ms: float = 0
    fusion_ms: float = 0
    rerank_ms: float = 0
    source_ms: float = 0


@dataclass(frozen=True, slots=True)
class HybridResult:
    retrieval_mode: RetrievalMode
    coverage_mode: CoverageMode
    channel_status: ChannelStatus
    hits: tuple[HybridHit, ...]
    documents: tuple[DocumentCoverage, ...]
    diagnostic: RetrievalDiagnostic | None
    timings: RetrievalTimings
    query_prompt_tokens: int | None = None
    query_total_tokens: int | None = None
    rerank_used: bool = False
    rerank_diagnostic: RetrievalDiagnostic | None = None
    candidates: tuple[CandidateRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class _Document:
    document_id: str
    version_id: str | None
    scope: VectorScope | None
    keyword_indexed: bool
    vector_indexed: bool | None


@dataclass(frozen=True, slots=True)
class _Request:
    raw_query: str
    phrase: str | None
    query_variants: tuple[str, ...]
    extra_terms: tuple[str, ...]
    exact_phrase: bool
    top_k: int
    retrieval_mode: RetrievalMode
    coverage_mode: CoverageMode
    profile: EmbeddingProfile | None
    document_ids: tuple[str, ...] | None
    filter_sql: str
    filter_values: tuple
    pinned: bool
    min_vector_similarity: float
    candidate_k: int
    per_document_quota: int


@dataclass(slots=True)
class _Candidate:
    hit: KeywordHit | VectorHit
    ranks: list[ChannelRank]
    score: float = 0.0
    rerank_score: float | None = None
    rerank_rank: int | None = None
    is_reference_chunk: bool = False

    def max_channel_score(self, channel: Channel) -> float:
        vals = [r.score for r in self.ranks if r.channel == channel]
        return max(vals) if vals else -1.0

    def rank_for(self, channel: Channel) -> int | None:
        vals = [r.rank for r in self.ranks if r.channel == channel]
        return min(vals) if vals else None

    @property
    def order(self):
        hit = self.hit
        if self.rerank_score is not None:
            return (0, -self.rerank_score, -self.score, hit.chunk_set_id, hit.chunk_index, hit.chunk_id)
        # Break RRF ties by raw vector cosine & BM25 score rather than chunk_set_id (import order)!
        return (
            1,
            -round(self.score, 12),
            -round(self.max_channel_score("vector"), 6),
            -round(self.max_channel_score("keyword"), 6),
            min(rank.rank for rank in self.ranks),
            hit.chunk_set_id,
            hit.chunk_index,
            hit.chunk_id,
        )


@dataclass(slots=True)
class _LocalRetrievalBundle:
    documents: tuple[_Document, ...]
    ranked: list[_Candidate]
    excluded_overlap: set[str]
    sources: dict[tuple[int, int], list[ChunkSourceSpan]]
    vector_ran: bool
    diagnostic: RetrievalDiagnostic | None
    timings: RetrievalTimings
    scopes: dict[str, VectorScope | None]
    active_compare_doc_ids: set[str] | None


def _selected_documents(db: sqlite3.Connection, request: _Request) -> dict[str, str | None]:
    join_sets = "JOIN chunk_sets AS s USING (document_version_id)" if request.pinned else ""
    rows = db.execute(f"""SELECT v.document_id, v.document_version_id, count(*)
        FROM document_versions AS v JOIN documents AS d USING (document_id) {join_sets}
        WHERE 1 = 1 {request.filter_sql}
        GROUP BY v.document_id ORDER BY v.document_id""", request.filter_values).fetchall()
    if any(count > 1 for _, _, count in rows):
        raise ValueError("Hybrid retrieval requires one pinned Chunk Set per textbook.")
    documents = {doc_id: version_id for doc_id, version_id, _ in rows}
    if not request.pinned:
        for doc_id in request.document_ids or ():
            documents.setdefault(doc_id, None)
    return documents


def _select_scope(db: sqlite3.Connection, request: _Request) -> tuple[tuple[_Document, ...], RetrievalDiagnostic | None]:
    documents = _selected_documents(db, request)
    if not documents:
        return (), None
    keyword_sets = {}
    vector_sets = {}
    diagnostic = None
    for row in db.execute(f"{SELECT_SETS}{request.filter_sql}", request.filter_values):
        scope = VectorScope(*row[:-1])
        keyword_sets[scope.chunk_set_id] = scope
    if request.profile is not None:
        try:
            check_vector_profile_active(db, request.profile.key)
            sql, params = complete_vector_scope(request.profile, request.filter_sql, request.filter_values)
            for row in db.execute(f"{sql} SELECT {VECTOR_SCOPE_FIELDS} FROM eligible WHERE chunk_count > 0", params):
                scope = VectorScope(*row)
                vector_sets[scope.chunk_set_id] = scope
        except EmbeddingError as error:
            if request.retrieval_mode == "vector":
                raise
            diagnostic = RetrievalDiagnostic.from_error(error)
        except sqlite3.DatabaseError:
            error = VectorRetrievalError("VECTOR_QUERY_FAILED", "无法读取本地向量索引状态。")
            if request.retrieval_mode == "vector":
                raise error from None
            diagnostic = RetrievalDiagnostic.from_error(error)
            vector_sets.clear()

    latest = {}
    # Prefer a keyword-ready set for hybrid so an API failure can use the same set.
    for eligible in (vector_sets, keyword_sets) if request.retrieval_mode != "vector" else (vector_sets,):
        for scope in sorted(eligible.values(), key=lambda item: item.chunk_set_id):
            latest[scope.document_id] = scope
    return tuple(
        _Document(doc_id, documents[doc_id], latest.get(doc_id),
                  latest[doc_id].chunk_set_id in keyword_sets if doc_id in latest else False,
                  (doc_id in latest and latest[doc_id].chunk_set_id in vector_sets)
                  if request.profile is not None else None)
        for doc_id in sorted(documents)
    ), diagnostic


def _merge_keyword_variants(variant_lists: Sequence[tuple[KeywordHit, ...]], limit: int) -> tuple[KeywordHit, ...]:
    """Merge multi-variant BM25 lists taking each chunk's best score without inflating ranks."""
    best: dict[str, KeywordHit] = {}
    for hits in variant_lists:
        for hit in hits:
            prev = best.get(hit.chunk_id)
            if prev is None or hit.score > prev.score:
                best[hit.chunk_id] = hit
    ordered = sorted(best.values(), key=lambda h: (-h.score, h.chunk_set_id, h.chunk_index, h.chunk_id))
    return tuple(ordered[:limit])


def _merge_vector_variants(variant_lists: Sequence[tuple[VectorHit, ...]], limit: int) -> tuple[VectorHit, ...]:
    """Merge multi-variant Vector lists taking each chunk's max cosine similarity."""
    best: dict[str, VectorHit] = {}
    for hits in variant_lists:
        for hit in hits:
            prev = best.get(hit.chunk_id)
            if prev is None or hit.similarity > prev.similarity:
                best[hit.chunk_id] = hit
    ordered = sorted(best.values(), key=lambda h: (-h.similarity, h.chunk_set_id, h.chunk_index, h.chunk_id))
    return tuple(ordered[:limit])


def _fuse(
    keywords: tuple[KeywordHit, ...],
    vectors: tuple[VectorHit, ...],
    rrf_k: int,
    *,
    keyword_weight: float = 1.0,
    demote_references: bool = False,
    ref_factor: float = 0.55,
) -> list[_Candidate]:
    candidates: dict[str, _Candidate] = {}
    for rank, hit in enumerate(keywords, 1):
        candidates[hit.chunk_id] = _Candidate(hit, [ChannelRank("keyword", rank, hit.score)])
    for rank, hit in enumerate(vectors, 1):
        candidate = candidates.setdefault(hit.chunk_id, _Candidate(hit, []))
        candidate.ranks.append(ChannelRank("vector", rank, hit.similarity))
    for candidate in candidates.values():
        candidate.is_reference_chunk = _is_reference_chunk(candidate.hit.section_path, candidate.hit.body_text)
        base = math.fsum(
            (keyword_weight if rank.channel == "keyword" else 1.0) / (rrf_k + rank.rank)
            for rank in candidate.ranks
        )
        if demote_references and candidate.is_reference_chunk:
            base *= ref_factor
        candidate.score = base
    return sorted(candidates.values(), key=lambda item: item.order)


def _overlap(first: Sequence[ChunkSourceSpan], second: Sequence[ChunkSourceSpan]) -> float:
    lengths = [sum(span.block_end - span.block_start for span in spans) for spans in (first, second)]
    shorter = min(lengths)
    shared = sum(max(0, min(a.block_end, b.block_end) - max(a.block_start, b.block_start))
                 for a in first for b in second if a.block_id == b.block_id)
    return shared / shorter if shorter else 0


def _deduplicate(ranked: Sequence[_Candidate], sources: dict, threshold: float) -> tuple[list[_Candidate], set[str]]:
    kept = []
    excluded = set()
    accepted = set()
    for item in ranked:
        hit = item.hit
        location = (hit.chunk_set_id, hit.chunk_index)
        neighbors = ((hit.chunk_set_id, hit.chunk_index + offset) for offset in (-1, 1))
        if any(other in accepted and _overlap(sources[location], sources[other]) >= threshold for other in neighbors):
            excluded.add(hit.chunk_id)
            continue
        accepted.add(location)
        kept.append(item)
    return kept, excluded


def _read_sources(db: sqlite3.Connection, locations) -> dict[tuple[int, int], list[ChunkSourceSpan]]:
    indices = defaultdict(set)
    for set_id, index in locations:
        indices[set_id].add(index)
    sources = {}
    for set_id, group in indices.items():
        spans = read_chunk_source_spans(db, set_id, sorted(group))
        if spans.keys() != group:
            raise HybridRetrievalError("HYBRID_SOURCE_UNAVAILABLE", "无法读取检索结果的完整来源。")
        sources.update({(set_id, index): spans[index] for index in group})
    return sources


def _locations(items: Sequence[_Candidate]) -> set[tuple[int, int]]:
    return {(item.hit.chunk_set_id, item.hit.chunk_index) for item in items}


def _select_hits_unified(
    ranked: Sequence[_Candidate],
    top_k: int,
    quota: int,
    coverage_mode: CoverageMode,
) -> tuple[list[_Candidate], dict[str, str]]:
    """Select final hits using unified global ranking + soft per-document quota for compare mode."""
    exclusions: dict[str, str] = {}
    if not ranked:
        return [], exclusions

    # Filter clearly irrelevant candidates when reranker found strong matches
    reranked_scores = [c.rerank_score for c in ranked if c.rerank_score is not None]
    has_strong_rerank = bool(reranked_scores and max(reranked_scores) >= RERANK_STRONG_SIGNAL_THRESHOLD)
    eligible: list[_Candidate] = []
    for item in ranked:
        if has_strong_rerank and item.rerank_score is not None and item.rerank_score < RERANK_LOW_RELEVANCE_FLOOR:
            exclusions[item.hit.chunk_id] = "rerank_low_relevance"
            continue
        eligible.append(item)
    if not eligible:
        eligible = list(ranked)
        exclusions.clear()

    if coverage_mode == "global":
        selected = eligible[:top_k]
        for item in eligible[top_k:]:
            exclusions[item.hit.chunk_id] = "topk_cutoff"
        return selected, exclusions

    # Compare / per_document mode: soft per-document allocation ordered by UNIFIED relevance!
    # Step 1: group by document preserving unified rank order.
    by_doc: dict[str, list[_Candidate]] = defaultdict(list)
    for item in eligible:
        by_doc[item.hit.document_id].append(item)

    # Order documents by their strongest unified candidate (never by chunk_set_id / import order!)
    doc_order = sorted(by_doc.keys(), key=lambda did: by_doc[did][0].order)

    # Round-robin up to soft quota per relevant document, then fill surplus by unified order.
    selected: list[_Candidate] = []
    selected_ids: set[str] = set()
    for slot in range(quota):
        for did in doc_order:
            items = by_doc[did]
            if slot < len(items):
                cand = items[slot]
                if len(selected) < top_k:
                    selected.append(cand)
                    selected_ids.add(cand.hit.chunk_id)
                else:
                    exclusions[cand.hit.chunk_id] = "topk_cutoff"

    if len(selected) < top_k:
        for cand in eligible:
            if cand.hit.chunk_id in selected_ids:
                continue
            if len(selected) < top_k:
                selected.append(cand)
                selected_ids.add(cand.hit.chunk_id)
            else:
                exclusions[cand.hit.chunk_id] = "topk_cutoff"
    else:
        for cand in eligible:
            if cand.hit.chunk_id not in selected_ids:
                exclusions.setdefault(cand.hit.chunk_id, "per_document_quota")

    selected.sort(key=lambda item: item.order)
    return selected, exclusions


def _select_top_compare_books(
    documents: Sequence[_Document],
    coarse_kw: tuple[KeywordHit, ...],
    coarse_vec: tuple[VectorHit, ...],
    max_books: int,
) -> list[_Document]:
    """Pick the most relevant books for full-library comparison without forcing irrelevant books."""
    book_scores: dict[str, float] = defaultdict(float)
    book_hits: dict[str, int] = defaultdict(int)
    for rank, hit in enumerate(coarse_vec, 1):
        # Reward strongest vector match + multi-hit support
        book_scores[hit.document_id] = max(book_scores[hit.document_id], hit.similarity) + 0.03 / (1 + rank * 0.1)
        book_hits[hit.document_id] += 1
    for rank, hit in enumerate(coarse_kw, 1):
        # Ensure BM25-only books with genuine lexical hits can also qualify
        kw_contrib = 0.48 + 0.12 / (1 + rank * 0.15)
        book_scores[hit.document_id] = max(book_scores[hit.document_id], kw_contrib) + 0.02
        book_hits[hit.document_id] += 1

    if not book_scores:
        return [doc for doc in documents if doc.keyword_indexed or doc.vector_indexed][:max_books]

    ranked_ids = sorted(book_scores.keys(), key=lambda did: (-book_scores[did], -book_hits[did], did))[:max_books]
    by_id = {doc.document_id: doc for doc in documents}
    return [by_id[did] for did in ranked_ids if did in by_id]


def _channels(document: _Document, request: _Request, vector_ran: bool) -> tuple[Channel, ...]:
    channels: list[Channel] = []
    if document.keyword_indexed and request.retrieval_mode != "vector":
        channels.append("keyword")
    if document.vector_indexed and vector_ran:
        channels.append("vector")
    return tuple(channels)


def _coverage_status(channels, count, candidates, mode):
    if not channels:
        return "unavailable"
    if count:
        return "evidence"
    if candidates:
        return "not_returned"
    return "no_evidence" if mode == "per_document" else "not_assessed"


def _channel_status(coverage: Sequence[DocumentCoverage]) -> ChannelStatus:
    channels = {item.searched_channels for item in coverage if item.searched_channels}
    if not channels:
        return "unavailable"
    if channels == {("keyword",)}:
        return "keyword_only"
    if channels == {("vector",)}:
        return "vector_only"
    if channels == {("keyword", "vector")}:
        return "hybrid"
    return "partial"


class HybridRetrievalService:
    def __init__(
        self,
        keyword: KeywordSearchService,
        vector: VectorRetrievalService,
        *,
        config: HybridRetrievalConfig | None = None,
        reranker: SiliconFlowReranker | None = None,
    ):
        if keyword.store.path != vector.store.path:
            raise ValueError("Hybrid retrieval requires both services to use the same knowledge database.")
        self.keyword = keyword
        self.vector = vector
        self.config = config if config is not None else HybridRetrievalConfig()
        self.reranker = reranker

    async def search(
        self,
        query: str,
        top_k: int = 10,
        *,
        document_ids: Sequence[str] | None = None,
        chunk_set_ids: Sequence[int] | None = None,
        retrieval_mode: RetrievalMode = "hybrid",
        coverage_mode: CoverageMode = "global",
        profile: EmbeddingProfile | None = None,
        query_vector: Sequence[float] | None = None,
        query_variants: Sequence[str] = (),
        extra_terms: Sequence[str] = (),
        exact_phrase: bool = False,
        use_rerank: bool = False,
        rerank_query: str | None = None,
        min_vector_similarity: float | None = None,
        candidate_k: int | None = None,
        per_document_quota: int | None = None,
    ) -> HybridResult:
        """Search current published sets, or pin selected historical Chunk Set IDs."""
        if retrieval_mode not in ("keyword", "vector", "hybrid") or coverage_mode not in ("global", "per_document"):
            raise ValueError("Invalid retrieval or coverage mode.")
        phrase = prepare_keyword_query(query)
        if not query.strip() or type(top_k) is not int or not 1 <= top_k <= MAX_TOP_K:
            raise ValueError("Invalid hybrid query or result limit.")
        if retrieval_mode == "keyword":
            if query_vector is not None:
                raise ValueError("Keyword retrieval does not accept a query vector.")
            profile = None
        elif not isinstance(profile, EmbeddingProfile):
            raise ValueError("An explicit embedding profile is required.")
        if query_vector is not None and profile is not None:
            import numpy as np
            from app.services.vector_retrieval import _unit_query
            _unit_query(np, query_vector, profile.dimensions)
        filter_sql, values = scope_filters(document_ids, chunk_set_ids, label="hybrid")
        documents = None if document_ids is None else tuple(dict.fromkeys(document_ids))
        pinned = chunk_set_ids is not None

        # Deduplicate query variants (up to 3) keeping primary query first
        variants_clean: list[str] = []
        seen_v: set[str] = set()
        for qv in (query, *query_variants):
            s = " ".join(qv.strip().split())
            if s and s.casefold() not in seen_v and len(s) <= 256:
                seen_v.add(s.casefold())
                variants_clean.append(s)
                if len(variants_clean) >= 3:
                    break

        eff_min_sim = self.config.min_vector_similarity if min_vector_similarity is None else float(min_vector_similarity)
        eff_cand_k = self.config.candidate_k if candidate_k is None else int(candidate_k)
        eff_quota = self.config.per_document_quota if per_document_quota is None else int(per_document_quota)
        request = _Request(
            raw_query=query,
            phrase=phrase,
            query_variants=tuple(variants_clean),
            extra_terms=tuple(extra_terms),
            exact_phrase=exact_phrase,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            coverage_mode=coverage_mode,
            profile=profile,
            document_ids=documents,
            filter_sql=filter_sql if pinned else filter_sql + CURRENT_VERSION,
            filter_values=values,
            pinned=pinned,
            min_vector_similarity=eff_min_sim,
            candidate_k=eff_cand_k,
            per_document_quota=eff_quota,
        )

        needs_embedding = profile is not None and query_vector is None
        initial_vectors = [query_vector] if query_vector is not None else None
        local = await asyncio.to_thread(self._search_local, request, initial_vectors, needs_embedding, None, False)

        query_embedding_ms = 0.0
        prompt_tokens: int | None = None
        total_tokens: int | None = None
        if isinstance(local, float):
            assert profile is not None
            preflight_ms = local
            batch, diagnostic = None, None
            start = perf_counter()
            try:
                batch = await self.vector.embed_queries(request.query_variants or (query,), profile=profile)
            except (EmbeddingError, VectorRetrievalError) as error:
                if retrieval_mode == "vector":
                    raise
                diagnostic = RetrievalDiagnostic.from_error(error)
            query_embedding_ms = (perf_counter() - start) * 1000
            if batch is not None:
                prompt_tokens = batch.prompt_tokens
                total_tokens = batch.total_tokens
            local = await asyncio.to_thread(
                self._search_local,
                request,
                batch.vectors if batch else None,
                False,
                diagnostic,
                True,
            )
            assert isinstance(local, _LocalRetrievalBundle)
            local.timings = replace(
                local.timings,
                scope_ms=local.timings.scope_ms + preflight_ms,
                query_embedding_ms=query_embedding_ms,
            )

        assert isinstance(local, _LocalRetrievalBundle)

        # Optional SiliconFlow Reranker step (outside SQLite transaction!)
        rerank_used = False
        rerank_diag: RetrievalDiagnostic | None = None
        rerank_ms = 0.0
        ranked = local.ranked
        demote_refs = not _query_wants_references(request.raw_query)
        if use_rerank and ranked and self.reranker is not None:
            pool = ranked[: self.config.rerank_candidates]
            doc_texts = [
                ((" > ".join(c.hit.title_context) + "\n\n") if c.hit.title_context else "") + c.hit.body_text
                for c in pool
            ]
            try:
                rr = await self.reranker.rerank(
                    rerank_query or query,
                    doc_texts,
                    top_n=len(pool),
                )
                rerank_ms = rr.elapsed_ms
                if rr.items:
                    rerank_used = True
                    for rr_rank, item in enumerate(rr.items, 1):
                        cand = pool[item.index]
                        score = item.relevance_score
                        if demote_refs and cand.is_reference_chunk:
                            score *= 0.65
                        cand.rerank_score = score
                        cand.rerank_rank = rr_rank
                    ranked.sort(key=lambda c: c.order)
            except RerankError as error:
                rerank_diag = RetrievalDiagnostic.from_error(error)

        # Final selection & source hydration
        selected, exclusions = _select_hits_unified(
            ranked, request.top_k, request.per_document_quota, request.coverage_mode,
        )
        selected_ids = {c.hit.chunk_id for c in selected}

        # Hydrate missing sources for selected hits
        missing_locs = _locations(selected) - local.sources.keys()
        if missing_locs:
            t_src = perf_counter()
            extra_sources = await asyncio.to_thread(self._load_sources_sync, missing_locs)
            local.sources.update(extra_sources)
            local.timings = replace(local.timings, source_ms=local.timings.source_ms + (perf_counter() - t_src) * 1000)

        available = Counter(item.hit.document_id for item in ranked)
        counts = Counter(item.hit.document_id for item in selected)
        coverage = []
        for doc in local.documents:
            channels = _channels(doc, request, local.vector_ran)
            count = counts[doc.document_id]
            status = _coverage_status(channels, count, available[doc.document_id], request.coverage_mode)
            coverage.append(
                DocumentCoverage(
                    doc.document_id,
                    doc.version_id,
                    doc.scope.chunk_set_id if doc.scope else None,
                    doc.keyword_indexed,
                    doc.vector_indexed,
                    channels,
                    status,
                    count,
                )
            )

        hits = tuple(self._hit(item, local.scopes[item.hit.document_id], local.sources) for item in selected)

        # Build CandidateRecord trace (top 50 candidates for Retrieval Lab / diagnostics)
        candidate_records: list[CandidateRecord] = []
        for rrf_idx, item in enumerate(ranked[:50], 1):
            cid = item.hit.chunk_id
            bm25_r = item.rank_for("keyword")
            vec_r = item.rank_for("vector")
            bm25_s = item.max_channel_score("keyword") if bm25_r is not None else None
            vec_s = item.max_channel_score("vector") if vec_r is not None else None
            candidate_records.append(
                CandidateRecord(
                    chunk_id=cid,
                    document_id=item.hit.document_id,
                    chunk_index=item.hit.chunk_index,
                    section_path=item.hit.section_path,
                    bm25_raw=bm25_s,
                    bm25_rank=bm25_r,
                    cosine=vec_s,
                    vector_rank=vec_r,
                    rrf_score=item.score,
                    rrf_rank=rrf_idx,
                    rerank_score=item.rerank_score,
                    rerank_rank=item.rerank_rank,
                    selected=cid in selected_ids,
                    exclusion_reason=None if cid in selected_ids else exclusions.get(cid, "topk_cutoff"),
                    is_reference_chunk=item.is_reference_chunk,
                    snippet=item.hit.body_text[:240],
                )
            )

        timings = replace(local.timings, rerank_ms=rerank_ms)
        return HybridResult(
            retrieval_mode=request.retrieval_mode,
            coverage_mode=request.coverage_mode,
            channel_status=_channel_status(coverage),
            hits=hits,
            documents=tuple(coverage),
            diagnostic=local.diagnostic,
            timings=timings,
            query_prompt_tokens=prompt_tokens,
            query_total_tokens=total_tokens,
            rerank_used=rerank_used,
            rerank_diagnostic=rerank_diag,
            candidates=tuple(candidate_records),
        )

    def _load_sources_sync(self, locations: set[tuple[int, int]]) -> dict[tuple[int, int], list[ChunkSourceSpan]]:
        with self.keyword.store.connect() as db, db:
            db.execute("BEGIN")
            return _read_sources(db, locations)

    def _search_local(
        self,
        request: _Request,
        query_vectors: Sequence[Sequence[float]] | None,
        needs_embedding: bool,
        diagnostic: RetrievalDiagnostic | None,
        remote_vector: bool = False,
    ) -> _LocalRetrievalBundle | float:
        try:
            start = perf_counter()
            with self.keyword.store.connect() as db, db:
                db.execute("BEGIN")
                documents, scope_diagnostic = _select_scope(db, request)
                diagnostic = diagnostic or scope_diagnostic
                vector_ready = any(doc.vector_indexed and doc.scope.chunk_count for doc in documents)
                scope_ms = (perf_counter() - start) * 1000
                if needs_embedding and vector_ready:
                    return scope_ms
                return self._retrieve(db, request, documents, query_vectors, remote_vector, diagnostic, scope_ms)
        except sqlite3.DatabaseError:
            raise HybridRetrievalError("HYBRID_QUERY_FAILED", "无法查询本地知识库。") from None

    def _run_keyword_variants(
        self,
        db: sqlite3.Connection,
        request: _Request,
        set_ids: Sequence[int],
        limit: int,
    ) -> tuple[KeywordHit, ...]:
        if request.retrieval_mode == "vector" or not set_ids:
            return ()
        variant_lists: list[tuple[KeywordHit, ...]] = []
        for idx, qv in enumerate(request.query_variants or (request.raw_query,)):
            phr = request.phrase if idx == 0 else prepare_keyword_query(qv)
            if phr is None:
                continue
            extra = request.extra_terms if idx == 0 else ()
            # Call search_indexed positionally when exact/simple so legacy monkeypatches work cleanly,
            # and run independent BM25 OR expansion via keyword service when needed.
            hits = self.keyword.search_indexed(db, phr, limit, set_ids)
            if not request.exact_phrase and (
                not hits or extra or self.keyword.needs_bm25_expansion(qv)
            ):
                expanded = self.keyword.search_bm25_or_indexed(db, qv, limit, set_ids, extra_terms=extra)
                if expanded:
                    hits = self.keyword._merge_hits(hits, expanded, limit) if hits else expanded
            if hits:
                variant_lists.append(hits)
        if not variant_lists:
            return ()
        if len(variant_lists) == 1:
            return variant_lists[0]
        return _merge_keyword_variants(variant_lists, limit)

    def _run_vector_variants(
        self,
        db: sqlite3.Connection,
        request: _Request,
        scopes: Sequence[VectorScope],
        query_vectors: Sequence[Sequence[float]],
        limit: int,
    ) -> tuple[VectorHit, ...]:
        if not scopes or not query_vectors:
            return ()
        variant_lists: list[tuple[VectorHit, ...]] = []
        for qv in query_vectors:
            groups_res = self.vector.search_published(
                db, qv, request.profile, [tuple(scopes)], limit, min_similarity=request.min_vector_similarity,
            )
            if groups_res and groups_res[0]:
                variant_lists.append(groups_res[0])
        if not variant_lists:
            return ()
        if len(variant_lists) == 1:
            return variant_lists[0]
        return _merge_vector_variants(variant_lists, limit)

    def _retrieve(self, db, request: _Request, documents: tuple[_Document, ...], query_vectors, remote_vector, diagnostic, scope_ms):
        limit = request.candidate_k
        kw_set_ids = [doc.scope.chunk_set_id for doc in documents if doc.keyword_indexed and doc.scope]
        vec_scopes = [doc.scope for doc in documents if doc.vector_indexed and doc.scope]

        t_fts = perf_counter()
        global_kw = self._run_keyword_variants(db, request, kw_set_ids, limit)
        fts_ms = (perf_counter() - t_fts) * 1000

        t_vec = perf_counter()
        global_vec: tuple[VectorHit, ...] = ()
        vector_ran = False
        if query_vectors is not None and diagnostic is None and vec_scopes:
            try:
                global_vec = self._run_vector_variants(db, request, vec_scopes, query_vectors, limit)
                vector_ran = True
            except (EmbeddingError, VectorRetrievalError) as error:
                if request.retrieval_mode == "vector" or (isinstance(error, EmbeddingError) and not remote_vector):
                    raise
                diagnostic = RetrievalDiagnostic.from_error(error)
        vector_ms = (perf_counter() - t_vec) * 1000

        active_compare_ids: set[str] | None = None
        if request.coverage_mode == "per_document":
            searchable_docs = [doc for doc in documents if doc.keyword_indexed or doc.vector_indexed]
            # Stage 1: compare ALL searchable books when explicitly specified or when library is small (<= compare_auto_books);
            # otherwise select top-N relevant books from global coarse recall (never crashing on 34+ books!).
            if request.document_ids is not None or len(searchable_docs) <= self.config.compare_auto_books:
                target_books = searchable_docs
            else:
                target_books = _select_top_compare_books(
                    documents, global_kw, global_vec, self.config.compare_auto_books,
                )
            active_compare_ids = {doc.document_id for doc in target_books}

            # Stage 2: ensure per-book recall coverage for each selected book, then merge into ONE global pool!
            per_book_k = min(MAX_CANDIDATES, max(limit, 8))
            kw_pool: list[tuple[KeywordHit, ...]] = [global_kw] if global_kw else []
            vec_pool: list[tuple[VectorHit, ...]] = [global_vec] if global_vec else []
            for doc in target_books:
                if doc.keyword_indexed and doc.scope and request.retrieval_mode != "vector":
                    t1 = perf_counter()
                    bh = self._run_keyword_variants(db, request, [doc.scope.chunk_set_id], per_book_k)
                    fts_ms += (perf_counter() - t1) * 1000
                    if bh:
                        kw_pool.append(bh)
                if doc.vector_indexed and doc.scope and vector_ran and query_vectors is not None:
                    t2 = perf_counter()
                    vh = self._run_vector_variants(db, request, [doc.scope], query_vectors, per_book_k)
                    vector_ms += (perf_counter() - t2) * 1000
                    if vh:
                        vec_pool.append(vh)

            merged_kw = _merge_keyword_variants(kw_pool, max(limit, per_book_k * max(1, len(target_books))))
            merged_vec = _merge_vector_variants(vec_pool, max(limit, per_book_k * max(1, len(target_books))))
            # Restrict pooled candidates to the chosen comparison books when auto-selected across full library
            if active_compare_ids:
                merged_kw = tuple(h for h in merged_kw if h.document_id in active_compare_ids)
                merged_vec = tuple(h for h in merged_vec if h.document_id in active_compare_ids)
        else:
            merged_kw = global_kw
            merged_vec = global_vec

        t_fuse = perf_counter()
        # Unified global RRF across ALL candidates (never independent per-book RRF!)
        # Gently soften natural-language BM25 OR weight when vector hits are present to curb OR noise without hurting exact queries.
        kw_weight = (
            self.config.bm25_or_rrf_weight
            if (merged_kw and merged_vec and not request.exact_phrase and self.keyword.needs_bm25_expansion(request.raw_query))
            else 1.0
        )
        demote_refs = not _query_wants_references(request.raw_query)
        ranked = _fuse(
            merged_kw,
            merged_vec,
            self.config.rrf_k,
            keyword_weight=kw_weight,
            demote_references=demote_refs,
            ref_factor=self.config.reference_demotion_factor,
        )
        locations = _locations(ranked)
        fusion_ms = (perf_counter() - t_fuse) * 1000

        t_src = perf_counter()
        # Load all candidate source spans within the SAME read snapshot as keyword/vector search!
        sources = _read_sources(db, locations)
        source_ms = (perf_counter() - t_src) * 1000

        t_dedup = perf_counter()
        unique, excluded_overlap = _deduplicate(ranked, sources, self.config.overlap_threshold)
        fusion_ms += (perf_counter() - t_dedup) * 1000

        scopes = {doc.document_id: doc.scope for doc in documents}
        timings = RetrievalTimings(
            scope_ms=scope_ms, fts_ms=fts_ms, vector_ms=vector_ms, fusion_ms=fusion_ms, source_ms=source_ms,
        )
        return _LocalRetrievalBundle(
            documents=documents,
            ranked=unique,
            excluded_overlap=excluded_overlap,
            sources=sources,
            vector_ran=vector_ran,
            diagnostic=diagnostic,
            timings=timings,
            scopes=scopes,
            active_compare_doc_ids=active_compare_ids,
        )

    @staticmethod
    def _hit(item: _Candidate, scope: VectorScope, sources: dict) -> HybridHit:
        hit = item.hit
        return HybridHit(
            hit.chunk_id, hit.document_id, hit.document_version_id, hit.chunk_set_id,
            hit.parsed_document_id, hit.chunk_index, scope.content_sha256, hit.body_text,
            hit.section_heading_block_id, hit.section_path, hit.title_context,
            hit.matched_in if isinstance(hit, KeywordHit) else (),
            tuple(item.ranks), item.score, tuple(sources[(hit.chunk_set_id, hit.chunk_index)]),
            rerank_score=item.rerank_score,
            rerank_rank=item.rerank_rank,
        )
