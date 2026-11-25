"""Version-consistent keyword and semantic retrieval for the Sidecar."""

import asyncio
import math
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Literal

from app.chunking.models import ChunkSourceSpan
from app.models import ModelError
from app.models.embeddings import EmbeddingError, EmbeddingProfile
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

MAX_COMPARISON_DOCUMENTS = 8


class HybridRetrievalError(ModelError):
    pass


@dataclass(frozen=True, slots=True)
class HybridRetrievalConfig:
    candidate_k: int = 60
    rrf_k: int = 60
    per_document_quota: int = 2
    min_vector_similarity: float = 0.45
    overlap_threshold: float = 0.65

    def __post_init__(self):
        if (type(self.candidate_k) is not int or not 1 <= self.candidate_k <= MAX_CANDIDATES
                or type(self.rrf_k) is not int or not 1 <= self.rrf_k <= 10_000
                or type(self.per_document_quota) is not int or not 1 <= self.per_document_quota <= MAX_TOP_K
                or type(self.min_vector_similarity) not in (int, float) or not 0 < self.min_vector_similarity <= 1
                or type(self.overlap_threshold) not in (int, float) or not 0 < self.overlap_threshold <= 1):
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
        return cls(error.code, str(error))


@dataclass(frozen=True, slots=True)
class RetrievalTimings:
    scope_ms: float = 0
    fts_ms: float = 0
    vector_ms: float = 0
    query_embedding_ms: float = 0
    fusion_ms: float = 0
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


@dataclass(frozen=True, slots=True)
class _Document:
    document_id: str
    version_id: str | None
    scope: VectorScope | None
    keyword_indexed: bool
    vector_indexed: bool | None


@dataclass(frozen=True, slots=True)
class _Request:
    phrase: str | None
    top_k: int
    retrieval_mode: RetrievalMode
    coverage_mode: CoverageMode
    profile: EmbeddingProfile | None
    document_ids: tuple[str, ...] | None
    filter_sql: str
    filter_values: tuple
    pinned: bool


@dataclass(slots=True)
class _Candidate:
    hit: KeywordHit | VectorHit
    ranks: list[ChannelRank]
    score: float = 0

    @property
    def order(self):
        hit = self.hit
        return (-self.score, min(rank.rank for rank in self.ranks), hit.chunk_set_id, hit.chunk_index, hit.chunk_id)


def _selected_documents(db: sqlite3.Connection, request: _Request) -> dict[str, str | None]:
    join_sets = "JOIN chunk_sets AS s USING (document_version_id)" if request.pinned else ""
    limit = f"LIMIT {MAX_COMPARISON_DOCUMENTS + 1}" if request.coverage_mode == "per_document" else ""
    rows = db.execute(f"""SELECT v.document_id, v.document_version_id, count(*)
        FROM document_versions AS v JOIN documents AS d USING (document_id) {join_sets}
        WHERE 1 = 1 {request.filter_sql}
        GROUP BY v.document_id ORDER BY v.document_id {limit}""", request.filter_values).fetchall()
    if request.coverage_mode == "per_document" and len(rows) > MAX_COMPARISON_DOCUMENTS:
        raise ValueError(f"At most {MAX_COMPARISON_DOCUMENTS} textbooks can be compared at once.")
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


def _fuse(keywords: tuple[KeywordHit, ...], vectors: tuple[VectorHit, ...], rrf_k: int) -> list[_Candidate]:
    candidates = {}
    for rank, hit in enumerate(keywords, 1):
        candidates[hit.chunk_id] = _Candidate(hit, [ChannelRank("keyword", rank, hit.score)])
    for rank, hit in enumerate(vectors, 1):
        candidate = candidates.setdefault(hit.chunk_id, _Candidate(hit, []))
        candidate.ranks.append(ChannelRank("vector", rank, hit.similarity))
    for candidate in candidates.values():
        candidate.score = math.fsum(1 / (rrf_k + rank.rank) for rank in candidate.ranks)
    return sorted(candidates.values(), key=lambda item: item.order)


def _overlap(first: Sequence[ChunkSourceSpan], second: Sequence[ChunkSourceSpan]) -> float:
    lengths = [sum(span.block_end - span.block_start for span in spans) for spans in (first, second)]
    shorter = min(lengths)
    shared = sum(max(0, min(a.block_end, b.block_end) - max(a.block_start, b.block_start))
                 for a in first for b in second if a.block_id == b.block_id)
    return shared / shorter if shorter else 0


def _deduplicate(ranked: Sequence[_Candidate], sources: dict, threshold: float) -> list[_Candidate]:
    kept = []
    accepted = set()
    for item in ranked:
        hit = item.hit
        location = (hit.chunk_set_id, hit.chunk_index)
        neighbors = ((hit.chunk_set_id, hit.chunk_index + offset) for offset in (-1, 1))
        if any(other in accepted and _overlap(sources[location], sources[other]) >= threshold for other in neighbors):
            continue
        accepted.add(location)
        kept.append(item)
    return kept


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


def _select_hits(groups: Sequence[list[_Candidate]], top_k: int, quota: int,
                 coverage_mode: CoverageMode) -> list[_Candidate]:
    if coverage_mode == "global":
        return groups[0][:top_k]
    reserved = sorted(((index, item) for group in groups for index, item in enumerate(group[:quota])),
                      key=lambda entry: (entry[0], entry[1].order))
    selected = [item for _, item in reserved[:top_k]]
    if len(selected) < top_k:
        surplus = sorted((item for group in groups for item in group[quota:]), key=lambda item: item.order)
        selected.extend(surplus[:top_k - len(selected)])
    return sorted(selected, key=lambda item: item.order)


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
    def __init__(self, keyword: KeywordSearchService, vector: VectorRetrievalService, *,
                 config: HybridRetrievalConfig | None = None):
        if keyword.store.path != vector.store.path:
            raise ValueError("Hybrid retrieval requires both services to use the same knowledge database.")
        self.keyword = keyword
        self.vector = vector
        self.config = config if config is not None else HybridRetrievalConfig()

    async def search(self, query: str, top_k: int = 10, *,
                     document_ids: Sequence[str] | None = None, chunk_set_ids: Sequence[int] | None = None,
                     retrieval_mode: RetrievalMode = "hybrid", coverage_mode: CoverageMode = "global",
                     profile: EmbeddingProfile | None = None,
                     query_vector: Sequence[float] | None = None) -> HybridResult:
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
        filter_sql, values = scope_filters(document_ids, chunk_set_ids, label="hybrid")
        documents = None if document_ids is None else tuple(dict.fromkeys(document_ids))
        if coverage_mode == "per_document" and documents and len(documents) > MAX_COMPARISON_DOCUMENTS:
            raise ValueError(f"At most {MAX_COMPARISON_DOCUMENTS} textbooks can be compared at once.")
        pinned = chunk_set_ids is not None
        request = _Request(phrase, top_k, retrieval_mode, coverage_mode, profile, documents,
                           filter_sql if pinned else filter_sql + CURRENT_VERSION, values, pinned)

        needs_embedding = profile is not None and query_vector is None
        result = await asyncio.to_thread(self._search_local, request, query_vector, needs_embedding, None)
        if isinstance(result, HybridResult):
            return result

        # Reopen a fresh snapshot after the network wait. No transaction spans an API call.
        assert profile is not None
        preflight_ms = result
        batch, diagnostic = None, None
        start = perf_counter()
        try:
            batch = await self.vector.embed_query(query, profile=profile)
        except (EmbeddingError, VectorRetrievalError) as error:
            if retrieval_mode == "vector":
                raise
            diagnostic = RetrievalDiagnostic.from_error(error)
        embedding_ms = (perf_counter() - start) * 1000
        result = await asyncio.to_thread(self._search_local, request, batch.vectors[0] if batch else None,
                                         False, diagnostic, True)
        assert isinstance(result, HybridResult)
        timings = replace(result.timings, scope_ms=result.timings.scope_ms + preflight_ms,
                          query_embedding_ms=embedding_ms)
        return replace(result, timings=timings, query_prompt_tokens=batch.prompt_tokens if batch else None,
                       query_total_tokens=batch.total_tokens if batch else None)

    def _search_local(self, request: _Request, query_vector, needs_embedding: bool,
                      diagnostic: RetrievalDiagnostic | None, remote_vector: bool = False) -> HybridResult | float:
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
                return self._retrieve(db, request, documents, query_vector, remote_vector, diagnostic, scope_ms)
        except sqlite3.DatabaseError:
            raise HybridRetrievalError("HYBRID_QUERY_FAILED", "无法查询本地知识库。") from None

    def _retrieve(self, db, request, documents, query_vector, remote_vector, diagnostic, scope_ms):
        groups = (documents,) if request.coverage_mode == "global" else tuple((doc,) for doc in documents)
        limit = self.config.candidate_k
        start = perf_counter()
        keyword_hits = tuple(
            self.keyword.search_indexed(db, request.phrase, limit,
                                        [doc.scope.chunk_set_id for doc in group if doc.keyword_indexed])
            if request.retrieval_mode != "vector" else () for group in groups
        )
        fts_ms = (perf_counter() - start) * 1000
        start = perf_counter()
        vector_hits = ((),) * len(groups)
        vector_ran = False
        if query_vector is not None and diagnostic is None:
            try:
                vector_hits = self.vector.search_published(
                    db, query_vector, request.profile,
                    [tuple(doc.scope for doc in group if doc.vector_indexed) for group in groups],
                    limit, min_similarity=self.config.min_vector_similarity,
                )
                vector_ran = True
            except (EmbeddingError, VectorRetrievalError) as error:
                if request.retrieval_mode == "vector" or (isinstance(error, EmbeddingError) and not remote_vector):
                    raise
                diagnostic = RetrievalDiagnostic.from_error(error)
        vector_ms = (perf_counter() - start) * 1000

        start = perf_counter()
        ranked = [_fuse(kw, vec, self.config.rrf_k) for kw, vec in zip(keyword_hits, vector_hits, strict=True)]
        locations = set().union(*(_locations(group) for group in ranked))
        adjacent = {pos for pos in locations if (pos[0], pos[1] - 1) in locations or (pos[0], pos[1] + 1) in locations}
        fusion_ms = (perf_counter() - start) * 1000
        start = perf_counter()
        sources = _read_sources(db, adjacent)
        source_ms = (perf_counter() - start) * 1000
        start = perf_counter()
        unique = [_deduplicate(items, sources, self.config.overlap_threshold) for items in ranked]
        selected = _select_hits(unique, request.top_k, self.config.per_document_quota, request.coverage_mode)
        available = Counter(item.hit.document_id for group in unique for item in group)
        counts = Counter(item.hit.document_id for item in selected)
        coverage = []
        for doc in documents:
            channels = _channels(doc, request, vector_ran)
            count = counts[doc.document_id]
            status = _coverage_status(channels, count, available[doc.document_id], request.coverage_mode)
            coverage.append(DocumentCoverage(doc.document_id, doc.version_id,
                                             doc.scope.chunk_set_id if doc.scope else None, doc.keyword_indexed,
                                             doc.vector_indexed, channels, status, count))
        fusion_ms += (perf_counter() - start) * 1000
        start = perf_counter()
        sources.update(_read_sources(db, _locations(selected) - sources.keys()))
        scopes = {doc.document_id: doc.scope for doc in documents}
        hits = tuple(self._hit(item, scopes[item.hit.document_id], sources) for item in selected)
        source_ms += (perf_counter() - start) * 1000
        timings = RetrievalTimings(scope_ms=scope_ms, fts_ms=fts_ms, vector_ms=vector_ms,
                                   fusion_ms=fusion_ms, source_ms=source_ms)
        return HybridResult(request.retrieval_mode, request.coverage_mode, _channel_status(coverage),
                            hits, tuple(coverage), diagnostic, timings)

    @staticmethod
    def _hit(item: _Candidate, scope: VectorScope, sources: dict) -> HybridHit:
        hit = item.hit
        return HybridHit(hit.chunk_id, hit.document_id, hit.document_version_id, hit.chunk_set_id,
                         hit.parsed_document_id, hit.chunk_index, scope.content_sha256, hit.body_text,
                         hit.section_heading_block_id, hit.section_path, hit.title_context,
                         hit.matched_in if isinstance(hit, KeywordHit) else (),
                         tuple(item.ranks), item.score, tuple(sources[(hit.chunk_set_id, hit.chunk_index)]))
