"""Structured RetrievalTrace and QueryPlan models shared by RAG 2.0 chat and retrieval lab."""

from dataclasses import asdict, dataclass, field
from typing import Any, Literal
from uuid import uuid4

RetrievalStrategy = Literal["auto", "global", "scoped", "compare", "book_summary"]
EffectiveStrategy = Literal["global", "scoped", "compare", "book_summary"]
RewriteMode = Literal["off", "rules", "llm"]
StageStatus = Literal["ok", "skipped", "degraded", "failed"]


@dataclass(frozen=True, slots=True)
class QueryPlan:
    original_query: str
    resolved_query: str
    query_variants: tuple[str, ...]
    requested_strategy: RetrievalStrategy
    actual_strategy: EffectiveStrategy
    selected_document_ids: tuple[str, ...] | None
    rewrite_mode: RewriteMode = "rules"
    rewrite_reason: str | None = None
    chinese_terms: tuple[str, ...] = ()
    english_terms: tuple[str, ...] = ()
    subqueries: tuple[str, ...] = ()
    rewrite_elapsed_ms: float = 0.0
    rewrite_error: str | None = None


@dataclass(frozen=True, slots=True)
class StageTrace:
    stage: str
    status: StageStatus
    candidate_count: int = 0
    elapsed_ms: float = 0.0
    error_code: str | None = None
    message: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateTrace:
    chunk_id: str
    document_id: str
    textbook_name: str | None = None
    chunk_index: int = 0
    page_range: str | None = None
    section_path: tuple[str, ...] = ()
    bm25_raw: float | None = None
    bm25_rank: int | None = None
    cosine: float | None = None
    vector_rank: int | None = None
    rrf_score: float | None = None
    rrf_rank: int | None = None
    rerank_score: float | None = None
    rerank_rank: int | None = None
    selected: bool = False
    exclusion_reason: str | None = None
    is_reference_chunk: bool = False
    is_neighbor_expansion: bool = False
    snippet: str | None = None


@dataclass(frozen=True, slots=True)
class DocumentTrace:
    document_id: str
    textbook_name: str | None
    version_id: str | None
    keyword_indexed: bool
    vector_indexed: bool | None
    searched_channels: tuple[str, ...]
    evidence_status: str
    returned_count: int
    skip_reason: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievalTrace:
    trace_id: str
    original_query: str
    resolved_query: str
    query_variants: tuple[str, ...]
    requested_strategy: str
    actual_strategy: EffectiveStrategy
    requested_mode: str
    effective_channels: tuple[str, ...]
    in_scope_documents: int
    searchable_documents: int
    hit_documents: int
    cited_documents: int
    documents: tuple[DocumentTrace, ...] = ()
    stages: tuple[StageTrace, ...] = ()
    candidates: tuple[CandidateTrace, ...] = ()
    final_evidence_ids: tuple[str, ...] = ()
    prompt_evidence_count: int = 0
    estimated_tokens: int = 0
    degraded: bool = False
    degrade_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    chinese_terms: tuple[str, ...] = ()
    english_terms: tuple[str, ...] = ()
    rewrite_mode: str = "off"
    rewrite_reason: str | None = None
    bm25_used: bool = False
    vector_used: bool = False
    rerank_used: bool = False
    total_elapsed_ms: float = 0.0
    context_expansion: int = 0
    summary_coverage: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None
    schema_version: int = 1

    @staticmethod
    def new_id() -> str:
        return uuid4().hex

    def to_compact_dict(self, *, max_candidates: int = 40, snippet_chars: int = 180) -> dict[str, Any]:
        """Serialize compact trace for SQLite persistence without duplicating full chunk bodies."""
        raw = asdict(self)
        cands = raw.get("candidates") or []
        trimmed = []
        for item in cands[:max_candidates]:
            snip = item.get("snippet")
            if isinstance(snip, str) and len(snip) > snippet_chars:
                item["snippet"] = snip[:snippet_chars].rstrip() + "…"
            trimmed.append(item)
        raw["candidates"] = trimmed
        raw["total_candidates_recorded"] = len(cands)
        return raw

    def to_summary_dict(self) -> dict[str, Any]:
        """Lightweight summary for RunSummary and collapsed chat retrieval card."""
        return {
            "trace_id": self.trace_id,
            "actual_strategy": self.actual_strategy,
            "requested_mode": self.requested_mode,
            "effective_channels": list(self.effective_channels),
            "in_scope_documents": self.in_scope_documents,
            "searchable_documents": self.searchable_documents,
            "hit_documents": self.hit_documents,
            "cited_documents": self.cited_documents,
            "prompt_evidence_count": self.prompt_evidence_count,
            "bm25_used": self.bm25_used,
            "vector_used": self.vector_used,
            "rerank_used": self.rerank_used,
            "total_elapsed_ms": round(self.total_elapsed_ms, 1),
            "context_expansion": self.context_expansion,
            "degraded": self.degraded,
            "degrade_reasons": list(self.degrade_reasons),
            "warnings": list(self.warnings),
            "error_code": self.error_code,
            "error_message": self.error_message,
            "summary_coverage": self.summary_coverage,
        }


def make_failed_trace(
    query: str,
    *,
    requested_strategy: str,
    requested_mode: str,
    error_code: str,
    error_message: str,
    elapsed_ms: float = 0.0,
) -> RetrievalTrace:
    """Create a structured diagnostic trace when retrieval fails before completion."""
    actual: EffectiveStrategy = (
        requested_strategy if requested_strategy in ("global", "scoped", "compare", "book_summary") else "global"
    )
    return RetrievalTrace(
        trace_id=RetrievalTrace.new_id(),
        original_query=query,
        resolved_query=query,
        query_variants=(query,),
        requested_strategy=requested_strategy,
        actual_strategy=actual,
        requested_mode=requested_mode,
        effective_channels=(),
        in_scope_documents=0,
        searchable_documents=0,
        hit_documents=0,
        cited_documents=0,
        stages=(StageTrace("retrieval", "failed", 0, round(elapsed_ms, 1), error_code, error_message),),
        degraded=True,
        degrade_reasons=(f"{error_message}（{error_code}）",),
        total_elapsed_ms=round(elapsed_ms, 1),
        error_code=error_code,
        error_message=error_message,
    )
