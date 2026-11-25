"""HTTP endpoints for local textbook retrieval, RAG Retrieval Lab, Chunk Browser, and citation verification."""

from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query, Request, Response
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints

from app.api.chat import validate_utf8
from app.api.library import require_library
from app.models import ModelError
from app.security import ApiError
from app.services.chunk_browser import ChunkBrowserResponse, ChunkBrowserService, ChunkNeighborsResponse
from app.services.citation_resolver import CitationScope, CitationStatus, CitationView, format_citation_view
from app.services.hybrid_retrieval import CoverageMode, RetrievalMode
from app.services.knowledge_qa import KnowledgeQAService, KnowledgeSearchResponse
from app.services.library import DOCUMENT_ID_PATTERN
from app.services.retrieval_trace import RetrievalStrategy, RewriteMode


class KnowledgeSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: Annotated[
        str,
        StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=256),
        AfterValidator(validate_utf8),
    ]
    top_k: Annotated[int, Field(ge=1, le=50)] | None = None
    document_ids: Annotated[
        list[Annotated[str, StringConstraints(pattern=DOCUMENT_ID_PATTERN)]],
        Field(max_length=128),
    ] | None = None
    retrieval_mode: RetrievalMode = "hybrid"
    coverage_mode: CoverageMode = "global"
    strategy: RetrievalStrategy | None = None
    rewrite_mode: RewriteMode = "off"
    use_rerank: bool | None = None
    exact_phrase: bool = False


class DebugSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: Annotated[
        str,
        StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=256),
        AfterValidator(validate_utf8),
    ]
    document_ids: Annotated[
        list[Annotated[str, StringConstraints(pattern=DOCUMENT_ID_PATTERN)]],
        Field(max_length=128),
    ] | None = None
    strategy: RetrievalStrategy = "global"
    mode: RetrievalMode = "hybrid"
    coverage_mode: CoverageMode | None = None
    rerank: bool = True
    rewrite: RewriteMode = "off"
    top_k: Annotated[int, Field(ge=1, le=30)] = 8
    candidate_k: Annotated[int, Field(ge=10, le=200)] = 60
    min_vector_similarity: Annotated[float, Field(gt=0.0, le=1.0)] = 0.25
    exact_phrase: bool = False
    expand_neighbors: bool = False


class CitationValidationItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    evidence_id: Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=256)]
    chunk_set_id: Annotated[int, Field(ge=1, le=2**63 - 1)] | None = None


class ValidateCitationsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    items: Annotated[list[CitationValidationItem], Field(min_length=1, max_length=50)]


@dataclass(frozen=True, slots=True)
class CitationValidationResult:
    evidence_id: str
    status: CitationStatus
    citation: CitationView | None = None


DocumentId = Annotated[str, Path(pattern=DOCUMENT_ID_PATTERN)]


def _require_qa(request: Request) -> KnowledgeQAService:
    qa: KnowledgeQAService | None = getattr(request.app.state, "knowledge_qa", None)
    if qa is None:
        raise ApiError(503, "KNOWLEDGE_STORE_UNAVAILABLE", "教材检索服务暂不可用。")
    return qa


def _map_model_error(error: ModelError) -> ApiError:
    status = {
        "EMBEDDING_NOT_CONFIGURED": 409,
        "EMBEDDING_AUTH_FAILED": 401,
        "EMBEDDING_FORBIDDEN": 403,
        "EMBEDDING_RATE_LIMITED": 429,
        "EMBEDDING_TIMEOUT": 504,
    }.get(error.code, 502)
    return ApiError(status, error.code, str(error))


def knowledge_search_router(require_ui, require_ready) -> APIRouter:
    router = APIRouter(
        prefix="/api/v1/knowledge",
        dependencies=[Depends(require_ui), Depends(require_ready), Depends(require_library)],
    )

    @router.post("/search")
    async def search_knowledge(
        body: KnowledgeSearchRequest,
        request: Request,
        response: Response,
    ) -> KnowledgeSearchResponse:
        qa = _require_qa(request)
        try:
            result = await qa.search(
                body.query,
                top_k=body.top_k,
                document_ids=body.document_ids,
                retrieval_mode=body.retrieval_mode,
                coverage_mode=body.coverage_mode,
                strategy=body.strategy,
                rewrite_mode=body.rewrite_mode,
                use_rerank=body.use_rerank,
                exact_phrase=body.exact_phrase,
            )
        except ModelError as error:
            raise _map_model_error(error) from None
        except ValueError as error:
            raise ApiError(422, "VALIDATION_ERROR", str(error)) from None
        response.headers["Cache-Control"] = "no-store"
        return result

    @router.post("/debug-search")
    async def debug_search_knowledge(
        body: DebugSearchRequest,
        request: Request,
        response: Response,
    ) -> KnowledgeSearchResponse:
        """Unified RAG Retrieval Lab endpoint reusing the exact production retrieval pipeline."""
        qa = _require_qa(request)
        cov: CoverageMode = body.coverage_mode or ("per_document" if body.strategy == "compare" else "global")
        try:
            result = await qa.search(
                body.query,
                top_k=body.top_k,
                document_ids=body.document_ids,
                retrieval_mode=body.mode,
                coverage_mode=cov,
                strategy=body.strategy,
                rewrite_mode=body.rewrite,
                use_rerank=body.rerank,
                min_vector_similarity=body.min_vector_similarity,
                candidate_k=body.candidate_k,
                exact_phrase=body.exact_phrase,
                expand_neighbors=body.expand_neighbors,
            )
        except ModelError as error:
            raise _map_model_error(error) from None
        except ValueError as error:
            raise ApiError(422, "VALIDATION_ERROR", str(error)) from None
        response.headers["Cache-Control"] = "no-store"
        return result

    @router.get("/documents/{document_id}/chunks")
    def browse_document_chunks(
        document_id: DocumentId,
        request: Request,
        response: Response,
        page: Annotated[int | None, Query(ge=1, le=10000)] = None,
        section: Annotated[str | None, Query(max_length=160)] = None,
        q: Annotated[str | None, Query(max_length=160)] = None,
        offset: Annotated[int, Query(ge=0, le=100000)] = 0,
        limit: Annotated[int, Query(ge=1, le=100)] = 30,
    ) -> ChunkBrowserResponse:
        qa = _require_qa(request)
        browser = ChunkBrowserService(qa.store, profile=qa.profile)
        result = browser.list_chunks(document_id, page=page, section=section, query=q, offset=offset, limit=limit)
        if result is None:
            raise ApiError(404, "TEXTBOOK_NOT_FOUND", "未找到该教材。")
        response.headers["Cache-Control"] = "no-store"
        return result

    @router.get("/documents/{document_id}/chunks/{chunk_index}/neighbors")
    def get_chunk_neighbors(
        document_id: DocumentId,
        chunk_index: Annotated[int, Path(ge=0, le=100000)],
        request: Request,
        response: Response,
    ) -> ChunkNeighborsResponse:
        qa = _require_qa(request)
        browser = ChunkBrowserService(qa.store, profile=qa.profile)
        result = browser.get_neighbors(document_id, chunk_index)
        if result is None:
            raise ApiError(404, "TEXTBOOK_NOT_FOUND", "未找到该教材或切片索引。")
        response.headers["Cache-Control"] = "no-store"
        return result

    @router.post("/citations/validate")
    def validate_citations(
        body: ValidateCitationsRequest,
        request: Request,
        response: Response,
    ) -> list[CitationValidationResult]:
        qa = _require_qa(request)
        results: list[CitationValidationResult] = []
        for item in body.items:
            scope = CitationScope(chunk_set_ids=(item.chunk_set_id,) if item.chunk_set_id is not None else None)
            resolution = qa.citation_resolver.validate_evidence(item.evidence_id, scope=scope)
            view = (
                format_citation_view(resolution.evidence, status="resolved")
                if resolution.status == "resolved" and resolution.evidence is not None
                else None
            )
            results.append(CitationValidationResult(
                evidence_id=item.evidence_id,
                status=resolution.status,
                citation=view,
            ))
        response.headers["Cache-Control"] = "no-store"
        return results

    return router
