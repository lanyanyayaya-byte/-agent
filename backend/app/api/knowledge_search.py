"""HTTP endpoints for local textbook retrieval and citation verification."""

from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints

from app.api.chat import validate_utf8
from app.api.library import require_library
from app.models import ModelError
from app.security import ApiError
from app.services.citation_resolver import CitationScope, CitationStatus, CitationView, format_citation_view
from app.services.hybrid_retrieval import CoverageMode, MAX_COMPARISON_DOCUMENTS, RetrievalMode
from app.services.knowledge_qa import KnowledgeQAService, KnowledgeSearchResponse
from app.services.library import DOCUMENT_ID_PATTERN


class KnowledgeSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: Annotated[
        str,
        StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=256),
        AfterValidator(validate_utf8),
    ]
    top_k: Annotated[int, Field(ge=1, le=50)] = 10
    document_ids: Annotated[
        list[Annotated[str, StringConstraints(pattern=DOCUMENT_ID_PATTERN)]],
        Field(max_length=128),
    ] | None = None
    retrieval_mode: RetrievalMode = "hybrid"
    coverage_mode: CoverageMode = "global"


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


def _require_qa(request: Request) -> KnowledgeQAService:
    qa: KnowledgeQAService | None = getattr(request.app.state, "knowledge_qa", None)
    if qa is None:
        raise ApiError(503, "KNOWLEDGE_STORE_UNAVAILABLE", "教材检索服务暂不可用。")
    return qa


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
        if body.coverage_mode == "per_document" and body.document_ids and len(set(body.document_ids)) > MAX_COMPARISON_DOCUMENTS:
            raise ApiError(422, "VALIDATION_ERROR", f"跨教材对比最多同时选择 {MAX_COMPARISON_DOCUMENTS} 本教材。")
        try:
            result = await qa.search(
                body.query,
                top_k=body.top_k,
                document_ids=body.document_ids,
                retrieval_mode=body.retrieval_mode,
                coverage_mode=body.coverage_mode,
            )
        except ModelError as error:
            status = {
                "EMBEDDING_NOT_CONFIGURED": 409,
                "EMBEDDING_AUTH_FAILED": 401,
                "EMBEDDING_FORBIDDEN": 403,
                "EMBEDDING_RATE_LIMITED": 429,
                "EMBEDDING_TIMEOUT": 504,
            }.get(error.code, 502)
            raise ApiError(status, error.code, str(error)) from None
        except ValueError as error:
            raise ApiError(422, "VALIDATION_ERROR", str(error)) from None
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
