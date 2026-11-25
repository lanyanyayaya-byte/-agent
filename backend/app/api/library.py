import sqlite3
from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, Depends, Path, Request, Response
from pydantic import BaseModel, ConfigDict, StringConstraints

from app.diagnostics import record
from app.security import ApiError
from app.services.library import DOCUMENT_ID_PATTERN, IMPORT_ID_PATTERN, LibraryService
from app.storage.library import DocumentDetails, DocumentSummary, ImportResult, KnowledgeStoreError


class ImportTextbook(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    import_id: Annotated[str, StringConstraints(pattern=IMPORT_ID_PATTERN)]
    filename: Annotated[str, StringConstraints(min_length=1, max_length=255)]
    document_id: Annotated[str, StringConstraints(pattern=DOCUMENT_ID_PATTERN)] | None = None


def require_library(request: Request) -> Iterator[LibraryService]:
    library = request.app.state.library
    if library is None:
        raise ApiError(503, "KNOWLEDGE_STORE_UNAVAILABLE", "教材库暂不可用。")
    try:
        yield library
    except (OSError, sqlite3.Error, KnowledgeStoreError) as error:
        record(
            "knowledge_request_error", request_id=request.state.request_id, exception_type=type(error).__name__,
            sqlite_error=getattr(error, "sqlite_errorname", None), errno=getattr(error, "errno", None),
        )
        raise ApiError(503, "KNOWLEDGE_STORE_UNAVAILABLE", "教材库暂不可用。") from None


Library = Annotated[LibraryService, Depends(require_library)]
DocumentId = Annotated[str, Path(pattern=DOCUMENT_ID_PATTERN)]


def library_router(require_ui, require_ready) -> APIRouter:
    router = APIRouter(dependencies=[Depends(require_ready)])
    ui = APIRouter(prefix="/api/v1/knowledge/documents", dependencies=[Depends(require_ui)])

    # Host authentication runs in middleware before the route reads the request body.
    @router.post("/_host/knowledge/imports")
    def import_textbook(body: ImportTextbook, library: Library, response: Response) -> ImportResult:
        response.headers["Cache-Control"] = "no-store"
        return library.import_textbook(body.import_id, body.filename, body.document_id)

    @ui.get("")
    def list_textbooks(library: Library, response: Response) -> list[DocumentSummary]:
        response.headers["Cache-Control"] = "no-store"
        return library.list_documents()

    @ui.get("/{document_id}")
    def get_textbook(document_id: DocumentId, library: Library, response: Response) -> DocumentDetails:
        response.headers["Cache-Control"] = "no-store"
        return library.get_document(document_id)

    @ui.post("/{document_id}/delete", status_code=204)
    def delete_textbook(document_id: DocumentId, library: Library) -> Response:
        library.delete_document(document_id)
        return Response(status_code=204, headers={"Cache-Control": "no-store"})

    router.include_router(ui)
    return router
