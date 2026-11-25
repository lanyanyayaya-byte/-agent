"""FastAPI routes for session workspace files and artifacts.

Channel separation:
- UI Token (`/api/v1/sessions/{session_id}/...`): read-only workspace summary,
  artifact metadata, and controlled artifact preview content.
- Host Token (`/api/v1/_host/sessions/{session_id}/...`): local file import copy,
  attachment deletion, and host-only artifact path resolution for native Open/Reveal/Save As.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Annotated, Any
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, ConfigDict, StringConstraints

from app.security import ApiError
from app.workspace.artifacts import ArtifactService
from app.workspace.service import WorkspaceError, WorkspaceService

_ERROR_HTTP_STATUS: dict[str, int] = {
    "WORKSPACE_SESSION_NOT_FOUND": 404,
    "WORKSPACE_NOT_FOUND": 404,
    "WORKSPACE_ARTIFACT_MISSING": 404,
    "WORKSPACE_INVALID_ID": 422,
    "WORKSPACE_PATH_INVALID": 422,
    "WORKSPACE_EMPTY_ARTIFACT": 422,
    "WORKSPACE_FILE_TOO_LARGE": 413,
    "WORKSPACE_PATH_EXISTS": 409,
    "WORKSPACE_INVALID_RUN_STATE": 409,
    "WORKSPACE_IO_ERROR": 500,
}


def _raise_api_error(exc: WorkspaceError) -> None:
    raise ApiError(_ERROR_HTTP_STATUS.get(exc.code, 500), exc.code, exc.message) from None


def _require_services(request: Request) -> tuple[WorkspaceService, ArtifactService]:
    workspace: WorkspaceService | None = getattr(request.app.state, "workspace_service", None)
    artifacts: ArtifactService | None = getattr(request.app.state, "artifact_service", None)
    if workspace is None or artifacts is None:
        raise ApiError(503, "WORKSPACE_UNAVAILABLE", "工作区服务尚未初始化。")
    return workspace, artifacts


class ImportWorkspaceFileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    path: Annotated[str, StringConstraints(min_length=1, max_length=4096)]
    filename: Annotated[str, StringConstraints(min_length=1, max_length=255)] | None = None


def workspace_router(host_namespace: str, require_ui, require_ready) -> APIRouter:
    router = APIRouter()

    # ------------------------------------------------------------------
    # UI Token: read-only workspace & artifact queries / preview
    # ------------------------------------------------------------------

    @router.get(
        "/api/v1/sessions/{session_id}/workspace",
        dependencies=[Depends(require_ui), Depends(require_ready)],
    )
    def get_session_workspace(session_id: UUID, request: Request, response: Response) -> dict[str, Any]:
        workspace, artifacts = _require_services(request)
        sid = str(session_id)
        try:
            files = workspace.list_files(sid)
            published = artifacts.list_artifacts(sid, include_non_published=False)
        except WorkspaceError as exc:
            _raise_api_error(exc)
        file_views = []
        for item in files:
            _, path = workspace.get_file_path(sid, item.file_id)
            file_views.append({**asdict(item), "file_exists": path.is_file()})
        response.headers["Cache-Control"] = "no-store"
        return {
            "session_id": sid,
            "files": file_views,
            "artifacts": [asdict(a) for a in published],
        }

    @router.get(
        "/api/v1/sessions/{session_id}/artifacts/{artifact_id}",
        dependencies=[Depends(require_ui), Depends(require_ready)],
    )
    def get_artifact_metadata(
        session_id: UUID, artifact_id: UUID, request: Request, response: Response,
    ) -> dict[str, Any]:
        _, artifacts = _require_services(request)
        try:
            view, _ = artifacts.get_artifact(str(session_id), str(artifact_id), include_non_published=False)
        except WorkspaceError as exc:
            _raise_api_error(exc)
        response.headers["Cache-Control"] = "no-store"
        return asdict(view)

    @router.get(
        "/api/v1/sessions/{session_id}/artifacts/{artifact_id}/content",
        dependencies=[Depends(require_ui), Depends(require_ready)],
    )
    def get_artifact_content(
        session_id: UUID, artifact_id: UUID, request: Request,
    ) -> Response:
        _, artifacts = _require_services(request)
        try:
            view, data = artifacts.read_artifact_bytes(str(session_id), str(artifact_id))
        except WorkspaceError as exc:
            _raise_api_error(exc)
        encoded_name = quote(view.filename, safe="")
        return Response(
            content=data,
            media_type=view.media_type,
            headers={
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "Content-Disposition": f"inline; filename*=UTF-8''{encoded_name}",
            },
        )

    # ------------------------------------------------------------------
    # Host Token: privileged import, delete, and native path lookup
    # ------------------------------------------------------------------

    @router.post(
        f"{host_namespace}/sessions/{{session_id}}/workspace/files",
        status_code=201,
        dependencies=[Depends(require_ready)],
    )
    def host_import_workspace_file(
        session_id: UUID,
        body: ImportWorkspaceFileRequest,
        request: Request,
        response: Response,
    ) -> dict[str, Any]:
        workspace, _ = _require_services(request)
        try:
            stored = workspace.import_local_file(
                str(session_id), body.path, filename=body.filename,
            )
        except WorkspaceError as exc:
            _raise_api_error(exc)
        response.headers["Cache-Control"] = "no-store"
        return {**asdict(stored), "file_exists": True}

    @router.delete(
        f"{host_namespace}/sessions/{{session_id}}/workspace/files/{{file_id}}",
        status_code=204,
        dependencies=[Depends(require_ready)],
    )
    def host_delete_workspace_file(
        session_id: UUID, file_id: UUID, request: Request,
    ) -> Response:
        workspace, _ = _require_services(request)
        try:
            workspace.delete_file(str(session_id), str(file_id))
        except WorkspaceError as exc:
            _raise_api_error(exc)
        return Response(status_code=204, headers={"Cache-Control": "no-store"})

    @router.get(
        f"{host_namespace}/sessions/{{session_id}}/artifacts/{{artifact_id}}/path",
        dependencies=[Depends(require_ready)],
    )
    def host_get_artifact_path(
        session_id: UUID, artifact_id: UUID, request: Request, response: Response,
    ) -> dict[str, Any]:
        """Host-only lookup returning absolute artifact path for native Open / Reveal / Save As."""
        _, artifacts = _require_services(request)
        try:
            view, disk_path = artifacts.get_artifact(str(session_id), str(artifact_id), include_non_published=False)
        except WorkspaceError as exc:
            _raise_api_error(exc)
        response.headers["Cache-Control"] = "no-store"
        return {
            **asdict(view),
            "absolute_path": str(disk_path),
        }

    return router
