from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, ConfigDict, StringConstraints

from app.security import ApiError
from app.services.grants import GrantLifetime, GrantRegistry, GrantScope, GrantView
from app.storage.business import BusinessStore, SessionNotFound


class CreateGrant(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    scope: GrantScope
    path: Annotated[str, StringConstraints(min_length=1)]
    operations: list[Literal["read"]]
    lifetime: GrantLifetime


def grant_router(store: BusinessStore, grants: GrantRegistry, require_ui, require_ready) -> APIRouter:
    router = APIRouter(prefix="/api/v1")

    def require_session(session_id: UUID) -> None:
        try:
            store.require_session(str(session_id))
        except SessionNotFound:
            raise ApiError(404, "NOT_FOUND", "未找到该会话。") from None

    @router.get("/sessions/{session_id}/grants", dependencies=[Depends(require_ui), Depends(require_ready)])
    def list_grants(session_id: UUID, response: Response) -> list[GrantView]:
        require_session(session_id)
        response.headers["Cache-Control"] = "no-store"
        return grants.list(session_id)

    # The entire /_host namespace is authenticated before routing or validating its input.
    @router.post("/_host/sessions/{session_id}/grants", status_code=201, dependencies=[Depends(require_ready)])
    def create_grant(session_id: UUID, body: CreateGrant, response: Response) -> GrantView:
        require_session(session_id)
        if body.operations != ["read"]:
            raise ApiError(400, "INVALID_REQUEST", "文件授权仅支持读取。")
        grant = grants.create(session_id, body.scope, body.path)
        response.headers["Cache-Control"] = "no-store"
        return grant

    @router.delete("/_host/grants/{grant_id}", status_code=204, dependencies=[Depends(require_ready)])
    def revoke_grant(grant_id: UUID, session_id: UUID | None = None) -> Response:
        grants.revoke(grant_id, session_id)
        return Response(status_code=204, headers={"Cache-Control": "no-store"})

    return router
