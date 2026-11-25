"""FastAPI routes for independent Runtime probing and user-confirmed dependency preparation.

Channel separation:
- UI Token (`GET /api/v1/runtime/status`): read-only environment probe.
- Host Token (`POST /api/v1/_host/runtime/prepare`): user-confirmed venv & dependency installation.
- No arbitrary HTTP command execution endpoint is exposed; command execution enters Runtime via Agent Runs.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.runtime.manager import DEFAULT_PREPARE_TIMEOUT_SECONDS, RuntimeError_, RuntimeManager
from app.security import ApiError


def _require_runtime_manager(request: Request) -> RuntimeManager:
    manager: RuntimeManager | None = getattr(request.app.state, "runtime_manager", None)
    if manager is None:
        raise ApiError(503, "RUNTIME_UNAVAILABLE", "运行环境服务尚未初始化。")
    return manager


class PrepareRuntimeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    python_packages: Annotated[
        list[Annotated[str, StringConstraints(min_length=1, max_length=128)]],
        Field(max_length=32),
    ] = Field(default_factory=list)
    install_mermaid_cli: bool = False
    ensure_python_venv: bool = True
    timeout_seconds: Annotated[float, Field(ge=10.0, le=900.0)] = DEFAULT_PREPARE_TIMEOUT_SECONDS


def runtime_router(host_namespace: str, require_ui, require_ready) -> APIRouter:
    router = APIRouter()

    @router.get(
        "/api/v1/runtime/status",
        dependencies=[Depends(require_ui), Depends(require_ready)],
    )
    async def get_runtime_status(request: Request, response: Response) -> dict[str, Any]:
        manager = _require_runtime_manager(request)
        status = await asyncio.to_thread(manager.probe_status)
        response.headers["Cache-Control"] = "no-store"
        return status.to_dict()

    @router.post(
        f"{host_namespace}/runtime/prepare",
        dependencies=[Depends(require_ready)],
    )
    async def host_prepare_runtime(
        body: PrepareRuntimeRequest,
        request: Request,
        response: Response,
    ) -> dict[str, Any]:
        manager = _require_runtime_manager(request)
        try:
            result = await manager.prepare(
                python_packages=body.python_packages,
                install_mermaid_cli=body.install_mermaid_cli,
                ensure_python_venv=body.ensure_python_venv,
                timeout_seconds=body.timeout_seconds,
            )
        except RuntimeError_ as exc:
            raise ApiError(422, exc.code, exc.message) from None
        response.headers["Cache-Control"] = "no-store"
        return result.to_dict()

    return router
