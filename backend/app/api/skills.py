"""FastAPI routes for Skills listing, detail, GitHub/local/ZIP import, dependency prepare, update, enable/disable, and delete.

Channel separation:
- UI Token (`GET /api/v1/skills`, `GET /api/v1/skills/{skill_id}`): read-only queries.
- Host Token (`/api/v1/_host/skills/...`): import preview, selective install, local import,
  enable/disable, update, dependency preparation, and deletion.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.security import ApiError
from app.skills.dependencies import extract_dependency_spec, prepare_missing_dependencies
from app.skills.installer import ImportSourceKind, SkillInstaller
from app.skills.manager import SkillManager
from app.skills.models import SkillError
from app.skills.parser import parse_skill_directory

_ERROR_HTTP_STATUS: dict[str, int] = {
    "SKILL_NOT_FOUND": 404,
    "SKILL_FILES_MISSING": 404,
    "SKILL_RESOURCE_NOT_FOUND": 404,
    "SKILL_PREVIEW_EXPIRED": 404,
    "SKILL_INVALID_PATH": 422,
    "SKILL_INVALID_SOURCE": 422,
    "SKILL_INVALID": 422,
    "SKILL_PARSE_ERROR": 422,
    "SKILL_RESOURCE_INVALID": 422,
    "SKILL_UPDATE_UNSUPPORTED": 409,
    "SKILL_RESOURCE_TOO_LARGE": 413,
    "SKILL_IMPORT_FAILED": 502,
}


def _raise_api_error(exc: SkillError) -> None:
    raise ApiError(_ERROR_HTTP_STATUS.get(exc.code, 500), exc.code, exc.message) from None


def _require_skill_manager(request: Request) -> SkillManager:
    manager: SkillManager | None = getattr(request.app.state, "skill_manager", None)
    if manager is None:
        raise ApiError(503, "SKILLS_UNAVAILABLE", "技能管理服务尚未初始化。")
    return manager


def _require_skill_installer(request: Request) -> SkillInstaller:
    installer: SkillInstaller | None = getattr(request.app.state, "skill_installer", None)
    if installer is None:
        raise ApiError(503, "SKILLS_UNAVAILABLE", "技能安装服务尚未初始化。")
    return installer


class ImportLocalSkillRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    path: Annotated[str, StringConstraints(min_length=1, max_length=4096)]
    enabled: bool = False
    replace_existing_id: Annotated[str, StringConstraints(min_length=1, max_length=128)] | None = None


class ImportPreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    source_kind: ImportSourceKind
    source: Annotated[str, StringConstraints(min_length=1, max_length=4096)]
    ref: Annotated[str, StringConstraints(min_length=1, max_length=256)] | None = None
    subpath: Annotated[str, StringConstraints(min_length=1, max_length=512)] | None = None


class InstallFromPreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    preview_id: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    selected_relative_paths: Annotated[
        list[Annotated[str, StringConstraints(min_length=1, max_length=512)]],
        Field(min_length=1, max_length=64),
    ]
    enabled: bool = True
    replace_existing: bool = True


class UpdateSkillRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: bool


def skills_router(host_namespace: str, require_ui, require_ready) -> APIRouter:
    router = APIRouter()

    # ------------------------------------------------------------------
    # UI Token: read-only skill queries
    # ------------------------------------------------------------------

    @router.get(
        "/api/v1/skills",
        dependencies=[Depends(require_ui), Depends(require_ready)],
    )
    async def list_skills(request: Request, response: Response) -> list[dict[str, Any]]:
        manager = _require_skill_manager(request)
        views = await asyncio.to_thread(manager.list_skills)
        response.headers["Cache-Control"] = "no-store"
        return [v.to_dict() for v in views]

    @router.get(
        "/api/v1/skills/{skill_id}",
        dependencies=[Depends(require_ui), Depends(require_ready)],
    )
    async def get_skill_detail(skill_id: str, request: Request, response: Response) -> dict[str, Any]:
        manager = _require_skill_manager(request)
        try:
            detail = await asyncio.to_thread(manager.get_skill_detail, skill_id)
        except SkillError as exc:
            _raise_api_error(exc)
        response.headers["Cache-Control"] = "no-store"
        return detail.to_dict()

    # ------------------------------------------------------------------
    # Host Token: privileged preview, install, update, prepare, enable, delete
    # ------------------------------------------------------------------

    @router.post(
        f"{host_namespace}/skills/import-preview",
        dependencies=[Depends(require_ready)],
    )
    async def host_import_preview(
        body: ImportPreviewRequest,
        request: Request,
        response: Response,
    ) -> dict[str, Any]:
        installer = _require_skill_installer(request)
        try:
            preview = await installer.preview_import(
                source_kind=body.source_kind,
                source=body.source,
                ref=body.ref,
                subpath=body.subpath,
            )
        except SkillError as exc:
            _raise_api_error(exc)
        response.headers["Cache-Control"] = "no-store"
        return preview.to_dict()

    @router.post(
        f"{host_namespace}/skills/install",
        status_code=201,
        dependencies=[Depends(require_ready)],
    )
    async def host_install_skills(
        body: InstallFromPreviewRequest,
        request: Request,
        response: Response,
    ) -> dict[str, Any]:
        installer = _require_skill_installer(request)
        try:
            installed = await installer.install_from_preview(
                body.preview_id,
                body.selected_relative_paths,
                enabled=body.enabled,
                replace_existing=body.replace_existing,
            )
        except SkillError as exc:
            _raise_api_error(exc)
        response.headers["Cache-Control"] = "no-store"
        return {"skills": [v.to_dict() for v in installed]}

    @router.post(
        f"{host_namespace}/skills/import-local",
        status_code=201,
        dependencies=[Depends(require_ready)],
    )
    async def host_import_local_skills(
        body: ImportLocalSkillRequest,
        request: Request,
        response: Response,
    ) -> dict[str, Any]:
        manager = _require_skill_manager(request)
        try:
            installed = await asyncio.to_thread(
                manager.import_local_directory,
                body.path,
                enabled=body.enabled,
                replace_existing_id=body.replace_existing_id,
            )
        except SkillError as exc:
            _raise_api_error(exc)
        response.headers["Cache-Control"] = "no-store"
        return {"skills": [v.to_dict() for v in installed]}

    @router.patch(
        f"{host_namespace}/skills/{{skill_id}}",
        dependencies=[Depends(require_ready)],
    )
    async def host_update_skill(
        skill_id: str,
        body: UpdateSkillRequest,
        request: Request,
        response: Response,
    ) -> dict[str, Any]:
        manager = _require_skill_manager(request)
        try:
            view = await asyncio.to_thread(manager.set_enabled, skill_id, body.enabled)
        except SkillError as exc:
            _raise_api_error(exc)
        response.headers["Cache-Control"] = "no-store"
        return view.to_dict()

    @router.post(
        f"{host_namespace}/skills/{{skill_id}}/update",
        dependencies=[Depends(require_ready)],
    )
    async def host_refresh_skill_source(
        skill_id: str,
        request: Request,
        response: Response,
    ) -> dict[str, Any]:
        installer = _require_skill_installer(request)
        try:
            view = await installer.update_skill(skill_id)
        except SkillError as exc:
            _raise_api_error(exc)
        response.headers["Cache-Control"] = "no-store"
        return view.to_dict()

    @router.post(
        f"{host_namespace}/skills/{{skill_id}}/prepare",
        dependencies=[Depends(require_ready)],
    )
    async def host_prepare_skill_dependencies(
        skill_id: str,
        request: Request,
        response: Response,
    ) -> dict[str, Any]:
        manager = _require_skill_manager(request)
        try:
            _stored, disk_dir = await asyncio.to_thread(manager.resolve_skill_dir, skill_id)
            doc = await asyncio.to_thread(parse_skill_directory, disk_dir)
            spec = await asyncio.to_thread(extract_dependency_spec, disk_dir, doc)
            dep_status, prep_result = await prepare_missing_dependencies(spec, manager.runtime_manager)
            view = await asyncio.to_thread(manager.get_skill_detail, skill_id)
        except SkillError as exc:
            _raise_api_error(exc)
        response.headers["Cache-Control"] = "no-store"
        return {
            "ok": dep_status.ready if prep_result is None else (prep_result.ok and dep_status.ready),
            "dependency_status": dep_status.to_dict(),
            "prepare_result": prep_result.to_dict() if prep_result is not None else None,
            "skill": view.skill.to_dict(),
        }

    @router.delete(
        f"{host_namespace}/skills/{{skill_id}}",
        status_code=204,
        dependencies=[Depends(require_ready)],
    )
    async def host_delete_skill(skill_id: str, request: Request) -> Response:
        manager = _require_skill_manager(request)
        try:
            await asyncio.to_thread(manager.delete_skill, skill_id)
        except SkillError as exc:
            _raise_api_error(exc)
        return Response(status_code=204, headers={"Cache-Control": "no-store"})

    return router
