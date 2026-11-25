import asyncio
import base64
import binascii
import json
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints

from app.models import ModelError
from app.security import ApiError
from app.services.hybrid_retrieval import CoverageMode, MAX_COMPARISON_DOCUMENTS, RetrievalMode
from app.services.library import DOCUMENT_ID_PATTERN
from app.services.runs import KnowledgeRunConfig, RunNotCancellable, RunService, RunServiceStopping, RunStream
from app.storage.business import (
    BusinessStore, Message, RunConflict, RunNotFound, RunStatus, RunSummary, Session, SessionNotFound,
    SessionSummary,
)

KEEPALIVE_SECONDS = 15
MAX_MESSAGE_LENGTH = 20_000


def validate_utf8(value: str) -> str:
    value.encode("utf-8")
    return value


class CreateSession(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=100),
                     AfterValidator(validate_utf8)] = "新对话"


class KnowledgeQueryOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: bool = True
    document_ids: Annotated[
        list[Annotated[str, StringConstraints(pattern=DOCUMENT_ID_PATTERN)]],
        Field(max_length=128),
    ] | None = None
    retrieval_mode: RetrievalMode = "hybrid"
    coverage_mode: CoverageMode = "global"
    top_k: Annotated[int, Field(ge=1, le=20)] = 6


class StartRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_request_id: UUID
    message: Annotated[
        str,
        StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=MAX_MESSAGE_LENGTH),
        AfterValidator(validate_utf8),
    ]
    knowledge: KnowledgeQueryOptions | None = None


class RunPage(BaseModel):
    items: list[RunSummary]
    next_cursor: str | None


def encode_cursor(run: RunSummary) -> str:
    return base64.urlsafe_b64encode(f"{run.session_id}/{run.started_at}/{run.run_id}".encode()).decode()


def decode_cursor(cursor: str, session_id: str) -> tuple[str, str]:
    try:
        encoded = base64.b64decode(cursor, altchars=b"-_", validate=True)
        owner, started_at, run_id = encoded.decode("ascii").split("/")
        if (
            owner != session_id or str(UUID(run_id)) != run_id
            or datetime.strptime(started_at, "%Y-%m-%dT%H:%M:%S.%fZ").isoformat(timespec="milliseconds") + "Z"
            != started_at or base64.urlsafe_b64encode(encoded).decode() != cursor
        ):
            raise ValueError
        return started_at, run_id
    except (binascii.Error, ValueError):
        raise ApiError(422, "VALIDATION_ERROR", "运行列表分页游标无效。") from None


async def event_stream(stream: RunStream) -> AsyncIterator[bytes]:
    try:
        while True:
            try:
                event = await asyncio.wait_for(stream.events.get(), KEEPALIVE_SECONDS)
            except TimeoutError:
                yield b": keepalive\n\n"
                continue
            if event is None:
                return
            payload = {
                "version": 1,
                "type": event.type,
                "session_id": stream.run.session_id,
                "run_id": stream.run.run_id,
                "ts": event.ts,
                "data": event.data,
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n".encode()
    finally:
        stream.detach()


def chat_router(runs: RunService, store: BusinessStore, require_ui, require_ready) -> APIRouter:
    router = APIRouter(prefix="/api/v1/sessions", dependencies=[Depends(require_ui), Depends(require_ready)])

    @router.get("")
    def list_sessions(response: Response) -> list[SessionSummary]:
        response.headers["Cache-Control"] = "no-store"
        return store.list_sessions()

    @router.get("/{session_id}/messages")
    def list_messages(session_id: UUID, response: Response) -> list[Message]:
        try:
            messages = store.list_messages(str(session_id))
        except SessionNotFound:
            raise ApiError(404, "NOT_FOUND", "未找到该会话。") from None
        response.headers["Cache-Control"] = "no-store"
        return messages

    @router.get("/{session_id}/runs")
    def list_runs(
        session_id: UUID, response: Response, limit: Annotated[int, Query(ge=1, le=100)] = 50,
        cursor: Annotated[str | None, Query(min_length=1, max_length=160)] = None,
    ) -> RunPage:
        before = decode_cursor(cursor, str(session_id)) if cursor is not None else None
        try:
            items = store.list_runs(str(session_id), limit + 1, before)
        except SessionNotFound:
            raise ApiError(404, "NOT_FOUND", "未找到该会话。") from None
        page_items = items[:limit]
        qa = runs.knowledge_qa
        if qa is not None:
            page_items = [
                replace(item, citations=qa.refresh_stored_citations(item.citations))
                if item.citations else item
                for item in page_items
            ]
        response.headers["Cache-Control"] = "no-store"
        return RunPage(items=page_items, next_cursor=encode_cursor(items[limit - 1]) if len(items) > limit else None)

    @router.post("", status_code=201)
    async def create_session(body: CreateSession, response: Response) -> Session:
        try:
            session = await runs.create_session(body.title)
        except RunServiceStopping:
            raise ApiError(503, "SIDECAR_NOT_READY", "后端正在关闭。") from None
        response.headers["Cache-Control"] = "no-store"
        return session

    @router.post("/{session_id}/rename")
    async def rename_session(session_id: UUID, body: CreateSession, response: Response) -> Session:
        try:
            session = await runs.rename_session(str(session_id), body.title)
        except SessionNotFound:
            raise ApiError(404, "NOT_FOUND", "未找到该会话。") from None
        except RunServiceStopping:
            raise ApiError(503, "SIDECAR_NOT_READY", "后端正在关闭。") from None
        response.headers["Cache-Control"] = "no-store"
        return session

    @router.post("/{session_id}/delete", status_code=204)
    async def delete_session(session_id: UUID) -> Response:
        try:
            await runs.delete_session(str(session_id))
        except SessionNotFound:
            raise ApiError(404, "NOT_FOUND", "未找到该会话。") from None
        except RunConflict as error:
            raise ApiError(409, error.code, "此会话有正在执行的任务，请先停止生成后再删除。", run_id=error.run_id) from None
        except RunServiceStopping:
            raise ApiError(503, "SIDECAR_NOT_READY", "后端正在关闭。") from None
        return Response(status_code=204, headers={"Cache-Control": "no-store"})

    @router.post("/{session_id}/runs/stream")
    async def start_run(session_id: UUID, body: StartRun, request: Request) -> StreamingResponse:
        knowledge_cfg: KnowledgeRunConfig | None = None
        if body.knowledge is not None and body.knowledge.enabled:
            if (body.knowledge.coverage_mode == "per_document" and body.knowledge.document_ids
                    and len(set(body.knowledge.document_ids)) > MAX_COMPARISON_DOCUMENTS):
                raise ApiError(422, "VALIDATION_ERROR", f"跨教材对比最多同时选择 {MAX_COMPARISON_DOCUMENTS} 本教材。")
            knowledge_cfg = KnowledgeRunConfig(
                document_ids=tuple(body.knowledge.document_ids) if body.knowledge.document_ids is not None else None,
                retrieval_mode=body.knowledge.retrieval_mode,
                coverage_mode=body.knowledge.coverage_mode,
                top_k=body.knowledge.top_k,
            )
        try:
            stream = await runs.start(
                str(session_id), str(body.client_request_id), body.message,
                request_id=request.state.request_id,
                **({"knowledge": knowledge_cfg} if knowledge_cfg is not None else {}),
            )
        except SessionNotFound:
            raise ApiError(404, "NOT_FOUND", "未找到该会话。") from None
        except RunConflict as error:
            message = ("此请求已提交。" if error.code == "DUPLICATE_CLIENT_REQUEST"
                       else "此会话有正在执行的任务。")
            raise ApiError(409, error.code, message, run_id=error.run_id) from None
        except ModelError as error:
            raise ApiError(503, error.code, str(error)) from None
        except RunServiceStopping:
            raise ApiError(503, "SIDECAR_NOT_READY", "后端正在关闭。") from None
        request.state.run_id = stream.run.run_id
        return StreamingResponse(
            event_stream(stream), media_type="text/event-stream",
            headers={"Cache-Control": "no-store"},
        )

    return router


def run_router(runs: RunService, require_ui, require_ready) -> APIRouter:
    router = APIRouter(prefix="/api/v1/runs", dependencies=[Depends(require_ui), Depends(require_ready)])

    @router.get("/lookup")
    async def lookup_run(session_id: UUID, client_request_id: UUID, response: Response, request: Request) -> RunStatus:
        try:
            status = await runs.lookup_run(str(session_id), str(client_request_id))
        except RunNotFound:
            raise ApiError(404, "NOT_FOUND", "未找到该运行。") from None
        request.state.run_id = status.run_id
        request.state.session_id = status.session_id
        response.headers["Cache-Control"] = "no-store"
        return status

    @router.get("/{run_id}")
    async def get_run(run_id: UUID, response: Response, request: Request) -> RunStatus:
        try:
            status = await runs.get_run(str(run_id))
        except RunNotFound:
            raise ApiError(404, "NOT_FOUND", "未找到该运行。") from None
        request.state.session_id = status.session_id
        response.headers["Cache-Control"] = "no-store"
        return status

    @router.post("/{run_id}/cancel")
    async def cancel_run(run_id: UUID, response: Response, request: Request) -> dict:
        try:
            status = await runs.cancel(str(run_id))
        except RunNotFound:
            raise ApiError(404, "NOT_FOUND", "未找到该运行。") from None
        except RunNotCancellable:
            raise ApiError(409, "INVALID_RUN_STATE", "无法取消当前运行。", run_id=str(run_id)) from None
        except RunServiceStopping:
            raise ApiError(503, "SIDECAR_NOT_READY", "后端正在关闭。") from None
        request.state.session_id = status.session_id
        response.headers["Cache-Control"] = "no-store"
        if status.status == "cancelling":
            response.status_code = 202
            return {"run_id": status.run_id, "status": status.status}
        return {"run_id": status.run_id, "status": status.status, "already_finished": True}

    return router
