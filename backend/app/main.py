import asyncio
import json
import os
import re
import sqlite3
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Literal
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, SecretStr
from starlette.exceptions import HTTPException
from starlette.middleware.cors import CORSMiddleware

from app.api.chat import chat_router, run_router
from app.api.grants import grant_router
from app.api.index_jobs import index_job_router, index_management_router, index_overview_router
from app.api.knowledge_search import knowledge_search_router
from app.api.library import library_router
from app.api.mcp import mcp_router
from app.api.runtime import runtime_router
from app.api.skills import skills_router
from app.api.workspace import workspace_router
from app.diagnostics import record
from app.mcp.manager import McpManager
from app.mcp.process_registry import McpProcessRegistry
from app.models import ModelError
from app.models.deepseek import DEEPSEEK_BASE_URL
from app.models.reranker import SiliconFlowReranker
from app.models.siliconflow import EMBEDDING_MODEL, MAX_BATCH_SIZE, SiliconFlowEmbedding
from app.runtime.executor import RuntimeExecutor
from app.runtime.manager import RuntimeManager
from app.security import ApiError, require_bearer, validate_tokens
from app.services.citation_resolver import CitationResolver
from app.services.document_embeddings import DocumentEmbeddingService
from app.services.hybrid_retrieval import HybridRetrievalService
from app.services.index_status import IndexStatusService
from app.services.index_worker import IndexWorker
from app.services.keyword_search import KeywordSearchService
from app.services.knowledge_qa import KnowledgeQAService
from app.services.mcp import McpService
from app.services.vector_retrieval import VectorRetrievalService
from app.services.grants import GrantRegistry
from app.services.library import LibraryService, log_maintenance_failure
from app.services.models import ModelService
from app.services.runs import RunService
from app.skills.installer import SkillInstaller
from app.skills.manager import SkillManager
from app.storage.artifacts import WorkspaceArtifactStore
from app.storage.business import DATABASE_NAME, BusinessStore, DatabaseLockedError
from app.storage.checkpoints import CHECKPOINT_DATABASE_NAME
from app.storage.chunks import ChunkStore
from app.storage.index_jobs import IndexJobStore
from app.storage.knowledge_settings import KnowledgeSettingsStore
from app.storage.library import KnowledgeStore, KnowledgeStoreError
from app.storage.mcp import McpStore
from app.storage.parsed_documents import ParsedDocumentStore
from app.storage.skills import SkillStore
from app.workspace.artifacts import ArtifactService
from app.workspace.service import WorkspaceService

DEV_ORIGIN = "http://127.0.0.1:1420"
MACOS_APP_ORIGIN = "tauri://localhost"
HOST_NAMESPACE = "/api/v1/_host"
HOST_NAMESPACES = (HOST_NAMESPACE, "/_host")
LOOPBACK_HOST = re.compile(r"127\.0\.0\.1(?::[0-9]{1,5})?")
KNOWLEDGE_CLEANUP_SECONDS = 60
INDEX_BATCH_SIZE_ENV = "KNOWLEDGEAGENT_INDEX_EMBEDDING_BATCH_SIZE"


class Status(BaseModel):
    status: Literal["ready"] = "ready"
    database: Literal["ready"] = "ready"
    model_configured: bool = False
    siliconflow_configured: bool = False


class ModelCredential(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    api_key: SecretStr
    base_url: str = DEEPSEEK_BASE_URL


class ModelConfiguration(BaseModel):
    configured: bool


class EmbeddingConnection(BaseModel):
    model: Literal["BAAI/bge-m3"] = EMBEDDING_MODEL
    dimensions: int


@dataclass(frozen=True)
class SidecarControl:
    instance_id: UUID
    port: int
    stopping: asyncio.Event
    credentials_restored: asyncio.Event | None = None

    def require_ready(self) -> None:
        if self.stopping.is_set():
            raise ApiError(503, "SIDECAR_NOT_READY", "后端正在关闭。")


def error_response(
    request: Request, status_code: int, code: str, message: str, headers: dict[str, str] | None = None,
    *, run_id: str | None = None,
) -> JSONResponse:
    request.state.error_code = code
    if run_id:
        request.state.run_id = run_id
    return JSONResponse(
        status_code=status_code,
        headers={**(headers or {}), "Cache-Control": "no-store"},
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": request.state.request_id,
                "run_id": run_id,
                "details": None,
            }
        },
    )


def create_app(
    ui_token: str,
    host_token: str,
    *,
    database_path: Path,
    control: SidecarControl | None = None,
    ui_origin: str = DEV_ORIGIN,
    models: ModelService | None = None,
    embeddings: SiliconFlowEmbedding | None = None,
) -> FastAPI:
    ui_token_bytes, host_token_bytes = validate_tokens(ui_token, host_token)
    if models is None:
        models = ModelService()
    if embeddings is None:
        embeddings = SiliconFlowEmbedding()
    store = BusinessStore(database_path, initialize=control is None)
    grants = GrantRegistry()
    runs = RunService(store, models, grants)

    async def watch_shutdown() -> None:
        await control.stopping.wait()
        runs.stop()

    async def clean_library(library: LibraryService, jobs: IndexJobStore, stopping: asyncio.Event) -> None:
        parsed = ParsedDocumentStore(library.store)
        chunks = ChunkStore(library.store)
        while not stopping.is_set():
            try:
                await asyncio.to_thread(library.cleanup_orphans)
            except (OSError, sqlite3.Error, KnowledgeStoreError) as error:
                log_maintenance_failure(error)
            for cleanup_interrupted in (parsed.cleanup_interrupted, chunks.cleanup_interrupted):
                try:
                    await asyncio.to_thread(cleanup_interrupted, max_batches=16)
                except (OSError, sqlite3.Error, KnowledgeStoreError) as error:
                    log_maintenance_failure(error)
            try:
                await asyncio.to_thread(jobs.recover_expired)
            except (OSError, sqlite3.Error, KnowledgeStoreError) as error:
                log_maintenance_failure(error)
            with suppress(TimeoutError):
                await asyncio.wait_for(stopping.wait(), timeout=KNOWLEDGE_CLEANUP_SECONDS)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        async with AsyncExitStack() as stack:
            try:
                stack.enter_context(store.own_runs())
                await stack.enter_async_context(runs.lifespan(database_path.with_name(CHECKPOINT_DATABASE_NAME)))
                recovered = await asyncio.to_thread(store.recover_interrupted_runs)
            except DatabaseLockedError:
                record("startup_error", error_code="DATABASE_LOCKED")
                raise
            except (RuntimeError, sqlite3.Error):
                record("startup_error", error_code="DATABASE_MIGRATION_FAILED")
                raise
            for run_id, session_id in recovered:
                record("run_finished", run_id=run_id, session_id=session_id, status="failed",
                       error_code="RUN_INTERRUPTED")
            library = None
            cleanup = None
            worker_task = None
            mcp_service = None
            stop_background = control.stopping if control else asyncio.Event()
            instance_id = control.instance_id if control else uuid4()
            mcp_store = McpStore(store)
            mcp_registry = McpProcessRegistry(database_path.parent, str(instance_id))
            mcp_manager = McpManager(process_registry=mcp_registry)
            mcp_service = McpService(mcp_store, mcp_manager)
            await mcp_service.load_persisted_servers()
            application.state.mcp_store = mcp_store
            application.state.mcp_manager = mcp_manager
            application.state.mcp_service = mcp_service
            runs.mcp_service = mcp_service
            workspace_store = WorkspaceArtifactStore(store)
            workspace_service = WorkspaceService(database_path.parent, workspace_store)
            artifact_service = ArtifactService(workspace_service, workspace_store)
            runtime_manager = RuntimeManager(database_path.parent)
            runtime_executor = RuntimeExecutor(runtime_manager)
            skill_store = SkillStore(store)
            skill_manager = SkillManager(database_path.parent, skill_store, runtime_manager)
            skill_installer = SkillInstaller(skill_manager, runtime_manager)
            await asyncio.to_thread(skill_manager.sync_builtin_skills)
            application.state.workspace_store = workspace_store
            application.state.workspace_service = workspace_service
            application.state.artifact_service = artifact_service
            application.state.runtime_manager = runtime_manager
            application.state.runtime_executor = runtime_executor
            application.state.skill_store = skill_store
            application.state.skill_manager = skill_manager
            application.state.skill_installer = skill_installer
            runs.workspace_service = workspace_service
            runs.artifact_service = artifact_service
            runs.runtime_executor = runtime_executor
            runs.skill_manager = skill_manager
            try:
                knowledge_store = await asyncio.to_thread(KnowledgeStore, database_path.parent)
                library = await asyncio.to_thread(LibraryService, knowledge_store, instance_id)
                try:
                    batch_size = int(os.environ.get(INDEX_BATCH_SIZE_ENV, str(MAX_BATCH_SIZE)))
                    document_embeddings = DocumentEmbeddingService(knowledge_store, embeddings, batch_size=batch_size)
                except ValueError:
                    record("index_configuration_error", error_code="INDEX_BATCH_SIZE_INVALID")
                    document_embeddings = DocumentEmbeddingService(knowledge_store, embeddings)
                worker = IndexWorker(document_embeddings, instance_id)
                knowledge_settings = KnowledgeSettingsStore(knowledge_store)
                index_status = IndexStatusService(knowledge_store, knowledge_settings, worker)
                reranker = SiliconFlowReranker(embeddings)
                application.state.knowledge_store = knowledge_store
                application.state.knowledge_settings = knowledge_settings
                application.state.library = library
                application.state.index_jobs = worker.jobs
                application.state.index_worker = worker
                application.state.index_status = index_status
                application.state.document_embeddings = document_embeddings
                application.state.keyword_search = KeywordSearchService(knowledge_store)
                application.state.vector_retrieval = VectorRetrievalService(knowledge_store, embeddings)
                application.state.reranker = reranker
                application.state.hybrid_retrieval = HybridRetrievalService(
                    application.state.keyword_search,
                    application.state.vector_retrieval,
                    reranker=reranker,
                )
                application.state.citation_resolver = CitationResolver(knowledge_store)
                application.state.knowledge_qa = KnowledgeQAService(
                    knowledge_store,
                    application.state.hybrid_retrieval,
                    application.state.citation_resolver,
                    settings_store=knowledge_settings,
                    models=models,
                )
                runs.knowledge_qa = application.state.knowledge_qa
                cleanup = asyncio.create_task(clean_library(library, worker.jobs, stop_background))
                worker_task = asyncio.create_task(
                    worker.run(stop_background, control.credentials_restored if control else None),
                    name="index-worker",
                )
            except (OSError, sqlite3.Error, KnowledgeStoreError) as error:
                record(
                    "knowledge_startup_error", error_code="KNOWLEDGE_STORE_UNAVAILABLE",
                    exception_type=type(error).__name__,
                    reason=str(error) if isinstance(error, KnowledgeStoreError) else None,
                    sqlite_error=getattr(error, "sqlite_errorname", None), errno=getattr(error, "errno", None),
                )
            watcher = asyncio.create_task(watch_shutdown()) if control else None
            try:
                yield
            finally:
                stop_background.set()
                runs.mcp_service = None
                runs.workspace_service = None
                runs.artifact_service = None
                runs.runtime_executor = None
                runs.skill_manager = None
                if mcp_service is not None:
                    await mcp_service.shutdown()
                application.state.mcp_store = None
                application.state.mcp_manager = None
                application.state.mcp_service = None
                application.state.workspace_store = None
                application.state.workspace_service = None
                application.state.artifact_service = None
                application.state.runtime_manager = None
                application.state.runtime_executor = None
                application.state.skill_store = None
                application.state.skill_manager = None
                application.state.skill_installer = None
                if worker_task is not None:
                    await worker_task
                runs.knowledge_qa = None
                application.state.knowledge_store = None
                application.state.knowledge_settings = None
                application.state.library = None
                application.state.index_jobs = None
                application.state.index_worker = None
                application.state.index_status = None
                application.state.document_embeddings = None
                application.state.keyword_search = None
                application.state.vector_retrieval = None
                application.state.reranker = None
                application.state.hybrid_retrieval = None
                application.state.citation_resolver = None
                application.state.knowledge_qa = None
                if cleanup is not None:
                    await cleanup
                    await asyncio.to_thread(library.close)
                if watcher:
                    watcher.cancel()
                    with suppress(asyncio.CancelledError):
                        await watcher

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, redirect_slashes=False, lifespan=lifespan)
    app.state.knowledge_store = None
    app.state.knowledge_settings = None
    app.state.library = None
    app.state.index_jobs = None
    app.state.index_worker = None
    app.state.index_status = None
    app.state.document_embeddings = None
    app.state.keyword_search = None
    app.state.vector_retrieval = None
    app.state.reranker = None
    app.state.hybrid_retrieval = None
    app.state.citation_resolver = None
    app.state.knowledge_qa = None
    app.state.mcp_store = None
    app.state.mcp_manager = None
    app.state.mcp_service = None
    app.state.workspace_store = None
    app.state.workspace_service = None
    app.state.artifact_service = None
    app.state.runtime_manager = None
    app.state.runtime_executor = None
    app.state.skill_store = None
    app.state.skill_manager = None
    app.state.skill_installer = None
    app.state.runs = runs

    @app.exception_handler(ApiError)
    async def handle_api_error(request: Request, error: ApiError) -> JSONResponse:
        return error_response(request, error.status_code, error.code, error.message, error.headers, run_id=error.run_id)

    @app.exception_handler(RequestValidationError)
    async def handle_invalid_payload(request: Request, _error: RequestValidationError) -> JSONResponse:
        # Validation errors can contain complete input values, including the credential.
        return error_response(request, 422, "VALIDATION_ERROR", "请求字段无效。")

    @app.exception_handler(HTTPException)
    async def handle_http_error(request: Request, error: HTTPException) -> JSONResponse:
        if error.status_code == 404:
            return error_response(request, 404, "NOT_FOUND", "未找到该接口。")
        return error_response(request, error.status_code, "INVALID_REQUEST", "请求无效。", error.headers)

    # Authenticate the entire host namespace, including paths without a registered operation.
    @app.middleware("http")
    async def authenticate_host_namespace(request: Request, call_next):
        path = request.scope["path"]
        if any(path == namespace or path.startswith(f"{namespace}/") for namespace in HOST_NAMESPACES):
            try:
                require_bearer(request, host_token_bytes, ui_token_bytes)
            except ApiError as error:
                return error_response(request, error.status_code, error.code, error.message, error.headers)
        try:
            return await call_next(request)
        except Exception as error:
            request.state.exception_type = type(error).__name__
            return error_response(request, 500, "INTERNAL_ERROR", "服务暂时不可用。")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[ui_origin],
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.middleware("http")
    async def restrict_host_and_origin(request: Request, call_next):
        hosts = request.headers.getlist("host")
        if control:
            valid_host = hosts == [f"127.0.0.1:{control.port}"]
        else:
            valid_host = len(hosts) == 1 and LOOPBACK_HOST.fullmatch(hosts[0]) is not None
        if not valid_host:
            return error_response(request, 400, "INVALID_REQUEST", "请求的 Host 无效。")
        origins = request.headers.getlist("origin")
        if origins and origins != [ui_origin]:
            return error_response(request, 403, "FORBIDDEN", "不允许此请求来源。")
        return await call_next(request)

    @app.middleware("http")
    async def correlate_requests(request: Request, call_next):
        started = monotonic()
        request.state.request_id = str(uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        route = getattr(request.scope.get("route"), "path", None)
        if route != "/api/v1/status" or response.status_code >= 400:
            ids = {}
            for key in ("session_id", "run_id"):
                value = getattr(request.state, key, None) or request.path_params.get(key)
                if value:
                    with suppress(ValueError):
                        ids[key] = str(UUID(value))
            record(
                "http_request", request_id=request.state.request_id, route=route,
                method=request.method if request.method in {"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"} else "OTHER",
                status=response.status_code, elapsed_ms=round((monotonic() - started) * 1000),
                instance_id=str(control.instance_id) if control else None,
                error_code=getattr(request.state, "error_code", None),
                exception_type=getattr(request.state, "exception_type", None), **ids,
            )
        return response

    async def require_ui(request: Request) -> None:
        require_bearer(request, ui_token_bytes, host_token_bytes)

    async def require_ready() -> None:
        if control:
            control.require_ready()

    app.include_router(chat_router(runs, store, require_ui, require_ready))
    app.include_router(run_router(runs, require_ui, require_ready))
    app.include_router(grant_router(store, grants, require_ui, require_ready))
    app.include_router(library_router(require_ui, require_ready))
    app.include_router(knowledge_search_router(require_ui, require_ready))
    app.include_router(index_job_router(HOST_NAMESPACE, require_ready))
    app.include_router(index_management_router(require_ui, require_ready))
    app.include_router(index_overview_router(require_ui, require_ready))
    app.include_router(mcp_router(HOST_NAMESPACE, require_ui, require_ready))
    app.include_router(workspace_router(HOST_NAMESPACE, require_ui, require_ready))
    app.include_router(runtime_router(HOST_NAMESPACE, require_ui, require_ready))
    app.include_router(skills_router(HOST_NAMESPACE, require_ui, require_ready))

    @app.get("/api/v1/status", dependencies=[Depends(require_ui)])
    async def status(response: Response) -> Status:
        if control:
            control.require_ready()
        response.headers["Cache-Control"] = "no-store"
        return Status(model_configured=models.configured, siliconflow_configured=embeddings.configured)

    @app.put(f"{HOST_NAMESPACE}/model-credentials/{{provider}}")
    async def configure_model(
        provider: str, body: ModelCredential, response: Response,
    ) -> ModelConfiguration:
        if control:
            control.require_ready()
        try:
            match provider:
                case "deepseek":
                    models.configure_deepseek(body.api_key, body.base_url)
                case "siliconflow":
                    embeddings.configure(body.api_key, body.base_url)
                case _:
                    raise ApiError(404, "NOT_FOUND", "此模型供应商未启用。")
        except ModelError as error:
            raise ApiError(400, error.code, str(error)) from None
        response.headers["Cache-Control"] = "no-store"
        return ModelConfiguration(configured=True)

    @app.delete(f"{HOST_NAMESPACE}/model-credentials/{{provider}}")
    async def delete_model_credential(
        provider: str, response: Response,
    ) -> ModelConfiguration:
        if control:
            control.require_ready()
        match provider:
            case "deepseek":
                models.clear_deepseek()
            case "siliconflow":
                embeddings.clear()
            case _:
                raise ApiError(404, "NOT_FOUND", "此模型供应商未启用。")
        response.headers["Cache-Control"] = "no-store"
        return ModelConfiguration(configured=False)

    @app.post(f"{HOST_NAMESPACE}/model-credentials/siliconflow/test", dependencies=[Depends(require_ready)])
    async def test_siliconflow_connection(response: Response) -> EmbeddingConnection:
        try:
            dimensions = await embeddings.test_connection()
        except ModelError as error:
            raise ApiError({
                "EMBEDDING_NOT_CONFIGURED": 409,
                "EMBEDDING_AUTH_FAILED": 401,
                "EMBEDDING_FORBIDDEN": 403,
                "EMBEDDING_RATE_LIMITED": 429,
                "EMBEDDING_TIMEOUT": 504,
            }.get(error.code, 502), error.code, str(error)) from None
        response.headers["Cache-Control"] = "no-store"
        return EmbeddingConnection(dimensions=dimensions)

    if control:
        @app.get(f"{HOST_NAMESPACE}/ready")
        async def host_ready(response: Response) -> dict[str, str]:
            control.require_ready()
            response.headers["Cache-Control"] = "no-store"
            return {"status": "ready", "instance_id": str(control.instance_id)}

        @app.post(f"{HOST_NAMESPACE}/shutdown", status_code=204)
        async def host_shutdown() -> Response:
            control.stopping.set()
            return Response(status_code=204, headers={"Cache-Control": "no-store"})

        @app.post(f"{HOST_NAMESPACE}/model-credentials/restore-complete", status_code=204,
                  dependencies=[Depends(require_ready)])
        async def credentials_restore_complete() -> Response:
            if control.credentials_restored is not None:
                control.credentials_restored.set()
            return Response(status_code=204, headers={"Cache-Control": "no-store"})

    return app


def load_dev_app() -> FastAPI:
    """Explicit P1.1 entry point. Start Uvicorn on 127.0.0.1."""
    try:
        config_path = Path(os.environ["KNOWLEDGEAGENT_DEV_CONFIG"])
        config = json.loads(config_path.read_text(encoding="utf-8"))
        return create_app(
            config["ui_token"], config["host_token"], database_path=config_path.with_name(DATABASE_NAME),
        )
    except (KeyError, OSError, TypeError, ValueError):
        raise RuntimeError("缺少或无效的 KnowledgeAgent 本地开发配置。") from None
