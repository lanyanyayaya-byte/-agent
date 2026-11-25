from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query, Request, Response
from pydantic import BaseModel, ConfigDict

from app.api.library import DocumentId, require_library
from app.security import ApiError
from app.services.index_readiness import PublishedIndexReadiness
from app.services.index_worker import IndexJobProgress, IndexJobState, IndexWorker
from app.services.library import DOCUMENT_ID_PATTERN
from app.storage.index_jobs import PIPELINE_JOB_TYPE, IndexJob, IndexJobError, IndexJobStatus

JOB_ID_PATTERN = r"^[0-9a-f]{32}$"
JobId = Annotated[str, Path(pattern=JOB_ID_PATTERN)]


@dataclass(frozen=True)
class IndexJobPage:
    active_job_id: str | None
    jobs: list[IndexJobProgress]


@dataclass(frozen=True)
class DocumentIndex:
    document_id: str
    document_version_id: str
    published_readiness: PublishedIndexReadiness | None
    job: IndexJobState | None


class IndexConsent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    allow_cloud_embedding: bool


def _job_not_found() -> ApiError:
    return ApiError(404, "INDEX_JOB_NOT_FOUND", "未找到该索引任务。")


def _scoped_job(worker: IndexWorker, document_id: str, version_id: str, job_id: str) -> IndexJob:
    job = worker.jobs.get_job(job_id)
    if (job is None or (job.document_id, job.document_version_id, job.job_type) !=
            (document_id, version_id, PIPELINE_JOB_TYPE) or job.config_key not in worker.config_keys):
        raise _job_not_found()
    return job


def _submit(
    worker: IndexWorker, document_id: str, version_id: str, body: IndexConsent, *, retry_job_id: str | None = None,
) -> IndexJobState:
    if not body.allow_cloud_embedding:
        raise ApiError(422, "EMBEDDING_CONSENT_REQUIRED", "请确认教材分块文本将发送至 SiliconFlow。")
    current = worker.pipeline.store.latest_version_identity(document_id)
    if current is None:
        raise ApiError(404, "TEXTBOOK_NOT_FOUND", "该教材已不存在。")
    if current[0] != version_id:
        raise ApiError(409, "INDEX_VERSION_NOT_CURRENT", "教材版本已更新，请刷新后重试。")
    try:
        job = worker.pipeline.submit(document_id, version_id, current_only=True, restart_after=retry_job_id)
    except IndexJobError as error:
        code = str(error)
        if code == "INDEX_JOB_NOT_FOUND":
            raise _job_not_found() from None
        if code == "INDEX_VERSION_NOT_CURRENT":
            raise ApiError(409, code, "教材版本已变更，请刷新后重试。") from None
        if code in ("INDEX_JOB_NOT_TERMINAL", "INDEX_JOB_NOT_RETRYABLE", "INDEX_JOB_IDENTITY_MISMATCH"):
            raise ApiError(409, code, "该索引任务当前无法重试，请刷新状态。") from None
        raise
    return IndexJobState.from_job(job)


def index_management_router(require_ui, require_ready) -> APIRouter:
    router = APIRouter(prefix="/api/v1/knowledge/documents/{document_id}",
                       dependencies=[Depends(require_ui), Depends(require_ready), Depends(require_library)])
    jobs_path = "/versions/{version_id}/index-jobs"

    @router.get("/index")
    def document_index(document_id: DocumentId, request: Request, response: Response) -> DocumentIndex:
        worker: IndexWorker = request.app.state.index_worker
        current = worker.pipeline.store.latest_version_identity(document_id)
        if current is None:
            raise ApiError(404, "TEXTBOOK_NOT_FOUND", "该教材已不存在。")
        version_id, format = current
        job = worker.pipeline.latest_attempt(document_id, version_id, format)
        readiness = worker.pipeline.version_readiness(version_id, format)
        response.headers["Cache-Control"] = "no-store"
        return DocumentIndex(document_id, version_id, readiness, IndexJobState.from_job(job) if job else None)

    @router.post(jobs_path)
    def submit(document_id: DocumentId, version_id: DocumentId, body: IndexConsent,
               request: Request, response: Response) -> IndexJobState:
        response.headers["Cache-Control"] = "no-store"
        return _submit(request.app.state.index_worker, document_id, version_id, body)

    @router.get(f"{jobs_path}/{{job_id}}")
    def status(document_id: DocumentId, version_id: DocumentId, job_id: JobId,
               request: Request, response: Response) -> IndexJobState:
        response.headers["Cache-Control"] = "no-store"
        return IndexJobState.from_job(_scoped_job(request.app.state.index_worker, document_id, version_id, job_id))

    @router.post(f"{jobs_path}/{{job_id}}/cancel")
    def cancel(document_id: DocumentId, version_id: DocumentId, job_id: JobId,
               request: Request, response: Response) -> IndexJobState:
        worker: IndexWorker = request.app.state.index_worker
        _scoped_job(worker, document_id, version_id, job_id)
        try:
            job = worker.jobs.request_cancel(job_id)
        except IndexJobError as error:
            if str(error) == "INDEX_JOB_NOT_FOUND":
                raise _job_not_found() from None
            raise
        response.headers["Cache-Control"] = "no-store"
        return IndexJobState.from_job(job)

    @router.post(f"{jobs_path}/{{job_id}}/retry")
    def retry(document_id: DocumentId, version_id: DocumentId, job_id: JobId, body: IndexConsent,
              request: Request, response: Response) -> IndexJobState:
        worker: IndexWorker = request.app.state.index_worker
        _scoped_job(worker, document_id, version_id, job_id)
        response.headers["Cache-Control"] = "no-store"
        return _submit(worker, document_id, version_id, body, retry_job_id=job_id)

    return router


def index_job_router(host_namespace: str, require_ready) -> APIRouter:
    router = APIRouter(prefix=f"{host_namespace}/knowledge/index-jobs",
                       dependencies=[Depends(require_ready), Depends(require_library)])

    @router.get("")
    def list_jobs(
        request: Request, response: Response, *,
        document_id: Annotated[str | None, Query(pattern=DOCUMENT_ID_PATTERN)] = None,
        status: IndexJobStatus | None = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
        offset: Annotated[int, Query(ge=0, le=2**63 - 1)] = 0,
    ) -> IndexJobPage:
        worker = request.app.state.index_worker
        rows = worker.jobs.list_jobs(document_id=document_id, status=status, job_type=PIPELINE_JOB_TYPE,
                                     limit=limit, offset=offset)
        response.headers["Cache-Control"] = "no-store"
        return IndexJobPage(worker.active_job_id, [worker.progress(job) for job in rows])

    @router.get("/{job_id}")
    def get_job(request: Request, response: Response, job_id: Annotated[str, Path(pattern=JOB_ID_PATTERN)]) -> IndexJobProgress:
        worker = request.app.state.index_worker
        job = worker.jobs.get_job(job_id)
        if job is None or job.job_type != PIPELINE_JOB_TYPE:
            raise _job_not_found()
        response.headers["Cache-Control"] = "no-store"
        return worker.progress(job)

    return router
