"""Consume the local index queue within the Sidecar lifespan."""

import asyncio
import sqlite3
from dataclasses import asdict, dataclass
from uuid import UUID

from app.diagnostics import record
from app.models.siliconflow import SiliconFlowEmbedding
from app.services.document_embeddings import DocumentEmbeddingService
from app.services.index_pipeline import IndexPipeline
from app.services.index_readiness import IndexReadiness
from app.storage.index_jobs import (
    ALL_PIPELINE_JOB_TYPES, DEFAULT_LEASE_SECONDS, IndexJob, IndexJobClaim, IndexJobError, IndexJobStage, IndexJobStatus,
)

POLL_SECONDS = 1
CREDENTIAL_REPLAY_TIMEOUT = 15
INDEX_EMBEDDING_PROFILE = SiliconFlowEmbedding.profile(compatibility_epoch="bge-m3-v1")


@dataclass(frozen=True, slots=True)
class IndexJobState:
    job_id: str
    document_id: str
    document_version_id: str
    status: IndexJobStatus
    stage: IndexJobStage
    completed_work: int
    total_work: int | None
    error_code: str | None
    retry_count: int
    next_retry_at: int | None
    cancel_requested_at: int | None

    @classmethod
    def from_job(cls, job: IndexJob) -> "IndexJobState":
        return cls(
            job.job_id, job.document_id, job.document_version_id, job.status, job.stage,
            job.completed_work, job.total_work, job.error_code, job.retry_count, job.next_retry_at,
            job.cancel_requested_at,
        )


@dataclass(frozen=True, slots=True)
class IndexJobProgress(IndexJobState):
    published_readiness: IndexReadiness | None
    worker_supported: bool


class IndexWorker:
    def __init__(
        self, embeddings: DocumentEmbeddingService, instance_id: UUID, *,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
    ):
        self.pipeline = IndexPipeline(embeddings.results.store, embeddings, INDEX_EMBEDDING_PROFILE)
        self.jobs = self.pipeline.jobs
        self.instance_id = instance_id
        self.lease_seconds = lease_seconds
        self.config_keys = self.pipeline.config_keys
        self.active_job_id: str | None = None

    def progress(self, job: IndexJob) -> IndexJobProgress:
        try:
            readiness = self.pipeline.published_readiness(job)
        except sqlite3.Error as error:
            record("index_readiness_error", job_id=job.job_id, sqlite_error=getattr(error, "sqlite_errorname", None))
            readiness = None
        return IndexJobProgress(**asdict(IndexJobState.from_job(job)),
                                published_readiness=readiness, worker_supported=job.config_key in self.config_keys)

    async def run(self, stopping: asyncio.Event, credentials_restored: asyncio.Event | None = None) -> None:
        shutdown = asyncio.create_task(stopping.wait())
        try:
            if credentials_restored is not None and not credentials_restored.is_set():
                replay = asyncio.create_task(credentials_restored.wait())
                try:
                    done, _ = await asyncio.wait((replay, shutdown), timeout=CREDENTIAL_REPLAY_TIMEOUT,
                                                 return_when=asyncio.FIRST_COMPLETED)
                    if not done:
                        record("index_credential_replay_timeout", instance_id=str(self.instance_id))
                finally:
                    replay.cancel()
                    await asyncio.gather(replay, return_exceptions=True)

            while not stopping.is_set():
                try:
                    claim = await asyncio.to_thread(
                        self.jobs.claim_next, self.instance_id, job_type=ALL_PIPELINE_JOB_TYPES,
                        lease_seconds=self.lease_seconds, config_keys=self.config_keys, wait=False,
                    )
                    if claim is not None:
                        await self._execute(claim, stopping, shutdown)
                        continue
                except Exception as error:
                    record("index_worker_poll_error", instance_id=str(self.instance_id),
                           exception_type=type(error).__name__, sqlite_error=getattr(error, "sqlite_errorname", None))
                await asyncio.wait((shutdown,), timeout=POLL_SECONDS)
        finally:
            shutdown.cancel()
            await asyncio.gather(shutdown, return_exceptions=True)

    async def _execute(
        self, claim: IndexJobClaim, stopping: asyncio.Event, shutdown: asyncio.Task,
    ) -> None:
        job_id = claim.job.job_id
        self.active_job_id = job_id
        try:
            if stopping.is_set():
                await asyncio.to_thread(self.jobs.interrupt, claim)
                return
            record("index_job_claimed", job_id=job_id, instance_id=str(self.instance_id), stage=claim.job.stage)
            work = asyncio.create_task(self.pipeline.execute(claim, lease_seconds=self.lease_seconds))
            try:
                await asyncio.wait((work, shutdown), return_when=asyncio.FIRST_COMPLETED)
            finally:
                if not work.done():
                    work.cancel()
                    await asyncio.gather(work, return_exceptions=True)
            if work.cancelled():
                if not stopping.is_set():
                    raise RuntimeError("Index pipeline cancelled outside Sidecar shutdown.")
                result = await asyncio.to_thread(self.jobs.interrupt, claim)
            else:
                result = work.result()
            record("index_job_attempt_finished", job_id=job_id, status=result.status, stage=result.stage,
                   error_code=result.error_code, next_retry_at=result.next_retry_at)
        except Exception as error:
            if isinstance(error, IndexJobError) and str(error) == "INDEX_JOB_LEASE_LOST":
                record("index_job_lease_lost", job_id=job_id)
                return
            record("index_worker_attempt_error", job_id=job_id, exception_type=type(error).__name__)
            await asyncio.to_thread(self._settle_interrupted, claim, stopping.is_set())
        finally:
            self.active_job_id = None

    def _settle_interrupted(self, claim: IndexJobClaim, stopping: bool) -> None:
        try:
            if stopping:
                self.jobs.interrupt(claim)
            else:
                self.jobs.fail(claim, "INDEX_WORKER_ERROR")
        except IndexJobError as error:
            if str(error) == "INDEX_JOB_CANCEL_REQUESTED":
                self.jobs.acknowledge_cancel(claim)
            elif str(error) != "INDEX_JOB_LEASE_LOST":
                raise
