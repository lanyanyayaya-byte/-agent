"""Explicit execution of a claimed textbook indexing job."""

import asyncio
import sqlite3
from contextlib import asynccontextmanager
from dataclasses import asdict
from threading import Event

from app.chunking.hybrid import CHUNKER_VERSION
from app.chunking.models import ChunkerConfig
from app.models.embeddings import EmbeddingError, EmbeddingProfile, canonical_sha256
from app.parsing.models import DocumentParseError
from app.services.document_chunking import DocumentChunkingService
from app.services.document_embeddings import DocumentEmbeddingService
from app.services.document_parsing import PARSER_SPECS, DocumentParsingService
from app.services.index_readiness import IndexReadiness, PublishedIndexReadiness, get_index_readiness
from app.services.index_threads import run_index_sync
from app.storage.index_jobs import (
    ALL_PIPELINE_JOB_TYPES, DEFAULT_LEASE_SECONDS, LOCAL_PIPELINE_JOB_TYPE, LOCAL_PIPELINE_STAGES,
    PIPELINE_JOB_TYPE, PIPELINE_STAGES, IndexJob, IndexJobClaim, IndexJobError, IndexJobStage, IndexJobStore,
)
from app.storage.keywords import KeywordIndex, KeywordIndexError
from app.storage.library import DocumentFormat, KnowledgeStore, KnowledgeStoreError


class IndexPipeline:
    """Run one job using the caller's chunk configuration and embedding profile."""

    def __init__(
        self, store: KnowledgeStore, embeddings: DocumentEmbeddingService, profile: EmbeddingProfile, *,
        chunk_config: ChunkerConfig = ChunkerConfig(),
    ):
        self.store = store
        self.jobs = IndexJobStore(store)
        self.parsing = DocumentParsingService(store)
        self.chunking = DocumentChunkingService(store)
        self.keywords = KeywordIndex(store)
        self.embeddings = embeddings
        self.profile = profile
        self.chunk_config = chunk_config

    @property
    def config_keys(self) -> tuple[str, ...]:
        return tuple(
            self.config_key_for(format, job_type=job_type)
            for job_type in ALL_PIPELINE_JOB_TYPES
            for format in PARSER_SPECS
        )

    def published_readiness(self, job: IndexJob) -> IndexReadiness | None:
        version = self.store.get_document_version(job.document_version_id)
        if version is None or job.config_key != self.config_key_for(version[1], job_type=job.job_type):
            return None
        return self._readiness(job.document_version_id, version[1], job_id=job.job_id)

    def _readiness(
        self, version_id: str, format: DocumentFormat, *, job_id: str | None = None,
    ) -> IndexReadiness | None:
        parsed = self.parsing.results.get_reusable(version_id, PARSER_SPECS[format])
        chunks = self.chunking.get_chunk_set(parsed.parsed_document_id, self.chunk_config) if parsed else None
        return get_index_readiness(
            self.store, version_id, chunk_set_id=chunks.chunk_set_id if chunks else None,
            profile=self.profile, job_id=job_id,
        )

    def version_readiness(self, version_id: str, format: DocumentFormat) -> PublishedIndexReadiness | None:
        readiness = self._readiness(version_id, format)
        assert readiness != "failed"
        return readiness

    def latest_attempt(self, document_id: str, version_id: str, format: DocumentFormat) -> IndexJob | None:
        job = self.jobs.latest_attempt(document_id, version_id, self._config_key(format))
        return job if job is not None and job.job_type == PIPELINE_JOB_TYPE else None

    def _config_key(self, format: DocumentFormat) -> str:
        return canonical_sha256({
            "pipeline_version": 1,
            "format": format,
            "parse": asdict(PARSER_SPECS[format]),
            "chunker_version": CHUNKER_VERSION,
            "chunker_config": self.chunk_config.key,
            "embedding_profile": self.profile.key,
        })

    def config_key_for(self, format: DocumentFormat, *, job_type: str = PIPELINE_JOB_TYPE) -> str:
        if job_type == PIPELINE_JOB_TYPE:
            return self._config_key(format)
        return canonical_sha256({
            "pipeline_version": 1,
            "mode": "local_only",
            "format": format,
            "parse": asdict(PARSER_SPECS[format]),
            "chunker_version": CHUNKER_VERSION,
            "chunker_config": self.chunk_config.key,
        })

    def _version_format(self, document_id: str, version_id: str) -> DocumentFormat:
        version = self.store.get_document_version(version_id)
        if version is None or version[0] != document_id:
            raise IndexJobError("INDEX_VERSION_NOT_CURRENT")
        return version[1]

    def submit(
        self, document_id: str, document_version_id: str, *, restart_after: str | None = None, max_retries: int = 3,
        current_only: bool = False, job_type: str = PIPELINE_JOB_TYPE,
    ) -> IndexJob:
        format = self._version_format(document_id, document_version_id)
        return self.jobs.create_or_get(
            document_id, document_version_id, self.config_key_for(format, job_type=job_type), job_type=job_type,
            max_retries=max_retries, restart_after=restart_after, current_only=current_only,
        )

    async def execute(self, claim: IndexJobClaim, *, lease_seconds: int = DEFAULT_LEASE_SECONDS) -> IndexJob:
        """Execute a single attempt. A lost lease is reported to the caller as IndexJobError."""
        if claim.job.job_type not in ALL_PIPELINE_JOB_TYPES:
            raise IndexJobError("INDEX_PIPELINE_JOB_TYPE_MISMATCH")
        try:
            try:
                current = await run_index_sync(self.jobs.renew_lease, claim, lease_seconds=lease_seconds)
                claim = IndexJobClaim(current, claim.lease_token)
                await self._run_with_heartbeat(claim, lease_seconds)
                return await run_index_sync(self.jobs.complete, claim)
            except EmbeddingError as error:
                return await run_index_sync(
                    self.jobs.fail, claim, error.code, http_status=error.http_status, retry_after=error.retry_after,
                )
            except DocumentParseError as error:
                return await run_index_sync(self.jobs.fail, claim, error.code)
            except KeywordIndexError as error:
                return await run_index_sync(self.jobs.fail, claim, str(error))
            except (OSError, sqlite3.Error, KnowledgeStoreError) as error:
                sqlite_code = getattr(error, "sqlite_errorcode", 0) & 0xff
                code = ("INDEX_PIPELINE_STORAGE_BUSY" if sqlite_code in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
                        else "INDEX_PIPELINE_STORAGE_ERROR")
                return await run_index_sync(self.jobs.fail, claim, code)
            except IndexJobError as error:
                if str(error) not in ("INDEX_PIPELINE_CONFIG_MISMATCH", "INDEX_PUBLICATION_SCOPE_MISMATCH"):
                    raise
                return await run_index_sync(self.jobs.fail, claim, str(error))
        except IndexJobError as error:
            if str(error) == "INDEX_JOB_CANCEL_REQUESTED":
                return await run_index_sync(self.jobs.acknowledge_cancel, claim)
            raise

    async def _run_with_heartbeat(self, claim: IndexJobClaim, lease_seconds: int) -> None:
        cancelled = Event()

        async def heartbeat() -> None:
            loop = asyncio.get_running_loop()
            renew_interval = min(30, lease_seconds / 3)
            next_renewal = loop.time() + renew_interval
            while True:
                await asyncio.sleep(min(1, renew_interval))
                if loop.time() >= next_renewal:
                    await run_index_sync(self.jobs.renew_lease, claim, lease_seconds=lease_seconds)
                    next_renewal = loop.time() + renew_interval
                else:
                    await run_index_sync(self.jobs.check_claim, claim)

        work = asyncio.create_task(self._run_stages(claim, cancelled))
        renewal = asyncio.create_task(heartbeat())
        try:
            done, _ = await asyncio.wait((work, renewal), return_when=asyncio.FIRST_COMPLETED)
            if renewal in done:
                renewal.result()
            work.result()
        finally:
            cancelled.set()
            renewal.cancel()
            work.cancel()
            drained = asyncio.gather(work, renewal, return_exceptions=True)
            try:
                await asyncio.shield(drained)
            except asyncio.CancelledError:
                await drained
                raise

    @asynccontextmanager
    async def _stage(self, claim: IndexJobClaim, stage: IndexJobStage, *, total_work: int = 1):
        stages = LOCAL_PIPELINE_STAGES if claim.job.job_type == LOCAL_PIPELINE_JOB_TYPE else PIPELINE_STAGES
        # Re-resolve earlier artifacts on resume while leaving persisted progress monotonic.
        if stages.index(stage) < stages.index(claim.job.stage):
            await run_index_sync(self.jobs.check_claim, claim)
            yield
            return
        completed = claim.job.completed_work if stage == claim.job.stage else 0
        effective_total = claim.job.total_work if (stage == claim.job.stage and claim.job.total_work is not None) else total_work
        await run_index_sync(self.jobs.update_progress, claim, stage, completed_work=completed, total_work=effective_total)
        yield
        await run_index_sync(self.jobs.update_progress, claim, stage, completed_work=effective_total, total_work=effective_total)

    async def _run_stages(self, claim: IndexJobClaim, cancelled: Event) -> None:
        job = claim.job
        version_id = job.document_version_id
        local_only = job.job_type == LOCAL_PIPELINE_JOB_TYPE
        async with self._stage(claim, "parsing"):
            format = await run_index_sync(self._version_format, job.document_id, version_id)
            if job.config_key != self.config_key_for(format, job_type=job.job_type):
                raise IndexJobError("INDEX_PIPELINE_CONFIG_MISMATCH")
            parsed = await run_index_sync(self.parsing.ensure_parsed, version_id, cancelled=cancelled)
        async with self._stage(claim, "chunking"):
            chunks = await run_index_sync(
                self.chunking.ensure_chunked, parsed.parsed_document_id, self.chunk_config, cancelled=cancelled,
                claim=claim,
            )
        set_id = chunks.chunk_set_id
        async with self._stage(claim, "keyword_indexing"):
            readiness = await run_index_sync(get_index_readiness, self.store, version_id, chunk_set_id=set_id)
            searchable = readiness != "unsearchable"
            if searchable:
                await run_index_sync(self.keywords.ensure_indexed, set_id, claim=claim, cancelled=cancelled)
        if not local_only:
            emb_total = max(1, chunks.chunk_count) if searchable else 1
            async with self._stage(claim, "embedding", total_work=emb_total):
                if searchable:
                    await self.embeddings.ensure_embedded(set_id, self.profile, claim=claim, progress_total=emb_total)
        async with self._stage(claim, "publishing"):
            readiness = await run_index_sync(
                get_index_readiness, self.store, version_id, chunk_set_id=set_id,
                profile=None if local_only else self.profile,
            )
            if not searchable:
                expected_ok = readiness == "unsearchable"
            elif local_only:
                expected_ok = readiness in ("ready_keyword", "ready_hybrid")
            else:
                expected_ok = readiness == "ready_hybrid"
            if not expected_ok:
                raise KeywordIndexError("INDEX_PUBLICATION_INCOMPLETE")
