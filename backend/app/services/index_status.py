"""Real SQLite-aggregated textbook index status and P10.1 auto-index orchestration."""

import json
import sqlite3
from dataclasses import dataclass
from typing import Literal

from app.chunking.hybrid import CHUNKER_VERSION
from app.chunking.models import ChunkerConfig
from app.models.embeddings import EmbeddingProfile
from app.services.document_parsing import PARSER_SPECS
from app.services.index_readiness import PublishedIndexReadiness, get_index_readiness
from app.services.index_worker import INDEX_EMBEDDING_PROFILE, IndexWorker
from app.storage.index_jobs import (
    LOCAL_PIPELINE_JOB_TYPE,
    PIPELINE_JOB_TYPE,
    IndexJob,
    IndexJobError,
    IndexJobStage,
    IndexJobStatus,
)
from app.storage.knowledge_settings import KnowledgeSettings, KnowledgeSettingsStore
from app.storage.library import KnowledgeStore

ParseStatus = Literal["pending", "writing", "ready", "failed"]
KeywordStatus = Literal["pending", "ready", "unsearchable"]
EmbeddingStatus = Literal["pending", "running", "waiting_key", "disabled", "partial", "ready", "failed", "unsearchable"]

JOB_ERROR_MESSAGES: dict[str, str] = {
    "EMBEDDING_NOT_CONFIGURED": "SiliconFlow API Key 未配置，仅本地关键词可用。",
    "EMBEDDING_AUTH_FAILED": "SiliconFlow API Key 无效或已失效（401），请更新 Key。",
    "EMBEDDING_FORBIDDEN": "SiliconFlow 拒绝访问（403），请检查账户与模型权限。",
    "EMBEDDING_RATE_LIMITED": "SiliconFlow 请求限流（429），等待自动退避重试。",
    "EMBEDDING_TIMEOUT": "连接 SiliconFlow 超时，可稍后重试。",
    "EMBEDDING_NETWORK_ERROR": "无法连接 SiliconFlow，请检查网络。",
    "EMBEDDING_SERVICE_ERROR": "SiliconFlow 服务返回错误，可稍后重试。",
    "EMBEDDING_INPUT_TOO_LONG": "存在超出长度预算的切片，需重新分块。",
    "INDEX_LEASE_EXPIRED": "上次后台索引意外中断，已保存进度将在恢复时复用。",
    "INDEX_PIPELINE_STORAGE_BUSY": "数据库繁忙，等待重试。",
    "INDEX_PIPELINE_STORAGE_ERROR": "索引持久化读写失败。",
    "PDF_ENCRYPTED": "PDF 已加密，无法解析。",
    "PDF_INVALID": "PDF 文件损坏或结构无效。",
    "PDF_RESOURCE_LIMIT": "PDF 超出解析资源限制。",
    "PDF_PARSE_TIMEOUT": "PDF 解析或 OCR 超时。",
    "PDF_PARSE_FAILED": "PDF 本地解析失败。",
    "DOCX_PARSE_FAILED": "DOCX 本地解析失败。",
    "MARKDOWN_PARSE_FAILED": "Markdown 解析失败。",
}


def human_job_error(code: str | None) -> str | None:
    if not code:
        return None
    if code in JOB_ERROR_MESSAGES:
        return JOB_ERROR_MESSAGES[code]
    if code.startswith(("PDF_", "DOCX_", "MARKDOWN_", "DOCUMENT_PARSE_")):
        return f"文档解析失败（{code}）。"
    return f"索引任务出错（{code}）。"


@dataclass(frozen=True, slots=True)
class IndexJobBrief:
    job_id: str
    job_type: str
    status: IndexJobStatus
    stage: IndexJobStage
    completed_work: int
    total_work: int | None
    error_code: str | None
    error_message: str | None
    retry_count: int
    next_retry_at: int | None
    cancel_requested_at: int | None


@dataclass(frozen=True, slots=True)
class DocumentIndexStatus:
    document_id: str
    name: str
    format: Literal["pdf", "docx", "md"]
    version_id: str
    version_count: int
    latest_size_bytes: int | None
    created_at: str
    updated_at: str
    parse_status: ParseStatus
    page_count: int | None
    parsed_pages: int | None
    chunk_count: int | None
    keyword_status: KeywordStatus
    embedding_status: EmbeddingStatus
    embedded_count: int
    vector_coverage_pct: float | None
    vector_searchable: bool
    embedding_model: str | None
    embedding_dimension: int | None
    published_readiness: PublishedIndexReadiness | None
    latest_job: IndexJobBrief | None
    last_indexed_at: str | None
    warnings_count: int
    parse_spec_current: bool = True
    chunker_current: bool = True
    needs_rebuild: bool = False
    stale_reason: str | None = None


@dataclass(frozen=True, slots=True)
class KnowledgeIndexOverview:
    total_documents: int
    ready_hybrid: int
    ready_keyword: int
    in_progress: int
    pending: int
    failed: int
    unsearchable: int
    missing_for_completion: int
    active_job_id: str | None
    settings: KnowledgeSettings
    siliconflow_configured: bool
    documents: list[DocumentIndexStatus]
    stale_count: int = 0


@dataclass(frozen=True, slots=True)
class BatchIndexResult:
    queued_count: int
    skipped_complete: int
    skipped_active: int
    mode: Literal["hybrid", "local_only"]
    queued_document_ids: list[str]


def _job_brief(job: IndexJob | None) -> IndexJobBrief | None:
    if job is None:
        return None
    return IndexJobBrief(
        job_id=job.job_id,
        job_type=job.job_type,
        status=job.status,
        stage=job.stage,
        completed_work=job.completed_work,
        total_work=job.total_work,
        error_code=job.error_code,
        error_message=human_job_error(job.error_code),
        retry_count=job.retry_count,
        next_retry_at=job.next_retry_at,
        cancel_requested_at=job.cancel_requested_at,
    )


def _is_active_job(job: IndexJob | None) -> bool:
    return job is not None and job.status in ("queued", "running", "retry_wait")


class IndexStatusService:
    """Aggregate real per-document index state directly from SQLite tables."""

    def __init__(
        self,
        store: KnowledgeStore,
        settings_store: KnowledgeSettingsStore,
        worker: IndexWorker,
        *,
        profile: EmbeddingProfile = INDEX_EMBEDDING_PROFILE,
        chunk_config: ChunkerConfig = ChunkerConfig(),
    ):
        self.store = store
        self.settings_store = settings_store
        self.worker = worker
        self.profile = profile
        self.chunk_config = chunk_config

    def get_overview(self) -> KnowledgeIndexOverview:
        settings = self.settings_store.get()
        sf_configured = self.worker.pipeline.embeddings.provider.configured
        profile_key = self.profile.key
        config_key = self.chunk_config.key

        docs_out: list[DocumentIndexStatus] = []
        ready_hybrid = 0
        ready_keyword = 0
        in_progress = 0
        pending = 0
        failed = 0
        unsearchable = 0
        missing_for_completion = 0
        want_cloud = settings.cloud_embedding_enabled and sf_configured

        with self.store.connect() as db:
            doc_rows = db.execute(
                """SELECT d.document_id, d.name, d.format, d.created_at, d.updated_at,
                          (SELECT count(*) FROM document_versions v WHERE v.document_id = d.document_id) AS vcnt,
                          v.document_version_id, v.size_bytes
                   FROM documents d
                   JOIN document_versions v ON v.rowid = (
                       SELECT max(rowid) FROM document_versions WHERE document_id = d.document_id
                   )
                   ORDER BY d.updated_at DESC, d.document_id"""
            ).fetchall()

            # Preload latest index jobs per version_id
            jobs_by_version: dict[str, IndexJob] = {}
            for row in db.execute(
                """SELECT job_id, document_id, document_version_id, job_type, config_key, previous_job_id,
                          stage, status, completed_work, total_work, created_at, started_at, updated_at, finished_at,
                          error_code, error_http_status, retry_count, max_retries, next_retry_at,
                          instance_id, heartbeat_at, lease_expires_at, cancel_requested_at
                   FROM index_jobs
                   WHERE job_type IN (?, ?)
                   ORDER BY rowid DESC""",
                (PIPELINE_JOB_TYPE, LOCAL_PIPELINE_JOB_TYPE),
            ):
                job = IndexJob(*row)
                if job.document_version_id not in jobs_by_version:
                    jobs_by_version[job.document_version_id] = job

            for doc_id, name, fmt, created_at, updated_at, vcnt, ver_id, size_bytes in doc_rows:
                spec = PARSER_SPECS[fmt]
                latest_job = jobs_by_version.get(ver_id)
                active = _is_active_job(latest_job)

                # Parsed document for current version & spec (fallback to any ready parse for version)
                p_row = db.execute(
                    """SELECT parsed_document_id, status, warnings, published_at,
                              parser_version, structure_version, config_key
                       FROM parsed_documents
                       WHERE document_version_id = ?
                       ORDER BY (parser_version = ? AND structure_version = ? AND config_key = ?) DESC,
                                (status = 'ready') DESC, parsed_document_id DESC
                       LIMIT 1""",
                    (ver_id, spec.parser_version, spec.structure_version, spec.config_key),
                ).fetchone()

                parsed_id: int | None = None
                parse_status: ParseStatus = "pending"
                parse_spec_current = True
                warnings_count = 0
                page_count: int | None = None
                parsed_pages: int | None = None

                if p_row is not None:
                    parsed_id, raw_pstatus, raw_warns, _, p_ver, p_struct, p_cfg = p_row
                    parse_status = "ready" if raw_pstatus == "ready" else "writing"
                    parse_spec_current = (
                        p_ver == spec.parser_version
                        and p_struct == spec.structure_version
                        and p_cfg == spec.config_key
                    )
                    try:
                        warnings_count = len(json.loads(raw_warns or "[]"))
                    except ValueError:
                        warnings_count = 0
                    if fmt == "pdf" and parse_status == "ready":
                        pg = db.execute(
                            """SELECT max(CAST(json_extract(details, '$.physical_page') AS INTEGER)),
                                      count(DISTINCT json_extract(details, '$.physical_page'))
                               FROM source_locations
                               WHERE parsed_document_id = ? AND kind = 'pdf'""",
                            (parsed_id,),
                        ).fetchone()
                        if pg and pg[0] is not None:
                            page_count = int(pg[0])
                            parsed_pages = int(pg[1] or 0)
                elif latest_job is not None and latest_job.status == "failed" and latest_job.stage == "parsing":
                    parse_status = "failed"

                # Chunk set for current version (prefer active config, fallback to latest ready chunk_set)
                cs_row = db.execute(
                    """SELECT chunk_set_id, status, chunk_count, published_at, chunker_version, chunker_config_key
                       FROM chunk_sets
                       WHERE document_version_id = ? AND status = 'ready'
                       ORDER BY (chunker_version = ? AND chunker_config_key = ?) DESC, chunk_set_id DESC
                       LIMIT 1""",
                    (ver_id, CHUNKER_VERSION, config_key),
                ).fetchone()

                chunk_set_id: int | None = None
                chunk_count: int | None = None
                chunker_current = True
                keyword_status: KeywordStatus = "pending"
                embedding_status: EmbeddingStatus = "pending"
                embedded_count = 0
                vector_coverage_pct: float | None = None
                vector_searchable = False
                embedding_model: str | None = None
                embedding_dimension: int | None = None
                last_indexed_at: str | None = None
                readiness: PublishedIndexReadiness | None = None

                if cs_row is not None:
                    chunk_set_id, _, chunk_count, cs_pub, cs_ver, cs_cfg = cs_row
                    chunker_current = (cs_ver == CHUNKER_VERSION and cs_cfg == config_key)
                    last_indexed_at = cs_pub
                    # Keyword index check
                    ks_row = db.execute(
                        "SELECT indexed_at FROM keyword_index_sets WHERE chunk_set_id = ?",
                        (chunk_set_id,),
                    ).fetchone()
                    kw_rows = db.execute(
                        "SELECT count(*) FROM keyword_index_rows WHERE chunk_set_id = ?",
                        (chunk_set_id,),
                    ).fetchone()[0]
                    if ks_row and ks_row[0] and chunk_count > 0 and kw_rows == chunk_count:
                        keyword_status = "ready"
                        last_indexed_at = ks_row[0]
                    elif chunk_count == 0:
                        keyword_status = "unsearchable"
                        embedding_status = "unsearchable"
                        readiness = "unsearchable"

                    # Embedding check
                    if keyword_status != "unsearchable":
                        es_row = db.execute(
                            """SELECT e.published_at, p.model_id, p.dimensions, p.invalidated_at
                               FROM embedding_sets e
                               JOIN embedding_profiles p USING (profile_key)
                               WHERE e.chunk_set_id = ? AND e.profile_key = ?""",
                            (chunk_set_id, profile_key),
                        ).fetchone()
                        embedded_count = db.execute(
                            """SELECT count(*)
                               FROM chunk_embeddings a
                               JOIN embeddings v USING (profile_key, cache_key)
                               WHERE a.chunk_set_id = ? AND a.profile_key = ? AND v.vector IS NOT NULL""",
                            (chunk_set_id, profile_key),
                        ).fetchone()[0]
                        if chunk_count > 0:
                            vector_coverage_pct = round(min(100.0, (embedded_count / chunk_count) * 100.0), 1)
                        if es_row is not None:
                            es_pub, m_id, m_dim, invalidated = es_row
                            embedding_model = m_id
                            embedding_dimension = m_dim
                            if es_pub and invalidated is None and chunk_count > 0 and embedded_count == chunk_count:
                                vector_searchable = True
                                embedding_status = "ready"
                                last_indexed_at = es_pub
                                readiness = "ready_hybrid" if keyword_status == "ready" else None
                            elif active and latest_job and latest_job.stage == "embedding":
                                embedding_status = "running"
                            elif embedded_count > 0:
                                embedding_status = "partial"
                        if not vector_searchable:
                            if keyword_status == "ready":
                                readiness = "ready_keyword"
                            if active and latest_job and latest_job.stage == "embedding":
                                embedding_status = "running"
                            elif latest_job and latest_job.status == "failed" and latest_job.stage == "embedding":
                                embedding_status = "waiting_key" if latest_job.error_code == "EMBEDDING_NOT_CONFIGURED" else "failed"
                            elif not settings.cloud_embedding_enabled:
                                embedding_status = "disabled"
                            elif not sf_configured:
                                embedding_status = "waiting_key"
                            elif embedded_count > 0:
                                embedding_status = "partial"
                            else:
                                embedding_status = "pending"
                else:
                    # Not yet chunked
                    if active:
                        embedding_status = "running" if (latest_job and latest_job.stage == "embedding") else "pending"
                    elif latest_job and latest_job.status == "failed":
                        embedding_status = "failed"
                    elif not settings.cloud_embedding_enabled:
                        embedding_status = "disabled"
                    elif not sf_configured:
                        embedding_status = "waiting_key"

                # Re-check unsearchable when parsed produced 0 searchable tokens
                if readiness is None and chunk_set_id is not None and keyword_status == "pending":
                    r_check = get_index_readiness(self.store, ver_id, chunk_set_id=chunk_set_id)
                    if r_check == "unsearchable":
                        readiness = "unsearchable"
                        keyword_status = "unsearchable"
                        embedding_status = "unsearchable"

                # Classify summary counters
                if active:
                    in_progress += 1
                elif readiness == "ready_hybrid":
                    ready_hybrid += 1
                elif readiness == "unsearchable":
                    unsearchable += 1
                elif latest_job is not None and latest_job.status == "failed":
                    failed += 1
                    if readiness == "ready_keyword":
                        ready_keyword += 1
                elif readiness == "ready_keyword":
                    ready_keyword += 1
                else:
                    pending += 1

                # Count how many books still need work for "一键补全索引"
                if not active and readiness != "unsearchable":
                    if want_cloud:
                        if readiness != "ready_hybrid":
                            missing_for_completion += 1
                    else:
                        if readiness not in ("ready_hybrid", "ready_keyword"):
                            missing_for_completion += 1

                needs_rebuild = False
                stale_reason: str | None = None
                if parse_status == "ready" and (not parse_spec_current or (chunk_set_id is not None and not chunker_current)):
                    needs_rebuild = True
                    parts_stale = []
                    if not parse_spec_current:
                        parts_stale.append("PDF 解析器已升级")
                    if chunk_set_id is not None and not chunker_current:
                        parts_stale.append("Chunk 切片策略已升级")
                    stale_reason = "、".join(parts_stale) + "，当前旧索引仍可检索，可按需重建"

                docs_out.append(
                    DocumentIndexStatus(
                        document_id=doc_id,
                        name=name,
                        format=fmt,
                        version_id=ver_id,
                        version_count=vcnt,
                        latest_size_bytes=size_bytes,
                        created_at=created_at,
                        updated_at=updated_at,
                        parse_status=parse_status,
                        page_count=page_count,
                        parsed_pages=parsed_pages,
                        chunk_count=chunk_count,
                        keyword_status=keyword_status,
                        embedding_status=embedding_status,
                        embedded_count=embedded_count,
                        vector_coverage_pct=vector_coverage_pct,
                        vector_searchable=vector_searchable,
                        embedding_model=embedding_model,
                        embedding_dimension=embedding_dimension,
                        published_readiness=readiness,
                        latest_job=_job_brief(latest_job),
                        last_indexed_at=last_indexed_at,
                        warnings_count=warnings_count,
                        parse_spec_current=parse_spec_current,
                        chunker_current=chunker_current,
                        needs_rebuild=needs_rebuild,
                        stale_reason=stale_reason,
                    )
                )

        stale_count = sum(1 for d in docs_out if d.needs_rebuild)
        return KnowledgeIndexOverview(
            total_documents=len(docs_out),
            ready_hybrid=ready_hybrid,
            ready_keyword=ready_keyword,
            in_progress=in_progress,
            pending=pending,
            failed=failed,
            unsearchable=unsearchable,
            missing_for_completion=missing_for_completion,
            active_job_id=self.worker.active_job_id,
            settings=settings,
            siliconflow_configured=sf_configured,
            documents=docs_out,
            stale_count=stale_count,
        )

    def enqueue_for_document(
        self,
        document_id: str,
        version_id: str,
        *,
        force_rebuild: bool = False,
        require_cloud: bool | None = None,
    ) -> IndexJob | None:
        """Submit the appropriate pipeline job (hybrid or local-only) for the latest version."""
        settings = self.settings_store.get()
        sf_configured = self.worker.pipeline.embeddings.provider.configured
        use_cloud = (
            require_cloud
            if require_cloud is not None
            else (settings.cloud_embedding_enabled and sf_configured)
        )
        job_type = PIPELINE_JOB_TYPE if use_cloud else LOCAL_PIPELINE_JOB_TYPE

        current = self.store.latest_version_identity(document_id)
        if current is None or current[0] != version_id:
            return None
        _, fmt = current

        # Skip if already complete (unless force_rebuild)
        readiness = self.worker.pipeline.version_readiness(version_id, fmt)
        if not force_rebuild:
            if readiness == "unsearchable":
                return None
            if use_cloud and readiness == "ready_hybrid":
                return None
            if not use_cloud and readiness in ("ready_hybrid", "ready_keyword"):
                return None

        # Check existing latest job for this version & job_type
        config_key = self.worker.pipeline.config_key_for(fmt, job_type=job_type)
        existing = self.worker.jobs.latest_attempt(document_id, version_id, config_key)
        if existing is not None:
            if existing.status in ("queued", "running", "retry_wait"):
                return existing
            if existing.status == "succeeded" and not force_rebuild:
                # Re-check readiness in case embeddings were missing after a prior local job
                if (use_cloud and readiness == "ready_hybrid") or (
                    not use_cloud and readiness in ("ready_hybrid", "ready_keyword")
                ):
                    return existing
            # Restart successor after terminal job
            try:
                return self.worker.pipeline.submit(
                    document_id,
                    version_id,
                    restart_after=existing.job_id,
                    current_only=False,
                    job_type=job_type,
                )
            except IndexJobError:
                return existing

        return self.worker.pipeline.submit(
            document_id,
            version_id,
            current_only=False,
            job_type=job_type,
        )

    def index_all_missing(self) -> BatchIndexResult:
        """Enqueue all documents missing local or cloud index according to current settings."""
        settings = self.settings_store.get()
        sf_configured = self.worker.pipeline.embeddings.provider.configured
        use_cloud = settings.cloud_embedding_enabled and sf_configured
        mode: Literal["hybrid", "local_only"] = "hybrid" if use_cloud else "local_only"

        overview = self.get_overview()
        queued_ids: list[str] = []
        skipped_complete = 0
        skipped_active = 0

        for doc in overview.documents:
            if doc.latest_job is not None and doc.latest_job.status in ("queued", "running", "retry_wait"):
                skipped_active += 1
                continue
            if doc.published_readiness == "unsearchable":
                skipped_complete += 1
                continue
            if use_cloud and doc.published_readiness == "ready_hybrid":
                skipped_complete += 1
                continue
            if not use_cloud and doc.published_readiness in ("ready_hybrid", "ready_keyword"):
                skipped_complete += 1
                continue
            job = self.enqueue_for_document(doc.document_id, doc.version_id, force_rebuild=False, require_cloud=use_cloud)
            if job is not None and job.status in ("queued", "running", "retry_wait"):
                queued_ids.append(doc.document_id)

        return BatchIndexResult(
            queued_count=len(queued_ids),
            skipped_complete=skipped_complete,
            skipped_active=skipped_active,
            mode=mode,
            queued_document_ids=queued_ids,
        )

    def rebuild_stale(self) -> BatchIndexResult:
        """Batch-enqueue rebuild jobs only for documents whose parse or chunker version is stale."""
        settings = self.settings_store.get()
        sf_configured = self.worker.pipeline.embeddings.provider.configured
        use_cloud = settings.cloud_embedding_enabled and sf_configured
        mode: Literal["hybrid", "local_only"] = "hybrid" if use_cloud else "local_only"

        overview = self.get_overview()
        queued_ids: list[str] = []
        skipped_complete = 0
        skipped_active = 0
        for doc in overview.documents:
            if doc.latest_job is not None and doc.latest_job.status in ("queued", "running", "retry_wait"):
                skipped_active += 1
                continue
            if not doc.needs_rebuild:
                skipped_complete += 1
                continue
            job = self.enqueue_for_document(doc.document_id, doc.version_id, force_rebuild=True, require_cloud=use_cloud)
            if job is not None and job.status in ("queued", "running", "retry_wait"):
                queued_ids.append(doc.document_id)
        return BatchIndexResult(
            queued_count=len(queued_ids),
            skipped_complete=skipped_complete,
            skipped_active=skipped_active,
            mode=mode,
            queued_document_ids=queued_ids,
        )
