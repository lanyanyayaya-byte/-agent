import hashlib
import sqlite3
from contextlib import AsyncExitStack, suppress

from app.models.embeddings import EmbeddingError, EmbeddingProfile
from app.models.siliconflow import MAX_BATCH_SIZE, SiliconFlowEmbedding, validate_input
from app.services.index_threads import run_index_sync
from app.storage.embeddings import ChunkVector, EmbeddingInput, EmbeddingSet, EmbeddingStore
from app.storage.index_jobs import IndexJobClaim
from app.storage.library import KnowledgeStore


class DocumentEmbeddingService:
    """Build and resume document embeddings for an explicit profile and published chunk set."""

    def __init__(self, store: KnowledgeStore, provider: SiliconFlowEmbedding, *, batch_size: int = MAX_BATCH_SIZE):
        if type(batch_size) is not int or not 1 <= batch_size <= MAX_BATCH_SIZE:
            raise ValueError("Invalid embedding batch size.")
        self.provider = provider
        self.results = EmbeddingStore(store)
        self.batch_size = batch_size

    def get_embedding_set(self, set_id: int, profile: EmbeddingProfile) -> EmbeddingSet | None:
        return self.results.get_ready(set_id, profile)

    def read_vectors(self, set_id: int, profile: EmbeddingProfile) -> tuple[ChunkVector, ...] | None:
        return self.results.read_vectors(set_id, profile)

    async def ensure_embedded(self, set_id: int, profile: EmbeddingProfile, *,
                              claim: IndexJobClaim | None = None,
                              progress_total: int | None = None) -> EmbeddingSet:
        self.provider.check_profile(profile)
        ready = await run_index_sync(self.results.get_ready, set_id, profile)
        if ready is not None:
            return ready
        await run_index_sync(self.results.stage, set_id, profile, claim=claim)
        cursor = -1
        jobs = None
        if claim is not None and progress_total is not None and progress_total > 0:
            from app.storage.index_jobs import IndexJobStore
            jobs = IndexJobStore(self.results.store)
        async with AsyncExitStack() as stack:
            client = None
            while rows := await run_index_sync(self.results.missing_chunks, set_id, profile, cursor, self.batch_size):
                entries = await run_index_sync(self._build_inputs, rows, profile)
                lease = self.results.batch_lease_token(claim)
                try:
                    missing = await run_index_sync(self.results.claim, set_id, profile, entries, lease, job_claim=claim)
                    if missing:
                        if client is None:
                            client = await stack.enter_async_context(self.provider.open_client())
                        batch = await self.provider.embed(client, [entry.text for entry in missing], profile=profile)
                        await run_index_sync(
                            self.results.commit, set_id, batch.profile, lease, missing, batch.vectors, claim=claim,
                        )
                    if jobs is not None and claim is not None and progress_total is not None:
                        done = await run_index_sync(self.results.count_embedded, set_id, profile)
                        await run_index_sync(
                            jobs.update_progress, claim, "embedding",
                            completed_work=min(done, progress_total), total_work=progress_total,
                        )
                except BaseException as error:
                    try:
                        if isinstance(error, EmbeddingError) and error.code == "EMBEDDING_REVISION_CHANGED":
                            await run_index_sync(self.results.invalidate_profile, profile, claim=claim)
                    finally:
                        retry_after = error.retry_after if isinstance(error, EmbeddingError) else 0
                        with suppress(sqlite3.Error):
                            await run_index_sync(self.results.release, lease, retry_after)
                    raise
                cursor = entries[-1].chunk_index
        return await run_index_sync(self.results.publish, set_id, profile, claim=claim)

    @staticmethod
    def _build_inputs(rows: list[tuple[int, str, tuple[str, ...]]], profile: EmbeddingProfile) -> list[EmbeddingInput]:
        entries = []
        for index, body, titles in rows:
            text = profile.build_input(body, titles)
            try:
                validate_input(text)
            except EmbeddingError as error:
                raise EmbeddingError(error.code, f"Chunk {index}：{error}") from None
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            entries.append(EmbeddingInput(index, text, digest, profile.cache_key(digest)))
        return entries
