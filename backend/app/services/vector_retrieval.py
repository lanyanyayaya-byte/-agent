"""Exact local cosine retrieval over published embeddings in knowledge.db."""

import asyncio
import json
import sqlite3
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from heapq import nlargest

from app.models import ModelError
from app.models.embeddings import KEY_VERSION, VECTOR_FORMAT, EmbeddingBatch, EmbeddingError, EmbeddingProfile, encode_vector
from app.models.siliconflow import SiliconFlowEmbedding, validate_input
from app.storage.chunks import CHUNK_FIELDS, PUBLISHED_SETS
from app.storage.embeddings import SAVED_CHUNKS, SAVED_VECTORS, EmbeddingStore
from app.storage.library import KnowledgeStore
from app.storage.retrieval_scope import CURRENT_VERSION, LATEST_PER_DOCUMENT, scope_filters

MAX_TOP_K = 50
VECTOR_BATCH_SIZE = 256

PUBLISHED_VECTOR_SETS = f"""SELECT v.document_id, v.document_version_id, s.chunk_set_id,
    s.parsed_document_id, v.content_sha256, v.rowid AS version_order, s.chunk_count, e.profile_key
    {PUBLISHED_SETS}
    JOIN document_versions AS v ON v.document_version_id = s.document_version_id
    JOIN documents AS d ON d.document_id = v.document_id
    JOIN embedding_sets AS e ON e.chunk_set_id = s.chunk_set_id
    JOIN embedding_profiles AS ep ON ep.profile_key = e.profile_key
    WHERE s.status = 'ready' AND e.published_at IS NOT NULL
        AND ep.profile_key = ? AND ep.dimensions = ? AND ep.key_version = ? AND ep.vector_format = ?
        AND ep.invalidated_at IS NULL"""

COMPLETE_VECTOR_SETS = f"""SELECT * FROM published AS s
    WHERE s.chunk_count = (SELECT count(*) FROM chunks AS c WHERE c.chunk_set_id = s.chunk_set_id)
        AND s.chunk_count = (SELECT count(*) {SAVED_VECTORS}
            AND a.chunk_set_id = s.chunk_set_id AND a.profile_key = s.profile_key)"""

class VectorRetrievalError(ModelError):
    pass


@dataclass(frozen=True, slots=True)
class VectorHit:
    chunk_id: str
    document_id: str
    document_version_id: str
    chunk_set_id: int
    parsed_document_id: int
    chunk_index: int
    content_sha256: str
    similarity: float
    body_text: str
    section_heading_block_id: str | None
    section_path: tuple[str, ...]
    title_context: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VectorScope:
    document_id: str
    document_version_id: str
    chunk_set_id: int
    parsed_document_id: int
    content_sha256: str
    chunk_count: int


VECTOR_SCOPE_FIELDS = """document_id, document_version_id, chunk_set_id, parsed_document_id,
    content_sha256, chunk_count"""


def complete_vector_scope(profile: EmbeddingProfile, filter_sql: str, values: tuple) -> tuple[str, tuple]:
    sql = f"""WITH published AS MATERIALIZED ({PUBLISHED_VECTOR_SETS}{filter_sql}),
        eligible AS ({COMPLETE_VECTOR_SETS})"""
    return sql, (profile.key, profile.dimensions, KEY_VERSION, VECTOR_FORMAT, *values)


def _select_scope(profile: EmbeddingProfile, top_k: int, document_ids: Sequence[str] | None,
                  chunk_set_ids: Sequence[int] | None) -> tuple[str, tuple]:
    if not isinstance(profile, EmbeddingProfile):
        raise ValueError("An explicit embedding profile is required.")
    if type(top_k) is not int or not 1 <= top_k <= MAX_TOP_K:
        raise ValueError("Invalid vector result limit.")
    filter_sql, values = scope_filters(document_ids, chunk_set_ids, label="vector")
    if chunk_set_ids is None:
        filter_sql += CURRENT_VERSION
    scope = LATEST_PER_DOCUMENT if chunk_set_ids is None else "SELECT * FROM eligible"
    sql, params = complete_vector_scope(profile, filter_sql, values)
    return f"{sql}, scope AS ({scope}) SELECT {VECTOR_SCOPE_FIELDS} FROM scope", params


def check_vector_profile_active(db: sqlite3.Connection, profile_key: str) -> None:
    if db.execute("""SELECT 1 FROM embedding_profiles
        WHERE profile_key = ? AND invalidated_at IS NOT NULL""", (profile_key,)).fetchone():
        raise EmbeddingError("EMBEDDING_PROFILE_INVALIDATED", "该向量配置已停用，请切换兼容代次后重建。")


def _unit_query(np, query, dimensions: int):
    if isinstance(query, np.ndarray):
        if query.shape != (dimensions,) or query.dtype.kind not in "iuf":
            raise EmbeddingError("EMBEDDING_INVALID_VECTOR", "查询向量维度或数值类型无效。")
        query = query.tolist()
    if not isinstance(query, Sequence) or isinstance(query, (str, bytes, bytearray)) or len(query) != dimensions:
        raise EmbeddingError("EMBEDDING_INVALID_VECTOR", "查询向量维度或数值类型无效。")
    values = np.frombuffer(encode_vector(list(query), dimensions), dtype="<f4").astype(np.float64)
    return values / np.linalg.norm(values)


def _score_batches(np, rows: sqlite3.Cursor, unit_query) -> Iterator[tuple[int, float]]:
    dimensions = unit_query.size
    while batch := rows.fetchmany(VECTOR_BATCH_SIZE):
        blobs = []
        for _, blob in batch:
            if not isinstance(blob, bytes) or len(blob) != dimensions * 4:
                raise EmbeddingError("EMBEDDING_INVALID_VECTOR", "已存向量字节长度或编码无效。")
            blobs.append(blob)
        matrix = np.frombuffer(b"".join(blobs), dtype="<f4").reshape(-1, dimensions).astype(np.float64)
        squared_norms = np.einsum("ij,ij->i", matrix, matrix)
        if not (np.isfinite(squared_norms).all() and (squared_norms > 0).all()):
            raise EmbeddingError("EMBEDDING_INVALID_VECTOR", "已存向量包含非有限数值或为零向量。")
        similarities = np.einsum("ij,j->i", matrix, unit_query) / np.sqrt(squared_norms)
        similarities = np.clip(similarities, -1.0, 1.0)
        for (index, _), similarity in zip(batch, similarities, strict=True):
            yield index, float(similarity)


class VectorRetrievalService:
    def __init__(self, store: KnowledgeStore, provider: SiliconFlowEmbedding):
        self.store = store
        self.provider = provider
        self.embeddings = EmbeddingStore(store)

    async def search(self, query, top_k: int = 10, *, profile: EmbeddingProfile,
                     document_ids: Sequence[str] | None = None,
                     chunk_set_ids: Sequence[int] | None = None) -> tuple[VectorHit, ...]:
        """Embed text or accept a vector. Search at most one published set per textbook."""
        scope = _select_scope(profile, top_k, document_ids, chunk_set_ids)
        if isinstance(query, str):
            query = (await self.embed_query(query, profile=profile)).vectors[0]
        return await asyncio.to_thread(self._search, query, top_k, profile, scope)

    def search_vector(self, query_vector, top_k: int = 10, *, profile: EmbeddingProfile,
                      document_ids: Sequence[str] | None = None,
                      chunk_set_ids: Sequence[int] | None = None) -> tuple[VectorHit, ...]:
        """Search offline. The caller must supply the profile that produced the query vector."""
        scope = _select_scope(profile, top_k, document_ids, chunk_set_ids)
        return self._search(query_vector, top_k, profile, scope)

    async def embed_query(self, query: str, *, profile: EmbeddingProfile) -> EmbeddingBatch:
        """Use the same upload, validation, and invalidation path for every text query."""
        validate_input(query)
        self.provider.check_profile(profile)
        await asyncio.to_thread(self._check_active_profile, profile.key)
        try:
            async with self.provider.open_client() as client:
                return await self.provider.embed(client, [query], profile=profile)
        except EmbeddingError as error:
            if error.code == "EMBEDDING_REVISION_CHANGED":
                await asyncio.to_thread(self.embeddings.invalidate_profile, profile)
            raise

    def _check_active_profile(self, profile_key: str) -> None:
        try:
            with self.store.connect() as db:
                check_vector_profile_active(db, profile_key)
        except sqlite3.DatabaseError:
            raise VectorRetrievalError("VECTOR_QUERY_FAILED", "无法读取本地向量配置。") from None

    def _search(self, vector, top_k: int, profile: EmbeddingProfile, scope: tuple[str, tuple]) -> tuple[VectorHit, ...]:
        np, unit_query = self._prepare_vector(vector, profile)
        sql, params = scope
        try:
            with self.store.connect() as db, db:
                db.execute("BEGIN")
                check_vector_profile_active(db, profile.key)
                selected = tuple(VectorScope(*row) for row in db.execute(sql, params))
                if len({item.document_id for item in selected}) != len(selected):
                    raise ValueError("Vector search requires one published Chunk Set per textbook.")
                return self._rank(db, np, unit_query, selected, profile.key, top_k, -1.0)
        except sqlite3.DatabaseError:
            raise VectorRetrievalError("VECTOR_QUERY_FAILED", "无法查询本地向量数据库。") from None

    def search_published(self, db: sqlite3.Connection, vector, profile: EmbeddingProfile,
                         groups: Sequence[Sequence[VectorScope]], limit: int, *,
                         min_similarity: float) -> tuple[tuple[VectorHit, ...], ...]:
        """Score exact published groups already selected in the caller's read snapshot."""
        np, unit_query = self._prepare_vector(vector, profile)
        profile_key = profile.key
        try:
            return tuple(self._rank(db, np, unit_query, group, profile_key, limit, min_similarity)
                         for group in groups)
        except EmbeddingError as error:
            # Query validation happens above. Failures here concern the saved index.
            raise VectorRetrievalError(error.code, str(error)) from None
        except sqlite3.DatabaseError:
            raise VectorRetrievalError("VECTOR_QUERY_FAILED", "无法查询本地向量数据库。") from None

    @staticmethod
    def _prepare_vector(vector, profile: EmbeddingProfile):
        try:
            # The Sidecar and its other services can start even if this native module cannot load.
            import numpy as np
        except (ImportError, OSError, RuntimeError, ValueError):
            raise VectorRetrievalError("VECTOR_ENGINE_UNAVAILABLE", "本地向量计算组件不可用。") from None
        return np, _unit_query(np, vector, profile.dimensions)

    @classmethod
    def _rank(cls, db, np, unit_query, selected, profile_key, limit, min_similarity):
        def candidates():
            for scope in selected:
                rows = db.execute(f"{SAVED_CHUNKS} ORDER BY a.chunk_index", (scope.chunk_set_id, profile_key))
                for index, similarity in _score_batches(np, rows, unit_query):
                    if similarity >= min_similarity:
                        yield similarity, scope, index

        ranked = nlargest(limit, candidates(), key=lambda item: (item[0], -item[1].chunk_set_id, -item[2]))
        return tuple(cls._read_hit(db, *item) for item in ranked)

    @staticmethod
    def _read_hit(db: sqlite3.Connection, similarity: float, scope: VectorScope, index: int) -> VectorHit:
        row = db.execute(f"""SELECT {CHUNK_FIELDS} FROM chunks AS c
            WHERE c.chunk_set_id = ? AND c.chunk_index = ?""", (scope.chunk_set_id, index)).fetchone()
        chunk_index, chunk_id, body, heading, section, title = row
        return VectorHit(chunk_id, scope.document_id, scope.document_version_id, scope.chunk_set_id,
                         scope.parsed_document_id, chunk_index, scope.content_sha256, similarity, body,
                         heading, tuple(json.loads(section)), tuple(json.loads(title)))
