import json
import sqlite3
from dataclasses import dataclass
from math import ceil
from uuid import uuid4

from app.models.embeddings import KEY_VERSION, VECTOR_FORMAT, EmbeddingError, EmbeddingProfile, decode_vector, encode_vector
from app.storage.business import utc_now
from app.storage.index_jobs import IndexJobClaim, IndexJobStore
from app.storage.keywords import COMPLETE_KEYWORD_ROWS
from app.storage.library import KnowledgeStore

# Longer than the provider's entire batch budget, including queued time and all retries.
LEASE_SECONDS = 120

SET_INFO = """SELECT s.chunk_count, e.published_at, p.invalidated_at
    FROM chunk_sets AS s
    LEFT JOIN embedding_sets AS e ON e.chunk_set_id = s.chunk_set_id AND e.profile_key = ?
    LEFT JOIN embedding_profiles AS p ON p.profile_key = ?
    WHERE s.chunk_set_id = ? AND s.status = 'ready'"""

SAVED_VECTORS = """FROM chunk_embeddings AS a JOIN embeddings AS v USING (profile_key, cache_key)
    WHERE v.vector IS NOT NULL"""
SAVED_CHUNKS = f"""SELECT a.chunk_index, v.vector {SAVED_VECTORS}
    AND a.chunk_set_id = ? AND a.profile_key = ?"""


@dataclass(frozen=True, slots=True)
class EmbeddingSet:
    chunk_set_id: int
    profile_key: str
    chunk_count: int
    published_at: str


@dataclass(frozen=True, slots=True)
class EmbeddingInput:
    chunk_index: int
    text: str
    input_sha256: str
    cache_key: str


@dataclass(frozen=True, slots=True)
class ChunkVector:
    chunk_index: int
    chunk_id: str
    vector: tuple[float, ...]


def _unavailable() -> EmbeddingError:
    return EmbeddingError("EMBEDDING_CHUNK_SET_UNAVAILABLE", "已发布的 Chunk Set 不可用。")


def _interrupted() -> EmbeddingError:
    return EmbeddingError("EMBEDDING_WRITE_INTERRUPTED", "向量批次已失去写入租约，请重新运行。")


def _set_info(db: sqlite3.Connection, set_id: int, profile_key: str) -> tuple[int, str | None]:
    row = db.execute(SET_INFO, (profile_key, profile_key, set_id)).fetchone()
    if row is None:
        raise _unavailable()
    count, published_at, invalidated_at = row
    if invalidated_at is not None:
        raise EmbeddingError("EMBEDDING_PROFILE_INVALIDATED", "该向量配置已停用，请切换兼容代次后重建。")
    return count, published_at


class EmbeddingStore:
    def __init__(self, store: KnowledgeStore):
        self.store = store

    @staticmethod
    def batch_lease_token(claim: IndexJobClaim | None) -> str:
        token = uuid4().hex
        return f"index:{claim.lease_token}:{token}" if claim is not None else token

    def get_ready(self, set_id: int, profile: EmbeddingProfile) -> EmbeddingSet | None:
        key = profile.key
        with self.store.connect() as db, db:
            db.execute("BEGIN")
            count, published_at = _set_info(db, set_id, key)
            if published_at is None:
                return None
            self._check_count(db, set_id, key, count)
        return EmbeddingSet(set_id, key, count, published_at)

    def read_vectors(self, set_id: int, profile: EmbeddingProfile) -> tuple[ChunkVector, ...] | None:
        """Read only a published set, in chunk order, from one consistent SQLite snapshot."""
        key = profile.key
        with self.store.connect() as db, db:
            db.execute("BEGIN")
            count, published_at = _set_info(db, set_id, key)
            if published_at is None:
                return None
            rows = db.execute(f"""SELECT c.chunk_index, c.chunk_id, ready.vector FROM ({SAVED_CHUNKS}) AS ready
                              JOIN chunks AS c ON c.chunk_set_id = ? AND c.chunk_index = ready.chunk_index
                              ORDER BY c.chunk_index""", (set_id, key, set_id)).fetchall()
        if len(rows) != count:
            raise _interrupted()
        return tuple(ChunkVector(index, chunk_id, decode_vector(blob, profile.dimensions))
                     for index, chunk_id, blob in rows)

    @staticmethod
    def _register_profile(db: sqlite3.Connection, profile: EmbeddingProfile) -> None:
        db.execute("""INSERT INTO embedding_profiles (profile_key, key_version, provider, service_id, model_id,
                          service_revision, compatibility_epoch, dimensions, input_strategy_version, vector_format)
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (profile_key) DO NOTHING""",
                   (profile.key, KEY_VERSION, profile.provider, profile.service_id, profile.model_id,
                    profile.service_revision, profile.compatibility_epoch, profile.dimensions,
                    profile.input_strategy_version, VECTOR_FORMAT))

    def invalidate_profile(self, profile: EmbeddingProfile, *, claim: IndexJobClaim | None = None) -> None:
        """Fence reads and writes for this profile, including any concurrent in-flight batch."""
        with self.store.write() as db:
            if claim is not None:
                IndexJobStore(self.store).require_active_claim(db, claim)
            self._register_profile(db, profile)
            db.execute("""UPDATE embedding_profiles SET invalidated_at = ?
                          WHERE profile_key = ? AND invalidated_at IS NULL""", (utc_now(), profile.key))

    def stage(self, set_id: int, profile: EmbeddingProfile, *, claim: IndexJobClaim | None = None) -> None:
        with self.store.write() as db:
            _set_info(db, set_id, profile.key)
            IndexJobStore(self.store).require_index_publication(db, set_id, claim)
            self._register_profile(db, profile)
            db.execute("""INSERT INTO embedding_sets (chunk_set_id, profile_key, created_at) VALUES (?, ?, ?)
                          ON CONFLICT (chunk_set_id, profile_key) DO NOTHING""", (set_id, profile.key, utc_now()))

    def missing_chunks(self, set_id: int, profile: EmbeddingProfile, after_index: int,
                       limit: int) -> list[tuple[int, str, tuple[str, ...]]]:
        key = profile.key
        with self.store.connect() as db, db:
            db.execute("BEGIN")
            _set_info(db, set_id, key)
            rows = db.execute("""SELECT c.chunk_index, c.text, c.title_context FROM chunks AS c
                WHERE c.chunk_set_id = ? AND c.chunk_index > ? AND NOT EXISTS (
                    SELECT 1 FROM chunk_embeddings AS a JOIN embeddings AS v USING (profile_key, cache_key)
                    WHERE a.chunk_set_id = c.chunk_set_id AND a.chunk_index = c.chunk_index
                        AND a.profile_key = ? AND v.vector IS NOT NULL
                ) ORDER BY c.chunk_index LIMIT ?""", (set_id, after_index, key, limit)).fetchall()
        return [(index, body, tuple(json.loads(titles))) for index, body, titles in rows]

    def claim(self, set_id: int, profile: EmbeddingProfile, entries: list[EmbeddingInput],
              lease: str, *, job_claim: IndexJobClaim | None = None) -> list[EmbeddingInput]:
        key = profile.key
        seen = set()
        missing = []
        with self.store.write() as db:
            _set_info(db, set_id, key)
            if job_claim is not None:
                IndexJobStore(self.store).require_index_publication(db, set_id, job_claim)
            for entry in entries:
                if entry.cache_key not in seen:
                    seen.add(entry.cache_key)
                    db.execute("""INSERT INTO embeddings (cache_key, profile_key, dimensions, input_sha256)
                                  VALUES (?, ?, ?, ?) ON CONFLICT (cache_key) DO NOTHING""",
                               (entry.cache_key, key, profile.dimensions, entry.input_sha256))
                    cached, retry_after = db.execute("""SELECT vector IS NOT NULL, retry_not_before - unixepoch()
                        FROM embeddings WHERE cache_key = ?""", (entry.cache_key,)).fetchone()
                    if not cached:
                        if retry_after > 0:
                            raise EmbeddingError("EMBEDDING_RATE_LIMITED", "SiliconFlow 正在冷却，请稍后继续处理。",
                                                 retry_after=retry_after)
                        # An expired, cancelled, or replaced job no longer owns its unfinished vector leases.
                        changed = db.execute("""UPDATE embeddings SET lease_token = ?, lease_expires_at = unixepoch() + ?
                            WHERE cache_key = ? AND (lease_token IS NULL OR lease_expires_at <= unixepoch()
                                OR (lease_token GLOB 'index:*' AND NOT EXISTS (
                                    SELECT 1 FROM index_jobs WHERE status = 'running' AND lease_expires_at > unixepoch()
                                        AND cancel_requested_at IS NULL
                                        AND lease_token = substr(embeddings.lease_token, 7, 32)
                                )))""",
                                             (lease, LEASE_SECONDS, entry.cache_key)).rowcount
                        if changed != 1:
                            raise EmbeddingError("EMBEDDING_BUILD_BUSY", "相同输入的向量正在计算，请稍后继续。")
                        missing.append(entry)
                db.execute("""INSERT INTO chunk_embeddings (chunk_set_id, chunk_index, profile_key, cache_key)
                              VALUES (?, ?, ?, ?) ON CONFLICT (chunk_set_id, profile_key, chunk_index) DO NOTHING""",
                           (set_id, entry.chunk_index, key, entry.cache_key))
        return missing

    def commit(self, set_id: int, profile: EmbeddingProfile, lease: str,
               entries: list[EmbeddingInput], vectors: list[list[float]], *,
               claim: IndexJobClaim | None = None) -> None:
        if len(entries) != len(vectors):
            raise EmbeddingError("EMBEDDING_INVALID_RESPONSE", "向量数量与批次输入数量不匹配。")
        now = utc_now()
        key = profile.key
        rows = [(encode_vector(vector, profile.dimensions), now, entry.cache_key, key, lease)
                for entry, vector in zip(entries, vectors, strict=True)]
        with self.store.write() as db:
            _set_info(db, set_id, key)
            if claim is not None:
                IndexJobStore(self.store).require_index_publication(db, set_id, claim)
            written = db.executemany("""UPDATE embeddings SET vector = ?, created_at = ?,
                    lease_token = NULL, lease_expires_at = NULL, retry_not_before = 0
                WHERE cache_key = ? AND profile_key = ? AND vector IS NULL
                    AND lease_token = ? AND lease_expires_at > unixepoch()""", rows).rowcount
            if written != len(rows):
                raise _interrupted()

    def release(self, lease: str, retry_after: float = 0) -> None:
        # A persisted deadline allows a fresh process to honor a rejected batch's Retry-After.
        seconds = ceil(min(retry_after, 2**62)) + 1 if retry_after > 0 else 0
        with self.store.write(wait=bool(seconds)) as db:
            db.execute("""UPDATE embeddings SET lease_token = NULL, lease_expires_at = NULL,
                              retry_not_before = max(retry_not_before, CASE WHEN ? > 0 THEN unixepoch() + ? ELSE 0 END)
                          WHERE lease_token = ?""", (seconds, seconds, lease))

    @staticmethod
    def _check_count(db: sqlite3.Connection, set_id: int, key: str, count: int) -> None:
        complete, = db.execute(f"SELECT count(*) FROM ({SAVED_CHUNKS})", (set_id, key)).fetchone()
        if count != complete:
            raise _interrupted()

    def publish(self, set_id: int, profile: EmbeddingProfile, *,
                claim: IndexJobClaim | None = None) -> EmbeddingSet:
        key = profile.key
        with self.store.write() as db:
            count, published_at = _set_info(db, set_id, key)
            self._check_count(db, set_id, key, count)
            if published_at is None:
                if claim is not None and db.execute(f"""SELECT 1 FROM chunk_sets AS s
                    JOIN keyword_index_sets AS i USING (chunk_set_id)
                    WHERE s.chunk_set_id = ? AND s.chunk_count > 0
                      AND i.indexed_at IS NOT NULL AND {COMPLETE_KEYWORD_ROWS}""", (set_id,)).fetchone() is None:
                    raise EmbeddingError("EMBEDDING_KEYWORD_INDEX_UNAVAILABLE", "关键词索引尚未就绪。")
                IndexJobStore(self.store).require_index_publication(db, set_id, claim)
                published_at = utc_now()
                changed = db.execute("""UPDATE embedding_sets SET published_at = ?
                                        WHERE chunk_set_id = ? AND profile_key = ?""",
                                     (published_at, set_id, key)).rowcount
                if changed != 1:
                    raise _interrupted()
        return EmbeddingSet(set_id, key, count, published_at)
