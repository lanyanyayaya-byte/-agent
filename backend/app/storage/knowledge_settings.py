"""Persistent RAG 2.0 indexing and retrieval settings stored in knowledge.db."""

import json
import sqlite3
from dataclasses import asdict, dataclass

from app.storage.business import utc_now
from app.storage.library import KnowledgeStore

SETTINGS_KEY = "rag_settings_v1"


@dataclass(frozen=True, slots=True)
class KnowledgeSettings:
    auto_index_on_import: bool = True
    cloud_embedding_enabled: bool = False
    cloud_rerank_enabled: bool = True
    enable_bilingual_rewrite: bool = True
    default_top_k: int = 8
    compare_top_k: int = 12
    rerank_candidates: int = 30
    min_vector_similarity: float = 0.45

    def __post_init__(self) -> None:
        if not 1 <= self.default_top_k <= 50:
            raise ValueError("default_top_k must be between 1 and 50.")
        if not 1 <= self.compare_top_k <= 50:
            raise ValueError("compare_top_k must be between 1 and 50.")
        if not 1 <= self.rerank_candidates <= 100:
            raise ValueError("rerank_candidates must be between 1 and 100.")
        if not 0.0 <= self.min_vector_similarity <= 1.0:
            raise ValueError("min_vector_similarity must be between 0.0 and 1.0.")


def _has_existing_cloud_consent(db: sqlite3.Connection) -> bool:
    """Inherit prior explicit cloud-embedding consent when upgrading an existing DB."""
    if db.execute("SELECT 1 FROM embedding_sets LIMIT 1").fetchone() is not None:
        return True
    if db.execute(
        "SELECT 1 FROM index_jobs WHERE job_type = 'index_pipeline' LIMIT 1"
    ).fetchone() is not None:
        return True
    return False


class KnowledgeSettingsStore:
    def __init__(self, store: KnowledgeStore):
        self.store = store

    def has_explicit_settings(self) -> bool:
        with self.store.connect() as db:
            return (
                db.execute("SELECT 1 FROM knowledge_settings WHERE key = ?", (SETTINGS_KEY,)).fetchone()
                is not None
            )

    def get(self) -> KnowledgeSettings:
        with self.store.connect() as db:
            row = db.execute(
                "SELECT value_json FROM knowledge_settings WHERE key = ?",
                (SETTINGS_KEY,),
            ).fetchone()
            if row is None:
                inherited_cloud = _has_existing_cloud_consent(db)
                return KnowledgeSettings(cloud_embedding_enabled=inherited_cloud)
            raw = json.loads(row[0])
            if not isinstance(raw, dict):
                return KnowledgeSettings()
            defaults = asdict(KnowledgeSettings())
            merged = {k: raw.get(k, defaults[k]) for k in defaults}
            try:
                return KnowledgeSettings(**merged)
            except (TypeError, ValueError):
                return KnowledgeSettings()

    def update(self, **changes) -> KnowledgeSettings:
        with self.store.write() as db:
            row = db.execute(
                "SELECT value_json FROM knowledge_settings WHERE key = ?",
                (SETTINGS_KEY,),
            ).fetchone()
            if row is None:
                current = asdict(KnowledgeSettings(cloud_embedding_enabled=_has_existing_cloud_consent(db)))
            else:
                raw = json.loads(row[0])
                defaults = asdict(KnowledgeSettings())
                current = {k: raw.get(k, defaults[k]) for k in defaults} if isinstance(raw, dict) else defaults
            for key, value in changes.items():
                if key not in current:
                    raise ValueError(f"Unknown knowledge setting: {key}")
                if value is not None:
                    current[key] = value
            updated = KnowledgeSettings(**current)
            db.execute(
                """INSERT INTO knowledge_settings (key, value_json, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT (key) DO UPDATE SET value_json = excluded.value_json, updated_at = excluded.updated_at""",
                (SETTINGS_KEY, json.dumps(asdict(updated), separators=(",", ":")), utc_now()),
            )
            return updated
