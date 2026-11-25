"""Read the P7.5 index status for an explicitly selected version and artifact set."""

from typing import Literal

from app.models.embeddings import EmbeddingProfile
from app.services.keyword_search import SELECT_SETS
from app.services.vector_retrieval import complete_vector_scope
from app.storage.chunks import PUBLISHED_SETS
from app.storage.keyword_terms import keyword_terms, keyword_title_terms
from app.storage.library import KnowledgeStore

PublishedIndexReadiness = Literal["ready_hybrid", "ready_keyword", "unsearchable"]
IndexReadiness = PublishedIndexReadiness | Literal["failed"]
EXACT_SET = " AND s.document_version_id = ? AND s.chunk_set_id = ?"


def get_index_readiness(
    store: KnowledgeStore, document_version_id: str, *, chunk_set_id: int | None = None,
    profile: EmbeddingProfile | None = None, job_id: str | None = None,
) -> IndexReadiness | None:
    """Inspect published metadata in one snapshot. The caller selects compatible artifacts."""
    with store.connect() as db, db:
        db.execute("BEGIN")
        if chunk_set_id is not None:
            scope = (document_version_id, chunk_set_id)
            chunk_set = db.execute(f"""SELECT s.chunk_count {PUBLISHED_SETS}
                WHERE s.status = 'ready' {EXACT_SET}
                AND s.chunk_count = (SELECT count(*) FROM chunks WHERE chunk_set_id = s.chunk_set_id)""",
                scope,
            ).fetchone()
            if chunk_set is not None and db.execute(f"{SELECT_SETS} {EXACT_SET}", scope).fetchone():
                # Open the actual FTS postings as well as checking the relational coverage.
                db.execute("SELECT rowid FROM keyword_fts WHERE keyword_fts MATCH ? LIMIT 1",
                           ('"knowledgeagentreadinessprobe"',)).fetchone()
                if profile is not None:
                    sql, params = complete_vector_scope(profile, EXACT_SET, scope)
                    if db.execute(f"{sql} SELECT 1 FROM eligible", params).fetchone():
                        return "ready_hybrid"
                return "ready_keyword"
            if chunk_set is not None and not any(
                any(char.isalnum() for char in keyword_terms(body) + keyword_title_terms(titles))
                for body, titles in db.execute("SELECT text, title_context FROM chunks WHERE chunk_set_id = ?",
                                               (chunk_set_id,))
            ):
                return "unsearchable"
        if job_id is not None and db.execute("""SELECT 1 FROM index_jobs
            WHERE job_id = ? AND document_version_id = ? AND status = 'failed'""",
            (job_id, document_version_id),
        ).fetchone():
            return "failed"
        return None
