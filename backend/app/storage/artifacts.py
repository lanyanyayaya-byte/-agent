"""SQLite persistence layer for workspace attachments, run artifacts, and skill activations (Migration V7).

Reuses `BusinessStore`'s SQLite connection and `BEGIN IMMEDIATE` transaction
semantics inside `knowledgeagent.sqlite3`.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from uuid import uuid4

from app.storage.business import (
    ArtifactStatus,
    BusinessStore,
    SessionNotFound,
    StoredArtifact,
    StoredSkillActivation,
    utc_now,
)


@dataclass(frozen=True, slots=True)
class StoredWorkspaceFile:
    file_id: str
    session_id: str
    original_name: str
    relative_path: str
    media_type: str
    size_bytes: int
    created_at: str


class WorkspaceArtifactStore:
    """Persistence facade for workspace_files, artifacts, and run_skill_activations."""

    def __init__(self, business: BusinessStore) -> None:
        self._business = business

    # ------------------------------------------------------------------
    # Workspace files (user-imported input copies)
    # ------------------------------------------------------------------

    def add_workspace_file(
        self,
        *,
        file_id: str,
        session_id: str,
        original_name: str,
        relative_path: str,
        media_type: str,
        size_bytes: int,
    ) -> StoredWorkspaceFile:
        now = utc_now()
        with closing(self._business._connect()) as conn, self._business._write(conn):
            self._business._require_session(conn, session_id)
            conn.execute(
                """INSERT INTO workspace_files (
                       file_id, session_id, original_name, relative_path,
                       media_type, size_bytes, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (file_id, session_id, original_name, relative_path, media_type, size_bytes, now),
            )
        return StoredWorkspaceFile(
            file_id=file_id,
            session_id=session_id,
            original_name=original_name,
            relative_path=relative_path,
            media_type=media_type,
            size_bytes=size_bytes,
            created_at=now,
        )

    def list_workspace_files(self, session_id: str) -> list[StoredWorkspaceFile]:
        with closing(self._business._connect()) as conn:
            self._business._require_session(conn, session_id)
            rows = conn.execute(
                """SELECT file_id, session_id, original_name, relative_path,
                          media_type, size_bytes, created_at
                   FROM workspace_files
                   WHERE session_id = ?
                   ORDER BY created_at DESC, rowid DESC""",
                (session_id,),
            ).fetchall()
            return [StoredWorkspaceFile(*row) for row in rows]

    def get_workspace_file(self, session_id: str, file_id: str) -> StoredWorkspaceFile | None:
        with closing(self._business._connect()) as conn:
            self._business._require_session(conn, session_id)
            row = conn.execute(
                """SELECT file_id, session_id, original_name, relative_path,
                          media_type, size_bytes, created_at
                   FROM workspace_files
                   WHERE session_id = ? AND file_id = ?""",
                (session_id, file_id),
            ).fetchone()
            return StoredWorkspaceFile(*row) if row else None

    def delete_workspace_file(self, session_id: str, file_id: str) -> StoredWorkspaceFile | None:
        with closing(self._business._connect()) as conn, self._business._write(conn):
            self._business._require_session(conn, session_id)
            row = conn.execute(
                """SELECT file_id, session_id, original_name, relative_path,
                          media_type, size_bytes, created_at
                   FROM workspace_files
                   WHERE session_id = ? AND file_id = ?""",
                (session_id, file_id),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "DELETE FROM workspace_files WHERE session_id = ? AND file_id = ?",
                (session_id, file_id),
            )
            return StoredWorkspaceFile(*row)

    # ------------------------------------------------------------------
    # Artifacts (staged -> published / discarded)
    # ------------------------------------------------------------------

    def create_staged_artifact(
        self,
        *,
        artifact_id: str,
        session_id: str,
        run_id: str,
        filename: str,
        title: str | None,
        relative_path: str,
        media_type: str,
        size_bytes: int,
    ) -> StoredArtifact:
        now = utc_now()
        with closing(self._business._connect()) as conn, self._business._write(conn):
            self._business._require_session(conn, session_id)
            run_row = conn.execute(
                "SELECT status FROM runs WHERE session_id = ? AND run_id = ?",
                (session_id, run_id),
            ).fetchone()
            if run_row is None:
                raise ValueError("Artifact requires an existing run")
            if run_row[0] not in ("running", "awaiting_approval", "cancelling"):
                raise ValueError("Artifact staging requires an active run")
            conn.execute(
                """INSERT INTO artifacts (
                       artifact_id, session_id, run_id, filename, title,
                       relative_path, media_type, size_bytes, status, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'staged', ?)""",
                (
                    artifact_id, session_id, run_id, filename, title,
                    relative_path, media_type, size_bytes, now,
                ),
            )
        return StoredArtifact(
            artifact_id=artifact_id,
            session_id=session_id,
            run_id=run_id,
            filename=filename,
            title=title,
            relative_path=relative_path,
            media_type=media_type,
            size_bytes=size_bytes,
            status="staged",
            created_at=now,
        )

    def publish_run_artifacts(self, session_id: str, run_id: str) -> list[StoredArtifact]:
        with closing(self._business._connect()) as conn, self._business._write(conn):
            self._business._require_session(conn, session_id)
            conn.execute(
                """UPDATE artifacts SET status = 'published'
                   WHERE session_id = ? AND run_id = ? AND status = 'staged'""",
                (session_id, run_id),
            )
            rows = conn.execute(
                """SELECT artifact_id, session_id, run_id, filename, title,
                          relative_path, media_type, size_bytes, status, created_at
                   FROM artifacts
                   WHERE session_id = ? AND run_id = ? AND status = 'published'
                   ORDER BY created_at, rowid""",
                (session_id, run_id),
            ).fetchall()
            return [StoredArtifact(*row) for row in rows]

    def discard_run_artifacts(self, session_id: str, run_id: str) -> int:
        with closing(self._business._connect()) as conn, self._business._write(conn):
            self._business._require_session(conn, session_id)
            return conn.execute(
                """UPDATE artifacts SET status = 'discarded'
                   WHERE session_id = ? AND run_id = ? AND status = 'staged'""",
                (session_id, run_id),
            ).rowcount

    def list_artifacts(
        self,
        session_id: str,
        *,
        run_id: str | None = None,
        status: ArtifactStatus | None = "published",
    ) -> list[StoredArtifact]:
        with closing(self._business._connect()) as conn:
            self._business._require_session(conn, session_id)
            clauses = ["session_id = ?"]
            params: list[object] = [session_id]
            if run_id is not None:
                clauses.append("run_id = ?")
                params.append(run_id)
            if status is not None:
                clauses.append("status = ?")
                params.append(status)
            where = " AND ".join(clauses)
            rows = conn.execute(
                f"""SELECT artifact_id, session_id, run_id, filename, title,
                           relative_path, media_type, size_bytes, status, created_at
                    FROM artifacts
                    WHERE {where}
                    ORDER BY created_at DESC, rowid DESC""",
                tuple(params),
            ).fetchall()
            return [StoredArtifact(*row) for row in rows]

    def get_artifact(
        self,
        session_id: str,
        artifact_id: str,
        *,
        include_non_published: bool = False,
    ) -> StoredArtifact | None:
        with closing(self._business._connect()) as conn:
            self._business._require_session(conn, session_id)
            predicate = "" if include_non_published else " AND status = 'published'"
            row = conn.execute(
                f"""SELECT artifact_id, session_id, run_id, filename, title,
                           relative_path, media_type, size_bytes, status, created_at
                    FROM artifacts
                    WHERE session_id = ? AND artifact_id = ?{predicate}""",
                (session_id, artifact_id),
            ).fetchone()
            return StoredArtifact(*row) if row else None

    def list_known_artifact_ids(self, session_id: str) -> set[str]:
        with closing(self._business._connect()) as conn:
            self._business._require_session(conn, session_id)
            rows = conn.execute(
                "SELECT artifact_id FROM artifacts WHERE session_id = ?",
                (session_id,),
            ).fetchall()
            return {row[0] for row in rows}

    # ------------------------------------------------------------------
    # Skill activations (for P9.3/P9.4)
    # ------------------------------------------------------------------

    def record_skill_activation(
        self,
        *,
        session_id: str,
        run_id: str,
        skill_id: str,
        name_snapshot: str,
        revision: str,
    ) -> StoredSkillActivation:
        activation_id = str(uuid4())
        now = utc_now()
        with closing(self._business._connect()) as conn, self._business._write(conn):
            self._business._require_session(conn, session_id)
            conn.execute(
                """INSERT INTO run_skill_activations (
                       activation_id, session_id, run_id, skill_id,
                       name_snapshot, revision, activated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (activation_id, session_id, run_id, skill_id, name_snapshot, revision, now),
            )
        return StoredSkillActivation(
            activation_id=activation_id,
            skill_id=skill_id,
            name_snapshot=name_snapshot,
            revision=revision,
            activated_at=now,
        )

    def require_session(self, session_id: str) -> None:
        try:
            self._business.require_session(session_id)
        except SessionNotFound:
            raise
