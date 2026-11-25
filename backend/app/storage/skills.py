"""SQLite persistence layer for installed Agent Skills (Migration V7 `skills` table)."""

from __future__ import annotations

import json
from contextlib import closing
from typing import Any

from app.skills.models import SkillSourceType, StoredSkill
from app.storage.business import BusinessStore, utc_now


class SkillStore:
    """CRUD wrapper over the V7 `skills` table inside `knowledgeagent.sqlite3`."""

    def __init__(self, business: BusinessStore) -> None:
        self._business = business

    @staticmethod
    def _row_to_skill(row: tuple[Any, ...]) -> StoredSkill:
        (
            skill_id, name, description, source_type, source_url,
            revision, relative_dir, enabled, metadata_json, created_at, updated_at,
        ) = row
        try:
            meta = json.loads(metadata_json) if metadata_json else {}
            if not isinstance(meta, dict):
                meta = {}
        except ValueError:
            meta = {}
        return StoredSkill(
            skill_id=skill_id,
            name=name,
            description=description,
            source_type=source_type,
            source_url=source_url,
            revision=revision,
            relative_dir=relative_dir,
            enabled=bool(enabled),
            metadata=meta,
            created_at=created_at,
            updated_at=updated_at,
        )

    def list_skills(self, *, enabled_only: bool = False) -> list[StoredSkill]:
        predicate = "WHERE enabled = 1" if enabled_only else ""
        with closing(self._business._connect()) as conn:
            rows = conn.execute(
                f"""SELECT skill_id, name, description, source_type, source_url,
                           revision, relative_dir, enabled, metadata_json, created_at, updated_at
                    FROM skills
                    {predicate}
                    ORDER BY CASE source_type WHEN 'builtin' THEN 0 ELSE 1 END, name COLLATE NOCASE, skill_id""",
            ).fetchall()
            return [self._row_to_skill(r) for r in rows]

    def get_skill(self, skill_id: str) -> StoredSkill | None:
        with closing(self._business._connect()) as conn:
            row = conn.execute(
                """SELECT skill_id, name, description, source_type, source_url,
                          revision, relative_dir, enabled, metadata_json, created_at, updated_at
                   FROM skills WHERE skill_id = ?""",
                (skill_id,),
            ).fetchone()
            return self._row_to_skill(row) if row else None

    def find_by_name(self, name: str) -> list[StoredSkill]:
        with closing(self._business._connect()) as conn:
            rows = conn.execute(
                """SELECT skill_id, name, description, source_type, source_url,
                          revision, relative_dir, enabled, metadata_json, created_at, updated_at
                   FROM skills WHERE name = ? COLLATE NOCASE
                   ORDER BY created_at""",
                (name,),
            ).fetchall()
            return [self._row_to_skill(r) for r in rows]

    def upsert_skill(
        self,
        *,
        skill_id: str,
        name: str,
        description: str,
        source_type: SkillSourceType,
        source_url: str | None,
        revision: str,
        relative_dir: str,
        enabled: bool,
        metadata: dict[str, Any],
        preserve_enabled_if_exists: bool = False,
    ) -> StoredSkill:
        now = utc_now()
        meta_json = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
        with closing(self._business._connect()) as conn, self._business._write(conn):
            existing = conn.execute(
                "SELECT enabled, created_at FROM skills WHERE skill_id = ?",
                (skill_id,),
            ).fetchone()
            if existing is None:
                conn.execute(
                    """INSERT INTO skills (
                           skill_id, name, description, source_type, source_url,
                           revision, relative_dir, enabled, metadata_json, created_at, updated_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        skill_id, name, description, source_type, source_url,
                        revision, relative_dir, 1 if enabled else 0, meta_json, now, now,
                    ),
                )
                effective_enabled = enabled
                created_at = now
            else:
                effective_enabled = bool(existing[0]) if preserve_enabled_if_exists else enabled
                created_at = existing[1]
                conn.execute(
                    """UPDATE skills
                       SET name = ?, description = ?, source_type = ?, source_url = ?,
                           revision = ?, relative_dir = ?, enabled = ?, metadata_json = ?, updated_at = ?
                       WHERE skill_id = ?""",
                    (
                        name, description, source_type, source_url,
                        revision, relative_dir, 1 if effective_enabled else 0, meta_json, now, skill_id,
                    ),
                )
        return StoredSkill(
            skill_id=skill_id,
            name=name,
            description=description,
            source_type=source_type,
            source_url=source_url,
            revision=revision,
            relative_dir=relative_dir,
            enabled=effective_enabled,
            metadata=metadata,
            created_at=created_at,
            updated_at=now,
        )

    def set_enabled(self, skill_id: str, enabled: bool) -> StoredSkill | None:
        now = utc_now()
        with closing(self._business._connect()) as conn, self._business._write(conn):
            changed = conn.execute(
                "UPDATE skills SET enabled = ?, updated_at = ? WHERE skill_id = ?",
                (1 if enabled else 0, now, skill_id),
            ).rowcount
            if changed != 1:
                return None
            row = conn.execute(
                """SELECT skill_id, name, description, source_type, source_url,
                          revision, relative_dir, enabled, metadata_json, created_at, updated_at
                   FROM skills WHERE skill_id = ?""",
                (skill_id,),
            ).fetchone()
            return self._row_to_skill(row) if row else None

    def delete_skill(self, skill_id: str) -> StoredSkill | None:
        with closing(self._business._connect()) as conn, self._business._write(conn):
            row = conn.execute(
                """SELECT skill_id, name, description, source_type, source_url,
                          revision, relative_dir, enabled, metadata_json, created_at, updated_at
                   FROM skills WHERE skill_id = ?""",
                (skill_id,),
            ).fetchone()
            if row is None:
                return None
            conn.execute("DELETE FROM skills WHERE skill_id = ?", (skill_id,))
            return self._row_to_skill(row)
