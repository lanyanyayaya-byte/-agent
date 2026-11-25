"""Data models for Agent Skills parsing, storage, dependency status, and runtime activation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

SkillSourceType = Literal["builtin", "local", "github"]
SkillResourceKind = Literal["script", "reference", "asset", "other"]


class SkillError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class ParsedSkillDocument:
    name: str
    description: str
    body: str
    raw_markdown: str
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    allowed_tools: tuple[str, ...] = ()
    extra_frontmatter: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SkillDependencySpec:
    python_packages: tuple[str, ...] = ()
    node_tools: tuple[str, ...] = ()
    system_commands: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SkillDependencyStatus:
    ready: bool
    missing_python_packages: tuple[str, ...] = ()
    missing_node_tools: tuple[str, ...] = ()
    missing_system_commands: tuple[str, ...] = ()
    can_auto_prepare: bool = False
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StoredSkill:
    skill_id: str
    name: str
    description: str
    source_type: SkillSourceType
    source_url: str | None
    revision: str
    relative_dir: str
    enabled: bool
    metadata: dict[str, Any]
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class SkillResourceEntry:
    relative_path: str
    kind: SkillResourceKind
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SkillCatalogEntry:
    """Compact Progressive Disclosure entry injected into model context before activation."""

    skill_id: str
    name: str
    description: str
    revision: str
    source_type: SkillSourceType
    ready: bool

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "ready": self.ready,
        }


@dataclass(frozen=True, slots=True)
class SkillView:
    skill_id: str
    name: str
    description: str
    source_type: SkillSourceType
    source_url: str | None
    revision: str
    enabled: bool
    ready: bool
    dependency_status: SkillDependencyStatus
    warnings: tuple[str, ...]
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["dependency_status"] = self.dependency_status.to_dict()
        return data


@dataclass(frozen=True, slots=True)
class SkillDetailView:
    skill: SkillView
    license: str | None
    compatibility: str | None
    allowed_tools: tuple[str, ...]
    extra_frontmatter: dict[str, Any]
    metadata: dict[str, Any]
    instructions: str
    resources: tuple[SkillResourceEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.skill.to_dict(),
            "license": self.license,
            "compatibility": self.compatibility,
            "allowed_tools": list(self.allowed_tools),
            "extra_frontmatter": self.extra_frontmatter,
            "metadata": self.metadata,
            "instructions": self.instructions,
            "resources": [r.to_dict() for r in self.resources],
        }
