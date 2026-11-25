"""Standard Agent Skills (`SKILL.md`) YAML frontmatter parser and revision hashing.

Follows https://agentskills.io/specification:
- Parses YAML frontmatter delimited by `---` lines via PyYAML (`yaml.safe_load`).
- Extracts `name`, `description`, and optional `license`, `compatibility`, `metadata`, `allowed-tools`.
- Preserves unknown third-party frontmatter fields in `extra_frontmatter` with informational warnings
  rather than rejecting compatible third-party repositories.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml

from app.skills.models import ParsedSkillDocument, SkillError, SkillResourceEntry, SkillResourceKind

SKILL_FILENAME = "SKILL.md"
SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
FRONTMATTER_RE = re.compile(r"^\ufeff?---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|$)", re.DOTALL)

STANDARD_FRONTMATTER_KEYS = frozenset({
    "name",
    "description",
    "license",
    "compatibility",
    "metadata",
    "allowed-tools",
})

HOST_SPECIFIC_FRONTMATTER_KEYS = frozenset({
    "context",
    "agent",
    "hooks",
    "disable-model-invocation",
    "user-invocable",
})


def _parse_allowed_tools(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        return tuple(part for part in raw.split() if part)
    if isinstance(raw, list):
        return tuple(str(item).strip() for item in raw if str(item).strip())
    return ()


def parse_skill_markdown(raw_markdown: str, *, expected_dir_name: str | None = None) -> ParsedSkillDocument:
    """Parse a `SKILL.md` document and validate required Agent Skills fields."""
    if not isinstance(raw_markdown, str) or not raw_markdown.strip():
        raise SkillError("SKILL_PARSE_ERROR", "SKILL.md 内容为空。")

    match = FRONTMATTER_RE.match(raw_markdown)
    if match is None:
        raise SkillError("SKILL_PARSE_ERROR", "SKILL.md 缺少标准 YAML frontmatter（--- ... ---）。")

    yaml_text = match.group(1)
    body = raw_markdown[match.end():].lstrip("\r\n")

    try:
        loaded = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        raise SkillError("SKILL_PARSE_ERROR", f"SKILL.md YAML frontmatter 解析失败：{exc}") from None

    if not isinstance(loaded, dict):
        raise SkillError("SKILL_PARSE_ERROR", "SKILL.md frontmatter 必须为 YAML 映射对象。")

    raw_name = loaded.get("name")
    raw_description = loaded.get("description")
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise SkillError("SKILL_INVALID", "SKILL.md 缺少必需字段 name。")
    if not isinstance(raw_description, str) or not raw_description.strip():
        raise SkillError("SKILL_INVALID", "SKILL.md 缺少必需字段 description。")

    name = raw_name.strip()
    description = " ".join(raw_description.strip().split())
    warnings: list[str] = []

    if len(name) > 64 or not SKILL_NAME_PATTERN.fullmatch(name):
        warnings.append(f"技能名称 '{name}' 不完全符合标准 kebab-case 规范，已按兼容模式保留。")
    if expected_dir_name and expected_dir_name != name:
        warnings.append(f"目录名 '{expected_dir_name}' 与 frontmatter name '{name}' 不一致。")
    if len(description) > 1024:
        warnings.append("技能 description 超过 1024 字符建议上限。")
    if len(body) > 24_000:
        warnings.append("SKILL.md 指令正文较长，建议将详细规则拆分到 references/ 目录。")

    license_val = loaded.get("license")
    license_text = str(license_val).strip() if isinstance(license_val, str) and license_val.strip() else None

    compat_val = loaded.get("compatibility")
    compat_text = str(compat_val).strip() if isinstance(compat_val, str) and compat_val.strip() else None

    raw_meta = loaded.get("metadata")
    metadata: dict[str, Any] = dict(raw_meta) if isinstance(raw_meta, dict) else {}

    allowed_tools = _parse_allowed_tools(loaded.get("allowed-tools"))

    extra_frontmatter: dict[str, Any] = {}
    for key, value in loaded.items():
        key_str = str(key)
        if key_str not in STANDARD_FRONTMATTER_KEYS:
            extra_frontmatter[key_str] = value
            if key_str in HOST_SPECIFIC_FRONTMATTER_KEYS:
                warnings.append(f"包含宿主特定扩展字段 '{key_str}'，KnowledgeAgent 将按兼容模式保留。")

    return ParsedSkillDocument(
        name=name,
        description=description,
        body=body,
        raw_markdown=raw_markdown,
        license=license_text,
        compatibility=compat_text,
        metadata=metadata,
        allowed_tools=allowed_tools,
        extra_frontmatter=extra_frontmatter,
        warnings=tuple(warnings),
    )


def parse_skill_directory(skill_dir: Path) -> ParsedSkillDocument:
    """Read and parse `SKILL.md` inside a skill directory."""
    skill_md = skill_dir / SKILL_FILENAME
    if not skill_md.is_file():
        raise SkillError("SKILL_NOT_FOUND", f"目录中未找到 {SKILL_FILENAME}：{skill_dir.name}")
    try:
        raw = skill_md.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise SkillError("SKILL_PARSE_ERROR", f"无法读取 {SKILL_FILENAME}：{exc}") from None
    return parse_skill_markdown(raw, expected_dir_name=skill_dir.name)


def list_skill_resources(skill_dir: Path) -> tuple[SkillResourceEntry, ...]:
    """Enumerate resource files under a skill directory (excluding hidden files and __pycache__)."""
    if not skill_dir.is_dir():
        return ()
    entries: list[SkillResourceEntry] = []
    base = skill_dir.resolve()
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(base)
        parts = rel.parts
        if any(p.startswith(".") or p == "__pycache__" for p in parts):
            continue
        if rel.as_posix() == SKILL_FILENAME:
            continue
        top = parts[0] if len(parts) > 1 else ""
        kind: SkillResourceKind
        if top == "scripts":
            kind = "script"
        elif top == "references":
            kind = "reference"
        elif top == "assets":
            kind = "asset"
        else:
            kind = "other"
        entries.append(SkillResourceEntry(
            relative_path=rel.as_posix(),
            kind=kind,
            size_bytes=path.stat().st_size,
        ))
    return tuple(entries)


def compute_skill_revision(skill_dir: Path) -> str:
    """Compute a deterministic content revision string (e.g. `sha256:abcdef123456`)."""
    hasher = hashlib.sha256()
    base = skill_dir.resolve()
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(base)
        if any(p.startswith(".") or p == "__pycache__" for p in rel.parts):
            continue
        hasher.update(rel.as_posix().encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(path.read_bytes())
        hasher.update(b"\x00")
    return f"rev-{hasher.hexdigest()[:12]}"


def discover_skill_dirs(root: Path, *, max_depth: int = 4) -> list[Path]:
    """Find all directories containing `SKILL.md` under `root`."""
    if not root.is_dir():
        return []
    found: list[Path] = []
    root_resolved = root.resolve()

    def _walk(current: Path, depth: int) -> None:
        if depth > max_depth:
            return
        if (current / SKILL_FILENAME).is_file():
            found.append(current)
            return
        try:
            children = sorted(current.iterdir(), key=lambda p: p.name.casefold())
        except OSError:
            return
        for child in children:
            if not child.is_dir() or child.is_symlink():
                continue
            if child.name.startswith(".") or child.name in ("node_modules", "__pycache__", ".git", ".venv", "venv"):
                continue
            _walk(child, depth + 1)

    _walk(root_resolved, 0)
    return found
