"""Skill runtime dependency detection, readiness checking, and on-demand preparation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.runtime.manager import PrepareResult, RuntimeManager, RuntimeStatus
from app.skills.models import ParsedSkillDocument, SkillDependencySpec, SkillDependencyStatus

_REQ_NAME_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")


def _normalize_pkg_name(spec: str) -> str:
    match = _REQ_NAME_RE.match(spec.strip())
    if not match:
        return spec.strip().lower()
    return match.group(1).lower().replace("_", "-")


def _parse_requirements_txt(path: Path) -> list[str]:
    if not path.is_file():
        return []
    packages: list[str] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        packages.append(line)
    return packages


def _parse_package_json_tools(path: Path) -> list[str]:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    tools: list[str] = []
    for section in ("dependencies", "devDependencies"):
        deps = data.get(section)
        if isinstance(deps, dict):
            if "@mermaid-js/mermaid-cli" in deps:
                tools.append("mermaid-cli")
    return tools


def extract_dependency_spec(skill_dir: Path, doc: ParsedSkillDocument) -> SkillDependencySpec:
    """Detect Python packages, Node tools, and system commands required by a skill."""
    python_pkgs: list[str] = []
    node_tools: list[str] = []
    system_cmds: list[str] = []
    notes: list[str] = []

    def _add_unique(target: list[str], value: str) -> None:
        cleaned = value.strip()
        if cleaned and cleaned not in target:
            target.append(cleaned)

    # 1. Explicit runtime metadata in SKILL.md frontmatter (optional standard `metadata` map)
    runtime_meta: Any = doc.metadata.get("runtime") if isinstance(doc.metadata, dict) else None
    if isinstance(runtime_meta, dict):
        for pkg in runtime_meta.get("python_packages", ()) or ():
            if isinstance(pkg, str):
                _add_unique(python_pkgs, pkg)
        for tool in runtime_meta.get("node_tools", ()) or ():
            if isinstance(tool, str):
                _add_unique(node_tools, tool)
        for cmd in runtime_meta.get("system_commands", ()) or ():
            if isinstance(cmd, str):
                _add_unique(system_cmds, cmd)

    # 2. Standard requirements.txt / package.json inside skill directory
    for pkg in _parse_requirements_txt(skill_dir / "requirements.txt"):
        _add_unique(python_pkgs, pkg)
    for tool in _parse_package_json_tools(skill_dir / "package.json"):
        _add_unique(node_tools, tool)

    # 3. Lightweight compatibility hints (practical, non-fragile checks)
    compat_lower = (doc.compatibility or "").lower()
    body_lower = (doc.body or "").lower()
    combined_hints = f"{compat_lower}\n{body_lower}"

    known_py_hints = (
        ("python-docx", ("python-docx", "from docx import", "import docx")),
        ("openpyxl", ("openpyxl",)),
        ("pypdf", ("pypdf", "from pypdf import")),
        ("pdfplumber", ("pdfplumber",)),
        ("reportlab", ("reportlab",)),
        ("python-pptx", ("python-pptx", "from pptx import", "import pptx")),
        ("pillow", ("pillow", "from pil import")),
        ("defusedxml", ("defusedxml",)),
        ("lxml", ("lxml",)),
    )
    existing_py_norm = {_normalize_pkg_name(p) for p in python_pkgs}
    for pkg_name, markers in known_py_hints:
        if pkg_name not in existing_py_norm and any(m in combined_hints for m in markers):
            _add_unique(python_pkgs, pkg_name)
            existing_py_norm.add(pkg_name)

    if ("mermaid-cli" in compat_lower or "mmdc" in compat_lower) and "mermaid-cli" not in node_tools:
        _add_unique(node_tools, "mermaid-cli")
    if "libreoffice" in compat_lower or "soffice" in compat_lower:
        _add_unique(system_cmds, "soffice")
        notes.append("该技能声明需要 LibreOffice（soffice）以支持文档转换或视觉预览。")
    if "pandoc" in compat_lower:
        _add_unique(system_cmds, "pandoc")
        notes.append("该技能声明需要 Pandoc。")
    if "mermaid-cli" in node_tools and "node" not in system_cmds:
        _add_unique(system_cmds, "node")

    return SkillDependencySpec(
        python_packages=tuple(python_pkgs),
        node_tools=tuple(node_tools),
        system_commands=tuple(system_cmds),
        notes=tuple(notes),
    )


def evaluate_dependencies(spec: SkillDependencySpec, status: RuntimeStatus) -> SkillDependencyStatus:
    """Compare a skill's dependency spec against live RuntimeStatus."""
    installed_norm = {k.lower().replace("_", "-"): v for k, v in status.installed_python_packages.items()}

    missing_py: list[str] = []
    if spec.python_packages:
        if not status.venv_ready:
            missing_py.extend(spec.python_packages)
        else:
            for pkg in spec.python_packages:
                if _normalize_pkg_name(pkg) not in installed_norm:
                    missing_py.append(pkg)

    missing_node: list[str] = []
    for tool in spec.node_tools:
        if tool == "mermaid-cli" and not status.mermaid_cli.available:
            missing_node.append(tool)

    missing_sys: list[str] = []
    for cmd in spec.system_commands:
        if cmd == "node" and not status.node.available:
            missing_sys.append("node")
        elif cmd == "git" and not status.git.available:
            missing_sys.append("git")
        elif cmd == "chrome" and not status.chrome.available:
            missing_sys.append("chrome")
        elif cmd not in ("node", "git", "chrome", "python", "python3"):
            # Unprobed external CLI (e.g. soffice / pandoc) — check standard PATH quickly via shutil.which
            import shutil
            if shutil.which(cmd) is None:
                missing_sys.append(cmd)

    ready = not missing_py and not missing_node and not missing_sys
    can_auto_prepare = (
        not ready
        and not missing_sys
        and (
            (bool(missing_py) and (status.uv.available or status.python.available))
            or (bool(missing_node) and status.npm.available)
        )
    )

    if ready:
        summary = "运行依赖已就绪。"
    else:
        parts: list[str] = []
        if missing_py:
            parts.append(f"缺少 Python 包：{', '.join(missing_py)}")
        if missing_node:
            parts.append(f"缺少 Node 工具：{', '.join(missing_node)}")
        if missing_sys:
            parts.append(f"缺少系统程序：{', '.join(missing_sys)}")
        summary = "；".join(parts) + "。"

    return SkillDependencyStatus(
        ready=ready,
        missing_python_packages=tuple(missing_py),
        missing_node_tools=tuple(missing_node),
        missing_system_commands=tuple(missing_sys),
        can_auto_prepare=can_auto_prepare,
        summary=summary,
    )


async def prepare_missing_dependencies(
    spec: SkillDependencySpec,
    runtime_manager: RuntimeManager,
    *,
    timeout_seconds: float = 300.0,
) -> tuple[SkillDependencyStatus, PrepareResult | None]:
    """Ensure only missing predefined dependencies are installed into the independent Runtime."""
    import asyncio

    status = await asyncio.to_thread(runtime_manager.probe_status)
    dep_status = evaluate_dependencies(spec, status)
    if dep_status.ready or not dep_status.can_auto_prepare:
        return dep_status, None

    install_mmdc = "mermaid-cli" in dep_status.missing_node_tools
    prep = await runtime_manager.prepare(
        python_packages=dep_status.missing_python_packages,
        install_mermaid_cli=install_mmdc,
        ensure_python_venv=bool(dep_status.missing_python_packages),
        timeout_seconds=timeout_seconds,
    )
    updated_dep = evaluate_dependencies(spec, prep.status)
    return updated_dep, prep
