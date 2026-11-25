"""P9.3 verification suite: Agent Skills YAML parser, SkillManager, dependencies, and Skills API."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from app.main import create_app
from app.runtime.manager import RuntimeManager
from app.skills.dependencies import evaluate_dependencies, extract_dependency_spec
from app.skills.manager import SkillManager
from app.skills.models import SkillError
from app.skills.parser import parse_skill_markdown
from app.storage.business import BusinessStore
from app.storage.skills import SkillStore

UI_TOKEN = "p93-ui-token-1234567890"
HOST_TOKEN = "p93-host-token-abcdefghij"


def test_skill_markdown_parser_standard_and_extra_frontmatter():
    raw = """---
name: custom-research
description: >
  Helps summarize structured notes
  and export reports.
license: Apache-2.0
compatibility: Requires python-docx
allowed-tools: Bash Read
metadata:
  author: tester
  version: "0.2"
custom_vendor_flag: true
context: fork
---

# Custom Research Workflow
Step 1: Read input.
"""
    parsed = parse_skill_markdown(raw, expected_dir_name="custom-research")
    assert parsed.name == "custom-research"
    assert parsed.description == "Helps summarize structured notes and export reports."
    assert parsed.license == "Apache-2.0"
    assert parsed.compatibility == "Requires python-docx"
    assert parsed.allowed_tools == ("Bash", "Read")
    assert parsed.metadata == {"author": "tester", "version": "0.2"}
    assert parsed.extra_frontmatter == {"custom_vendor_flag": True, "context": "fork"}
    assert any("context" in w for w in parsed.warnings)
    assert "Step 1: Read input." in parsed.body

    with pytest.raises(SkillError) as exc_info:
        parse_skill_markdown("# No frontmatter here")
    assert exc_info.value.code == "SKILL_PARSE_ERROR"

    with pytest.raises(SkillError) as exc_info:
        parse_skill_markdown("---\nname: only-name\n---\nbody\n")
    assert exc_info.value.code == "SKILL_INVALID"


def test_skill_manager_builtin_sync_local_import_restart_and_progressive_disclosure(tmp_path):
    app_data = tmp_path / "app_data"
    db_path = app_data / "knowledgeagent.sqlite3"
    business = BusinessStore(db_path)
    skill_store = SkillStore(business)
    runtime_mgr = RuntimeManager(app_data)
    manager = SkillManager(app_data, skill_store, runtime_mgr)

    # 1. Sync built-in skills (`mind-map` & `word-formatting`)
    synced = manager.sync_builtin_skills()
    names = {s.name for s in synced}
    assert {"mind-map", "word-formatting"}.issubset(names)
    assert all(s.enabled for s in synced)

    # Catalog must expose lightweight metadata only (no instructions body)
    catalog = manager.build_catalog_snapshot()
    assert len(catalog) == 2
    prompt_entries = [c.to_prompt_dict() for c in catalog]
    assert all(set(item.keys()) == {"skill_id", "name", "description", "ready"} for item in prompt_entries)

    # Detail exposes full instructions and resources
    wf_detail = manager.get_skill_detail("builtin-word-formatting")
    assert "word-formatting" == wf_detail.skill.name
    assert any(r.relative_path == "scripts/format_docx.py" and r.kind == "script" for r in wf_detail.resources)
    assert any(r.relative_path == "references/chinese-typography.md" and r.kind == "reference" for r in wf_detail.resources)

    # Read reference resource + reject path traversal
    ref_read = manager.read_skill_file("builtin-word-formatting", "references/chinese-typography.md")
    assert "小四" in str(ref_read["content"])
    with pytest.raises(SkillError):
        manager.read_skill_file("builtin-word-formatting", "../../secret.txt")

    # Dependency extraction detects python-docx and mermaid-cli
    _, wf_dir = manager.resolve_skill_dir("builtin-word-formatting")
    wf_spec = extract_dependency_spec(wf_dir, parse_skill_markdown((wf_dir / "SKILL.md").read_text(encoding="utf-8")))
    assert "python-docx" in wf_spec.python_packages
    wf_dep_status = evaluate_dependencies(wf_spec, runtime_mgr.probe_status())
    assert wf_dep_status.ready is False  # fresh tmp_path has no venv yet
    assert wf_dep_status.can_auto_prepare is True

    # 2. Disable a built-in skill, then simulate app restart -> disabled state MUST persist
    manager.set_enabled("builtin-mind-map", False)
    business2 = BusinessStore(db_path)
    manager2 = SkillManager(app_data, SkillStore(business2), RuntimeManager(app_data))
    manager2.sync_builtin_skills()
    views_after = {v.skill_id: v for v in manager2.list_skills()}
    assert views_after["builtin-mind-map"].enabled is False
    assert views_after["builtin-word-formatting"].enabled is True

    # 3. Local skill directory import (with extra frontmatter) -> enable -> delete
    local_dir = tmp_path / "local_skill_pack" / "hello-skill"
    (local_dir / "references").mkdir(parents=True)
    (local_dir / "SKILL.md").write_text(
        "---\nname: hello-skill\ndescription: A local test skill.\nvendor_tag: v1\n---\n# Hello\nRun steps.\n",
        encoding="utf-8",
    )
    (local_dir / "references" / "guide.md").write_text("Guide line 1\n", encoding="utf-8")

    imported = manager2.import_local_directory(local_dir.parent, enabled=False)
    assert len(imported) == 1
    imp = imported[0]
    assert imp.name == "hello-skill"
    assert imp.source_type == "local"
    assert imp.enabled is False

    imp_detail = manager2.get_skill_detail(imp.skill_id)
    assert imp_detail.extra_frontmatter == {"vendor_tag": "v1"}
    manager2.set_enabled(imp.skill_id, True)
    assert any(c.skill_id == imp.skill_id for c in manager2.build_catalog_snapshot())

    manager2.delete_skill(imp.skill_id)
    with pytest.raises(SkillError):
        manager2.get_skill_detail(imp.skill_id)


@pytest.mark.anyio
async def test_skills_api_dual_token_lifecycle(tmp_path):
    app = create_app(UI_TOKEN, HOST_TOKEN, database_path=tmp_path / "knowledgeagent.sqlite3")
    ui_headers = {"Authorization": f"Bearer {UI_TOKEN}", "Host": "127.0.0.1:8000"}
    host_headers = {"Authorization": f"Bearer {HOST_TOKEN}", "Host": "127.0.0.1:8000"}

    local_skill = tmp_path / "demo-local"
    local_skill.mkdir()
    (local_skill / "SKILL.md").write_text(
        "---\nname: demo-local\ndescription: Local skill imported via API.\n---\n# Demo\nInstructions.\n",
        encoding="utf-8",
    )

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8000") as client:
            # UI lists built-in skills
            resp = await client.get("/api/v1/skills", headers=ui_headers)
            assert resp.status_code == 200
            names = {item["name"] for item in resp.json()}
            assert {"mind-map", "word-formatting"}.issubset(names)

            # UI reads skill detail
            detail = await client.get("/api/v1/skills/builtin-mind-map", headers=ui_headers)
            assert detail.status_code == 200
            assert "render_mindmap.py" in str(detail.json()["resources"])

            # UI token cannot call _host mutation endpoints
            denied = await client.post(
                "/api/v1/_host/skills/import-local",
                json={"path": str(local_skill)},
                headers=ui_headers,
            )
            assert denied.status_code == 403

            # Host imports local skill, toggles enabled, then deletes
            imp_resp = await client.post(
                "/api/v1/_host/skills/import-local",
                json={"path": str(local_skill), "enabled": False},
                headers=host_headers,
            )
            assert imp_resp.status_code == 201
            skill_id = imp_resp.json()["skills"][0]["skill_id"]

            patch_resp = await client.patch(
                f"/api/v1/_host/skills/{skill_id}",
                json={"enabled": True},
                headers=host_headers,
            )
            assert patch_resp.status_code == 200
            assert patch_resp.json()["enabled"] is True

            del_resp = await client.delete(f"/api/v1/_host/skills/{skill_id}", headers=host_headers)
            assert del_resp.status_code == 204
