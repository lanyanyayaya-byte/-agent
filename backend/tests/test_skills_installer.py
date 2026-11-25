"""Tests for P9.5 SkillInstaller: GitHub URL parsing, local dir & ZIP preview/install/update, and API endpoints."""

from __future__ import annotations

import zipfile
from pathlib import Path

import httpx
import pytest

from app.main import create_app
from app.runtime.manager import RuntimeManager
from app.skills.installer import SkillInstaller, parse_github_url
from app.skills.manager import SkillManager
from app.skills.models import SkillError
from app.storage.business import BusinessStore
from app.storage.skills import SkillStore


def _write_skill(root: Path, name: str, description: str, *, extra_script: str | None = None, reqs: str | None = None) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\ncustom-vendor-flag: enabled\n---\n\n# {name}\n\nUse this skill.\n",
        encoding="utf-8",
    )
    if extra_script:
        scripts = skill_dir / "scripts"
        scripts.mkdir(exist_ok=True)
        (scripts / "run.py").write_text(extra_script, encoding="utf-8")
    if reqs:
        (skill_dir / "requirements.txt").write_text(reqs, encoding="utf-8")
    return skill_dir


def test_parse_github_url_variants() -> None:
    root = parse_github_url("https://github.com/anthropics/skills")
    assert (root.owner, root.repo, root.repo_url, root.ref, root.subpath) == (
        "anthropics",
        "skills",
        "https://github.com/anthropics/skills",
        None,
        None,
    )

    shorthand = parse_github_url("anthropics/skills")
    assert shorthand.repo_url == "https://github.com/anthropics/skills"

    subdir = parse_github_url("https://github.com/anthropics/skills/tree/main/skills/xlsx")
    assert (subdir.ref, subdir.subpath) == ("main", "skills/xlsx")

    blob_skill = parse_github_url("https://github.com/anthropics/skills/blob/main/skills/pdf/SKILL.md")
    assert (blob_skill.ref, blob_skill.subpath) == ("main", "skills/pdf")

    override = parse_github_url(
        "https://github.com/anthropics/skills/tree/main/skills/pdf",
        explicit_ref="release/v1",
        explicit_subpath="skills/docx",
    )
    assert (override.ref, override.subpath) == ("release/v1", "skills/docx")

    with pytest.raises(SkillError) as exc:
        parse_github_url("https://gitlab.com/foo/bar")
    assert exc.value.code == "SKILL_INVALID_SOURCE"

    with pytest.raises(SkillError) as exc2:
        parse_github_url("https://github.com/anthropics/skills", explicit_subpath="../escape")
    assert exc2.value.code == "SKILL_INVALID_SOURCE"


@pytest.mark.anyio
async def test_local_dir_and_zip_selective_install_and_zip_slip_blocked(tmp_path: Path) -> None:
    app_data = tmp_path / "app_data"
    app_data.mkdir()
    db = BusinessStore(app_data / "knowledgeagent.db")
    runtime = RuntimeManager(app_data)
    manager = SkillManager(app_data, SkillStore(db), runtime)
    installer = SkillInstaller(manager, runtime)

    # 1) Multi-skill local folder: selective install of only one skill
    bundle = tmp_path / "multi_skills"
    _write_skill(bundle, "alpha-tool", "Alpha helper skill", extra_script="print('alpha')\n")
    _write_skill(bundle, "beta-tool", "Beta helper skill", extra_script="print('beta')\n")

    preview = await installer.preview_import(source_kind="local_dir", source=str(bundle))
    assert len(preview.candidates) == 2
    rel_paths = {c.name: c.relative_path for c in preview.candidates}
    assert set(rel_paths.keys()) == {"alpha-tool", "beta-tool"}

    installed = await installer.install_from_preview(
        preview.preview_id,
        [rel_paths["alpha-tool"]],
        enabled=True,
    )
    assert [s.name for s in installed] == ["alpha-tool"]
    assert installed[0].enabled is True
    detail = manager.get_skill_detail(installed[0].skill_id)
    assert detail.extra_frontmatter.get("custom-vendor-flag") == "enabled"
    assert any(r.relative_path == "scripts/run.py" for r in detail.resources)
    assert all(s.name != "beta-tool" for s in manager.list_skills())

    # 2) Update local skill after editing source directory
    (bundle / "alpha-tool" / "SKILL.md").write_text(
        "---\nname: alpha-tool\ndescription: Alpha updated description\n---\n\n# Updated\n",
        encoding="utf-8",
    )
    updated = await installer.update_skill(installed[0].skill_id)
    assert updated.description == "Alpha updated description"

    # 3) Local ZIP archive containing skills + selective install
    zip_src = tmp_path / "zip_src"
    _write_skill(zip_src / "repo-main" / "skills", "gamma-pack", "Gamma from ZIP", extra_script="print('gamma')\n")
    good_zip = tmp_path / "skills.zip"
    with zipfile.ZipFile(good_zip, "w") as zf:
        for file_path in zip_src.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(zip_src))

    zip_preview = await installer.preview_import(source_kind="local_zip", source=str(good_zip))
    assert len(zip_preview.candidates) == 1
    assert zip_preview.candidates[0].name == "gamma-pack"
    zip_installed = await installer.install_from_preview(
        zip_preview.preview_id,
        [zip_preview.candidates[0].relative_path],
        enabled=True,
    )
    assert zip_installed[0].name == "gamma-pack"

    # 4) Zip Slip traversal MUST be rejected
    evil_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil_zip, "w") as zf:
        zf.writestr("../../outside.txt", "escape")
    with pytest.raises(SkillError) as exc:
        await installer.preview_import(source_kind="local_zip", source=str(evil_zip))
    assert exc.value.code == "SKILL_INVALID_PATH"


@pytest.mark.anyio
async def test_skills_installer_host_api_channel_separation(tmp_path: Path) -> None:
    ui_tok = "p95-ui-token-1234567890"
    host_tok = "p95-host-token-abcdefghij"
    ui_headers = {"Authorization": f"Bearer {ui_tok}", "Host": "127.0.0.1:8000"}
    host_headers = {"Authorization": f"Bearer {host_tok}", "Host": "127.0.0.1:8000"}
    app = create_app(ui_tok, host_tok, database_path=tmp_path / "knowledgeagent.sqlite3")
    skill_src = _write_skill(tmp_path / "src", "api-skill", "Installed via Host API", extra_script="print('ok')\n")

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8000") as client:
            # UI token cannot call host preview/install
            r_denied = await client.post(
                "/api/v1/_host/skills/import-preview",
                headers=ui_headers,
                json={"source_kind": "local_dir", "source": str(skill_src)},
            )
            assert r_denied.status_code == 403

            # Host token previews and installs
            r_prev = await client.post(
                "/api/v1/_host/skills/import-preview",
                headers=host_headers,
                json={"source_kind": "local_dir", "source": str(skill_src)},
            )
            assert r_prev.status_code == 200
            prev_body = r_prev.json()
            assert len(prev_body["candidates"]) == 1

            r_inst = await client.post(
                "/api/v1/_host/skills/install",
                headers=host_headers,
                json={
                    "preview_id": prev_body["preview_id"],
                    "selected_relative_paths": [prev_body["candidates"][0]["relative_path"]],
                    "enabled": True,
                },
            )
            assert r_inst.status_code == 201
            installed_id = r_inst.json()["skills"][0]["skill_id"]

            # UI token lists & reads detail
            r_list = await client.get("/api/v1/skills", headers=ui_headers)
            assert r_list.status_code == 200
            assert any(s["skill_id"] == installed_id for s in r_list.json())

            # Host token prepares deps & deletes
            r_prep = await client.post(
                f"/api/v1/_host/skills/{installed_id}/prepare",
                headers=host_headers,
            )
            assert r_prep.status_code == 200
            assert r_prep.json()["ok"] is True

            r_del = await client.delete(
                f"/api/v1/_host/skills/{installed_id}",
                headers=host_headers,
            )
            assert r_del.status_code == 204
