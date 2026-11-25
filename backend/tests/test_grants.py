import os
import sys
from uuid import UUID, uuid4

import httpx
import pytest

from test_sidecar import SHUTDOWN, assert_clean_exit, headers, running_sidecar

SESSIONS = "/api/v1/sessions"
HOST_SESSIONS = "/api/v1/_host/sessions"
HOST_GRANTS = "/api/v1/_host/grants"


def client_for(ready, payload):
    return httpx.Client(base_url=f"http://127.0.0.1:{ready['port']}", trust_env=False,
                        headers=headers(payload["ui_token"]))


def session(client):
    response = client.post(SESSIONS, json={})
    assert response.status_code == 201
    return response.json()["session_id"]


def body(path, scope="file"):
    return dict(path=str(path), scope=scope, operations=["read"], lifetime="application_session")


def assert_error(response, status, code, private=""):
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert response.json()["error"]["details"] is None
    if private:
        assert private not in response.text


def test_live_grants_use_two_tokens_and_are_isolated_and_revocable(tmp_path):
    resources = tmp_path.resolve() / "local-resources"
    resources.mkdir()
    note = resources / "P41-note.TXT"
    note.write_text("sensitive fixture body", encoding="utf-8")
    with running_sidecar(tmp_path / "app") as (child, ready, payload), client_for(ready, payload) as api:
        host = headers(payload["host_token"])
        first, second = session(api), session(api)
        issue = f"{HOST_SESSIONS}/{first}/grants"
        listing = f"{SESSIONS}/{first}/grants"
        other = f"{SESSIONS}/{second}/grants"
        assert_error(api.post(issue, json=body(note)), 403, "FORBIDDEN", str(resources))
        assert_error(api.post(issue, headers=headers("invalid"), json=body(note)), 401, "AUTH_REQUIRED")
        assert_error(api.get(listing, headers=host), 403, "FORBIDDEN")
        assert_error(api.get(f"{SESSIONS}/{uuid4()}/grants"), 404, "NOT_FOUND")
        assert_error(api.post(f"{HOST_SESSIONS}/{uuid4()}/grants", headers=host, json=body(note)), 404, "NOT_FOUND")
        assert_error(api.post(issue, headers=host, json={**body(note), "grant_id": str(uuid4())}), 422, "VALIDATION_ERROR")
        assert_error(api.post(issue, headers=host, json={**body(note), "operations": ["write"]}), 422, "VALIDATION_ERROR")
        assert_error(api.post(issue, headers=host, json={**body(note), "operations": []}), 400, "INVALID_REQUEST")
        assert_error(api.post(issue, headers=host, json={**body(note), "lifetime": "persistent"}), 422, "VALIDATION_ERROR")
        assert api.get(listing).json() == []

        submitted = api.post(issue, headers=host, json=body(f"{resources}/./{note.name}"))
        assert submitted.status_code == 201
        grant = submitted.json()
        assert UUID(grant["grant_id"]).version == 4
        assert grant == dict(grant_id=grant["grant_id"], session_id=first, scope="file",
                             display_name=note.name, operations=["read"])
        assert api.post(issue, headers=host, json=body(note)).json() == grant
        if sys.platform == "darwin" and note.is_relative_to("/private/var"):
            alias = "/var" / note.relative_to("/private/var")
            assert api.post(issue, headers=host, json=body(alias)).json() == grant

        submitted = api.post(f"{HOST_SESSIONS}/{second}/grants", headers=host, json=body(resources, "directory"))
        assert submitted.status_code == 201
        directory = submitted.json()
        assert UUID(directory["grant_id"]).version == 4
        assert directory == dict(grant_id=directory["grant_id"], session_id=second, scope="directory",
                                 display_name=resources.name, operations=["read"])
        assert api.get(listing).json() == [grant]
        assert api.get(other).json() == [directory]

        revoke = f"{HOST_GRANTS}/{grant['grant_id']}"
        assert_error(api.delete(revoke), 403, "FORBIDDEN")
        assert_error(api.delete(revoke, headers=host, params={"session_id": second}), 404, "NOT_FOUND")
        assert api.get(listing).json() == [grant]
        response = api.delete(revoke, headers=host, params={"session_id": first})
        assert response.status_code == 204 and response.content == b""
        assert api.get(listing).json() == []
        assert api.get(other).json() == [directory]
        assert_error(api.delete(revoke, headers=host), 404, "NOT_FOUND")
        assert api.post(issue, headers=host, json=body(note)).json()["grant_id"] != grant["grant_id"]
        assert api.delete(f"{HOST_GRANTS}/{directory['grant_id']}", headers=host).status_code == 204
        assert api.post(SHUTDOWN, headers=host).status_code == 204
        stderr = assert_clean_exit(child, payload)
        diagnostics = (tmp_path / "app/logs/sidecar.jsonl").read_bytes()
        for private in (str(resources), note.name, "sensitive fixture body", *payload.values()):
            if isinstance(private, str) and private != "bootstrap":
                assert private.encode() not in stderr + diagnostics


def test_live_grants_reject_unsafe_paths_without_opening_resources(tmp_path):
    root = tmp_path.resolve()
    notes = root / "notes"
    sibling = root / "notes-private"
    notes.mkdir()
    sibling.mkdir()
    secret = sibling / "not-selected.md"
    secret.touch()
    link = notes / "escape.md"
    link.symlink_to(secret)
    linked_dir = notes / "escape-directory"
    linked_dir.symlink_to(sibling, target_is_directory=True)
    bad_type = notes / "unsupported.bin"
    bad_type.touch()
    disguised_directory = notes / "not-a-file.txt"
    disguised_directory.mkdir()
    special = notes / "pipe.md"
    os.mkfifo(special)

    with running_sidecar(tmp_path / "app") as (child, ready, payload), client_for(ready, payload) as api:
        owner = session(api)
        issue = f"{HOST_SESSIONS}/{owner}/grants"
        host = headers(payload["host_token"])
        for path, scope, status in [
            ("notes/note.md", "file", 400), (root.anchor, "directory", 400),
            (f"{notes}/../{sibling.name}/{secret.name}", "file", 400), (f"{notes}\x00/p.md", "file", 403),
            (link, "file", 403), (linked_dir, "directory", 403), (linked_dir / secret.name, "file", 403),
            (notes / "missing.txt", "file", 403), (secret, "directory", 403), (bad_type, "file", 403),
            (disguised_directory, "file", 403), (special, "file", 403),
        ]:
            result = api.post(issue, headers=host, json=body(path, scope))
            assert_error(result, status, "INVALID_REQUEST" if status == 400 else "FILE_PERMISSION_DENIED", str(root))
        assert api.get(f"{SESSIONS}/{owner}/grants").json() == []
        assert api.post(SHUTDOWN, headers=host).status_code == 204
        assert_clean_exit(child, payload)


@pytest.mark.parametrize("crash", [False, True], ids=["normal-restart", "sigkill"])
def test_live_grants_never_recover_with_the_session(tmp_path, crash):
    app_dir = tmp_path / "app"
    note = tmp_path.resolve() / "p41.md"
    note.touch()
    with running_sidecar(app_dir) as (child, ready, payload), client_for(ready, payload) as api:
        owner = session(api)
        listing = f"{SESSIONS}/{owner}/grants"
        issue = f"{HOST_SESSIONS}/{owner}/grants"
        host = headers(payload["host_token"])
        response = api.post(issue, headers=host, json=body(note))
        assert response.status_code == 201
        old = response.json()["grant_id"]
        if crash:
            child.kill()
            assert child.wait(timeout=6) != 0
        else:
            assert api.post(SHUTDOWN, headers=host).status_code == 204
            assert_clean_exit(child, payload)

    with running_sidecar(app_dir) as (child, current, fresh), client_for(current, fresh) as api:
        assert ready["instance_id"] != current["instance_id"]
        assert [row["session_id"] for row in api.get(SESSIONS).json()] == [owner]
        assert api.get(listing).json() == []
        assert_error(api.get(listing, headers=headers(payload["ui_token"])), 401, "AUTH_REQUIRED")
        host = headers(fresh["host_token"])
        assert_error(api.delete(f"{HOST_GRANTS}/{old}", headers=host), 404, "NOT_FOUND")
        assert api.post(issue, headers=host, json=body(note)).json()["grant_id"] != old
        assert api.post(SHUTDOWN, headers=host).status_code == 204
        assert_clean_exit(child, fresh)
