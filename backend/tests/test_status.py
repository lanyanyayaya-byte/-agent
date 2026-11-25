from secrets import token_urlsafe
from uuid import UUID

import httpx
import pytest

from app.main import DEV_ORIGIN, create_app, load_dev_app

STATUS = "/api/v1/status"
HOST_READY = "/api/v1/_host/ready"
pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def api(tmp_path):
    ui_token, host_token = token_urlsafe(32), token_urlsafe(32)
    transport = httpx.ASGITransport(app=create_app(ui_token, host_token, database_path=tmp_path / "business.sqlite3"))
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        yield client, f"Bearer {ui_token}", f"Bearer {host_token}"


def assert_api_error(response, expected_status, expected_code):
    assert response.status_code == expected_status
    assert response.headers["cache-control"] == "no-store"
    assert set(response.json()) == {"error"}
    error = response.json()["error"]
    assert set(error) == {"code", "message", "request_id", "run_id", "details"}
    assert error["code"] == expected_code
    assert error["message"]
    assert UUID(error["request_id"]).version == 4
    assert error["run_id"] is None
    assert error["details"] is None


async def test_status_uses_real_ui_authentication(api):
    client, ui_header, _ = api
    response = await client.get(STATUS, headers={"Authorization": ui_header, "Origin": DEV_ORIGIN})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == DEV_ORIGIN
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {
        "status": "ready", "database": "ready", "model_configured": False, "siliconflow_configured": False,
    }


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer invalid"}, {"Authorization": "Basic invalid"}])
async def test_missing_or_invalid_token(api, headers):
    client, _, _ = api
    response = await client.get(STATUS, headers=headers)
    assert_api_error(response, 401, "AUTH_REQUIRED")
    assert response.headers["www-authenticate"] == "Bearer"


async def test_trust_channels_are_not_interchangeable(api):
    client, ui_header, host_header = api
    assert_api_error(await client.get(STATUS, headers={"Authorization": host_header}), 403, "FORBIDDEN")
    assert_api_error(await client.get(HOST_READY, headers={"Authorization": ui_header}), 403, "FORBIDDEN")
    assert_api_error(await client.get(HOST_READY), 401, "AUTH_REQUIRED")
    assert_api_error(await client.get(HOST_READY, headers={"Authorization": host_header}), 404, "NOT_FOUND")


async def test_host_and_origin_are_enforced_even_with_ui_token(api):
    client, ui_header, _ = api
    headers = {"Authorization": ui_header}
    assert_api_error(await client.get(STATUS, headers={**headers, "Host": "invalid.example"}), 400, "INVALID_REQUEST")
    denied = await client.get(STATUS, headers={**headers, "Origin": "http://localhost:1420"})
    assert_api_error(denied, 403, "FORBIDDEN")
    assert "access-control-allow-origin" not in denied.headers


async def test_preflight_for_exact_dev_origin(api):
    client, _, _ = api
    response = await client.options(STATUS, headers={
        "Origin": DEV_ORIGIN,
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "authorization",
    })
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == DEV_ORIGIN
    assert "authorization" in response.headers["access-control-allow-headers"].lower()


def test_dev_config_is_required(monkeypatch):
    monkeypatch.delenv("KNOWLEDGEAGENT_DEV_CONFIG", raising=False)
    with pytest.raises(RuntimeError, match="本地开发配置"):
        load_dev_app()


def test_credentials_must_be_separate_and_nonempty(tmp_path):
    token = token_urlsafe(32)
    for pair in [("", token), (token, ""), (token, token)]:
        with pytest.raises(ValueError):
            create_app(*pair, database_path=tmp_path / "business.sqlite3")
