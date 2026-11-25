import json
from secrets import token_urlsafe

import httpx
import pytest

from app.main import create_app
from app.models.siliconflow import (
    CONNECTION_TEST_TEXT,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EMBEDDING_URL,
    SILICONFLOW_BASE_URL,
    SiliconFlowEmbedding,
)

CREDENTIAL = "/api/v1/_host/model-credentials/siliconflow"
TEST = f"{CREDENTIAL}/test"
STATUS = "/api/v1/status"
pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def embedding_response(dimensions=EMBEDDING_DIMENSIONS):
    return httpx.Response(200, json={
        "object": "list", "model": EMBEDDING_MODEL,
        "data": [{"index": 0, "embedding": [0.5] * dimensions}],
    })


@pytest.fixture
def local_api(tmp_path):
    def create(handler):
        ui, host = token_urlsafe(32), token_urlsafe(32)
        app = create_app(
            ui, host, database_path=tmp_path / "business.sqlite3",
            embeddings=SiliconFlowEmbedding(transport=httpx.MockTransport(handler)),
        )
        return (
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"),
            {"Authorization": f"Bearer {ui}"},
            {"Authorization": f"Bearer {host}"},
        )
    return create


async def test_siliconflow_credential_is_host_only_and_distinct_from_deepseek(local_api, monkeypatch, caplog):
    deepseek, initial, rotated = (token_urlsafe(32) for _ in range(3))
    monkeypatch.setenv("SILICONFLOW_API_KEY", rotated)
    keys = iter((initial, rotated))
    requests = []

    def respond(request):
        assert request.url == EMBEDDING_URL
        assert request.headers["authorization"] == f"Bearer {next(keys)}"
        assert json.loads(request.content) == {
            "model": EMBEDDING_MODEL, "input": CONNECTION_TEST_TEXT, "encoding_format": "float",
        }
        requests.append(request.url)
        return embedding_response()

    client, ui, host = local_api(respond)
    async with client:
        assert (await client.post(TEST, headers=host)).json()["error"]["code"] == "EMBEDDING_NOT_CONFIGURED"
        for headers, code in ((None, 401), (ui, 403)):
            assert (await client.post(TEST, headers=headers)).status_code == code
            assert (await client.put(CREDENTIAL, json={"api_key": initial}, headers=headers)).status_code == code
            assert (await client.delete(CREDENTIAL, headers=headers)).status_code == code
        assert (await client.get(CREDENTIAL, headers=host)).status_code == 405

        assert (await client.put(CREDENTIAL.replace("siliconflow", "deepseek"), json={"api_key": deepseek},
                                 headers=host)).status_code == 200
        invalid = await client.put(CREDENTIAL, headers=host, json={"api_key": initial, "base_url": EMBEDDING_URL})
        assert invalid.status_code == 400 and initial not in invalid.text
        assert not requests

        for key in (initial, rotated):
            saved = await client.put(CREDENTIAL, headers=host, json={"api_key": key, "base_url": SILICONFLOW_BASE_URL})
            assert saved.json() == {"configured": True}
            status = (await client.get(STATUS, headers=ui)).json()
            assert status["model_configured"] and status["siliconflow_configured"]
            tested = await client.post(TEST, headers=host)
            assert tested.status_code == 200 and tested.headers["cache-control"] == "no-store"
            assert tested.json() == {"model": EMBEDDING_MODEL, "dimensions": EMBEDDING_DIMENSIONS}

        assert (await client.delete(CREDENTIAL, headers=host)).json() == {"configured": False}
        status = (await client.get(STATUS, headers=ui)).json()
        assert status["model_configured"] and not status["siliconflow_configured"]
        assert (await client.post(TEST, headers=host)).status_code == 409
    assert len(requests) == 2
    assert all(key not in caplog.text for key in (initial, rotated, deepseek))


@pytest.mark.parametrize(("failure", "status", "code"), [
    (401, 401, "EMBEDDING_AUTH_FAILED"),
    (403, 403, "EMBEDDING_FORBIDDEN"),
    (429, 429, "EMBEDDING_RATE_LIMITED"),
    (503, 502, "EMBEDDING_SERVICE_ERROR"),
    (504, 504, "EMBEDDING_TIMEOUT"),
    (307, 502, "EMBEDDING_SERVICE_ERROR"),
    ("timeout", 504, "EMBEDDING_TIMEOUT"),
    ("network", 502, "EMBEDDING_NETWORK_ERROR"),
    ("dimensions", 502, "EMBEDDING_INVALID_RESPONSE"),
])
async def test_siliconflow_errors_are_sanitized(local_api, caplog, failure, status, code):
    key = token_urlsafe(32)

    def respond(request):
        assert request.url == EMBEDDING_URL
        if failure == "timeout":
            raise httpx.ReadTimeout(key, request=request)
        if failure == "network":
            raise httpx.ConnectError(key, request=request)
        if failure == "dimensions":
            return embedding_response(128)
        return httpx.Response(failure, text=key, headers={"Location": "https://example.com"})

    client, _, host = local_api(respond)
    async with client:
        assert (await client.put(CREDENTIAL, headers=host, json={
            "api_key": key, "base_url": SILICONFLOW_BASE_URL,
        })).status_code == 200
        result = await client.post(TEST, headers=host)
        assert result.status_code == status and result.json()["error"]["code"] == code
        assert key not in result.text and key not in caplog.text
