import asyncio
import json
import traceback
from secrets import token_urlsafe

import httpx
import pytest
from pydantic import SecretStr

from app.main import create_app
from app.models import ModelError
from app.models.deepseek import DEEPSEEK_BASE_URL, DeepSeekAdapter, DeepSeekSettings
from app.services.models import ModelService


CREDENTIAL = "/api/v1/_host/model-credentials/deepseek"
STATUS = "/api/v1/status"
pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def key():
    return token_urlsafe(32)


def service_with(transport, **settings):
    return ModelService(DeepSeekAdapter(
        DeepSeekSettings(**settings), transport=httpx.MockTransport(transport),
    ))


async def test_host_injection_and_agent_answer(key, monkeypatch, caplog, tmp_path):
    monkeypatch.setenv("DEEPSEEK_API_KEY", key)
    requests = []

    def respond(request):
        assert request.url == f"{DEEPSEEK_BASE_URL}/chat/completions"
        assert request.headers.get("authorization") == f"Bearer {key}"
        data = json.loads(request.content)
        assert data["model"] == "fixture-model"
        assert data["thinking"] == {"type": "disabled"}
        assert not data.get("tools")
        assert data["messages"] == [{"role": "user", "content": "fixed prompt"}]
        requests.append(request.url)
        return httpx.Response(200, json={
            "id": "fixture", "model": "fixture-model", "created": 1, "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": "fixture reply", "reasoning_content": "hidden",
            }}],
        })

    models = service_with(respond, model="fixture-model")
    with pytest.raises(ModelError) as error:
        await models.answer("fixed prompt")
    assert error.value.code == "MODEL_NOT_CONFIGURED" and not requests

    ui, host = token_urlsafe(32), token_urlsafe(32)
    ui_headers = {"Authorization": f"Bearer {ui}"}
    host_headers = {"Authorization": f"Bearer {host}"}
    app = create_app(ui, host, models=models, database_path=tmp_path / "business.sqlite3")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        payload = {"api_key": key, "base_url": DEEPSEEK_BASE_URL}
        for headers, code in [(None, 401), (ui_headers, 403)]:
            denied = await client.put(CREDENTIAL, headers=headers, json=payload)
            assert denied.status_code == code and key not in denied.text
        assert not models.configured

        invalid_payloads = [
            ({"api_key": [key]}, 422, "VALIDATION_ERROR"),
            ({**payload, "extra": key}, 422, "VALIDATION_ERROR"),
            ({**payload, "base_url": "https://api.deepseek.com.example.org"}, 400, "INVALID_REQUEST"),
        ]
        for invalid, code, label in invalid_payloads:
            denied = await client.put(CREDENTIAL, headers=host_headers, json=invalid)
            assert denied.status_code == code and denied.json()["error"]["code"] == label
            assert key not in denied.text
        assert not models.configured

        configured = await client.put(CREDENTIAL, headers=host_headers, json=payload)
        assert configured.status_code == 200 and configured.json() == {"configured": True}
        assert configured.headers["cache-control"] == "no-store"
        assert key not in configured.text and not requests
        status = await client.get(STATUS, headers=ui_headers)
        assert status.json()["model_configured"] is True and key not in status.text
        assert (await client.get(CREDENTIAL, headers=ui_headers)).status_code == 403
        denied_readback = await client.get(CREDENTIAL, headers=host_headers)
        assert denied_readback.status_code == 405 and key not in denied_readback.text
        assert (await client.put(STATUS, headers=ui_headers, json=payload)).status_code == 405
        assert (await client.put(CREDENTIAL.replace("deepseek", "other"), headers=host_headers,
                                 json=payload)).status_code == 404

    assert await models.answer("fixed prompt") == "fixture reply"
    assert len(requests) == 1

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        rotated_key = token_urlsafe(32)
        rotated = await client.put(CREDENTIAL, headers=host_headers,
                                   json={"api_key": rotated_key, "base_url": DEEPSEEK_BASE_URL})
        assert rotated.status_code == 200 and rotated.json() == {"configured": True}
        for headers, code in [(None, 401), (ui_headers, 403)]:
            denied_delete = await client.delete(CREDENTIAL, headers=headers)
            assert denied_delete.status_code == code and rotated_key not in denied_delete.text
        assert (await client.delete(CREDENTIAL.replace("deepseek", "other"), headers=host_headers)).status_code == 404
        assert models.configured

        cleared = await client.delete(CREDENTIAL, headers=host_headers)
        assert cleared.status_code == 200 and cleared.json() == {"configured": False}
        assert cleared.headers["cache-control"] == "no-store"
        assert not models.configured
        status_after = await client.get(STATUS, headers=ui_headers)
        assert status_after.json()["model_configured"] is False
        with pytest.raises(ModelError) as error:
            await models.answer("fixed prompt")
        assert error.value.code == "MODEL_NOT_CONFIGURED"

    assert key not in caplog.text and rotated_key not in caplog.text


@pytest.mark.parametrize(("mode", "code", "message"), [
    ("auth", "MODEL_UPSTREAM_ERROR", "认证失败"),
    ("network", "MODEL_UPSTREAM_ERROR", "无法连接"),
    ("interrupted", "MODEL_UPSTREAM_ERROR", "响应流中断"),
    ("timeout", "MODEL_TIMEOUT", "超时"),
    ("deadline", "MODEL_TIMEOUT", "超时"),
    ("gateway", "MODEL_TIMEOUT", "超时"),
    ("bad-response", "MODEL_UPSTREAM_ERROR", "响应无效"),
])
async def test_upstream_errors_are_sanitized(key, mode, code, message, caplog):
    class InterruptedBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"{"
            raise httpx.RemoteProtocolError(key)

    async def respond(request):
        if mode == "network":
            raise httpx.ConnectError(key, request=request)
        if mode == "interrupted":
            return httpx.Response(200, stream=InterruptedBody(), headers={"Content-Type": "application/json"})
        if mode == "timeout":
            raise httpx.ReadTimeout(key, request=request)
        if mode == "deadline":
            await asyncio.sleep(1)
            raise AssertionError("deadline did not interrupt the request")
        if mode == "bad-response":
            return httpx.Response(200, json={"error": key})
        return httpx.Response(401 if mode == "auth" else 504, json={"error": {"message": key}})

    models = service_with(respond, timeout_seconds=0.1 if mode == "deadline" else 10)
    models.configure_deepseek(SecretStr(key), DEEPSEEK_BASE_URL)
    with pytest.raises(ModelError) as error:
        await models.answer("fixed prompt")
    assert error.value.code == code and message in str(error.value)
    assert key not in "".join(traceback.format_exception(error.value))
    assert key not in caplog.text
