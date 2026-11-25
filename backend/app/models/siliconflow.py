import asyncio
import re
import unicodedata
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from random import uniform

import httpx
from pydantic import SecretStr

from app.models import ModelError
from app.models.embeddings import (
    EmbeddingBatch, EmbeddingError, EmbeddingProfile, InputStrategy, canonical_sha256, vector_is_valid,
)
from app.security import BEARER_TOKEN

SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
EMBEDDING_URL = f"{SILICONFLOW_BASE_URL}/embeddings"
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIMENSIONS = 1024
CONNECTION_TEST_TEXT = "KnowledgeAgent 向量连接测试。"
MAX_BATCH_SIZE = 32
MAX_INPUT_TOKENS = 8192
REQUEST_TIMEOUT = 25
BATCH_TIMEOUT = 85
MAX_ATTEMPTS = 3


class EmbeddingResponseError(ValueError):
    pass


def response_vectors(payload: object, count: int) -> tuple[list[list[float]], list[int]]:
    if not isinstance(payload, dict) or payload.get("object") != "list" or payload.get("model") != EMBEDDING_MODEL:
        raise EmbeddingResponseError("response_identity")
    rows = payload.get("data")
    if not isinstance(rows, list) or len(rows) != count:
        raise EmbeddingResponseError("response_count")
    indexed = {}
    for row in rows:
        index = row.get("index") if isinstance(row, dict) else None
        if type(index) is not int or not 0 <= index < count or index in indexed:
            raise EmbeddingResponseError("response_index")
        if not vector_is_valid(row.get("embedding"), EMBEDDING_DIMENSIONS):
            raise EmbeddingResponseError("response_vector")
        indexed[index] = row["embedding"]
    return [indexed[index] for index in range(count)], list(indexed)


def revision_fields(response: httpx.Response, payload: dict) -> dict[str, str]:
    candidates = {
        "model_version": payload.get("model_version"),
        "system_fingerprint": payload.get("system_fingerprint"),
        "x-model-version": response.headers.get("x-model-version"),
        "x-siliconcloud-model-version": response.headers.get("x-siliconcloud-model-version"),
    }
    return {name: value for name, value in candidates.items() if isinstance(value, str) and value.strip()}


def validate_input(text: str) -> None:
    if not text.strip():
        raise EmbeddingError("EMBEDDING_EMPTY_INPUT", "向量输入必须包含非空白文本。")
    # The service tokenizer is opaque. Bound both original and compatibility-normalized
    # UTF-8 size, reserve two special tokens, and send the original string unchanged.
    byte_budget = MAX_INPUT_TOKENS - 2
    try:
        too_long = (len(text.encode("utf-8")) > byte_budget
                    or len(unicodedata.normalize("NFKC", text).encode("utf-8")) > byte_budget)
    except UnicodeEncodeError:
        raise EmbeddingError("EMBEDDING_INVALID_INPUT", "向量输入无法编码为有效的 UTF-8 文本。") from None
    if too_long:
        raise EmbeddingError("EMBEDDING_INPUT_TOO_LONG", "标题与正文超出本地长度预算，请重新分块或调整输入策略。")


def retry_after_seconds(value: str | None) -> float:
    if value is None:
        return 0
    value = value.strip()
    if re.fullmatch(r"[0-9]+", value):
        return float(value)
    try:
        when = parsedate_to_datetime(value)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
    except (ValueError, TypeError, OverflowError):
        return 0


class _RequestError(EmbeddingError):
    def __init__(self, code: str, message: str, *, retryable: bool = False, retry_after: float = 0,
                 http_status: int | None = None):
        super().__init__(code, message, retry_after=retry_after, http_status=http_status)
        self.retryable = retryable


def _http_error(response: httpx.Response) -> _RequestError:
    status = response.status_code

    def error(code: str, message: str, **options) -> _RequestError:
        return _RequestError(code, message, http_status=status, **options)

    if status == 401:
        return error("EMBEDDING_AUTH_FAILED", "SiliconFlow API Key 无效或已失效，请更新后重试。")
    if status == 403:
        return error("EMBEDDING_FORBIDDEN", "SiliconFlow 拒绝访问，请检查账户和模型权限。")
    if status == 429:
        return error("EMBEDDING_RATE_LIMITED", "SiliconFlow 请求达到限额，请稍后重试。",
                     retryable=True, retry_after=retry_after_seconds(response.headers.get("Retry-After")))
    retry_after = retry_after_seconds(response.headers.get("Retry-After")) if 500 <= status < 600 else 0
    if status == 504:
        return error("EMBEDDING_TIMEOUT", "连接 SiliconFlow 超时，请稍后重试。", retryable=True,
                     retry_after=retry_after)
    if status == 400:
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict) and body.get("code") == 20015:
            return error("EMBEDDING_INPUT_TOO_LONG", "SiliconFlow 拒绝超长输入，请重新分块或调整输入策略。")
    return error("EMBEDDING_SERVICE_ERROR", "SiliconFlow 服务返回错误，请稍后重试。",
                 retryable=500 <= status < 600, retry_after=retry_after)


def _usage(payload: dict, field: str) -> int | None:
    usage = payload.get("usage")
    value = usage.get(field) if isinstance(usage, dict) else None
    return value if type(value) is int and value >= 0 else None


class SiliconFlowEmbedding:
    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None):
        self._transport = transport
        self._key: SecretStr | None = None
        self._request_lock = asyncio.Lock()
        self._cooldown_until = 0.0

    @property
    def configured(self) -> bool:
        return self._key is not None

    def configure(self, key: SecretStr, base_url: str) -> None:
        value = key.get_secret_value()
        if base_url != SILICONFLOW_BASE_URL or not 1 <= len(value) <= 512 or not BEARER_TOKEN.fullmatch(value):
            raise ModelError("INVALID_REQUEST", "SiliconFlow 凭据或服务地址无效。")
        if key != self._key:
            self._cooldown_until = 0
        self._key = key

    def clear(self) -> None:
        self._key = None
        self._cooldown_until = 0

    @staticmethod
    def profile(*, compatibility_epoch: str, service_revision: str | None = None,
                input_strategy: InputStrategy = "title-body-v1") -> EmbeddingProfile:
        return EmbeddingProfile("siliconflow", SILICONFLOW_BASE_URL, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS,
                                compatibility_epoch, service_revision, input_strategy)

    def check_profile(self, profile: EmbeddingProfile) -> None:
        if profile != self.profile(compatibility_epoch=profile.compatibility_epoch,
                                   service_revision=profile.service_revision,
                                   input_strategy=profile.input_strategy_version):
            raise EmbeddingError("EMBEDDING_PROFILE_UNSUPPORTED", "该模型配置与 SiliconFlow 向量服务不匹配。")

    @asynccontextmanager
    async def open_client(self):
        async with httpx.AsyncClient(timeout=httpx.Timeout(15, connect=5, write=5, pool=2),
                                     trust_env=False, follow_redirects=False, transport=self._transport) as client:
            yield client

    async def _post(self, client: httpx.AsyncClient, inputs: str | list[str], *, timeout: float) -> httpx.Response:
        key = self._key
        if key is None:
            raise EmbeddingError("EMBEDDING_NOT_CONFIGURED", "请先保存 SiliconFlow API Key。")
        try:
            async with asyncio.timeout(timeout):
                response = await client.post(
                    EMBEDDING_URL, headers={"Authorization": f"Bearer {key.get_secret_value()}"},
                    json={"model": EMBEDDING_MODEL, "input": inputs, "encoding_format": "float"},
                )
        except (TimeoutError, httpx.TimeoutException):
            raise _RequestError("EMBEDDING_TIMEOUT", "连接 SiliconFlow 超时，请稍后重试。", retryable=True) from None
        except httpx.RequestError:
            raise _RequestError("EMBEDDING_NETWORK_ERROR", "无法连接 SiliconFlow，请检查网络后重试。") from None
        if response.status_code != 200:
            raise _http_error(response)
        return response

    @staticmethod
    def _parse(response: httpx.Response, count: int) -> tuple[dict, list[list[float]]]:
        try:
            payload = response.json()
            vectors, _ = response_vectors(payload, count)
        except ValueError:
            raise EmbeddingError("EMBEDDING_INVALID_RESPONSE", "SiliconFlow 返回的模型或向量无效。") from None
        return payload, vectors

    async def test_connection(self) -> int:
        try:
            async with asyncio.timeout(REQUEST_TIMEOUT), self._request_lock, self.open_client() as client:
                response = await self._post(client, CONNECTION_TEST_TEXT, timeout=REQUEST_TIMEOUT)
        except TimeoutError:
            raise EmbeddingError("EMBEDDING_TIMEOUT", "连接 SiliconFlow 超时，请稍后重试。") from None
        _, vectors = await asyncio.to_thread(self._parse, response, 1)
        return len(vectors[0])

    async def embed(self, client: httpx.AsyncClient, texts: list[str], *, profile: EmbeddingProfile) -> EmbeddingBatch:
        self.check_profile(profile)
        if not 1 <= len(texts) <= MAX_BATCH_SIZE:
            raise ValueError("Invalid embedding batch size.")
        for text in texts:
            validate_input(text)
        inputs = texts[0] if len(texts) == 1 else texts
        loop = asyncio.get_running_loop()
        deadline = loop.time() + BATCH_TIMEOUT
        try:
            async with asyncio.timeout_at(deadline), self._request_lock:
                response = await self._retry_batch(client, inputs, deadline)
        except TimeoutError:
            raise EmbeddingError("EMBEDDING_TIMEOUT", "SiliconFlow 批次超时，请稍后继续处理。") from None
        payload, vectors = await asyncio.to_thread(self._parse, response, len(texts))
        fields = revision_fields(response, payload)
        revision = canonical_sha256(fields) if fields else None
        if revision != profile.service_revision:
            raise EmbeddingError("EMBEDDING_REVISION_CHANGED", "SiliconFlow 模型版本标识变化，请切换配置后重建。",
                                 observed_revision=revision)
        return EmbeddingBatch(profile, vectors, _usage(payload, "prompt_tokens"), _usage(payload, "total_tokens"))

    async def _retry_batch(self, client: httpx.AsyncClient, inputs: str | list[str], deadline: float) -> httpx.Response:
        loop = asyncio.get_running_loop()
        delay = max(0.0, self._cooldown_until - loop.time())
        if delay >= deadline - loop.time():
            raise _RequestError("EMBEDDING_RATE_LIMITED", "SiliconFlow 正在冷却，请稍后继续处理。", retry_after=delay)
        if delay:
            await asyncio.sleep(delay)
        for attempt in range(MAX_ATTEMPTS):
            key = self._key
            try:
                return await self._post(client, inputs, timeout=min(REQUEST_TIMEOUT, deadline - loop.time()))
            except _RequestError as error:
                delay = max(error.retry_after, uniform(0, 0.5 * 2**attempt)) if error.retryable else 0
                if error.code == "EMBEDDING_RATE_LIMITED" and key == self._key:
                    self._cooldown_until = loop.time() + delay
                    error.retry_after = delay
                if not error.retryable or attempt + 1 == MAX_ATTEMPTS or delay >= deadline - loop.time():
                    raise
                await asyncio.sleep(delay)
        raise AssertionError("Invalid embedding retry budget.")
