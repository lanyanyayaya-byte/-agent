"""SiliconFlow BAAI/bge-reranker-v2-m3 client reusing the existing SiliconFlow API Key."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

import httpx

from app.models import ModelError
from app.models.siliconflow import SILICONFLOW_BASE_URL, SiliconFlowEmbedding

RERANK_URL = f"{SILICONFLOW_BASE_URL}/rerank"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_RERANK_TIMEOUT = 20.0
MAX_DOC_CHARS = 2400
MAX_RERANK_ATTEMPTS = 2


class RerankError(ModelError):
    def __init__(self, code: str, message: str, *, http_status: int | None = None, retryable: bool = False):
        super().__init__(code, message)
        self.http_status = http_status
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class RerankItem:
    index: int
    relevance_score: float


@dataclass(frozen=True, slots=True)
class RerankResult:
    model: str
    items: tuple[RerankItem, ...]
    elapsed_ms: float
    input_tokens: int | None = None


def _map_http_error(response: httpx.Response) -> RerankError:
    status = response.status_code
    if status == 401:
        return RerankError("RERANK_AUTH_FAILED", "SiliconFlow API Key 无效或已失效（401）。", http_status=status)
    if status == 403:
        return RerankError("RERANK_FORBIDDEN", "SiliconFlow 拒绝访问 Reranker（403）。", http_status=status)
    if status == 429:
        return RerankError("RERANK_RATE_LIMITED", "SiliconFlow Reranker 请求限流（429）。", http_status=status, retryable=True)
    if status == 504:
        return RerankError("RERANK_TIMEOUT", "SiliconFlow Reranker 请求超时（504）。", http_status=status, retryable=True)
    return RerankError(
        "RERANK_SERVICE_ERROR",
        f"SiliconFlow Reranker 返回 HTTP {status}。",
        http_status=status,
        retryable=500 <= status < 600,
    )


class SiliconFlowReranker:
    """Thin multilingual reranker backed by the same SiliconFlow credential."""

    def __init__(
        self,
        embedding_provider: SiliconFlowEmbedding,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        model: str = RERANK_MODEL,
    ):
        self._provider = embedding_provider
        self._transport = transport
        self.model = model

    @property
    def configured(self) -> bool:
        return self._provider.configured

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_n: int | None = None,
        timeout: float = DEFAULT_RERANK_TIMEOUT,
    ) -> RerankResult:
        if not query.strip() or not documents:
            return RerankResult(model=self.model, items=(), elapsed_ms=0.0)
        key = self._provider._key
        if key is None:
            raise RerankError("RERANK_NOT_CONFIGURED", "SiliconFlow API Key 未配置，无法执行语义重排序。")

        trimmed_docs = [doc[:MAX_DOC_CHARS] for doc in documents]
        effective_top_n = len(trimmed_docs) if top_n is None else max(1, min(top_n, len(trimmed_docs)))
        payload = {
            "model": self.model,
            "query": query[:1024],
            "documents": trimmed_docs,
            "top_n": effective_top_n,
            "return_documents": False,
        }
        headers = {"Authorization": f"Bearer {key.get_secret_value()}"}

        started = perf_counter()
        last_error: RerankError | None = None
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=5.0, write=5.0, pool=2.0),
            trust_env=False,
            follow_redirects=False,
            transport=self._transport,
        ) as client:
            for attempt in range(MAX_RERANK_ATTEMPTS):
                try:
                    async with asyncio.timeout(timeout):
                        response = await client.post(RERANK_URL, headers=headers, json=payload)
                except (TimeoutError, httpx.TimeoutException):
                    last_error = RerankError("RERANK_TIMEOUT", "SiliconFlow Reranker 请求超时。", retryable=True)
                except httpx.RequestError:
                    last_error = RerankError("RERANK_NETWORK_ERROR", "无法连接 SiliconFlow Reranker 服务。", retryable=True)
                else:
                    if response.status_code == 200:
                        return self._parse(response, len(trimmed_docs), (perf_counter() - started) * 1000.0)
                    last_error = _map_http_error(response)

                if not last_error.retryable or attempt + 1 >= MAX_RERANK_ATTEMPTS:
                    raise last_error
                await asyncio.sleep(0.4 * (attempt + 1))

        assert last_error is not None
        raise last_error

    def _parse(self, response: httpx.Response, doc_count: int, elapsed_ms: float) -> RerankResult:
        try:
            body = response.json()
            raw_results = body.get("results") if isinstance(body, dict) else None
            if not isinstance(raw_results, list):
                raise ValueError("missing results list")
            items: list[RerankItem] = []
            seen: set[int] = set()
            for entry in raw_results:
                if not isinstance(entry, dict):
                    raise ValueError("invalid item")
                idx = entry.get("index")
                score = entry.get("relevance_score")
                if type(idx) is not int or not 0 <= idx < doc_count or idx in seen:
                    raise ValueError("invalid index")
                if type(score) not in (int, float):
                    raise ValueError("invalid score")
                seen.add(idx)
                items.append(RerankItem(index=idx, relevance_score=float(score)))
            items.sort(key=lambda x: (-x.relevance_score, x.index))
            input_tokens = None
            meta = body.get("meta")
            if isinstance(meta, dict) and isinstance(meta.get("tokens"), dict):
                tok = meta["tokens"].get("input_tokens")
                if type(tok) is int:
                    input_tokens = tok
            return RerankResult(
                model=self.model,
                items=tuple(items),
                elapsed_ms=round(elapsed_ms, 1),
                input_tokens=input_tokens,
            )
        except ValueError:
            raise RerankError("RERANK_INVALID_RESPONSE", "SiliconFlow Reranker 返回格式无效。") from None
