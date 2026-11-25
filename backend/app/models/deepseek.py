import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass
from typing import Any

from pydantic import PrivateAttr, SecretStr

from app.diagnostics import record
from app.models import ModelError, RetryableModelError

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_CONTEXT_LIMIT_HINTS = (
    "context length",
    "maximum context",
    "too many tokens",
    "token limit",
    "context_length_exceeded",
    "max_prompt_tokens",
)


@dataclass(frozen=True)
class DeepSeekSettings:
    model: str = "deepseek-flash"
    base_url: str = DEEPSEEK_BASE_URL
    timeout_seconds: float = 25
    connect_timeout_seconds: float | None = None
    max_tokens: int = 1024
    # Identity prompt for plain chat. None keeps requests prompt-free (used by fixture settings).
    persona: str | None = None

    @classmethod
    def from_environment(cls) -> "DeepSeekSettings":
        from app.agent.persona import DEFAULT_ASSISTANT_PERSONA

        return cls(
            model=os.environ.get("KNOWLEDGEAGENT_DEEPSEEK_MODEL", cls.model).strip(),
            timeout_seconds=120,
            connect_timeout_seconds=15,
            max_tokens=8192,
            persona=DEFAULT_ASSISTANT_PERSONA,
        )


def _find_cause(error: BaseException, target_type: type[BaseException]) -> BaseException | None:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        if isinstance(current, target_type):
            return current
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None


def _translate_deepseek_error(
    error: Exception,
    *,
    allow_remote_protocol_passthrough: bool = False,
) -> ModelError | None:
    """Map DeepSeek/OpenAI/httpx request exceptions into classified, sanitized ModelError values.

    - Transient errors (network/timeout/stream-disconnect/HTTP 429/408/5xx) map to `RetryableModelError`.
    - Permanent errors (HTTP 401/402/400/422, context limit) map to non-retryable `ModelError`.
    - Returns None if caller should re-raise `httpx.RemoteProtocolError` for stream retry.
    """
    import httpx
    from openai import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError

    if isinstance(error, ModelError):
        return error

    remote_proto = _find_cause(error, httpx.RemoteProtocolError)
    if remote_proto is not None:
        record("deepseek_api_error", exception_type="RemoteProtocolError", reason_kind="stream_disconnect", retryable=True)
        if allow_remote_protocol_passthrough:
            return None
        return RetryableModelError(
            "MODEL_UPSTREAM_ERROR",
            "DeepSeek 响应流中断（网络连接波动）。",
            reason_kind="stream_disconnect",
            prefer_non_streaming=True,
        )

    if _find_cause(error, httpx.ConnectTimeout) is not None:
        record("deepseek_api_error", exception_type="ConnectTimeout", reason_kind="timeout", retryable=True)
        return RetryableModelError("MODEL_TIMEOUT", "连接 DeepSeek 网络超时。", reason_kind="timeout")

    if isinstance(error, (TimeoutError, APITimeoutError, httpx.TimeoutException)):
        record("deepseek_api_error", exception_type=type(error).__name__, reason_kind="timeout", retryable=True)
        return RetryableModelError("MODEL_TIMEOUT", "DeepSeek 响应超时。", reason_kind="timeout")

    if isinstance(error, AuthenticationError):
        record("deepseek_api_error", exception_type=type(error).__name__, status_code=401, reason_kind="auth", retryable=False)
        return ModelError("MODEL_UPSTREAM_ERROR", "DeepSeek 认证失败（HTTP 401），请检查 API Key。")

    if isinstance(error, (APIConnectionError, httpx.ConnectError, httpx.ReadError)):
        record("deepseek_api_error", exception_type=type(error).__name__, reason_kind="network", retryable=True)
        return RetryableModelError("MODEL_UPSTREAM_ERROR", "无法连接 DeepSeek 网络服务，请检查网络或代理设置。", reason_kind="network")

    if isinstance(error, APIStatusError):
        status = getattr(error, "status_code", None)
        body_text = str(getattr(error, "body", "") or str(error)).lower()
        if status == 401:
            record("deepseek_api_error", exception_type="APIStatusError", status_code=401, reason_kind="auth", retryable=False)
            return ModelError("MODEL_UPSTREAM_ERROR", "DeepSeek 认证失败（HTTP 401），请检查 API Key。")
        if status == 402:
            record("deepseek_api_error", exception_type="APIStatusError", status_code=402, reason_kind="insufficient_balance", retryable=False)
            return ModelError("MODEL_UPSTREAM_ERROR", "DeepSeek 账户余额不足（HTTP 402），请检查账户额度。")
        if status == 429:
            record("deepseek_api_error", exception_type="APIStatusError", status_code=429, reason_kind="rate_limit", retryable=True)
            return RetryableModelError(
                "MODEL_UPSTREAM_ERROR",
                "DeepSeek 上游接口触发限流（HTTP 429），请稍后重试。",
                reason_kind="rate_limit",
                status_code=429,
            )
        if status in (408, 504):
            record("deepseek_api_error", exception_type="APIStatusError", status_code=status, reason_kind="timeout", retryable=True)
            return RetryableModelError(
                "MODEL_TIMEOUT",
                f"DeepSeek 响应超时（HTTP {status}）。",
                reason_kind="timeout",
                status_code=status,
            )
        if status in (500, 502, 503):
            record("deepseek_api_error", exception_type="APIStatusError", status_code=status, reason_kind="server_error", retryable=True)
            return RetryableModelError(
                "MODEL_UPSTREAM_ERROR",
                f"DeepSeek 服务端暂时不可用（HTTP {status}），请稍后重试。",
                reason_kind="server_error",
                status_code=status,
            )
        if status in (400, 422):
            is_context_limit = any(hint in body_text for hint in _CONTEXT_LIMIT_HINTS)
            reason = "context_length_exceeded" if is_context_limit else "invalid_request"
            record("deepseek_api_error", exception_type="APIStatusError", status_code=status, reason_kind=reason, retryable=False)
            if is_context_limit:
                return ModelError(
                    "MODEL_UPSTREAM_ERROR",
                    f"当前对话上下文超过 DeepSeek 最大长度限制（HTTP {status}），请开启新会话或精简输入。",
                )
            return ModelError(
                "MODEL_UPSTREAM_ERROR",
                f"DeepSeek 拒绝了本次请求格式（HTTP {status}），请检查输入内容或参数后重试。",
            )
        record("deepseek_api_error", exception_type="APIStatusError", status_code=status, reason_kind="upstream_status", retryable=False)
        return ModelError("MODEL_UPSTREAM_ERROR", f"DeepSeek 拒绝了本次请求（HTTP {status}）。")

    record("deepseek_api_error", exception_type=type(error).__name__, reason_kind="unknown", retryable=False)
    return ModelError("MODEL_UPSTREAM_ERROR", "DeepSeek 调用失败或响应无效。")


class DeepSeekAdapter:
    def __init__(self, settings: DeepSeekSettings, *, transport=None):
        self.settings = settings
        self._transport = transport

    @asynccontextmanager
    async def model(self, key: SecretStr):
        import httpx
        from langchain_deepseek import ChatDeepSeek

        settings = self.settings
        per_request_timeout = settings.timeout_seconds
        connect_timeout = (
            settings.connect_timeout_seconds
            if settings.connect_timeout_seconds is not None
            else min(per_request_timeout, 15.0)
        )
        http_timeout = httpx.Timeout(
            timeout=per_request_timeout,
            connect=connect_timeout,
        )
        options: dict[str, Any] = {
            "trust_env": False,
            "follow_redirects": False,
            "timeout": http_timeout,
        }

        class _ScopedChatDeepSeek(ChatDeepSeek):
            """Applies the configured timeout strictly per LLM request rather than per whole Run."""

            _per_request_timeout: float = PrivateAttr(default=per_request_timeout)

            async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
                try:
                    async with asyncio.timeout(self._per_request_timeout):
                        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    mapped = _translate_deepseek_error(error, allow_remote_protocol_passthrough=False)
                    assert mapped is not None
                    raise mapped from None

            async def _astream(
                self,
                messages,
                stop=None,
                run_manager=None,
                *,
                stream_usage=None,
                **kwargs,
            ) -> AsyncIterator[Any]:
                try:
                    async with asyncio.timeout(self._per_request_timeout):
                        async with aclosing(
                            super()._astream(
                                messages,
                                stop=stop,
                                run_manager=run_manager,
                                stream_usage=stream_usage,
                                **kwargs,
                            )
                        ) as stream:
                            async for chunk in stream:
                                yield chunk
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    mapped = _translate_deepseek_error(error, allow_remote_protocol_passthrough=True)
                    if mapped is None:
                        remote_proto = _find_cause(error, httpx.RemoteProtocolError)
                        assert isinstance(remote_proto, httpx.RemoteProtocolError)
                        raise remote_proto from None
                    raise mapped from None

        try:
            with httpx.Client(**options) as sync_client:
                async with httpx.AsyncClient(transport=self._transport, **options) as async_client:
                    yield _ScopedChatDeepSeek(
                        model=settings.model,
                        base_url=settings.base_url,
                        api_key=key,
                        timeout=per_request_timeout,
                        max_tokens=settings.max_tokens,
                        max_retries=0,
                        use_responses_api=False,
                        extra_body={"thinking": {"type": "disabled"}},
                        http_client=sync_client,
                        http_async_client=async_client,
                        cache=False,
                        verbose=False,
                    )
        except asyncio.CancelledError:
            raise
        except ModelError:
            raise
        except httpx.RemoteProtocolError:
            # Unhandled stream disconnect (plain chat or non-retryable turn).
            raise ModelError("MODEL_UPSTREAM_ERROR", "DeepSeek 响应流中断。") from None
