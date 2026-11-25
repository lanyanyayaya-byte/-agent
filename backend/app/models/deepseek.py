import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass
from typing import Any

from pydantic import PrivateAttr, SecretStr

from app.models import ModelError

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


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
    """Map DeepSeek/OpenAI/httpx request exceptions into sanitized ModelError values.

    Returns None if the caller should re-raise `httpx.RemoteProtocolError` for tool-graph retry.
    """
    import httpx
    from openai import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError

    if isinstance(error, ModelError):
        return error

    remote_proto = _find_cause(error, httpx.RemoteProtocolError)
    if remote_proto is not None:
        if allow_remote_protocol_passthrough:
            return None
        return ModelError("MODEL_UPSTREAM_ERROR", "DeepSeek 响应流中断。")

    if _find_cause(error, httpx.ConnectTimeout) is not None:
        return ModelError("MODEL_TIMEOUT", "连接 DeepSeek 网络超时。")

    if isinstance(error, (TimeoutError, APITimeoutError, httpx.TimeoutException)):
        return ModelError("MODEL_TIMEOUT", "DeepSeek 响应超时。")

    if isinstance(error, AuthenticationError):
        return ModelError("MODEL_UPSTREAM_ERROR", "DeepSeek 认证失败，请检查 API Key。")

    if isinstance(error, (APIConnectionError, httpx.ConnectError)):
        return ModelError("MODEL_UPSTREAM_ERROR", "无法连接 DeepSeek。")

    if isinstance(error, APIStatusError):
        if error.status_code in (408, 504):
            return ModelError("MODEL_TIMEOUT", "DeepSeek 响应超时。")
        return ModelError("MODEL_UPSTREAM_ERROR", "DeepSeek 拒绝了本次请求。")

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
