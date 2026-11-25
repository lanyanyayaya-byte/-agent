import asyncio
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass

from pydantic import SecretStr

from app.models import ModelError

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


@dataclass(frozen=True)
class DeepSeekSettings:
    model: str = "deepseek-flash"
    base_url: str = DEEPSEEK_BASE_URL
    timeout_seconds: float = 25
    max_tokens: int = 1024
    # Identity prompt for plain chat. None keeps requests prompt-free (used by fixture settings).
    persona: str | None = None

    @classmethod
    def from_environment(cls) -> "DeepSeekSettings":
        from app.agent.persona import DEFAULT_ASSISTANT_PERSONA

        return cls(
            model=os.environ.get("KNOWLEDGEAGENT_DEEPSEEK_MODEL", cls.model).strip(),
            timeout_seconds=120,
            max_tokens=8192,
            persona=DEFAULT_ASSISTANT_PERSONA,
        )


class DeepSeekAdapter:
    def __init__(self, settings: DeepSeekSettings, *, transport=None):
        self.settings = settings
        self._transport = transport

    @asynccontextmanager
    async def model(self, key: SecretStr):
        import httpx
        from langchain_deepseek import ChatDeepSeek
        from openai import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError

        settings = self.settings
        options = {"trust_env": False, "follow_redirects": False, "timeout": settings.timeout_seconds}
        try:
            async with asyncio.timeout(settings.timeout_seconds):
                with httpx.Client(**options) as sync_client:
                    async with httpx.AsyncClient(transport=self._transport, **options) as async_client:
                        yield ChatDeepSeek(
                            model=settings.model,
                            base_url=settings.base_url,
                            api_key=key,
                            timeout=settings.timeout_seconds,
                            max_tokens=settings.max_tokens,
                            max_retries=0,
                            use_responses_api=False,
                            extra_body={"thinking": {"type": "disabled"}},
                            http_client=sync_client,
                            http_async_client=async_client,
                            cache=False,
                            verbose=False,
                        )
        except (TimeoutError, APITimeoutError, httpx.TimeoutException):
            raise ModelError("MODEL_TIMEOUT", "DeepSeek 响应超时。") from None
        except AuthenticationError:
            raise ModelError("MODEL_UPSTREAM_ERROR", "DeepSeek 认证失败，请检查 API Key。") from None
        except (APIConnectionError, httpx.RemoteProtocolError) as error:
            cause: BaseException | None = error
            while cause is not None and not isinstance(cause, httpx.RemoteProtocolError):
                cause = cause.__cause__
            message = "DeepSeek 响应流中断。" if cause is not None else "无法连接 DeepSeek。"
            raise ModelError("MODEL_UPSTREAM_ERROR", message) from None
        except APIStatusError as error:
            if error.status_code in (408, 504):
                raise ModelError("MODEL_TIMEOUT", "DeepSeek 响应超时。") from None
            raise ModelError("MODEL_UPSTREAM_ERROR", "DeepSeek 拒绝了本次请求。") from None
        except ModelError:
            raise
        except Exception:
            raise ModelError("MODEL_UPSTREAM_ERROR", "DeepSeek 调用失败或响应无效。") from None
