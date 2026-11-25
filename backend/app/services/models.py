from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from pydantic import SecretStr

from app.models import ModelError
from app.agent.answer import answer_with_agent
from app.models.deepseek import DeepSeekAdapter, DeepSeekSettings
from app.security import BEARER_TOKEN

if TYPE_CHECKING:
    from langchain_deepseek import ChatDeepSeek


class ModelService:
    def __init__(self, adapter: DeepSeekAdapter | None = None):
        self._adapter = adapter or DeepSeekAdapter(DeepSeekSettings.from_environment())
        self._key: SecretStr | None = None

    @property
    def persona(self) -> str | None:
        return getattr(self._adapter.settings, "persona", None)

    @property
    def configured(self) -> bool:
        return self._key is not None

    def configure_deepseek(self, key: SecretStr, base_url: str) -> None:
        settings = self._adapter.settings
        value = key.get_secret_value()
        if (
            base_url != settings.base_url
            or not settings.model
            or not 1 <= len(value) <= 512
            or not BEARER_TOKEN.fullmatch(value)
        ):
            raise ModelError("INVALID_REQUEST", "DeepSeek 凭据或模型配置无效。")
        self._key = key

    def clear_deepseek(self) -> None:
        self._key = None

    async def answer(self, message: str, *, on_delta: Callable[[str], None] | None = None) -> str:
        async with self.open_model() as model:
            return (await answer_with_agent(model, message, on_delta=on_delta)).text

    @asynccontextmanager
    async def open_model(self) -> AsyncIterator["ChatDeepSeek"]:
        key = self._key
        if key is None:
            raise ModelError("MODEL_NOT_CONFIGURED", "请先设置 DeepSeek API Key。")
        async with self._adapter.model(key) as model:
            yield model
