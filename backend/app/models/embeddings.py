import hashlib
import json
import math
import struct
from dataclasses import dataclass
from typing import Literal

from app.models import ModelError

InputStrategy = Literal["title-body-v1", "body-v1"]
VECTOR_FORMAT = "float32-le"
KEY_VERSION = 1


class EmbeddingError(ModelError):
    def __init__(self, code: str, message: str, *, observed_revision: str | None = None,
                 retry_after: float = 0, http_status: int | None = None):
        super().__init__(code, message)
        self.observed_revision = observed_revision
        self.retry_after = retry_after
        self.http_status = http_status


def canonical_sha256(fields: dict) -> str:
    payload = json.dumps(fields, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class EmbeddingProfile:
    provider: str
    service_id: str
    model_id: str
    dimensions: int
    compatibility_epoch: str
    service_revision: str | None = None
    input_strategy_version: InputStrategy = "title-body-v1"

    def __post_init__(self) -> None:
        if (type(self.dimensions) is not int or self.dimensions < 1
                or not self.compatibility_epoch.strip()
                or self.input_strategy_version not in ("title-body-v1", "body-v1")):
            raise ValueError("Invalid embedding profile.")

    def _key_fields(self) -> dict:
        return {
            "key_version": KEY_VERSION,
            "provider": self.provider,
            "service_id": self.service_id,
            "model_id": self.model_id,
            "service_revision": self.service_revision,
            "compatibility_epoch": self.compatibility_epoch,
            "dimensions": self.dimensions,
            "input_strategy_version": self.input_strategy_version,
            "vector_format": VECTOR_FORMAT,
        }

    @property
    def key(self) -> str:
        return canonical_sha256(self._key_fields())

    def cache_key(self, input_sha256: str) -> str:
        return canonical_sha256({**self._key_fields(), "input_sha256": input_sha256})

    def build_input(self, body: str, title_context: tuple[str, ...]) -> str:
        if self.input_strategy_version == "title-body-v1" and title_context:
            return " > ".join(title_context) + "\n\n" + body
        return body


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    profile: EmbeddingProfile
    vectors: list[list[float]]
    prompt_tokens: int | None
    total_tokens: int | None


def _valid_values(vector) -> bool:
    try:
        return all(type(value) in (int, float) and math.isfinite(value) for value in vector) and (
            0 < math.hypot(*vector) < math.inf
        )
    except OverflowError:
        return False


def vector_is_valid(vector: object, dimensions: int) -> bool:
    return isinstance(vector, list) and len(vector) == dimensions and _valid_values(vector)


def decode_vector(blob: bytes, dimensions: int) -> tuple[float, ...]:
    if not isinstance(blob, bytes) or len(blob) != dimensions * 4:
        raise EmbeddingError("EMBEDDING_INVALID_VECTOR", "向量字节长度或编码无效。")
    values = struct.unpack(f"<{dimensions}f", blob)
    if not _valid_values(values):
        raise EmbeddingError("EMBEDDING_INVALID_VECTOR", "向量包含非有限数值或为零向量。")
    return values


def encode_vector(vector: list[float], dimensions: int) -> bytes:
    if not vector_is_valid(vector, dimensions):
        raise EmbeddingError("EMBEDDING_INVALID_VECTOR", "向量维度或数值无效。")
    try:
        blob = struct.pack(f"<{dimensions}f", *vector)
    except (OverflowError, struct.error):
        raise EmbeddingError("EMBEDDING_INVALID_VECTOR", "向量数值超出 float32 范围。") from None
    decode_vector(blob, dimensions)
    return blob
