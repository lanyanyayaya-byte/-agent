class ModelError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class RetryableModelError(ModelError):
    """Transient upstream/network/rate-limit/timeout model error eligible for retry."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        reason_kind: str = "transient",
        status_code: int | None = None,
        prefer_non_streaming: bool = False,
    ):
        super().__init__(code, message)
        self.reason_kind = reason_kind
        self.status_code = status_code
        self.prefer_non_streaming = prefer_non_streaming
