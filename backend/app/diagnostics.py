import json
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.storage.business import utc_now

LOGGER = logging.getLogger("knowledgeagent.sidecar")
MAX_LOG_BYTES = 1_048_576


def configure_log(data_dir: Path) -> None:
    log_dir = data_dir / "logs"
    log_dir.mkdir(mode=0o700, exist_ok=True)
    handler = RotatingFileHandler(
        log_dir / "sidecar.jsonl", maxBytes=MAX_LOG_BYTES, backupCount=2, encoding="utf-8",
    )
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


def record(event: str, **fields: object) -> None:
    # Callers supply only normalized operation names, IDs, timings and error codes.
    LOGGER.info(json.dumps({"ts": utc_now(), "pid": os.getpid(), "event": event, **fields}))
