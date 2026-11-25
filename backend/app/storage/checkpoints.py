from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.storage.business import BUSY_TIMEOUT_MS

CHECKPOINT_DATABASE_NAME = "checkpoints.sqlite3"


@asynccontextmanager
async def open_checkpointer(path: Path):
    async with aiosqlite.connect(path, timeout=BUSY_TIMEOUT_MS / 1000) as connection:
        async with connection.execute("PRAGMA journal_mode=WAL") as cursor:
            if await cursor.fetchone() != ("wal",):
                raise RuntimeError("Checkpoint SQLite WAL initialization failed")
        async with connection.execute("PRAGMA synchronous=FULL"):
            pass
        saver = AsyncSqliteSaver(connection, serde=JsonPlusSerializer(allowed_msgpack_modules=None))
        await saver.setup()
        yield saver
