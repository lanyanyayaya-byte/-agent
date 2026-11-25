import asyncio
from contextlib import suppress


async def run_index_sync(operation, *args, **kwargs):
    """Drain a started synchronous operation before the attempt releases its slot."""
    pending = asyncio.create_task(asyncio.to_thread(operation, *args, **kwargs))
    try:
        return await asyncio.shield(pending)
    except asyncio.CancelledError:
        with suppress(Exception):
            await pending
        raise
