import asyncio
import logging
import time

from app.kv_client import KVClient

logger = logging.getLogger(__name__)


class CleanupJob:
    """
    Background TTL cleanup for expired sessions.

    Problem: KV store has no TTL and no scan endpoint.
    Solution: Maintain a registry of known session keys. On each tick,
              GET each known key, check updated_at, DELETE if expired.

    Registry population: MemoryService calls register() on every
    append_message() and get_session() call. This means:
    - New sessions are always registered
    - Accessed sessions are always registered
    - Sessions from before the last service restart are NOT registered
      until they are accessed again

    This is an intentional limitation. Production solution: store session
    keys in a Redis SET or maintain a secondary index in the KV store itself
    using a known "index key" (e.g. "index:sessions:{agent_id}").

    The run_forever() coroutine is started as an asyncio.Task in lifespan
    and cancelled on shutdown.
    """

    def __init__(self, kv: KVClient, ttl_hours: int = 24):
        self._kv = kv
        self._ttl_seconds = ttl_hours * 3600
        self._registry: set[str] = set()
        self._interval = 60

    def register(self, key: str) -> None:
        """Register a session key for TTL tracking. Called by MemoryService."""
        self._registry.add(key)

    async def run_forever(self) -> None:
        """
        Main cleanup loop. Runs until cancelled.
        On each tick: check all registered keys, delete expired sessions.
        Logs: "TTL cleanup: checked N sessions, deleted M expired"
        """
        while True:
            await asyncio.sleep(self._interval)
            await self._cleanup_tick()

    async def _cleanup_tick(self) -> None:
        """Single cleanup pass. Safe to call manually in tests."""
        now = time.time()
        checked = 0
        deleted = 0
        expired_keys: list[str] = []

        for key in list(self._registry):
            checked += 1
            try:
                value, _ = await self._kv.get(key)
                if value is None:
                    expired_keys.append(key)
                    continue
                updated_at = value.get("updated_at", 0)
                if now - updated_at > self._ttl_seconds:
                    await self._kv.delete(key)
                    expired_keys.append(key)
                    deleted += 1
            except Exception:
                logger.warning("Cleanup error for key %s", key, exc_info=True)

        for key in expired_keys:
            self._registry.discard(key)

        logger.info("TTL cleanup: checked %d sessions, deleted %d expired", checked, deleted)
