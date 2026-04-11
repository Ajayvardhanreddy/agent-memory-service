import asyncio
import logging
import time

from app.kv_client import KVClient
from app.stream import ActivityStream
from app.cleanup import CleanupJob

logger = logging.getLogger(__name__)


class SessionNotFoundError(Exception):
    pass


class VersionConflictError(Exception):
    def __init__(self, session_id: str, retries: int):
        super().__init__(f"Version conflict on session {session_id} after {retries} retries")
        self.session_id = session_id
        self.retries = retries


class MemoryService:
    """
    Session memory for AI agents. Backed by the distributed KV store.

    Key format:  "mem:{agent_id}:{session_id}"

    Value schema:
    {
        "session_id": str,
        "agent_id": str,
        "messages": [{"role": str, "content": str, "ts": int}],
        "message_count": int,
        "created_at": int,   # unix timestamp
        "updated_at": int    # unix timestamp
    }

    Version conflict handling:
        Uses optimistic concurrency. On append, reads current version N,
        writes updated session, checks returned version == N+1.
        If not, another writer raced us — retry up to MAX_RETRIES times.

    Limitations (documented intentionally):
        - No session listing: KV store has no scan endpoint.
          Production would use Redis SCAN or a secondary index.
        - TTL via background job: sessions persist across service restarts
          but cleanup only runs while service is live.
    """

    MAX_RETRIES = 3
    RETRY_SLEEP = 0.05

    def __init__(self, kv: KVClient, stream: ActivityStream, cleanup: CleanupJob):
        self._kv = kv
        self._stream = stream
        self._cleanup = cleanup

    def _key(self, agent_id: str, session_id: str) -> str:
        return f"mem:{agent_id}:{session_id}"

    async def append_message(self, agent_id: str, session_id: str, role: str, content: str) -> dict:
        """
        Append a message to a session. Creates session if it doesn't exist.

        Uses optimistic concurrency control:
        1. GET current session (version=N). If not found, start fresh (version=0).
        2. Append new message to messages list.
        3. PUT updated session.
        4. Returned version should be N+1. If not, another writer raced us.
        5. On conflict: sleep RETRY_SLEEP, retry from step 1.
        6. After MAX_RETRIES failures: raise VersionConflictError.

        On success: register key with cleanup job, record event in stream.
        Returns: updated session dict (includes current version).
        """
        key = self._key(agent_id, session_id)
        now = int(time.time())

        for attempt in range(self.MAX_RETRIES):
            existing, version = await self._kv.get(key)

            if existing is None:
                session = {
                    "session_id": session_id,
                    "agent_id": agent_id,
                    "messages": [],
                    "message_count": 0,
                    "created_at": now,
                    "updated_at": now,
                }
                expected_version = 1
            else:
                session = existing
                expected_version = version + 1

            session["messages"].append({"role": role, "content": content, "ts": now})
            session["message_count"] = len(session["messages"])
            session["updated_at"] = now

            new_version = await self._kv.put(key, session)

            if new_version == expected_version:
                session["version"] = new_version
                self._cleanup.register(key)

                try:
                    await self._stream.record(agent_id, "append", session_id, {
                        "message_count": session["message_count"],
                        "version": new_version,
                        "role": role,
                    })
                except Exception:
                    logger.warning("Failed to record append event", exc_info=True)

                return session

            logger.warning(
                "Version conflict on %s: expected %d, got %d (attempt %d/%d)",
                key, expected_version, new_version, attempt + 1, self.MAX_RETRIES,
            )
            await asyncio.sleep(self.RETRY_SLEEP)

        raise VersionConflictError(session_id, self.MAX_RETRIES)

    async def get_session(self, agent_id: str, session_id: str) -> dict:
        """
        Get full session. Records a 'read' event in the activity stream.
        Raises SessionNotFoundError if session does not exist.
        """
        key = self._key(agent_id, session_id)
        value, version = await self._kv.get(key)

        if value is None:
            raise SessionNotFoundError(f"Session {session_id} not found for agent {agent_id}")

        value["version"] = version
        self._cleanup.register(key)

        try:
            await self._stream.record(agent_id, "read", session_id, {
                "message_count": value.get("message_count"),
                "version": version,
                "role": None,
            })
        except Exception:
            logger.warning("Failed to record read event", exc_info=True)

        return value

    async def get_window(self, agent_id: str, session_id: str, last_n: int = 10) -> dict:
        """
        Get last N messages from session.
        Returns: {"session_id": str, "messages": list, "total_messages": int, "window_size": int}
        Raises SessionNotFoundError if session does not exist.
        """
        key = self._key(agent_id, session_id)
        value, _ = await self._kv.get(key)

        if value is None:
            raise SessionNotFoundError(f"Session {session_id} not found for agent {agent_id}")

        messages = value.get("messages", [])
        window = messages[-last_n:]

        return {
            "session_id": session_id,
            "messages": window,
            "total_messages": len(messages),
            "window_size": len(window),
        }

    async def delete_session(self, agent_id: str, session_id: str) -> bool:
        """
        Delete a session. Records a 'delete' event in the activity stream.
        Returns True if deleted, False if not found.
        """
        key = self._key(agent_id, session_id)
        deleted = await self._kv.delete(key)

        if deleted:
            try:
                await self._stream.record(agent_id, "delete", session_id, {
                    "message_count": None,
                    "version": None,
                    "role": None,
                })
            except Exception:
                logger.warning("Failed to record delete event", exc_info=True)

        return deleted
