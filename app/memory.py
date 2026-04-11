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
    Index key:   "index:{agent_id}"  — holds list of session_id strings for that agent

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

    Session index:
        A secondary index key "index:{agent_id}" stores the list of known
        session_ids for that agent. Updated on session create and delete.
        Uses the same optimistic concurrency pattern as session writes.
        Index writes are best-effort — they never fail the parent operation.
    """

    MAX_RETRIES = 3
    RETRY_SLEEP = 0.05

    def __init__(self, kv: KVClient, stream: ActivityStream, cleanup: CleanupJob):
        self._kv = kv
        self._stream = stream
        self._cleanup = cleanup

    def _key(self, agent_id: str, session_id: str) -> str:
        return f"mem:{agent_id}:{session_id}"

    def _index_key(self, agent_id: str) -> str:
        return f"index:{agent_id}"

    async def _add_to_index(self, agent_id: str, session_id: str) -> None:
        """
        Add session_id to the agent's session index.
        Uses optimistic concurrency — retries on conflict.
        Best-effort: never raises, logs warning on failure.
        """
        key = self._index_key(agent_id)
        for _ in range(self.MAX_RETRIES):
            try:
                existing, version = await self._kv.get(key)
                if existing is None:
                    sessions = []
                    expected_version = 1
                else:
                    sessions = existing.get("sessions", [])
                    expected_version = version + 1

                if session_id in sessions:
                    return

                sessions.append(session_id)
                new_version = await self._kv.put(key, {"agent_id": agent_id, "sessions": sessions})
                if new_version == expected_version:
                    return
                await asyncio.sleep(self.RETRY_SLEEP)
            except Exception:
                logger.warning("Failed to add session %s to index for agent %s", session_id, agent_id, exc_info=True)
                return

    async def _remove_from_index(self, agent_id: str, session_id: str) -> None:
        """
        Remove session_id from the agent's session index.
        Uses optimistic concurrency — retries on conflict.
        Best-effort: never raises.
        """
        key = self._index_key(agent_id)
        for _ in range(self.MAX_RETRIES):
            try:
                existing, version = await self._kv.get(key)
                if existing is None:
                    return
                sessions = existing.get("sessions", [])
                if session_id not in sessions:
                    return
                sessions = [s for s in sessions if s != session_id]
                expected_version = version + 1
                new_version = await self._kv.put(key, {"agent_id": agent_id, "sessions": sessions})
                if new_version == expected_version:
                    return
                await asyncio.sleep(self.RETRY_SLEEP)
            except Exception:
                logger.warning("Failed to remove session %s from index for agent %s", session_id, agent_id, exc_info=True)
                return

    async def list_sessions(self, agent_id: str) -> list[dict]:
        """
        List all known sessions for an agent, with summary metadata.
        Reads from the session index key, then fetches each session.
        Sessions created before the index existed are not listed.
        Returns list of dicts with session_id, message_count, created_at, updated_at, version.
        """
        key = self._index_key(agent_id)
        existing, _ = await self._kv.get(key)
        if existing is None:
            return []

        session_ids = existing.get("sessions", [])
        summaries = []
        for sid in session_ids:
            session_key = self._key(agent_id, sid)
            value, version = await self._kv.get(session_key)
            if value is not None:
                summaries.append({
                    "session_id": sid,
                    "message_count": value.get("message_count", 0),
                    "created_at": value.get("created_at", 0),
                    "updated_at": value.get("updated_at", 0),
                    "version": version,
                })
        return summaries

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

                if existing is None:
                    await self._add_to_index(agent_id, session_id)

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
            await self._remove_from_index(agent_id, session_id)
            try:
                await self._stream.record(agent_id, "delete", session_id, {
                    "message_count": None,
                    "version": None,
                    "role": None,
                })
            except Exception:
                logger.warning("Failed to record delete event", exc_info=True)

        return deleted
