import logging
import time
import uuid
from collections import defaultdict

from app.kv_client import KVClient

logger = logging.getLogger(__name__)


class ActivityStream:
    """
    Event log for agent memory operations.

    Every append, read, and delete in MemoryService automatically records
    an event here. Events are stored in the KV store AND indexed in memory.

    Key format: "event:{agent_id}:{timestamp_ms}:{short_uuid}"

    Event schema:
    {
        "event_id": str,
        "agent_id": str,
        "action": "append" | "read" | "delete",
        "session_id": str,
        "ts": int,           # unix timestamp ms
        "metadata": {
            "message_count": int | None,
            "version": int | None,
            "role": str | None
        }
    }

    In-memory index: dict[agent_id, list[str]] of KV event keys, sorted by
    insertion order (which is chronological since ts is in the key).

    IMPORTANT LIMITATION:
        The in-memory index is empty on service restart.
        Events written before the last restart are stored in the KV store
        but are NOT in the index. They cannot be retrieved via the API
        until the index is rebuilt, which only happens as new events come in.
        Production solution: Redis sorted sets keyed by timestamp, or a
        dedicated time-series store like InfluxDB or TimescaleDB.
        This is an intentional scope decision, not a bug.
    """

    def __init__(self, kv: KVClient):
        self._kv = kv
        self._index: dict[str, list[str]] = defaultdict(list)
        self._MAX_INDEX_SIZE = 1000

    async def record(self, agent_id: str, action: str, session_id: str, metadata: dict) -> None:
        """
        Record an event. Called automatically by MemoryService.
        Stores event in KV store and adds key to in-memory index.
        Never raises — if KV write fails, log warning and continue.
        """
        ts = int(time.time() * 1000)
        short_id = uuid.uuid4().hex[:8]
        event_id = f"{ts}-{short_id}"
        kv_key = f"event:{agent_id}:{ts}:{short_id}"

        event = {
            "event_id": event_id,
            "agent_id": agent_id,
            "action": action,
            "session_id": session_id,
            "ts": ts,
            "metadata": metadata,
        }

        try:
            await self._kv.put(kv_key, event)
            agent_keys = self._index[agent_id]
            agent_keys.append(kv_key)
            if len(agent_keys) > self._MAX_INDEX_SIZE:
                self._index[agent_id] = agent_keys[-self._MAX_INDEX_SIZE:]
        except Exception:
            logger.warning("Failed to record event for agent %s action %s", agent_id, action, exc_info=True)

    async def get_events(self, agent_id: str, limit: int = 50) -> list[dict]:
        """
        Get most recent events for an agent, newest first.
        Reads from KV store using keys from in-memory index.
        Returns empty list if no events recorded since last restart.
        """
        keys = self._index.get(agent_id, [])
        selected = list(reversed(keys[-limit:]))

        events = []
        for key in selected:
            value, _ = await self._kv.get(key)
            if value is not None:
                events.append(value)
        return events

    async def filter_events(self, agent_id: str, action: str | None = None, limit: int = 20) -> list[dict]:
        """
        Get filtered events by action type.
        Filters from the in-memory index — same restart limitation applies.
        """
        all_events = await self.get_events(agent_id, limit=self._MAX_INDEX_SIZE)
        if action:
            all_events = [e for e in all_events if e.get("action") == action]
        return all_events[:limit]
