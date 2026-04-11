import json
from unittest.mock import AsyncMock, patch

import pytest

from app.stream import ActivityStream


@pytest.fixture
def mock_kv():
    kv = AsyncMock()
    kv.put = AsyncMock(return_value=1)
    kv.get = AsyncMock(return_value=(None, None))
    return kv


@pytest.fixture
def stream(mock_kv):
    return ActivityStream(mock_kv)


# ── 1. record() stores event in KV store with correct key format ────────────

async def test_record_stores_event_with_correct_key_format(stream, mock_kv):
    await stream.record("agent1", "append", "session1", {"message_count": 1, "version": 1, "role": "user"})

    mock_kv.put.assert_called_once()
    kv_key = mock_kv.put.call_args[0][0]
    assert kv_key.startswith("event:agent1:")
    parts = kv_key.split(":")
    assert len(parts) == 4
    assert parts[0] == "event"
    assert parts[1] == "agent1"

    event_data = mock_kv.put.call_args[0][1]
    assert event_data["agent_id"] == "agent1"
    assert event_data["action"] == "append"
    assert event_data["session_id"] == "session1"
    assert event_data["metadata"]["role"] == "user"


# ── 2. record() adds key to in-memory index ────────────────────────────────

async def test_record_adds_key_to_index(stream, mock_kv):
    await stream.record("agent1", "append", "session1", {})

    assert len(stream._index["agent1"]) == 1
    key = stream._index["agent1"][0]
    assert key.startswith("event:agent1:")


# ── 3. get_events() returns events newest first ────────────────────────────

async def test_get_events_returns_newest_first(stream, mock_kv):
    event1 = {"event_id": "1", "agent_id": "a1", "action": "append", "session_id": "s1", "ts": 1000, "metadata": {}}
    event2 = {"event_id": "2", "agent_id": "a1", "action": "read", "session_id": "s1", "ts": 2000, "metadata": {}}
    event3 = {"event_id": "3", "agent_id": "a1", "action": "delete", "session_id": "s1", "ts": 3000, "metadata": {}}

    stream._index["a1"] = ["event:a1:1000:aaa", "event:a1:2000:bbb", "event:a1:3000:ccc"]

    mock_kv.get = AsyncMock(side_effect=[
        (event3, 1),  # newest first (reversed order)
        (event2, 1),
        (event1, 1),
    ])

    events = await stream.get_events("a1", limit=50)

    assert len(events) == 3
    assert events[0]["ts"] == 3000
    assert events[1]["ts"] == 2000
    assert events[2]["ts"] == 1000


# ── 4. get_events() respects limit parameter ───────────────────────────────

async def test_get_events_respects_limit(stream, mock_kv):
    stream._index["a1"] = [f"event:a1:{i}:x" for i in range(10)]

    event = {"event_id": "x", "agent_id": "a1", "action": "append", "session_id": "s1", "ts": 0, "metadata": {}}
    mock_kv.get = AsyncMock(return_value=(event, 1))

    events = await stream.get_events("a1", limit=3)

    assert len(events) == 3
    assert mock_kv.get.call_count == 3


# ── 5. filter_events() returns only events matching action ─────────────────

async def test_filter_events_by_action(stream, mock_kv):
    stream._index["a1"] = ["event:a1:1:a", "event:a1:2:b", "event:a1:3:c"]

    events_data = [
        ({"event_id": "3", "agent_id": "a1", "action": "delete", "session_id": "s1", "ts": 3, "metadata": {}}, 1),
        ({"event_id": "2", "agent_id": "a1", "action": "append", "session_id": "s1", "ts": 2, "metadata": {}}, 1),
        ({"event_id": "1", "agent_id": "a1", "action": "append", "session_id": "s1", "ts": 1, "metadata": {}}, 1),
    ]
    mock_kv.get = AsyncMock(side_effect=events_data)

    events = await stream.filter_events("a1", action="append", limit=20)

    assert len(events) == 2
    assert all(e["action"] == "append" for e in events)


# ── 6. record() does NOT raise if KV write fails (best-effort) ─────────────

async def test_record_does_not_raise_on_kv_failure(stream, mock_kv):
    mock_kv.put = AsyncMock(side_effect=Exception("KV down"))

    await stream.record("agent1", "append", "session1", {})
    # Should not raise — event is best-effort


# ── 7. get_events() returns empty list when index is empty (fresh start) ───

async def test_get_events_empty_on_fresh_start(stream, mock_kv):
    events = await stream.get_events("nonexistent_agent")

    assert events == []
    mock_kv.get.assert_not_called()
