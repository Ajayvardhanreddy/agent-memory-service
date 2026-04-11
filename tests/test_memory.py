import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.memory import MemoryService, SessionNotFoundError, VersionConflictError


@pytest.fixture
def mock_kv():
    kv = AsyncMock()
    kv.get = AsyncMock(return_value=(None, None))
    kv.put = AsyncMock(return_value=1)
    kv.delete = AsyncMock(return_value=True)
    return kv


@pytest.fixture
def mock_stream():
    stream = AsyncMock()
    stream.record = AsyncMock()
    return stream


@pytest.fixture
def mock_cleanup():
    cleanup = MagicMock()
    cleanup.register = MagicMock()
    return cleanup


@pytest.fixture
def memory(mock_kv, mock_stream, mock_cleanup):
    svc = MemoryService(mock_kv, mock_stream, mock_cleanup)
    svc.RETRY_SLEEP = 0  # no waiting in tests
    return svc


# ── 1. append_message() creates new session when key not found ──────────────

async def test_append_creates_new_session(memory, mock_kv):
    mock_kv.get = AsyncMock(return_value=(None, None))
    mock_kv.put = AsyncMock(return_value=1)

    result = await memory.append_message("agent1", "session1", "user", "hello")

    assert result["session_id"] == "session1"
    assert result["agent_id"] == "agent1"
    assert result["message_count"] == 1
    assert result["version"] == 1
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][0]["content"] == "hello"


# ── 2. append_message() increments message_count on each call ───────────────

async def test_append_increments_message_count(memory, mock_kv):
    existing = {
        "session_id": "s1",
        "agent_id": "a1",
        "messages": [{"role": "user", "content": "first", "ts": 100}],
        "message_count": 1,
        "created_at": 100,
        "updated_at": 100,
    }
    mock_kv.get = AsyncMock(return_value=(existing, 1))
    mock_kv.put = AsyncMock(return_value=2)

    result = await memory.append_message("a1", "s1", "assistant", "reply")

    assert result["message_count"] == 2
    assert len(result["messages"]) == 2
    assert result["version"] == 2


# ── 3. append_message() sets created_at on first call only ──────────────────

async def test_append_preserves_created_at(memory, mock_kv):
    existing = {
        "session_id": "s1",
        "agent_id": "a1",
        "messages": [{"role": "user", "content": "hi", "ts": 100}],
        "message_count": 1,
        "created_at": 42,
        "updated_at": 100,
    }
    mock_kv.get = AsyncMock(return_value=(existing, 1))
    mock_kv.put = AsyncMock(return_value=2)

    result = await memory.append_message("a1", "s1", "assistant", "hey")

    assert result["created_at"] == 42
    assert result["updated_at"] != 42


# ── 4. get_window() returns only last N messages ───────────────────────────

async def test_get_window_returns_last_n(memory, mock_kv):
    messages = [{"role": "user", "content": f"msg{i}", "ts": i} for i in range(10)]
    session = {
        "session_id": "s1",
        "agent_id": "a1",
        "messages": messages,
        "message_count": 10,
        "created_at": 0,
        "updated_at": 9,
    }
    mock_kv.get = AsyncMock(return_value=(session, 10))

    result = await memory.get_window("a1", "s1", last_n=3)

    assert result["total_messages"] == 10
    assert result["window_size"] == 3
    assert len(result["messages"]) == 3
    assert result["messages"][0]["content"] == "msg7"
    assert result["messages"][2]["content"] == "msg9"


# ── 5. get_window() returns all messages when last_n > message_count ───────

async def test_get_window_returns_all_when_less_than_n(memory, mock_kv):
    messages = [{"role": "user", "content": f"msg{i}", "ts": i} for i in range(3)]
    session = {
        "session_id": "s1",
        "agent_id": "a1",
        "messages": messages,
        "message_count": 3,
        "created_at": 0,
        "updated_at": 2,
    }
    mock_kv.get = AsyncMock(return_value=(session, 3))

    result = await memory.get_window("a1", "s1", last_n=100)

    assert result["total_messages"] == 3
    assert result["window_size"] == 3


# ── 6. get_session() raises SessionNotFoundError when key not found ─────────

async def test_get_session_raises_not_found(memory, mock_kv):
    mock_kv.get = AsyncMock(return_value=(None, None))

    with pytest.raises(SessionNotFoundError):
        await memory.get_session("a1", "nonexistent")


# ── 7. delete_session() returns True when key exists, False when not ────────

async def test_delete_returns_true_when_found(memory, mock_kv):
    mock_kv.delete = AsyncMock(return_value=True)
    assert await memory.delete_session("a1", "s1") is True


async def test_delete_returns_false_when_not_found(memory, mock_kv):
    mock_kv.delete = AsyncMock(return_value=False)
    assert await memory.delete_session("a1", "missing") is False


# ── 8. append_message() retries on version conflict ─────────────────────────

async def test_append_retries_on_version_conflict(memory, mock_kv):
    existing = {
        "session_id": "s1",
        "agent_id": "a1",
        "messages": [{"role": "user", "content": "hi", "ts": 100}],
        "message_count": 1,
        "created_at": 100,
        "updated_at": 100,
    }
    # First two puts return wrong version (conflict), third succeeds
    mock_kv.get = AsyncMock(return_value=(existing, 1))
    mock_kv.put = AsyncMock(side_effect=[99, 99, 2])

    result = await memory.append_message("a1", "s1", "user", "retry msg")

    assert result["version"] == 2
    assert mock_kv.put.call_count == 3


# ── 9. append_message() raises VersionConflictError after MAX_RETRIES ──────

async def test_append_raises_after_max_retries(memory, mock_kv):
    existing = {
        "session_id": "s1",
        "agent_id": "a1",
        "messages": [],
        "message_count": 0,
        "created_at": 100,
        "updated_at": 100,
    }
    mock_kv.get = AsyncMock(return_value=(existing, 1))
    mock_kv.put = AsyncMock(return_value=99)  # always wrong version

    with pytest.raises(VersionConflictError) as exc_info:
        await memory.append_message("a1", "s1", "user", "conflict")

    assert exc_info.value.retries == 3
