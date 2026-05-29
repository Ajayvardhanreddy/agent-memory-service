from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.embedder import Embedder, EmbedderUnavailableError


_FAKE_VECTOR = [0.1] * 1536


@pytest.fixture
def mock_pg():
    pg = MagicMock()
    pg.execute = AsyncMock()
    pg.fetchall = AsyncMock(return_value=[
        {"role": "user", "content": "I went to that Italian place", "message_index": 2, "created_at": 1000},
        {"role": "assistant", "content": "Sounds great!", "message_index": 3, "created_at": 1001},
    ])
    return pg


@pytest.fixture
def embedder(mock_pg):
    emb = Embedder(mock_pg, api_key="test-key")
    return emb


# ── 1. embed() returns list of floats ──────────────────────────────────────

async def test_embed_returns_floats(embedder):
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=_FAKE_VECTOR)]

    with patch.object(embedder._openai.embeddings, "create", new=AsyncMock(return_value=mock_response)):
        result = await embedder.embed("hello world")

    assert isinstance(result, list)
    assert len(result) == 1536
    assert all(isinstance(v, float) for v in result)


# ── 2. embed_and_store() calls pg.execute with correct params ──────────────

async def test_embed_and_store_inserts_correct_row(embedder, mock_pg):
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=_FAKE_VECTOR)]

    with patch.object(embedder._openai.embeddings, "create", new=AsyncMock(return_value=mock_response)):
        await embedder.embed_and_store(
            agent_id="agent1",
            session_id="sess1",
            message_index=0,
            role="user",
            content="hello",
            created_at=9999,
        )

    mock_pg.execute.assert_called_once()
    call_args = mock_pg.execute.call_args
    params = call_args[0][1]
    assert params[0] == "agent1"
    assert params[1] == "sess1"
    assert params[2] == 0
    assert params[3] == "user"
    assert params[4] == "hello"
    assert isinstance(params[5], np.ndarray)
    assert params[6] == 9999


# ── 3. embed_and_store() on duplicate — pg.execute is still called (ON CONFLICT is SQL-level) ──

async def test_embed_and_store_duplicate_does_not_raise(embedder, mock_pg):
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=_FAKE_VECTOR)]
    mock_pg.execute = AsyncMock()  # no exception = ON CONFLICT DO NOTHING succeeded

    with patch.object(embedder._openai.embeddings, "create", new=AsyncMock(return_value=mock_response)):
        await embedder.embed_and_store("a", "s", 0, "user", "text", 1000)
        await embedder.embed_and_store("a", "s", 0, "user", "text", 1000)

    assert mock_pg.execute.call_count == 2  # both calls go through; DB silently ignores second


# ── 4. semantic_search() returns messages in correct shape ──────────────────

async def test_semantic_search_returns_messages(embedder, mock_pg):
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=_FAKE_VECTOR)]

    with patch.object(embedder._openai.embeddings, "create", new=AsyncMock(return_value=mock_response)):
        results = await embedder.semantic_search("agent1", "sess1", "restaurant", top_k=2)

    assert len(results) == 2
    assert results[0]["role"] == "user"
    assert results[0]["content"] == "I went to that Italian place"
    assert results[0]["ts"] == 1000
    assert "message_index" not in results[0]


# ── 5. embed() failure raises EmbedderUnavailableError ─────────────────────

async def test_embed_failure_raises_embedder_unavailable(embedder):
    with patch.object(
        embedder._openai.embeddings, "create",
        new=AsyncMock(side_effect=Exception("network error"))
    ):
        with pytest.raises(EmbedderUnavailableError):
            await embedder.embed("hello")


# ── 6. append_message best-effort — embed failure does not bubble up ────────

async def test_append_message_continues_if_embed_fails():
    from unittest.mock import AsyncMock, MagicMock
    from app.memory import MemoryService

    mock_kv = MagicMock()
    mock_kv.get = AsyncMock(return_value=(None, None))
    mock_kv.put = AsyncMock(return_value=1)

    mock_stream = MagicMock()
    mock_stream.record = AsyncMock()

    mock_cleanup = MagicMock()
    mock_cleanup.register = MagicMock()

    mock_embedder = MagicMock()
    mock_embedder.embed_and_store = AsyncMock(side_effect=Exception("OpenAI down"))

    svc = MemoryService(mock_kv, mock_stream, mock_cleanup, mock_embedder)
    svc.RETRY_SLEEP = 0

    # Should not raise even though embedder fails
    result = await svc.append_message("agent1", "sess1", "user", "hello")

    assert result["session_id"] == "sess1"
    assert result["message_count"] == 1
    mock_embedder.embed_and_store.assert_called_once()


# ── 7. append_message calls embed_and_store on success ─────────────────────

async def test_append_message_calls_embed_and_store(mock_pg):
    from app.memory import MemoryService

    mock_kv = MagicMock()
    mock_kv.get = AsyncMock(return_value=(None, None))
    mock_kv.put = AsyncMock(return_value=1)

    mock_stream = MagicMock()
    mock_stream.record = AsyncMock()

    mock_cleanup = MagicMock()
    mock_cleanup.register = MagicMock()

    mock_embedder = MagicMock()
    mock_embedder.embed_and_store = AsyncMock()

    svc = MemoryService(mock_kv, mock_stream, mock_cleanup, mock_embedder)
    svc.RETRY_SLEEP = 0

    await svc.append_message("agent1", "sess1", "user", "hello")

    mock_embedder.embed_and_store.assert_called_once()
    call_args = mock_embedder.embed_and_store.call_args[0]
    assert call_args[0] == "agent1"   # agent_id
    assert call_args[1] == "sess1"    # session_id
    assert call_args[2] == 0          # message_index (first message = index 0)
    assert call_args[3] == "user"     # role
    assert call_args[4] == "hello"    # content
