import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.kv_client import KVClient, KVStoreUnavailableError


def _mock_response(status_code: int, json_data: dict | None = None) -> httpx.Response:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    if json_data is not None:
        resp.json.return_value = json_data
    return resp


@pytest.fixture
def kv():
    client = AsyncMock(spec=httpx.AsyncClient)
    return KVClient(client, node_urls=["http://node0:8000", "http://node1:8001", "http://node2:8002"])


# ── 1. get() returns (dict, version) on 200 ────────────────────────────────

@pytest.mark.asyncio
async def test_get_returns_dict_and_version(kv):
    payload = {"session_id": "s1", "messages": []}
    kv._client.get = AsyncMock(return_value=_mock_response(200, {
        "value": json.dumps(payload),
        "version": 3,
    }))

    value, version = await kv.get("mem:agent1:s1")

    assert value == payload
    assert version == 3


# ── 2. get() returns (None, None) on 404 — does NOT try other nodes ────────

@pytest.mark.asyncio
async def test_get_returns_none_on_404(kv):
    kv._client.get = AsyncMock(return_value=_mock_response(404, {"detail": "Key not found"}))

    value, version = await kv.get("mem:agent1:missing")

    assert value is None
    assert version is None
    assert kv._client.get.call_count == 1


# ── 3. get() falls back to second node when first raises RequestError ──────

@pytest.mark.asyncio
async def test_get_failover_on_request_error(kv):
    payload = {"session_id": "s1"}
    kv._client.get = AsyncMock(side_effect=[
        httpx.ConnectError("connection refused"),
        _mock_response(200, {"value": json.dumps(payload), "version": 2}),
    ])

    value, version = await kv.get("mem:agent1:s1")

    assert value == payload
    assert version == 2
    assert kv._client.get.call_count == 2


# ── 4. get() raises KVStoreUnavailableError when ALL nodes fail ─────────────

@pytest.mark.asyncio
async def test_get_all_nodes_fail(kv):
    kv._client.get = AsyncMock(side_effect=httpx.ConnectError("down"))

    with pytest.raises(KVStoreUnavailableError):
        await kv.get("mem:agent1:s1")

    assert kv._client.get.call_count == 3


# ── 5. put() serializes dict value to JSON string before sending ────────────

@pytest.mark.asyncio
async def test_put_serializes_value(kv):
    session = {"session_id": "s1", "messages": [{"role": "user", "content": "hi"}]}
    kv._client.put = AsyncMock(return_value=_mock_response(200, {"message": "success", "key": "k", "version": 1}))

    await kv.put("mem:agent1:s1", session)

    call_kwargs = kv._client.put.call_args
    sent_body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert sent_body["value"] == json.dumps(session)
    assert sent_body["key"] == "mem:agent1:s1"


# ── 6. put() returns correct version integer from response ──────────────────

@pytest.mark.asyncio
async def test_put_returns_version(kv):
    kv._client.put = AsyncMock(return_value=_mock_response(200, {"message": "success", "key": "k", "version": 5}))

    version = await kv.put("mem:agent1:s1", {"data": "x"})

    assert version == 5


# ── 7. delete() returns True on 200, False on 404 ──────────────────────────

@pytest.mark.asyncio
async def test_delete_returns_true_on_success(kv):
    kv._client.delete = AsyncMock(return_value=_mock_response(200, {"message": "deleted", "key": "k"}))
    assert await kv.delete("mem:agent1:s1") is True


@pytest.mark.asyncio
async def test_delete_returns_false_on_404(kv):
    kv._client.delete = AsyncMock(return_value=_mock_response(404, {"detail": "not found"}))
    assert await kv.delete("mem:agent1:missing") is False


# ── 8. Round-robin: consecutive calls use different nodes ───────────────────

@pytest.mark.asyncio
async def test_round_robin_cycles_nodes(kv):
    payload_resp = _mock_response(200, {"value": json.dumps({"x": 1}), "version": 1})
    kv._client.get = AsyncMock(return_value=payload_resp)

    await kv.get("key1")
    call1_url = kv._client.get.call_args_list[0].args[0]

    await kv.get("key2")
    call2_url = kv._client.get.call_args_list[1].args[0]

    await kv.get("key3")
    call3_url = kv._client.get.call_args_list[2].args[0]

    urls = [call1_url, call2_url, call3_url]
    assert urls[0] != urls[1]
    assert urls[1] != urls[2]
    assert len(set(urls)) == 3
