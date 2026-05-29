# Agent Memory Service

Session memory and activity stream for AI agents, built on a [distributed KV store](https://github.com/Ajayvardhanreddy/distributed-kv-store).

## Architecture

```
   AI Agents (support-agent, sales-agent, onboarding-agent, ...)
      │
      │ HTTP  POST /memory/{agent}/{session}/append
      │       GET  /agents/{agent}/sessions
      │       GET  /memory/{agent}/{session}/window
      ▼
   ┌─────────────────────────────────────────────────────┐
   │            Agent Memory Service  :8081              │
   │                                                     │
   │  MemoryService          ActivityStream  CleanupJob  │
   │  ├─ append_message()    ├─ record()     ├─ register │
   │  ├─ get_session()       ├─ get_events() └─ run_tick │
   │  ├─ get_window()        └─ filter()                 │
   │  ├─ list_sessions()                                 │
   │  └─ delete_session()                                │
   │                                                     │
   │  KVClient  (httpx, round-robin, automatic failover) │
   └──────┬──────────────┬──────────────┬───────────────┘
          │              │              │
          ▼              ▼              ▼
     node-0:8000    node-1:8001    node-2:8002
          │              │              │
         WAL            WAL            WAL
    (consistent hashing, REPLICATION_FACTOR=2, leader failover)
```

## Key Storage Patterns

```
SESSION KEY
  mem:{agent_id}:{session_id}
  │
  └─ value: { session_id, agent_id, messages: [...], message_count,
               created_at, updated_at }
  └─ version: auto-incremented integer on every PUT (1, 2, 3, ...)

SESSION INDEX KEY  (cross-session lookup — no KV scan needed)
  index:{agent_id}
  │
  └─ value: { agent_id, sessions: ["user-alice", "user-bob", ...] }
             updated on every session create + delete

ACTIVITY STREAM KEY
  event:{agent_id}:{timestamp_ms}:{short_uuid}
  │
  └─ value: { event_id, agent_id, action, session_id, ts, metadata }
             in-memory index rebuilt as events come in
```

## Request → Storage Flow

```
POST /memory/support-agent/user-alice/append
  {"role": "user", "content": "My order hasn't arrived."}
          │
          ▼
    MemoryService.append_message()
          │
          ├─1─ KVClient.get("mem:support-agent:user-alice")
          │       → returns (session_dict, version=N) or (None, None)
          │
          ├─2─ Append message, bump message_count + updated_at
          │
          ├─3─ KVClient.put("mem:support-agent:user-alice", session)
          │       → KV store returns new version N+1
          │
          ├─4─ Assert returned version == N+1
          │       if mismatch → retry up to 3× (optimistic concurrency)
          │
          ├─5─ If new session → update index key "index:support-agent"
          │
          ├─6─ CleanupJob.register(key)  (TTL tracking)
          │
          └─7─ ActivityStream.record("append", metadata)  (audit trail)
                  → KVClient.put("event:support-agent:{ts}:{uid}", event)
```

## Fault Tolerance Flow

```
KVClient.get("mem:support-agent:user-alice")
   │
   ├─ Try node-0  →  RequestError / 5xx?  ──→  try next node
   ├─ Try node-1  →  200 OK               ──→  return (dict, version)
   │
   └─ All nodes fail → raise KVStoreUnavailableError → HTTP 503

node-0 DOWN:
  Writes → node-1 becomes leader (heartbeat detected in 5s)
  Reads  → node-2 replica serves from local store
  Result → zero data loss, automatic, transparent

2 nodes DOWN (only 1 alive):
  Writes → 503 (replication_factor=2 cannot be met)
  Reads  → surviving node serves its local copy
  Result → graceful rejection, no silent under-replication
```

## Why a custom KV store?

This service is built on [distributed-kv-store](https://github.com/Ajayvardhanreddy/distributed-kv-store) rather than Redis or Postgres to demonstrate the full infrastructure stack — from consistent hashing and WAL-based replication to application-layer session management. The architectural separation (application layer on top of infrastructure) mirrors how production AI companies build on their own storage systems. The KV store provides versioned keys, synchronous replication, and automatic failover — exactly the primitives a memory service needs.

## API Reference

| Method | Path | Description | Returns |
|--------|------|-------------|---------|
| `POST` | `/memory/{agent_id}/{session_id}/append` | Append a message to a session | `SessionResponse` |
| `GET` | `/agents/{agent_id}/sessions` | List all sessions for an agent | `SessionListResponse` |
| `GET` | `/memory/{agent_id}/{session_id}` | Get full session with all messages | `SessionResponse` |
| `GET` | `/memory/{agent_id}/{session_id}/window?last_n=10` | Get last N messages | `WindowResponse` |
| `DELETE` | `/memory/{agent_id}/{session_id}` | Delete a session | `{"message": "deleted"}` |
| `GET` | `/stream/{agent_id}?limit=50` | Get activity stream events | `StreamResponse` |
| `GET` | `/stream/{agent_id}/filter?action=append&limit=20` | Filter events by action type | `StreamResponse` |
| `GET` | `/health` | Service and KV cluster health | Health status |
| `GET` | `/` | Service info and endpoint list | Service metadata |

## Running Locally

**1. Start the KV cluster:**

```bash
git clone https://github.com/Ajayvardhanreddy/distributed-kv-store.git
cd distributed-kv-store
docker-compose up -d
```

**2. Start the memory service:**

```bash
cd agent-memory-service
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8081
```

**3. Quick test:**

```bash
# Append a message
curl -X POST http://localhost:8081/memory/agent1/session1/append \
  -H "Content-Type: application/json" \
  -d '{"role": "user", "content": "Hello, I need help with my order."}'

# List all sessions for an agent
curl http://localhost:8081/memory/agent1/sessions

# Get sliding window (last 5 messages)
curl "http://localhost:8081/memory/agent1/session1/window?last_n=5"

# Check health
curl http://localhost:8081/health
```

## Running the Demos

**Simple demo** — single agent, single session, node kill/restart:

```bash
python demo/demo.py
```

**Full production demo** — 3 agents, 7 users, concurrent reads, 2-node failure, recovery:

```bash
python demo/demo_full.py
```

The full demo covers 9 scenes and completes in under 3 minutes with zero manual steps.

## Running with Docker

```bash
# Make sure the KV cluster is running first
docker-compose up --build
```

## Tests

```bash
pytest tests/ -v --tb=short
```

26 unit tests across 3 test files covering the KV client, memory service, and activity stream.

## Key Design Decisions

### Optimistic concurrency with version counters

Every session write reads the current version N, appends the message, then writes and verifies the returned version is N+1. If another writer raced us, the version will be higher — we retry up to 3 times with a short backoff. This avoids distributed locks while guaranteeing no lost updates.

### Session index via secondary index key

The KV store has no scan endpoint, so session listing is implemented with a secondary index key `index:{agent_id}` that holds the ordered list of session IDs. This key is updated on every session creation and deletion using the same optimistic concurrency pattern. **Production alternative:** Redis SET or a metadata table in Postgres for larger scale.

### In-memory activity stream index

Events are stored in the KV store but indexed in memory (`dict[agent_id, list[kv_key]]`). This means the index is empty after a service restart — events from before the last restart exist in the KV store but cannot be queried until new events rebuild the index. **Production alternative:** Redis sorted sets keyed by timestamp, or a dedicated time-series store like TimescaleDB.

### TTL via background registry

The KV store has no TTL support and no scan endpoint. The cleanup job maintains a set of known session keys and periodically checks each one's `updated_at` timestamp. Sessions created before a service restart are not tracked until they are accessed again. **Production alternative:** Store session keys in a Redis SET or maintain a secondary index key in the KV store.

## Known Limitations

| Limitation | Impact | Production Solution |
|------------|--------|-------------------|
| Index only tracks sessions created after deploy | Pre-existing sessions not listed until accessed | Backfill job, or rebuild from activity stream |
| In-memory stream index | Empty after restart; old events not queryable | Redis sorted sets or TimescaleDB |
| Registry-based TTL | Misses sessions from before last restart | Redis SET or KV-stored index key |
| Single-service deployment | No horizontal scaling of memory service | Stateless design already supports it — add a load balancer |
| No authentication | Open API endpoints | API key middleware or OAuth2 |
