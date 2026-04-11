# Agent Memory Service

Session memory and activity stream for AI agents, built on a [distributed KV store](https://github.com/Ajayvardhanreddy/distributed-kv-store).

```
   AI Agent
      │
      │ HTTP
      ▼
   Agent Memory Service (:8080)
   MemoryService │ ActivityStream │ CleanupJob
      │
      │ HTTP (httpx, round-robin, failover)
      ▼
   Distributed KV Store Cluster
   node-0:8000  node-1:8001  node-2:8002
      │              │              │
      WAL           WAL           WAL
```

## Why a custom KV store?

This service is built on [distributed-kv-store](https://github.com/Ajayvardhanreddy/distributed-kv-store) rather than Redis or Postgres to demonstrate the full infrastructure stack — from consistent hashing and WAL-based replication to application-layer session management. The architectural separation (application layer on top of infrastructure) mirrors how production AI companies build on their own storage systems. The KV store provides versioned keys, synchronous replication, and automatic failover — exactly the primitives a memory service needs.

## API Reference

| Method | Path | Description | Returns |
|--------|------|-------------|---------|
| `POST` | `/memory/{agent_id}/{session_id}/append` | Append a message to a session | `SessionResponse` |
| `GET` | `/memory/{agent_id}/sessions` | List all sessions for an agent | `SessionListResponse` |
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
pip install -r requirements.txt
uvicorn app.main:app --port 8080
```

**3. Test it:**

```bash
# Append a message
curl -X POST http://localhost:8080/memory/agent1/session1/append \
  -H "Content-Type: application/json" \
  -d '{"role": "user", "content": "Hello, I need help with my order."}'

# Get the session
curl http://localhost:8080/memory/agent1/session1

# Get sliding window (last 5 messages)
curl http://localhost:8080/memory/agent1/session1/window?last_n=5

# Check health
curl http://localhost:8080/health
```

## Running the Demo

The demo script runs a fully automated 6-scene demonstration that proves versioned memory, sliding windows, fault tolerance, and activity streaming.

```bash
python demo/demo.py
```

Expected output: 6 scenes showing version-consistent appends, sliding window retrieval, node failure survival, writes with a node down, WAL-based rejoin sync, and activity stream audit trail. Completes in under 60 seconds.

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

### In-memory activity stream index

Events are stored in the KV store but indexed in memory (`dict[agent_id, list[kv_key]]`). This means the index is empty after a service restart — events from before the last restart exist in the KV store but cannot be queried until new events rebuild the index. **Production alternative:** Redis sorted sets keyed by timestamp, or a dedicated time-series store like TimescaleDB.

### TTL via background registry

The KV store has no TTL support and no scan endpoint. The cleanup job maintains a set of known session keys and periodically checks each one's `updated_at` timestamp. Sessions created before a service restart are not tracked until they are accessed again. **Production alternative:** Store session keys in a Redis SET or maintain a secondary index key in the KV store.

### Session index via secondary index key

The KV store has no scan endpoint, so session listing is implemented with a secondary index key `index:{agent_id}` that holds the list of known session IDs for that agent. This key is updated on every session creation and deletion using the same optimistic concurrency pattern as session writes. **Production alternative:** Redis SET or a metadata table in Postgres for larger scale.

## Known Limitations

| Limitation | Impact | Production Solution |
|------------|--------|-------------------|
| Index only tracks sessions created after first deploy | Sessions that existed before the index key was introduced are not listed | Backfill job, or use activity stream events to rebuild |
| In-memory stream index | Empty after restart; old events not queryable | Redis sorted sets or TimescaleDB |
| Registry-based TTL | Missed sessions from before restart | Redis SET or KV-stored index key |
| Single-service deployment | No horizontal scaling of memory service | Stateless design already supports it — just add a load balancer |
| No authentication | Open API endpoints | API key middleware or OAuth2 |
