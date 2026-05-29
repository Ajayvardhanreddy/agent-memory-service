# Agent Memory Service

Session memory and activity stream for AI agents, built on a [distributed KV store](https://github.com/Ajayvardhanreddy/distributed-kv-store). Supports both sliding-window recency retrieval and embedding-based semantic similarity search via pgvector. Powers the [Agent Execution Engine](https://github.com/Ajayvardhanreddy/agent-execution-engine) - the Layer 3 runtime that drives agent orchestration, tool execution, and observability.

## Architecture

```
   AI Agents (support-agent, sales-agent, onboarding-agent, ...)
      │
      │ HTTP  POST /memory/{agent}/{session}/append
      │       GET  /agents/{agent}/sessions
      │       GET  /memory/{agent}/{session}/window
      │       GET  /memory/{agent}/{session}/semantic?query=...
      ▼
   ┌──────────────────────────────────────────────────────────────┐
   │              Agent Memory Service  :8081                     │
   │                                                              │
   │  MemoryService          ActivityStream    CleanupJob         │
   │  ├─ append_message()    ├─ record()       ├─ register        │
   │  ├─ get_session()       ├─ get_events()   └─ run_tick        │
   │  ├─ get_window()        └─ filter()                          │
   │  ├─ semantic_search()                                        │
   │  ├─ list_sessions()                                          │
   │  └─ delete_session()                                         │
   │                                                              │
   │  KVClient  (httpx, round-robin, automatic failover)          │
   │  Embedder  (OpenAI text-embedding-3-small, best-effort)      │
   │  PostgresClient  (psycopg3 async pool)                       │
   └──────┬──────────────┬──────────────┬──────────┬─────────────┘
          │              │              │          │
          ▼              ▼              ▼          ▼
     node-0:8000    node-1:8001    node-2:8002   postgres:5432
          │              │              │          │
         WAL            WAL            WAL       pgvector
    (consistent hashing, REPLICATION_FACTOR=2,  (message_embeddings
     leader failover)                            + ivfflat index)
```

## The Problem Semantic Search Solves

Sliding-window retrieval (`last_n=10`) is fast and requires no external dependencies, but it has a hard limit: anything beyond the last N messages is invisible to the agent. In a 200-message session, a user mentioning a restaurant in message 5 is permanently lost from context once `last_n=10` slides past it.

Semantic search solves this by indexing every message as a vector on write. At query time, the query itself is embedded and the closest matching messages are returned by cosine similarity — regardless of when they were sent. The two modes are complementary:


| Mode                     | How it works                                           | When to use                                   |
| ------------------------ | ------------------------------------------------------ | --------------------------------------------- |
| `window(last_n=N)`       | Returns the N most recent messages                     | Recent conversational context                 |
| `semantic(query, top_k)` | Returns top-K messages closest in meaning to the query | Finding relevant old context in long sessions |


Layer 3 (agent-execution-engine) uses both in a hybrid `_load_memory()` call: `window(10)` for recency + `semantic(5)` for relevance, merged and deduplicated before building the LLM prompt.

## Storage Layout

```
KV STORE (session state — source of truth)
──────────────────────────────────────────
SESSION KEY
  mem:{agent_id}:{session_id}
  └─ value: { session_id, agent_id, messages: [...], message_count,
               created_at, updated_at }
  └─ version: auto-incremented integer on every PUT (1, 2, 3, ...)

SESSION INDEX KEY  (cross-session lookup — no KV scan needed)
  index:{agent_id}
  └─ value: { agent_id, sessions: ["user-alice", "user-bob", ...] }
             updated on every session create + delete

ACTIVITY STREAM KEY
  stream:{agent_id}:{timestamp_ms}:{uuid4}
  └─ value: { event_id, agent_id, action, session_id, ts, metadata }
             in-memory index rebuilt as events come in

PGVECTOR (semantic search index — eventually consistent with KV)
────────────────────────────────────────────────────────────────
TABLE: message_embeddings
  (agent_id, session_id, message_index, role, content,
   embedding vector(1536), created_at)
  UNIQUE (agent_id, session_id, message_index)
  INDEX: ivfflat cosine distance (lists=100)
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
          ├─7─ ActivityStream.record("append", metadata)  (audit trail)
          │       → KVClient.put("stream:support-agent:{ts}:{uid}", event)
          │
          └─8─ Embedder.embed_and_store(...)  (best-effort, async)
                  → OpenAI text-embedding-3-small API
                  → INSERT INTO message_embeddings ... ON CONFLICT DO NOTHING
                  (failure logged and skipped — KV write already succeeded)
```

Step 8 is **fire-and-forget**: if OpenAI is unavailable, the message is already durably stored in the KV store. The semantic index lags until the service recovers. Sliding-window retrieval is unaffected.

## Semantic Query Flow

```
GET /memory/support-agent/user-alice/semantic?query=restaurant&top_k=5
          │
          ▼
    MemoryService.semantic_search()
          │
          ├─1─ Embedder.embed("restaurant")
          │       → OpenAI API → [0.21, -0.04, 0.11, ...]  (1536 floats)
          │
          └─2─ PostgresClient.fetchall()
                  SELECT content, role, created_at
                  FROM message_embeddings
                  WHERE agent_id = $1 AND session_id = $2
                  ORDER BY embedding <=> $3   -- cosine distance
                  LIMIT $4
                  → top-5 semantically closest messages, regardless of recency
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

OpenAI API DOWN:
  append_message() → KV write succeeds normally; embedding skipped with warning
  semantic_search() → 503 EmbedderUnavailableError
  window retrieval → completely unaffected
```

## Why a custom KV store?

This service is built on [distributed-kv-store](https://github.com/Ajayvardhanreddy/distributed-kv-store) rather than Redis or Postgres to demonstrate the full infrastructure stack — from consistent hashing and WAL-based replication to application-layer session management. The architectural separation (application layer on top of infrastructure) mirrors how production AI companies build on their own storage systems. The KV store provides versioned keys, synchronous replication, and automatic failover — exactly the primitives a memory service needs.

## API Reference


| Method   | Path                                                         | Description                         | Returns                  |
| -------- | ------------------------------------------------------------ | ----------------------------------- | ------------------------ |
| `POST`   | `/memory/{agent_id}/{session_id}/append`                     | Append a message to a session       | `SessionResponse`        |
| `GET`    | `/agents/{agent_id}/sessions`                                | List all sessions for an agent      | `SessionListResponse`    |
| `GET`    | `/memory/{agent_id}/{session_id}`                            | Get full session with all messages  | `SessionResponse`        |
| `GET`    | `/memory/{agent_id}/{session_id}/window?last_n=10`           | Get last N messages (recency)       | `WindowResponse`         |
| `GET`    | `/memory/{agent_id}/{session_id}/semantic?query=...&top_k=5` | Top-K semantically similar messages | `SemanticSearchResponse` |
| `DELETE` | `/memory/{agent_id}/{session_id}`                            | Delete a session                    | `{"message": "deleted"}` |
| `GET`    | `/stream/{agent_id}?limit=50`                                | Get activity stream events          | `StreamResponse`         |
| `GET`    | `/stream/{agent_id}/filter?action=append&limit=20`           | Filter events by action type        | `StreamResponse`         |
| `GET`    | `/health`                                                    | Service and KV cluster health       | Health status            |
| `GET`    | `/`                                                          | Service info and endpoint list      | Service metadata         |


## Running Locally

**1. Start the KV cluster:**

```bash
git clone https://github.com/Ajayvardhanreddy/distributed-kv-store.git
cd distributed-kv-store
docker-compose up -d
```

**2. Start Postgres + pgvector:**

```bash
cd agent-memory-service
docker-compose up -d postgres
```

**3. Configure environment:**

```bash
cp .env.example .env
# Edit .env and set:
#   OPENAI_API_KEY=sk-...
#   DATABASE_URL=postgresql://memory:memory@localhost:5432/memory_service
```

**4. Start the memory service:**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8081
```

Semantic search is **opt-in**: if `DATABASE_URL` or `OPENAI_API_KEY` are unset, the service starts normally and all endpoints except `/semantic` work as before.

**5. Quick test:**

```bash
# Append a message
curl -X POST http://localhost:8081/memory/agent1/session1/append \
  -H "Content-Type: application/json" \
  -d '{"role": "user", "content": "I visited that Italian place on 5th Ave."}'

# Sliding window (last 5 messages)
curl "http://localhost:8081/memory/agent1/session1/window?last_n=5"

# Semantic search
curl "http://localhost:8081/memory/agent1/session1/semantic?query=restaurant&top_k=3"

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
# Starts postgres + memory-service (KV cluster must be running separately)
docker-compose up --build
```

## Retrieval Eval

Compares sliding-window vs semantic retrieval over a seeded 50-message session (5 topics: restaurant, Japan trip, work deadline, movie, headaches). Three-way comparison: `window_10` / `window_50` (oracle) / `semantic_5`.

```bash
# Seed the session (requires running service)
python evals/seed.py --clean

# Run the eval (assertion-based + LLM-as-judge)
ANTHROPIC_API_KEY=sk-... python evals/run_eval.py
```

See [evals/README.md](evals/README.md) for grading methodology, cost breakdown, and tradeoff analysis.

## Tests

```bash
pytest tests/ -v --tb=short
```

33 unit tests across 4 test files. All external dependencies (KV store, Postgres, OpenAI) are mocked — tests run offline with no infrastructure.


| File                | What it covers                                                 |
| ------------------- | -------------------------------------------------------------- |
| `test_kv_client.py` | Round-robin, failover, GET/PUT/DELETE                          |
| `test_memory.py`    | Optimistic concurrency, session CRUD, version conflict retries |
| `test_stream.py`    | Event recording, ordering, filtering                           |
| `test_embedder.py`  | Embed + store, semantic search, best-effort failure handling   |


## Key Design Decisions

### Optimistic concurrency with version counters

Every session write reads the current version N, appends the message, then writes and verifies the returned version is N+1. If another writer raced us, the version will be higher — we retry up to 3 times with a short backoff. This avoids distributed locks while guaranteeing no lost updates.

### Session index via secondary index key

The KV store has no scan endpoint, so session listing is implemented with a secondary index key `index:{agent_id}` that holds the ordered list of session IDs. This key is updated on every session creation and deletion using the same optimistic concurrency pattern. **Production alternative:** Redis SET or a metadata table in Postgres for larger scale.

### Semantic index as an eventually-consistent sidecar

The pgvector table is a search index over the KV store's data — not the source of truth. Embeddings are written after the KV write succeeds, as a best-effort background step. If OpenAI is unavailable, the message is safely stored; the semantic index lags until recovery. This matches the fault model of the activity stream and keeps `append_message()` latency independent of OpenAI API latency.

Duplicate-safe by design: the table has `UNIQUE (agent_id, session_id, message_index)` and inserts use `ON CONFLICT DO NOTHING`, so retries from optimistic concurrency conflicts cannot produce duplicate embeddings.

### Dual-store retrieval for long sessions

KV is optimal for recency queries (O(1) key lookup, slice last N). pgvector is optimal for relevance queries (cosine similarity over all messages regardless of position). Neither replaces the other. The recommended pattern — used by Layer 3 — is to merge both: `window(last_n=10)` + `semantic(top_k=5)`, deduped by content, injected as the LLM's context window.

### In-memory activity stream index

Events are stored in the KV store but indexed in memory (`dict[agent_id, list[kv_key]]`). This means the index is empty after a service restart — events from before the last restart exist in the KV store but cannot be queried until new events rebuild the index. **Production alternative:** Redis sorted sets keyed by timestamp, or a dedicated time-series store like TimescaleDB.

### TTL via background registry

The KV store has no TTL support and no scan endpoint. The cleanup job maintains a set of known session keys and periodically checks each one's `updated_at` timestamp. Sessions created before a service restart are not tracked until they are accessed again. **Production alternative:** Store session keys in a Redis SET or maintain a secondary index key in the KV store.

## Known Limitations


| Limitation                                      | Impact                                                      | Production Solution                                        |
| ----------------------------------------------- | ----------------------------------------------------------- | ---------------------------------------------------------- |
| Semantic index is eventually consistent         | Messages invisible to `/semantic` until embedding completes | Retry queue (e.g. Celery) for failed embeddings            |
| Index only tracks sessions created after deploy | Pre-existing sessions not listed until accessed             | Backfill job, or rebuild from activity stream              |
| In-memory stream index                          | Empty after restart; old events not queryable               | Redis sorted sets or TimescaleDB                           |
| Registry-based TTL                              | Misses sessions from before last restart                    | Redis SET or KV-stored index key                           |
| Single-service deployment                       | No horizontal scaling of memory service                     | Stateless design already supports it — add a load balancer |
| No authentication                               | Open API endpoints                                          | API key middleware or OAuth2                               |


