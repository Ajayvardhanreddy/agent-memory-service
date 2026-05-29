# Semantic Retrieval Eval

Compares sliding-window retrieval vs embedding-based semantic search over a seeded 50-message session.

## How to Run

```bash
# 1. Start the full stack
docker-compose up          # postgres + pgvector
uvicorn app.main:app --port 8081   # memory service

# 2. Seed the session (50 messages, 5 topics)
python evals/seed.py --clean

# 3. Run the eval
ANTHROPIC_API_KEY=sk-... python evals/run_eval.py
```

## What It Tests

50-message session across 5 topics:

| Messages | Topic |
|---|---|
| 1–10 | Italian restaurant on 5th Ave |
| 11–20 | Japan trip (Tokyo + Kyoto) |
| 21–30 | Q3 work report deadline stress |
| 31–40 | Dune Part Two movie discussion |
| 41–50 | Headache symptoms + remedies |

8 queries, three-way comparison:
- **`window_10`** — `GET /window?last_n=10` — what agents actually use today
- **`window_50`** — `GET /window?last_n=50` — oracle (full history)
- **`semantic_5`** — `GET /semantic?top_k=5` — embedding-based

## Grading

**Assertion-based** (queries 1–4, 6, 8): keyword matching — did the target topic appear in results?
Simple, deterministic, zero cost.

**LLM-as-judge** (queries 5, 7): Claude Haiku rates each result set 1–5 on relevance.
Catches nuanced relevance that keyword matching misses.

## Tradeoffs

### When semantic wins
- Query targets a message older than `last_n` — semantic finds it, window misses completely
- The relevant message isn't the most recent but is the most topically relevant
- Long sessions (100+ messages) where any fixed window loses context

### When sliding window wins
- Query is explicitly about recency ("what did I just say?", "latest message")
- Session is short — window and semantic return the same results
- The relevant messages happen to be in the last N

### Hybrid (production recommendation)
Use both: `window(last_n=10)` for recent conversational context + `semantic(top_k=5)` for relevant
historical context. Merge and deduplicate by content before injecting into the LLM prompt.
This is what Layer 3 (`step_runner._load_memory()`) does after this upgrade.

## Cost

- **Embedding on write**: `text-embedding-3-small`, ~50 tokens/message × $0.02/1M = ~$0.000001/message
- **Embedding on query**: same rate, ~$0.000002 per `/semantic` call
- **LLM-as-judge**: `claude-haiku-4-5`, ~200 tokens per judgment = ~$0.00005 per query

For 50 messages: total embedding cost ≈ $0.00005. Negligible.

## LLM Judge Limitations

- Nondeterministic: running twice may give different scores
- Haiku may rate a partially relevant result higher than it deserves
- Cost scales linearly with number of judged queries
- Not suitable for regression testing without score thresholds and multiple runs
