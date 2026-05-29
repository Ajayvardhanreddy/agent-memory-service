"""
Comparative eval: sliding window vs semantic retrieval.

Three-way comparison for 8 queries against the 50-message seeded session:
  window_10   : GET /window?last_n=10   (baseline — what agents actually use)
  window_50   : GET /window?last_n=50   (oracle — full history, shows ceiling)
  semantic_5  : GET /semantic?top_k=5   (experimental)

Grading:
  Queries 1–4, 6, 8 : assertion-based — target topic must appear in results
  Queries 5, 7      : LLM-as-judge — Claude Haiku rates each result 1–5

Usage:
    python evals/run_eval.py [--base-url http://localhost:8081]
"""

import argparse
import asyncio
import os
import textwrap

import httpx
from anthropic import AsyncAnthropic

AGENT_ID = "eval-agent"
SESSION_ID = "eval-session-50"

QUERIES = [
    # (query, grading, target_keywords, description)
    (
        "What restaurant did I mention visiting?",
        "assertion",
        ["italian", "5th avenue", "carbonara", "naples", "tiramisu"],
        "Old context (msgs 1-10) — window_10 misses entirely",
    ),
    (
        "What travel plans did I discuss?",
        "assertion",
        ["japan", "tokyo", "kyoto", "cherry blossom", "ryokan"],
        "Old context (msgs 11-20) — window_10 misses entirely",
    ),
    (
        "What was I stressed about at work?",
        "assertion",
        ["q3 report", "deadline", "manager", "appendix"],
        "Old context (msgs 21-30) — window_10 misses entirely",
    ),
    (
        "What movie did we discuss watching?",
        "assertion",
        ["dune", "zendaya", "sandworm", "part two"],
        "Old context (msgs 31-40) — window_10 misses entirely",
    ),
    (
        "What health issue did I recently mention?",
        "llm_judge",
        ["headache", "eye strain", "screen", "20-20-20"],
        "Recent context (msgs 41-50) — both should find it",
    ),
    (
        "What was the first thing I talked about at the start of our conversation?",
        "assertion",
        ["italian", "restaurant", "pasta", "carbonara"],
        "Explicitly old — semantic wins, window_10 completely blind",
    ),
    (
        "What remedies did you suggest for my symptoms?",
        "llm_judge",
        ["20-20-20", "hydrated", "caffeine", "monitor"],
        "Recent context — LLM judges relevance quality",
    ),
    (
        "What travel destinations did I mention?",
        "assertion",
        ["japan", "tokyo", "kyoto", "seoul"],
        "Old context (msgs 11-20) — window_10 misses entirely",
    ),
]


def _hits(results: list[dict], keywords: list[str]) -> int:
    combined = " ".join(m["content"].lower() for m in results)
    return sum(1 for kw in keywords if kw.lower() in combined)


def _format_results(results: list[dict]) -> str:
    lines = []
    for m in results:
        lines.append(f"  [{m['role']}] {m['content'][:80]}")
    return "\n".join(lines) if lines else "  (no results)"


async def llm_judge(
    client: AsyncAnthropic,
    query: str,
    window_results: list[dict],
    semantic_results: list[dict],
) -> tuple[int, int, str]:
    prompt = textwrap.dedent(f"""
        Query: "{query}"

        Result Set A (sliding window):
        {_format_results(window_results)}

        Result Set B (semantic search):
        {_format_results(semantic_results)}

        Rate each result set 1–5 on how well it answers the query.
        1 = completely irrelevant, 5 = perfectly answers the query.

        Respond with exactly this format:
        A: <score>
        B: <score>
        Reason: <one sentence>
    """).strip()

    resp = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()

    score_a, score_b, reason = 0, 0, text
    for line in text.splitlines():
        if line.startswith("A:"):
            score_a = int(line.split(":")[1].strip())
        elif line.startswith("B:"):
            score_b = int(line.split(":")[1].strip())
        elif line.startswith("Reason:"):
            reason = line[7:].strip()

    return score_a, score_b, reason


async def run_eval(base_url: str) -> None:
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_needed = any(g == "llm_judge" for _, g, _, _ in QUERIES)
    if openai_needed and not anthropic_key:
        raise RuntimeError("ANTHROPIC_API_KEY required for LLM-as-judge queries")

    llm = AsyncAnthropic(api_key=anthropic_key) if anthropic_key else None

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as http:
        print(f"\n{'='*80}")
        print(f"Semantic Retrieval Eval — {AGENT_ID}/{SESSION_ID}")
        print(f"{'='*80}\n")

        rows = []

        for i, (query, grading, keywords, description) in enumerate(QUERIES, 1):
            # window_10
            r = await http.get(f"/memory/{AGENT_ID}/{SESSION_ID}/window", params={"last_n": 10})
            r.raise_for_status()
            window_10 = r.json()["messages"]

            # window_50 (oracle)
            r = await http.get(f"/memory/{AGENT_ID}/{SESSION_ID}/window", params={"last_n": 50})
            r.raise_for_status()
            window_50 = r.json()["messages"]

            # semantic_5
            r = await http.get(
                f"/memory/{AGENT_ID}/{SESSION_ID}/semantic",
                params={"query": query, "top_k": 5},
            )
            r.raise_for_status()
            semantic_5 = r.json()["messages"]

            if grading == "assertion":
                w10_hits = _hits(window_10, keywords)
                w50_hits = _hits(window_50, keywords)
                s5_hits = _hits(semantic_5, keywords)
                winner = (
                    "semantic" if s5_hits > w10_hits
                    else "window_10" if w10_hits >= s5_hits
                    else "tie"
                )
                note = f"w10={w10_hits}/{len(keywords)} w50={w50_hits}/{len(keywords)} sem={s5_hits}/{len(keywords)} keyword hits"
            else:
                w10_score, s5_score, reason = await llm_judge(llm, query, window_10, semantic_5)
                winner = (
                    "semantic" if s5_score > w10_score
                    else "window_10" if w10_score > s5_score
                    else "tie"
                )
                note = f"LLM: w10={w10_score}/5 sem={s5_score}/5 — {reason}"

            rows.append((i, query[:55], winner, note, description))
            print(f"Q{i}: {query}")
            print(f"     Winner: {winner.upper()}")
            print(f"     {note}")
            print(f"     ({description})\n")

        # Summary table
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"{'#':<3} {'Query':<57} {'Winner':<12} {'Note'}")
        print(f"{'-'*3} {'-'*57} {'-'*12} {'-'*30}")
        for i, query, winner, note, _ in rows:
            print(f"{i:<3} {query:<57} {winner:<12} {note}")

        semantic_wins = sum(1 for _, _, w, _, _ in rows if w == "semantic")
        window_wins   = sum(1 for _, _, w, _, _ in rows if w == "window_10")
        ties          = sum(1 for _, _, w, _, _ in rows if w == "tie")
        print(f"\nResults: semantic={semantic_wins}  window_10={window_wins}  tie={ties}  (of {len(rows)} queries)")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8081")
    args = parser.parse_args()
    asyncio.run(run_eval(args.base_url))
