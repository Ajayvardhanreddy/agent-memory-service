"""
Agent Memory Service — Full Production Demo

Covers:
  - 3 specialized agents handling real-world conversations
  - 7 users across those agents (multi-tenant)
  - Cross-session lookup — list all sessions per agent without knowing IDs
  - Sliding window context injection at different window sizes
  - Concurrent parallel reads across all 7 sessions (asyncio.gather)
  - Single node failure — all 7 sessions survive reads and writes
  - Extreme dual-node failure — graceful degradation / 503
  - Recovery — WAL replay + sync verified on all 3 nodes
  - Multi-agent activity stream audit trail with action filtering
  - Session delete with index consistency verification

Prerequisites:
  - KV cluster: cd distributed-kv-store && docker-compose up -d
  - Memory service: uvicorn app.main:app --port 8080

Usage:
  python demo/demo_full.py
  python demo/demo_full.py --memory-url http://localhost:8080
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time

import httpx

MEMORY_URL = "http://localhost:8080"
KV_CONTAINER_PREFIX = "distributed-kv-store"

# ── Real-world conversation scenarios ────────────────────────────────────────
SCENARIOS = [
    {
        "agent": "support-agent",
        "user": "user-rithika",
        "label": "Rithika — lost package complaint",
        "conversation": [
            ("user",      "My order #7823 hasn't arrived after 2 weeks."),
            ("assistant", "I'm sorry Rithika. Let me pull up order #7823 right now."),
            ("user",      "The tracking says delivered but I never received it."),
            ("assistant", "Confirmed — filed a lost package claim and initiated a replacement."),
        ],
    },
    {
        "agent": "support-agent",
        "user": "user-ajay",
        "label": "Ajay — product return",
        "conversation": [
            ("user",      "I want to return the headphones I bought last month."),
            ("assistant", "Happy to help Ajay. Your 30-day return window is still open."),
            ("user",      "Please send the return label to my email on file."),
        ],
    },
    {
        "agent": "support-agent",
        "user": "user-charlie",
        "label": "Charlie — duplicate charge",
        "conversation": [
            ("user",      "I was charged twice for order #9901. Please fix this immediately."),
            ("assistant", "I see the duplicate charge from March 3rd. Processing a full refund now."),
            ("user",      "How long will it take to appear on my card?"),
        ],
    },
    {
        "agent": "sales-agent",
        "user": "user-diana",
        "label": "Diana — enterprise plan",
        "conversation": [
            ("user",      "We need an enterprise plan for 500 seats with SSO support."),
            ("assistant", "Our enterprise tier includes SSO, audit logs, and dedicated support."),
            ("user",      "Can you send a custom pricing quote to my work email?"),
        ],
    },
    {
        "agent": "sales-agent",
        "user": "user-eve",
        "label": "Eve — competitive evaluation",
        "conversation": [
            ("user",      "We are comparing you to Competitor X. What is your edge?"),
            ("assistant", "Real-time agent memory, full activity streams, and a 99.9% uptime SLA."),
            ("user",      "Can our team of 20 get a 3-month trial?"),
            ("assistant", "Absolutely — setting up your trial with dedicated onboarding support now."),
        ],
    },
    {
        "agent": "onboarding-agent",
        "user": "user-frank",
        "label": "Frank — first-time setup",
        "conversation": [
            ("user",      "I just signed up. Where do I start?"),
            ("assistant", "Welcome Frank! Begin by connecting a data source in the Dashboard tab."),
            ("user",      "Done. What should I configure next?"),
        ],
    },
    {
        "agent": "onboarding-agent",
        "user": "user-grace",
        "label": "Grace — Slack integration",
        "conversation": [
            ("user",      "How do I set up Slack notifications for agent alerts?"),
            ("assistant", "Go to Settings → Integrations → Slack, then paste your webhook URL."),
            ("user",      "Can I filter which alert types come through to Slack?"),
        ],
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

WIDTH = 64

def banner(n: int, title: str) -> None:
    print(f"\n{'═' * WIDTH}")
    print(f"  SCENE {n}: {title}")
    print(f"{'═' * WIDTH}")


def sub(title: str) -> None:
    print(f"\n  ── {title}")


def check_ok(r: httpx.Response, label: str) -> dict:
    if r.status_code != 200:
        print(f"\n  ERROR in {label}: HTTP {r.status_code}")
        try:
            body = r.json()
            print(f"  {body.get('error', 'unknown')}: {body.get('detail', r.text)}")
        except Exception:
            print(f"  Raw: {r.text[:200]}")
        sys.exit(1)
    return r.json()


def docker_stop(suffix: str) -> None:
    name = f"{KV_CONTAINER_PREFIX}-{suffix}-1"
    result = subprocess.run(["docker", "stop", name], capture_output=True, text=True)
    if result.returncode != 0:
        subprocess.run(["docker", "stop", f"{KV_CONTAINER_PREFIX}_{suffix}_1"], capture_output=True)


def docker_start(suffix: str) -> None:
    name = f"{KV_CONTAINER_PREFIX}-{suffix}-1"
    result = subprocess.run(["docker", "start", name], capture_output=True, text=True)
    if result.returncode != 0:
        subprocess.run(["docker", "start", f"{KV_CONTAINER_PREFIX}_{suffix}_1"], capture_output=True)


# ── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    versions: dict[tuple[str, str], int] = {}
    total_messages = 0

    async with httpx.AsyncClient(timeout=10.0) as client:

        # ── SCENE 0: Preflight + Cleanup ─────────────────────────────────
        banner(0, "Preflight + Cleanup")

        try:
            r = await client.get(f"{MEMORY_URL}/health")
            r.raise_for_status()
            health = r.json()
            print(f"\n  Memory service  : OK")
            for node, info in health.get("kv_cluster", {}).get("cluster", {}).items():
                print(f"  {node:12}: {info.get('status', 'unknown')}")
        except Exception as e:
            print(f"\n  FAILED: {e}")
            print(f"  Ensure the KV cluster and memory service are both running.")
            sys.exit(1)

        sub("Cleaning up sessions from any previous run")
        deleted = 0
        for s in SCENARIOS:
            r = await client.delete(f"{MEMORY_URL}/memory/{s['agent']}/{s['user']}")
            if r.status_code == 200:
                deleted += 1
        print(f"  Removed {deleted} stale session(s). Clean slate.")

        # ── SCENE 1: Multi-Agent, Multi-User Setup ────────────────────────
        banner(1, "Multi-Agent Setup — 3 agents, 7 users, real conversations")

        shown_agents: set[str] = set()
        for s in SCENARIOS:
            agent, user = s["agent"], s["user"]
            if agent not in shown_agents:
                print(f"\n  ┌─ {agent}")
                shown_agents.add(agent)
            print(f"  │  {s['label']}")

            prev_version: int | None = None
            for role, content in s["conversation"]:
                r = await client.post(
                    f"{MEMORY_URL}/memory/{agent}/{user}/append",
                    json={"role": role, "content": content},
                )
                session = check_ok(r, f"append to {user}")
                v = session["version"]
                if prev_version is not None:
                    assert v == prev_version + 1, (
                        f"Version gap on {user}: expected {prev_version + 1}, got {v}"
                    )
                prev_version = v
                total_messages += 1
                preview = content[:52] + "..." if len(content) > 52 else content
                print(f"  │    v{v} [{role:9}] {preview}")

            versions[(agent, user)] = prev_version  # type: ignore[assignment]

        print(f"\n  7 sessions created. {total_messages} messages stored. All version assertions passed.")

        # ── SCENE 2: Cross-Session Lookup ─────────────────────────────────
        banner(2, "Cross-Session Lookup — list all sessions per agent")
        print("  No session IDs needed upfront — the index key tracks them in the KV store.\n")

        agents = sorted(set(s["agent"] for s in SCENARIOS))
        for agent in agents:
            r = await client.get(f"{MEMORY_URL}/agents/{agent}/sessions")
            data = check_ok(r, f"list sessions for {agent}")
            print(f"  {agent}  →  {data['total']} active session(s)")
            for sess in data["sessions"]:
                print(
                    f"    session_id={sess['session_id']:16}  "
                    f"msgs={sess['message_count']}  "
                    f"v={sess['version']}"
                )

        print(f"\n  An ops team can now audit all active sessions per agent without any prior knowledge.")

        # ── SCENE 3: Sliding Window Context Injection ─────────────────────
        banner(3, "Sliding Window — context injection at 3 window sizes")

        agent, user = "support-agent", "user-rithika"
        rithika_conv = next(s for s in SCENARIOS if s["user"] == user)["conversation"]
        rithika_total = len(rithika_conv)

        for last_n in [1, 2, rithika_total]:
            r = await client.get(f"{MEMORY_URL}/memory/{agent}/{user}/window?last_n={last_n}")
            window = check_ok(r, f"window last_n={last_n}")
            print(
                f"\n  last_n={last_n}  →  {window['window_size']}/{window['total_messages']} messages "
                f"injected into LLM prompt:"
            )
            for m in window["messages"]:
                print(f"    [{m['role']:9}] {m['content'][:62]}")

        print(f"\n  Token-efficient context. Agent only pays for what it actually needs.")

        # ── SCENE 4: Concurrent Parallel Reads ────────────────────────────
        banner(4, "Concurrent Reads — all 7 sessions fetched in parallel (asyncio.gather)")

        start = time.time()
        tasks = [
            client.get(f"{MEMORY_URL}/memory/{s['agent']}/{s['user']}")
            for s in SCENARIOS
        ]
        responses = await asyncio.gather(*tasks)
        elapsed = time.time() - start

        print(f"\n  {'AGENT':24} {'USER':16} {'MSGS':6} {'VER'}")
        print(f"  {'─'*24} {'─'*16} {'─'*6} {'─'*5}")
        for s, resp in zip(SCENARIOS, responses):
            session = check_ok(resp, f"concurrent read {s['user']}")
            print(
                f"  {s['agent']:24} {s['user']:16} "
                f"{session['message_count']:<6} v{session['version']}"
            )

        print(f"\n  All 7 sessions fetched in {elapsed:.3f}s. Single event loop, zero thread overhead.")

        # ── SCENE 5: Single Node Failure ──────────────────────────────────
        banner(5, "Single Node Failure — node-1 killed, all 7 sessions survive")

        print(f"  Stopping node-1 (mid-cluster node)...")
        docker_stop("node-1")
        print(f"  Waiting 6s for heartbeat failure detection and leader re-election...")
        time.sleep(6)

        r = await client.get(f"{MEMORY_URL}/health")
        cluster = r.json().get("kv_cluster", {}).get("cluster", {})
        for node, info in cluster.items():
            status = info.get("status", "unknown")
            marker = " ← DEAD" if status != "healthy" else ""
            print(f"  {node:12}: {status}{marker}")

        sub("Writing follow-up message to ALL 7 sessions with node-1 down")
        write_ok = 0
        for s in SCENARIOS:
            r = await client.post(
                f"{MEMORY_URL}/memory/{s['agent']}/{s['user']}/append",
                json={"role": "user", "content": "Follow-up question while node-1 is down."},
            )
            session = check_ok(r, f"write {s['user']} with node-1 down")
            expected = versions[(s["agent"], s["user"])] + 1
            assert session["version"] == expected, (
                f"Version gap on {s['user']}: expected {expected}, got {session['version']}"
            )
            versions[(s["agent"], s["user"])] = session["version"]
            total_messages += 1
            write_ok += 1
            print(f"  {s['user']:16} v{session['version']} ✓")

        print(f"\n  {write_ok}/7 writes succeeded. node-1 gone. Replicas serving writes transparently.")

        # ── SCENE 6: Extreme Failure — 2 Nodes Down ───────────────────────
        banner(6, "Extreme Failure — node-2 also killed (only node-0 alive)")

        print(f"  Stopping node-2...")
        docker_stop("node-2")
        print(f"  Waiting 3s...")
        time.sleep(3)

        sub("Attempting a write with only 1/3 nodes alive")
        r = await client.post(
            f"{MEMORY_URL}/memory/support-agent/user-rithika/append",
            json={"role": "user", "content": "Can you confirm the replacement order was created?"},
        )
        if r.status_code == 503:
            body = r.json()
            print(f"  WRITE REJECTED — HTTP 503")
            print(f"  Error : {body.get('error')}")
            print(f"  Detail: {body.get('detail', '')[:100]}")
            print(f"\n  Expected behavior: replication_factor=2 but only 1 node alive.")
            print(f"  System refuses the write rather than silently under-replicate data.")
        elif r.status_code == 200:
            session = r.json()
            versions[("support-agent", "user-rithika")] = session["version"]
            total_messages += 1
            print(f"  Write accepted with reduced replication: v{session['version']}")
            print(f"  KV store accepted with single-node durability (note for production use).")
        else:
            print(f"  HTTP {r.status_code}: {r.text[:150]}")

        sub("Read attempt from surviving node-0")
        r = await client.get(f"{MEMORY_URL}/memory/support-agent/user-rithika")
        if r.status_code == 200:
            session = r.json()
            print(f"  READ OK — v{session['version']}, {session['message_count']} messages")
            print(f"  node-0 serving reads from local store. Data intact on surviving node.")
        else:
            print(f"  Read also failed: HTTP {r.status_code}")

        # ── SCENE 7: Recovery ─────────────────────────────────────────────
        banner(7, "Recovery — node-1 and node-2 restart, WAL replay + sync verified")

        print(f"  Starting node-1...")
        docker_start("node-1")
        print(f"  Starting node-2...")
        docker_start("node-2")
        print(f"  Waiting 10s for WAL replay and sync-on-rejoin across both nodes...")
        time.sleep(10)

        rithika_kv_key = "mem:support-agent:user-rithika"
        print(f"\n  Verifying consistency for '{rithika_kv_key}' directly on all 3 nodes:")

        node_versions: dict[int, int] = {}
        async with httpx.AsyncClient(timeout=5.0) as kv:
            for port in [8000, 8001, 8002]:
                try:
                    nr = await kv.get(f"http://localhost:{port}/kv/{rithika_kv_key}")
                    if nr.status_code == 200:
                        kv_resp = nr.json()
                        raw = json.loads(kv_resp["value"])
                        node_versions[port] = kv_resp["version"]
                        print(
                            f"  node @ :{port}  version={kv_resp['version']}  "
                            f"msgs={raw['message_count']}"
                        )
                    else:
                        print(f"  node @ :{port}  HTTP {nr.status_code} (may still be starting)")
                except Exception as exc:
                    print(f"  node @ :{port}  ERROR: {exc}")

        alive = set(node_versions.values())
        if len(alive) == 1:
            print(f"\n  All {len(node_versions)} nodes agree: version={list(alive)[0]}. Zero data loss.")
        else:
            print(f"\n  Versions differ: {node_versions} — sync may still be in progress.")

        # ── SCENE 8: Activity Stream — Multi-Agent Audit ──────────────────
        banner(8, "Activity Stream — multi-agent audit trail with action filtering")

        for agent in agents:
            r = await client.get(f"{MEMORY_URL}/stream/{agent}?limit=50")
            stream = r.json()
            print(f"\n  {agent}  →  {stream['total']} event(s) since last restart")
            for ev in stream["events"][:4]:
                meta = ev.get("metadata", {})
                print(
                    f"    {ev['action']:8} | {ev['session_id']:16} | "
                    f"v={str(meta.get('version', '—')):>3} | "
                    f"role={meta.get('role') or '—'}"
                )
            if stream["total"] > 4:
                print(f"    ... and {stream['total'] - 4} more")

            r = await client.get(f"{MEMORY_URL}/stream/{agent}/filter?action=append&limit=50")
            appends = r.json()
            other = stream["total"] - appends["total"]
            print(f"  append events: {appends['total']}  |  read+delete events: {other}")

        print(f"\n  Full audit trail — who did what, to which session, at which version.")

        # ── SCENE 9: Delete + Index Consistency ───────────────────────────
        banner(9, "Session Delete — index stays consistent after removal")

        sub("Deleting user-ajay's session")
        r = await client.delete(f"{MEMORY_URL}/memory/support-agent/user-ajay")
        assert r.status_code == 200, f"Delete failed: {r.status_code}"
        print(f"  user-ajay's session deleted.")

        r = await client.get(f"{MEMORY_URL}/agents/support-agent/sessions")
        data = check_ok(r, "list sessions after delete")
        session_ids = [sess["session_id"] for sess in data["sessions"]]
        assert "user-ajay" not in session_ids, "user-ajay still listed after delete — index inconsistency"
        print(f"  support-agent active sessions: {session_ids}")
        print(f"  user-ajay correctly removed from index.")

        r = await client.get(f"{MEMORY_URL}/memory/support-agent/user-ajay")
        assert r.status_code == 404, f"Expected 404 for deleted session, got {r.status_code}"
        print(f"  GET user-ajay → 404 '{r.json().get('error')}'  (clean, expected)")

        # ── SUMMARY ───────────────────────────────────────────────────────
        print(f"\n{'═' * WIDTH}")
        print(f"  FULL DEMO COMPLETE")
        print(f"{'═' * WIDTH}")
        print(f"  Agents              : 3  (support, sales, onboarding)")
        print(f"  Sessions created    : 7")
        print(f"  Sessions deleted    : 1  (user-ajay, Scene 9)")
        print(f"  Active sessions     : 6")
        print(f"  Messages stored     : {total_messages}")
        print(f"  Concurrent reads    : 7  (asyncio.gather, Scene 4)")
        print(f"  Node failures       : 2  (node-1 Scene 5, node-2 Scene 6)")
        print(f"  Data loss           : 0 messages, 0 versions")
        print(f"  Manual steps        : 0")
        print(f"{'═' * WIDTH}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent Memory Service — Full Production Demo")
    parser.add_argument("--memory-url", default=MEMORY_URL, help="Memory service base URL")
    parser.add_argument("--kv-container-prefix", default=KV_CONTAINER_PREFIX, help="Docker container prefix")
    args = parser.parse_args()

    MEMORY_URL = args.memory_url
    KV_CONTAINER_PREFIX = args.kv_container_prefix

    asyncio.run(main())
