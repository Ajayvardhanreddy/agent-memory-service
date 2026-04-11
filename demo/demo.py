"""
Agent Memory Service — End-to-End Demo

What this proves:
  1. Agent memory persists conversation context across messages (versioned)
  2. Sliding window retrieval works for context injection
  3. Memory survives a node crash — replica fallback is transparent
  4. Writes succeed with a node down — leader promotion is automatic
  5. Rejoining node syncs via WAL — version is consistent cluster-wide
  6. Activity stream gives full audit trail of agent behavior

Prerequisites:
  - KV cluster running: cd distributed-kv-store && docker-compose up -d
  - Memory service running: uvicorn app.main:app --port 8080
  - Docker available for node kill/restart

Usage:
  python demo/demo.py
  python demo/demo.py --memory-url http://localhost:8080 --kv-container-prefix distributed-kv-store
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
AGENT_ID = "support-agent"
SESSION_ID = "user-123"


def banner(n: int, title: str):
    print(f"\n{'═' * 62}")
    print(f"  SCENE {n}: {title}")
    print(f"{'═' * 62}")


def show(label: str, data: dict):
    print(f"\n  {label}")
    version = data.get("version", "—")
    msg_count = data.get("message_count", "—")
    print(f"    version       : {version}")
    print(f"    message_count : {msg_count}")
    messages = data.get("messages", [])
    if messages:
        print(f"    messages:")
        for m in messages[-3:]:
            preview = m["content"][:55] + "..." if len(m["content"]) > 55 else m["content"]
            print(f"      [{m['role']:9}] {preview}")


def check_ok(r: httpx.Response, label: str) -> dict:
    """Assert HTTP 200 and return parsed JSON, printing error detail on failure."""
    if r.status_code != 200:
        print(f"\n  ERROR in {label}: HTTP {r.status_code}")
        try:
            body = r.json()
            print(f"  {body.get('error', 'unknown')}: {body.get('detail', r.text)}")
        except Exception:
            print(f"  Raw response: {r.text[:300]}")
        sys.exit(1)
    return r.json()


def docker_stop(container_suffix: str):
    name = f"{KV_CONTAINER_PREFIX}-{container_suffix}-1"
    result = subprocess.run(["docker", "stop", name], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: could not stop {name}: {result.stderr.strip()}")
        print(f"  Trying alternative name format...")
        alt_name = f"{KV_CONTAINER_PREFIX}_{container_suffix}_1"
        subprocess.run(["docker", "stop", alt_name], capture_output=True)


def docker_start(container_suffix: str):
    name = f"{KV_CONTAINER_PREFIX}-{container_suffix}-1"
    result = subprocess.run(["docker", "start", name], capture_output=True, text=True)
    if result.returncode != 0:
        alt_name = f"{KV_CONTAINER_PREFIX}_{container_suffix}_1"
        subprocess.run(["docker", "start", alt_name], capture_output=True)


async def main():
    async with httpx.AsyncClient(timeout=10.0) as client:

        # ── SCENE 0: preflight ───────────────────────────────────────────
        banner(0, "Preflight — checking services are up")
        try:
            r = await client.get(f"{MEMORY_URL}/health")
            r.raise_for_status()
            health = r.json()
            print(f"  Memory service : OK")
            kv = health.get("kv_cluster", {})
            cluster = kv.get("cluster", {})
            for node, info in cluster.items():
                print(f"  {node:8}: {info.get('status', 'unknown')}")
        except Exception as e:
            print(f"  FAILED: {e}")
            print(f"  Make sure KV cluster and memory service are running.")
            sys.exit(1)

        # ── SCENE 1: start conversation ──────────────────────────────────
        banner(1, "Starting a conversation — version counter begins at 1")

        r = await client.post(
            f"{MEMORY_URL}/memory/{AGENT_ID}/{SESSION_ID}/append",
            json={"role": "user", "content": "My order #4521 hasn't arrived yet."},
        )
        session = check_ok(r, "append message 1")
        show("After message 1 (user)", session)
        assert session["version"] == 1, f"Expected version 1, got {session['version']}"

        r = await client.post(
            f"{MEMORY_URL}/memory/{AGENT_ID}/{SESSION_ID}/append",
            json={"role": "assistant", "content": "I can help with order #4521. Let me check the status for you."},
        )
        session = check_ok(r, "append message 2")
        show("After message 2 (assistant)", session)
        assert session["version"] == 2

        r = await client.post(
            f"{MEMORY_URL}/memory/{AGENT_ID}/{SESSION_ID}/append",
            json={"role": "user", "content": "Also, my account email address needs to be updated."},
        )
        session = check_ok(r, "append message 3")
        show("After message 3 (user)", session)
        assert session["version"] == 3
        print(f"\n  Version is now {session['version']}. Three messages, three increments. Consistent.")

        # ── SCENE 2: sliding window ──────────────────────────────────────
        banner(2, "Sliding window — last 2 messages only")
        r = await client.get(f"{MEMORY_URL}/memory/{AGENT_ID}/{SESSION_ID}/window?last_n=2")
        window = check_ok(r, "get window")
        print(f"  total_messages : {window['total_messages']}")
        print(f"  window_size    : {window['window_size']}")
        print(f"  messages returned:")
        for m in window["messages"]:
            print(f"    [{m['role']:9}] {m['content'][:60]}")
        print(f"\n  Agent injects only these 2 messages into LLM context. Memory bounded.")

        # ── SCENE 3: kill node-0 ─────────────────────────────────────────
        banner(3, "KILLING node-0 — replica fallback transparent to memory service")
        print(f"  Stopping node-0...")
        docker_stop("node-0")
        print(f"  Waiting 6s for heartbeat failure detection and leader promotion...")
        time.sleep(6)

        r = await client.get(f"{MEMORY_URL}/memory/{AGENT_ID}/{SESSION_ID}")
        session = check_ok(r, "get session with node-0 down")
        show("GET session with node-0 DOWN", session)
        assert session["version"] == 3, f"Version changed unexpectedly: {session['version']}"
        print(f"\n  node-0 is dead. Memory still returns version {session['version']}. Replica served it.")

        # ── SCENE 4: write with node down ───────────────────────────────
        banner(4, "Writing with node-0 DOWN — leader promotion automatic")
        r = await client.post(
            f"{MEMORY_URL}/memory/{AGENT_ID}/{SESSION_ID}/append",
            json={"role": "assistant", "content": "I've updated your email and am tracking order #4521."},
        )
        session = check_ok(r, "append message 4 with node-0 down")
        show("After message 4 — node-0 still down", session)
        assert session["version"] == 4
        print(f"\n  Write succeeded. Version {session['version']}. node-1 promoted to leader silently.")

        # ── SCENE 5: restart node-0, verify consistency ──────────────────
        banner(5, "Restarting node-0 — WAL replay + sync-on-rejoin")
        print(f"  Starting node-0...")
        docker_start("node-0")
        print(f"  Waiting 5s for WAL replay and sync-on-rejoin...")
        time.sleep(5)

        async with httpx.AsyncClient(timeout=5.0) as kv_client:
            kv_r = await kv_client.get(f"http://localhost:8000/kv/mem:{AGENT_ID}:{SESSION_ID}")
            kv_data = kv_r.json()
            raw_session = json.loads(kv_data["value"])
            kv_version = kv_data["version"]

        print(f"\n  Direct GET from node-0 (localhost:8000):")
        print(f"    version       : {kv_version}")
        print(f"    message_count : {raw_session['message_count']}")
        assert kv_version == 4, f"node-0 out of sync! Expected version 4, got {kv_version}"
        print(f"\n  node-0 rejoined and has version {kv_version}. WAL replayed. Sync complete. Zero data loss.")

        # ── SCENE 6: activity stream ─────────────────────────────────────
        banner(6, "Activity Stream — full audit trail of agent behavior")
        r = await client.get(f"{MEMORY_URL}/stream/{AGENT_ID}?limit=20")
        stream = r.json()
        print(f"\n  Total events recorded: {stream['total']}")
        print(f"  Events (newest first):")
        for ev in stream["events"][:8]:
            meta = ev.get("metadata", {})
            print(f"    {ev['action']:8} | session={ev['session_id']} | version={meta.get('version', '—')} | role={meta.get('role', '—')}")

        r = await client.get(f"{MEMORY_URL}/stream/{AGENT_ID}/filter?action=append&limit=10")
        appends = r.json()
        print(f"\n  Filtered to 'append' events only: {appends['total']} found")

        # ── SUMMARY ─────────────────────────────────────────────────────
        print(f"\n{'═' * 62}")
        print(f"  DEMO COMPLETE")
        print(f"{'═' * 62}")
        print(f"  Sessions created  : 1")
        print(f"  Messages stored   : 4")
        print(f"  Final version     : 4")
        print(f"  Node failures     : 1 (node-0)")
        print(f"  Data loss         : 0 messages, 0 versions")
        print(f"  Manual steps      : 0")
        print(f"{'═' * 62}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent Memory Service Demo")
    parser.add_argument("--memory-url", default=MEMORY_URL, help="Memory service URL")
    parser.add_argument("--kv-container-prefix", default=KV_CONTAINER_PREFIX, help="Docker container name prefix for KV nodes")
    args = parser.parse_args()

    MEMORY_URL = args.memory_url
    KV_CONTAINER_PREFIX = args.kv_container_prefix

    asyncio.run(main())
