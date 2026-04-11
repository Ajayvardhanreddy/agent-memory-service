import asyncio
import logging
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.cleanup import CleanupJob
from app.kv_client import KVClient, KVStoreUnavailableError
from app.memory import MemoryService, SessionNotFoundError, VersionConflictError
from app.stream import ActivityStream

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ── Pydantic models ────────────────────────────────────────────────────────


class AppendRequest(BaseModel):
    role: str
    content: str


class MessageModel(BaseModel):
    role: str
    content: str
    ts: int


class SessionResponse(BaseModel):
    session_id: str
    agent_id: str
    messages: list[MessageModel]
    message_count: int
    created_at: int
    updated_at: int
    version: int


class WindowResponse(BaseModel):
    session_id: str
    messages: list[MessageModel]
    total_messages: int
    window_size: int


class EventResponse(BaseModel):
    event_id: str
    agent_id: str
    action: str
    session_id: str
    ts: int
    metadata: dict


class StreamResponse(BaseModel):
    events: list[EventResponse]
    total: int


class ErrorResponse(BaseModel):
    error: str
    detail: str


# ── Lifespan ───────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(2.0),
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    )

    node_urls = os.getenv(
        "KV_NODE_URLS",
        "http://localhost:8000,http://localhost:8001,http://localhost:8002",
    ).split(",")
    ttl_hours = int(os.getenv("SESSION_TTL_HOURS", "24"))

    kv = KVClient(http_client, node_urls=[u.strip() for u in node_urls])
    stream = ActivityStream(kv)
    cleanup = CleanupJob(kv, ttl_hours=ttl_hours)
    memory = MemoryService(kv, stream, cleanup)

    app.state.memory = memory
    app.state.stream = stream
    app.state.kv = kv

    cleanup_task = asyncio.create_task(cleanup.run_forever())
    logger.info("Agent Memory Service started")

    yield

    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    await http_client.aclose()
    logger.info("Agent Memory Service shut down")


# ── App ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Agent Memory Service",
    description="Session memory and activity stream for AI agents, built on a distributed KV store.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Exception handlers ────────────────────────────────────────────────────


@app.exception_handler(SessionNotFoundError)
async def session_not_found_handler(request: Request, exc: SessionNotFoundError):
    return JSONResponse(
        status_code=404,
        content={"error": "session_not_found", "detail": str(exc)},
    )


@app.exception_handler(VersionConflictError)
async def version_conflict_handler(request: Request, exc: VersionConflictError):
    return JSONResponse(
        status_code=409,
        content={
            "error": "version_conflict",
            "detail": f"Version conflict on session {exc.session_id} after {exc.retries} retries",
        },
    )


@app.exception_handler(KVStoreUnavailableError)
async def kv_unavailable_handler(request: Request, exc: KVStoreUnavailableError):
    return JSONResponse(
        status_code=503,
        content={"error": "kv_store_unavailable", "detail": str(exc)},
    )


# ── Endpoints ──────────────────────────────────────────────────────────────


@app.post("/memory/{agent_id}/{session_id}/append", response_model=SessionResponse)
async def append_message(
    agent_id: str,
    session_id: str,
    request: Request,
    body: AppendRequest,
):
    """
    Append a message to a session. Creates session if it does not exist.
    Returns the full updated session including version and message_count.
    409 on version conflict after max retries.
    503 if KV cluster is unavailable.
    """
    memory: MemoryService = request.app.state.memory
    result = await memory.append_message(agent_id, session_id, body.role, body.content)
    return result


@app.get("/memory/{agent_id}/{session_id}", response_model=SessionResponse)
async def get_session(agent_id: str, session_id: str, request: Request):
    """
    Get full session with all messages.
    404 if session does not exist.
    """
    memory: MemoryService = request.app.state.memory
    return await memory.get_session(agent_id, session_id)


@app.get("/memory/{agent_id}/{session_id}/window", response_model=WindowResponse)
async def get_window(
    agent_id: str,
    session_id: str,
    request: Request,
    last_n: int = 10,
):
    """
    Get last N messages from session.
    404 if session does not exist.
    """
    memory: MemoryService = request.app.state.memory
    return await memory.get_window(agent_id, session_id, last_n=last_n)


@app.delete("/memory/{agent_id}/{session_id}")
async def delete_session(agent_id: str, session_id: str, request: Request):
    """
    Delete a session and all its messages.
    404 if session does not exist.
    """
    memory: MemoryService = request.app.state.memory
    deleted = await memory.delete_session(agent_id, session_id)
    if not deleted:
        raise SessionNotFoundError(f"Session {session_id} not found for agent {agent_id}")
    return {"message": "deleted", "session_id": session_id}


@app.get("/stream/{agent_id}", response_model=StreamResponse)
async def get_stream(agent_id: str, request: Request, limit: int = 50):
    """
    Get recent activity stream events for an agent.
    Returns empty list if no events since last restart (documented limitation).
    """
    stream: ActivityStream = request.app.state.stream
    events = await stream.get_events(agent_id, limit=limit)
    return {"events": events, "total": len(events)}


@app.get("/stream/{agent_id}/filter", response_model=StreamResponse)
async def filter_stream(
    agent_id: str,
    request: Request,
    action: str | None = None,
    limit: int = 20,
):
    """
    Filter activity stream by action type (append, read, delete).
    """
    stream: ActivityStream = request.app.state.stream
    events = await stream.filter_events(agent_id, action=action, limit=limit)
    return {"events": events, "total": len(events)}


@app.get("/health")
async def health(request: Request):
    """
    Service health including KV cluster status.
    Returns 503 if KV cluster is entirely unavailable.
    """
    kv: KVClient = request.app.state.kv
    try:
        cluster = await kv.cluster_health()
        return {"status": "healthy", "service": "agent-memory-service", "kv_cluster": cluster}
    except KVStoreUnavailableError:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "service": "agent-memory-service", "kv_cluster": "unavailable"},
        )


@app.get("/")
async def root():
    """Service info and endpoint reference."""
    return {
        "service": "Agent Memory Service",
        "version": "1.0.0",
        "description": "Session memory and activity stream for AI agents",
        "endpoints": {
            "POST /memory/{agent_id}/{session_id}/append": "Append a message to a session",
            "GET /memory/{agent_id}/{session_id}": "Get full session",
            "GET /memory/{agent_id}/{session_id}/window?last_n=10": "Get last N messages",
            "DELETE /memory/{agent_id}/{session_id}": "Delete a session",
            "GET /stream/{agent_id}?limit=50": "Get activity stream events",
            "GET /stream/{agent_id}/filter?action=append&limit=20": "Filter events by action",
            "GET /health": "Service and KV cluster health",
        },
    }
