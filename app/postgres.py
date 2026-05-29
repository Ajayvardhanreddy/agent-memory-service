import logging
from typing import Any

import psycopg
from psycopg_pool import AsyncConnectionPool
from pgvector.psycopg import register_vector_async

logger = logging.getLogger(__name__)

_INIT_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS message_embeddings (
    id             BIGSERIAL PRIMARY KEY,
    agent_id       TEXT NOT NULL,
    session_id     TEXT NOT NULL,
    message_index  INTEGER NOT NULL,
    role           TEXT NOT NULL,
    content        TEXT NOT NULL,
    embedding      vector(1536) NOT NULL,
    created_at     BIGINT NOT NULL,
    UNIQUE (agent_id, session_id, message_index)
);

CREATE INDEX IF NOT EXISTS message_embeddings_vec_idx
    ON message_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
"""


class PostgresClient:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: AsyncConnectionPool | None = None

    async def open(self) -> None:
        self._pool = AsyncConnectionPool(
            self._dsn,
            min_size=2,
            max_size=10,
            open=False,
        )
        await self._pool.open()
        await self._init_schema()
        logger.info("PostgresClient pool opened")

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            logger.info("PostgresClient pool closed")

    async def _init_schema(self) -> None:
        async with self._pool.connection() as conn:
            await register_vector_async(conn)
            await conn.execute(_INIT_SQL)

    async def execute(self, sql: str, params: tuple = ()) -> None:
        async with self._pool.connection() as conn:
            await register_vector_async(conn)
            await conn.execute(sql, params)

    async def fetchall(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        async with self._pool.connection() as conn:
            await register_vector_async(conn)
            async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                await cur.execute(sql, params)
                return await cur.fetchall()
