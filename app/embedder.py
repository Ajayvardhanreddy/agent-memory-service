import logging
from typing import TYPE_CHECKING

import numpy as np
from openai import AsyncOpenAI

if TYPE_CHECKING:
    from app.postgres import PostgresClient

logger = logging.getLogger(__name__)

_MODEL = "text-embedding-3-small"
_DIMENSIONS = 1536

_INSERT_SQL = """
INSERT INTO message_embeddings
    (agent_id, session_id, message_index, role, content, embedding, created_at)
VALUES
    (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (agent_id, session_id, message_index) DO NOTHING
"""

_SEARCH_SQL = """
SELECT content, role, message_index, created_at
FROM message_embeddings
WHERE agent_id = %s AND session_id = %s
ORDER BY embedding <=> %s
LIMIT %s
"""


class EmbedderUnavailableError(Exception):
    pass


class Embedder:
    def __init__(self, pg: "PostgresClient", api_key: str) -> None:
        self._pg = pg
        self._openai = AsyncOpenAI(api_key=api_key)

    async def embed(self, text: str) -> list[float]:
        try:
            response = await self._openai.embeddings.create(
                model=_MODEL,
                input=text,
            )
            return response.data[0].embedding
        except Exception as exc:
            raise EmbedderUnavailableError(f"OpenAI embedding failed: {exc}") from exc

    async def embed_and_store(
        self,
        agent_id: str,
        session_id: str,
        message_index: int,
        role: str,
        content: str,
        created_at: int,
    ) -> None:
        vector = await self.embed(content)
        await self._pg.execute(
            _INSERT_SQL,
            (agent_id, session_id, message_index, role, content,
             np.array(vector), created_at),
        )

    async def semantic_search(
        self,
        agent_id: str,
        session_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        vector = await self.embed(query)
        rows = await self._pg.fetchall(
            _SEARCH_SQL,
            (agent_id, session_id, np.array(vector), top_k),
        )
        return [
            {
                "role": row["role"],
                "content": row["content"],
                "ts": row["created_at"],
            }
            for row in rows
        ]
