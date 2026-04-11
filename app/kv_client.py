import json
import logging

import httpx

logger = logging.getLogger(__name__)


class KVStoreUnavailableError(Exception):
    """Raised when all KV nodes fail to respond."""
    pass


class KVClient:
    """
    Async HTTP client for the distributed KV store.

    Uses a single shared httpx.AsyncClient (injected at construction).
    Round-robins across nodes. On any node failure, tries the next node.
    Raises KVStoreUnavailableError only if ALL nodes fail.

    Value serialization: all values stored as JSON strings.
    Callers pass dicts; this client handles json.dumps/json.loads.
    """

    def __init__(self, client: httpx.AsyncClient, node_urls: list[str]):
        self._client = client
        self._nodes = node_urls
        self._current = 0

    async def get(self, key: str) -> tuple[dict | None, int | None]:
        """
        GET /kv/{key} from KV cluster.
        Returns (value_dict, version) or (None, None) if key not found.
        Never raises on 404.
        Raises KVStoreUnavailableError if all nodes fail.
        """
        errors: list[str] = []
        for _ in range(len(self._nodes)):
            url = self._next_node()
            try:
                resp = await self._client.get(f"{url}/kv/{key}")
                if resp.status_code == 404:
                    return None, None
                if resp.status_code >= 500:
                    errors.append(f"{url}: status {resp.status_code}")
                    logger.warning("KV GET %s from %s failed: status %d", key, url, resp.status_code)
                    continue
                data = resp.json()
                logger.debug("KV GET %s served by %s (version=%d)", key, url, data["version"])
                return json.loads(data["value"]), data["version"]
            except httpx.RequestError as exc:
                errors.append(f"{url}: {exc}")
                logger.warning("KV GET %s from %s failed: %s", key, url, exc)
                continue

        raise KVStoreUnavailableError(f"All nodes failed for GET /kv/{key}: {errors}")

    async def put(self, key: str, value: dict) -> int:
        """
        PUT /kv/{key} to KV cluster.
        Serializes value dict to JSON string before sending.
        Returns the new version number from the response.
        Raises KVStoreUnavailableError if all nodes fail.
        """
        body = {"key": key, "value": json.dumps(value)}
        errors: list[str] = []
        for _ in range(len(self._nodes)):
            url = self._next_node()
            try:
                resp = await self._client.put(f"{url}/kv/{key}", json=body)
                if resp.status_code >= 500:
                    errors.append(f"{url}: status {resp.status_code}")
                    logger.warning("KV PUT %s to %s failed: status %d", key, url, resp.status_code)
                    continue
                data = resp.json()
                logger.debug("KV PUT %s served by %s (version=%d)", key, url, data["version"])
                return data["version"]
            except httpx.RequestError as exc:
                errors.append(f"{url}: {exc}")
                logger.warning("KV PUT %s to %s failed: %s", key, url, exc)
                continue

        raise KVStoreUnavailableError(f"All nodes failed for PUT /kv/{key}: {errors}")

    async def delete(self, key: str) -> bool:
        """
        DELETE /kv/{key} from KV cluster.
        Returns True if deleted, False if not found.
        Raises KVStoreUnavailableError if all nodes fail.
        """
        errors: list[str] = []
        for _ in range(len(self._nodes)):
            url = self._next_node()
            try:
                resp = await self._client.delete(f"{url}/kv/{key}")
                if resp.status_code == 404:
                    return False
                if resp.status_code >= 500:
                    errors.append(f"{url}: status {resp.status_code}")
                    logger.warning("KV DELETE %s from %s failed: status %d", key, url, resp.status_code)
                    continue
                logger.debug("KV DELETE %s served by %s", key, url)
                return True
            except httpx.RequestError as exc:
                errors.append(f"{url}: {exc}")
                logger.warning("KV DELETE %s from %s failed: %s", key, url, exc)
                continue

        raise KVStoreUnavailableError(f"All nodes failed for DELETE /kv/{key}: {errors}")

    async def cluster_health(self) -> dict:
        """GET /cluster/health from first responding node."""
        errors: list[str] = []
        for _ in range(len(self._nodes)):
            url = self._next_node()
            try:
                resp = await self._client.get(f"{url}/cluster/health")
                if resp.status_code >= 500:
                    errors.append(f"{url}: status {resp.status_code}")
                    continue
                return resp.json()
            except httpx.RequestError as exc:
                errors.append(f"{url}: {exc}")
                continue

        raise KVStoreUnavailableError(f"All nodes failed for GET /cluster/health: {errors}")

    def _next_node(self) -> str:
        """Round-robin node selection."""
        url = self._nodes[self._current % len(self._nodes)]
        self._current += 1
        return url
