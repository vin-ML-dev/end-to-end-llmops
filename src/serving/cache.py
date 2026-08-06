"""Response caching (Day 7).

Identical prompts get identical answers at temperature 0 — so caching them cuts
BOTH cost (fewer upstream calls to the paid endpoint) and latency (a cache hit is
~1ms vs hundreds of ms upstream). This is the single highest-ROI optimization for
an LLM gateway.

Design:
  - key = hash(system_prompt + messages + generation params + model revision).
    Everything that affects the OUTPUT is in the key; nothing else. Change the
    system prompt or the model version -> different key -> no stale hits.
  - only cache DETERMINISTIC requests (temperature == 0). Caching a temperature>0
    response would pin one random sample forever — wrong.
  - Redis with a TTL (entries expire), so the cache can't grow unbounded and stale
    answers eventually refresh.
  - graceful degradation: if Redis is down, we skip the cache and serve normally.
    A cache outage must NEVER take down the gateway.
"""

import hashlib
import json
import os

CACHE_TTL_S = int(os.getenv("DOMAINBOT_CACHE_TTL_S", "3600"))  # 1h default
CACHE_ENABLED = os.getenv("DOMAINBOT_CACHE_ENABLED", "true").lower() == "true"


def cache_key(messages: list[dict], params: dict, revision: str) -> str:
    """Deterministic key from everything that affects the output."""
    payload = json.dumps(
        {"messages": messages, "params": params, "revision": revision},
        sort_keys=True,
        ensure_ascii=False,
    )
    return "domainbot:resp:" + hashlib.sha256(payload.encode()).hexdigest()


def is_cacheable(temperature: float) -> bool:
    """Only cache deterministic requests. temperature>0 -> random -> don't cache."""
    return CACHE_ENABLED and temperature == 0.0


class ResponseCache:
    """Thin async Redis wrapper. All ops fail-open (return None / no-op on error)."""

    def __init__(self, redis_client):
        self.redis = redis_client  # may be None -> cache disabled

    async def get(self, key: str) -> dict | None:
        if self.redis is None:
            return None
        try:
            raw = await self.redis.get(key)
            return json.loads(raw) if raw else None
        except Exception:  # noqa: BLE001 — cache errors never break serving
            return None

    async def set(self, key: str, value: dict) -> None:
        if self.redis is None:
            return
        try:
            await self.redis.set(key, json.dumps(value), ex=CACHE_TTL_S)
        except Exception:  # noqa: BLE001
            pass
