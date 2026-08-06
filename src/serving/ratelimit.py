"""Per-client rate limiting (Day 7).

Without limits, one client (or a bug, or an attack) can exhaust your upstream
budget and starve everyone else. A rate limit caps requests-per-window per client,
returning 429 when exceeded.

Design:
  - fixed-window counter in Redis: key = client id + current minute; INCR + EXPIRE.
    Simple, cheap, good enough for a gateway. (Sliding-window/token-bucket are more
    precise but heavier — fixed window is the standard first cut.)
  - client identity = API key (or IP if no key). So each API key gets its own quota.
  - fail-OPEN: if Redis is down, allow the request. A rate-limiter outage should not
    block legitimate traffic (availability > perfect enforcement for this control).
"""

import os
import time

RATE_LIMIT_ENABLED = os.getenv("DOMAINBOT_RATELIMIT_ENABLED", "true").lower() == "true"
RATE_LIMIT_PER_MIN = int(os.getenv("DOMAINBOT_RATELIMIT_PER_MIN", "60"))


class RateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client

    async def check(self, client_id: str) -> tuple[bool, int]:
        """Returns (allowed, remaining). Fail-open if Redis is unavailable."""
        if not RATE_LIMIT_ENABLED or self.redis is None:
            return True, RATE_LIMIT_PER_MIN
        window = int(time.time() // 60)  # current minute bucket
        key = f"domainbot:rl:{client_id}:{window}"
        try:
            count = await self.redis.incr(key)
            if count == 1:
                await self.redis.expire(key, 60)  # bucket lives 60s
            remaining = max(0, RATE_LIMIT_PER_MIN - count)
            return count <= RATE_LIMIT_PER_MIN, remaining
        except Exception:  # noqa: BLE001
            return True, RATE_LIMIT_PER_MIN  # fail-open
