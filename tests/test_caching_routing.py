"""Day 7 — cache keys, rate limiting, A/B routing. Pure logic, no Redis needed
(a fake in-memory Redis stands in). Tests the decisions that matter: cache only
deterministic requests, keys change with revision, sticky variant assignment,
fail-open behavior.
"""

import pytest

from src.serving.cache import ResponseCache, cache_key, is_cacheable
from src.serving.ratelimit import RateLimiter
from src.serving.routing import pick_variant


# ---------------------------------------------------------------- cache keys
def test_cache_key_stable_for_same_input():
    m = [{"role": "user", "content": "hi"}]
    p = {"temperature": 0.0, "max_tokens": 50}
    assert cache_key(m, p, "v1.1.0") == cache_key(m, p, "v1.1.0")


def test_cache_key_changes_with_revision():
    # a new model version must NOT serve the old model's cached answers
    m = [{"role": "user", "content": "hi"}]
    p = {"temperature": 0.0}
    assert cache_key(m, p, "v1.1.0") != cache_key(m, p, "v1.2.0")


def test_cache_key_changes_with_prompt():
    p = {"temperature": 0.0}
    a = cache_key([{"role": "user", "content": "hi"}], p, "v1")
    b = cache_key([{"role": "user", "content": "bye"}], p, "v1")
    assert a != b


def test_only_deterministic_requests_are_cacheable():
    assert is_cacheable(0.0) is True  # deterministic -> cache
    assert is_cacheable(0.7) is False  # random -> never cache


# ---------------------------------------------------------------- fake redis
class FakeRedis:
    def __init__(self):
        self.store = {}

    async def get(self, k):
        return self.store.get(k)

    async def set(self, k, v, ex=None):
        self.store[k] = v

    async def incr(self, k):
        self.store[k] = int(self.store.get(k, 0)) + 1
        return self.store[k]

    async def expire(self, k, s):
        pass


@pytest.mark.asyncio
async def test_cache_get_set_roundtrip():
    c = ResponseCache(FakeRedis())
    await c.set("k", {"content": "Tokyo"})
    assert (await c.get("k"))["content"] == "Tokyo"


@pytest.mark.asyncio
async def test_cache_fails_open_when_no_redis():
    # Redis is None -> get returns None, set is a no-op, gateway keeps serving
    c = ResponseCache(None)
    assert await c.get("k") is None
    await c.set("k", {"x": 1})  # must not raise


# ---------------------------------------------------------------- rate limit
@pytest.mark.asyncio
async def test_rate_limit_allows_then_blocks(monkeypatch):
    monkeypatch.setenv("DOMAINBOT_RATELIMIT_PER_MIN", "3")
    import importlib

    from src.serving import ratelimit

    importlib.reload(ratelimit)
    rl = ratelimit.RateLimiter(FakeRedis())
    results = [await rl.check("client-a") for _ in range(5)]
    allowed = [r[0] for r in results]
    assert allowed[:3] == [True, True, True]  # first 3 allowed
    assert allowed[3] is False  # 4th blocked


@pytest.mark.asyncio
async def test_rate_limit_fails_open_without_redis():
    rl = RateLimiter(None)
    allowed, _ = await rl.check("anyone")
    assert allowed is True  # no redis -> allow


# ---------------------------------------------------------------- A/B routing
def test_canary_zero_sends_all_to_stable(monkeypatch):
    monkeypatch.setenv("DOMAINBOT_CANARY_PERCENT", "0")
    assert all(pick_variant(f"c{i}") == "stable" for i in range(20))


def test_canary_hundred_sends_all_to_canary(monkeypatch):
    monkeypatch.setenv("DOMAINBOT_CANARY_PERCENT", "100")
    assert all(pick_variant(f"c{i}") == "canary" for i in range(20))


def test_canary_assignment_is_sticky(monkeypatch):
    # same client always gets the same variant within a rollout
    monkeypatch.setenv("DOMAINBOT_CANARY_PERCENT", "50")
    first = pick_variant("client-x")
    assert all(pick_variant("client-x") == first for _ in range(10))
