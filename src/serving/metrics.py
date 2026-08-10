"""Prometheus metrics (Day 8).

You can't improve what you can't see. These metrics turn "I think the cache helps"
into "cache hit ratio is 43%, p95 dropped from 800ms to 120ms." They instrument the
exact things Day 7 added (cache, rate limit, canary) plus the fundamentals every
service needs (request rate, latency, errors).

The four golden signals (Google SRE) guide what to measure:
  - LATENCY   — how long requests take (histogram -> percentiles)
  - TRAFFIC   — request rate (counter)
  - ERRORS    — failed request rate (counter, by status)
  - SATURATION— how full the system is (here: cache size / upstream health)

Everything is a module-level metric object; the app imports and updates them.
`/metrics` exposes them in Prometheus text format for scraping.
"""

from prometheus_client import Counter, Gauge, Histogram

# --- TRAFFIC + ERRORS: every request, labeled by route, status, and variant ---
REQUESTS = Counter(
    "domainbot_requests_total",
    "Total chat requests",
    ["route", "status", "variant"],  # variant = stable|canary -> compare A/B
)

# --- LATENCY: request duration histogram -> p50/p90/p99 in Grafana ---
LATENCY = Histogram(
    "domainbot_request_duration_seconds",
    "Request duration",
    ["route", "variant"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# --- CACHE: hits vs misses -> hit ratio (the Day 7 payoff, now measurable) ---
CACHE_EVENTS = Counter(
    "domainbot_cache_events_total",
    "Cache hits and misses",
    ["result"],  # result = hit|miss
)

# --- RATE LIMIT: how often clients get 429'd ---
RATE_LIMITED = Counter(
    "domainbot_rate_limited_total",
    "Requests rejected by the rate limiter",
)

# --- UPSTREAM: errors talking to the model endpoint, by kind ---
UPSTREAM_ERRORS = Counter(
    "domainbot_upstream_errors_total",
    "Errors calling the model endpoint",
    ["kind"],  # kind = timeout|unavailable|error
)

# --- TOKENS: usage -> cost tracking + capacity planning ---
TOKENS = Counter(
    "domainbot_tokens_total",
    "Tokens processed",
    ["kind", "variant"],  # kind = prompt|completion
)

# --- SATURATION: is the upstream reachable right now (readiness) ---
UPSTREAM_UP = Gauge(
    "domainbot_upstream_up",
    "1 if the model endpoint passed the last readiness check, else 0",
)


# --- GUARDRAILS: blocks by stage + reason (Day 9) ---
GUARDRAIL_BLOCKS = Counter(
    "domainbot_guardrail_blocks_total",
    "Requests/responses blocked by guardrails",
    ["stage", "reason"],  # stage = input|output
)

# --- RAG: retrievals + whether context was found (Day 9) ---
RAG_RETRIEVALS = Counter(
    "domainbot_rag_retrievals_total",
    "RAG retrieval attempts",
    ["result"],  # result = hit|empty
)
