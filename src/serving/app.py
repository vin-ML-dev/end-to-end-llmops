"""DomainBot serving gateway (Day 4).

An engine-agnostic FastAPI sidecar in front of an OpenAI-compatible inference
engine (vLLM on GPU, or llama.cpp on CPU). The gateway owns everything that is
*policy* rather than *inference*:

  - request validation (size caps, history caps, no client system prompt)
  - the server-side system prompt (clients cannot override the persona)
  - auth (API key)
  - health endpoints: /healthz (liveness) vs /readyz (readiness — pings the engine)
  - /v1/model-info (which model + revision is live — verifies rollbacks)
  - error mapping: timeout -> 504, engine down -> 503, engine error -> 502
  - structured JSON logs with request IDs

Why a separate gateway instead of calling vLLM directly? The engine speaks raw
completions; it does not know your auth, your limits, your persona, or your health
semantics. Keeping those in a thin gateway means you can swap vLLM <-> llama.cpp
<-> TGI without changing clients (ADR-012).
"""

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

import httpx
import yaml
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.serving.cache import ResponseCache, cache_key, is_cacheable
from src.serving.guardrails import check_input, check_output
from src.serving.metrics import (
    CACHE_EVENTS,
    GUARDRAIL_BLOCKS,
    LATENCY,
    RAG_RETRIEVALS,
    RATE_LIMITED,
    REQUESTS,
    TOKENS,
    UPSTREAM_ERRORS,
    UPSTREAM_UP,
)
from src.serving.rag import Retriever, build_context
from src.serving.ratelimit import RateLimiter
from src.serving.routing import pick_variant
from src.serving.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ModelInfo,
)

# --------------------------------------------------------------------------- config
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config() -> dict:
    path = os.path.join(ROOT, "configs", "serving.yaml")
    cfg = yaml.safe_load(open(path))
    # env overrides (12-factor): DOMAINBOT_ENGINE_URL, DOMAINBOT_REVISION, etc.
    cfg["model"]["engine_url"] = os.getenv("DOMAINBOT_ENGINE_URL", cfg["model"]["engine_url"])
    cfg["model"]["revision"] = os.getenv("DOMAINBOT_REVISION", cfg["model"]["revision"])
    cfg["model"]["repo_id"] = os.getenv("DOMAINBOT_REPO_ID", cfg["model"]["repo_id"])
    # optional auth token for the EXTERNAL model endpoint (managed APIs often need one).
    # If set, the gateway forwards it as `Authorization: Bearer <key>` on upstream calls.
    cfg["model"]["engine_api_key"] = os.getenv("DOMAINBOT_ENGINE_API_KEY", "")
    return cfg


CFG = load_config()

# --------------------------------------------------------------------------- logging
logger = logging.getLogger("domainbot")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def log_json(**kw):
    logger.info(json.dumps({"ts": time.time(), **kw}))


# --------------------------------------------------------------------------- lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # one shared async client for the whole process (connection pooling)
    app.state.client = httpx.AsyncClient(timeout=CFG["server"]["engine_timeout_s"])
    app.state.ready = False

    # Redis for cache + rate limiting (Day 7). Optional — if it can't connect,
    # we run WITHOUT cache/limits rather than failing. A cache is an optimization,
    # never a hard dependency.
    app.state.redis = None
    redis_url = os.getenv("DOMAINBOT_REDIS_URL", "")
    if redis_url:
        try:
            import redis.asyncio as aioredis

            r = aioredis.from_url(redis_url, decode_responses=True)
            await r.ping()
            app.state.redis = r
            log_json(event="redis_connected", url=redis_url)
        except Exception as e:  # noqa: BLE001
            log_json(event="redis_unavailable", error=str(e))

    app.state.cache = ResponseCache(app.state.redis)
    app.state.limiter = RateLimiter(app.state.redis)

    # RAG retriever (Day 9): loads the knowledge base once at startup.
    app.state.retriever = Retriever()
    log_json(event="rag_loaded", chunks=len(app.state.retriever.chunks))

    log_json(event="startup", model=CFG["model"]["repo_id"], revision=CFG["model"]["revision"])
    yield
    await app.state.client.aclose()
    if app.state.redis is not None:
        await app.state.redis.aclose()


app = FastAPI(title="DomainBot Gateway", version="1.0.0", lifespan=lifespan)


# --------------------------------------------------------------------------- auth
def require_api_key(authorization: str | None = Header(default=None)) -> None:
    expected = os.getenv(CFG["server"]["api_key_env"])
    if not expected:
        return  # no key configured -> auth disabled (dev). Set the env var in prod.
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="invalid or missing API key")


# --------------------------------------------------------------------------- request id
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = request.headers.get("x-request-id", str(uuid.uuid4())[:8])
    start = time.time()
    request.state.rid = rid
    try:
        response = await call_next(request)
    except Exception as e:  # noqa: BLE001
        log_json(event="unhandled", rid=rid, error=str(e))
        return JSONResponse(status_code=500, content={"detail": "internal error", "rid": rid})
    response.headers["x-request-id"] = rid
    log_json(
        event="request",
        rid=rid,
        path=request.url.path,
        status=response.status_code,
        ms=round((time.time() - start) * 1000),
    )
    return response


# --------------------------------------------------------------------------- helpers
def build_messages(user_messages: list, extra_system: str = "") -> list[dict]:
    """Prepend the server-side system prompt; enforce history cap.

    The system prompt lives here, not in the client request. This mirrors training
    (the model was fine-tuned with this exact system role) and prevents a client
    from overwriting the persona or safety instructions.
    """
    limits = CFG["limits"]
    trimmed = user_messages[-(limits["max_history_turns"] * 2) :]  # ~turns * (user+assistant)
    system = CFG["server"]["system_prompt"].strip()
    if extra_system:
        # RAG context is appended to the system prompt so the model grounds its answer.
        system = system + "\n\n" + extra_system
    out = [{"role": "system", "content": system}]
    out += [{"role": m.role, "content": m.content} for m in trimmed]
    return out


def validate_limits(req: ChatRequest) -> None:
    limits = CFG["limits"]
    total_chars = sum(len(m.content) for m in req.messages)
    if total_chars > limits["max_prompt_chars"]:
        raise HTTPException(status_code=413, detail="prompt too large")


def engine_payload(req: ChatRequest, messages: list[dict], stream: bool, model_name: str | None = None) -> dict:
    limits, gen = CFG["limits"], CFG["generation"]
    return {
        "model": model_name or CFG["model"]["engine_model_name"],
        "messages": messages,
        "max_tokens": min(req.max_tokens or limits["default_new_tokens"], limits["max_new_tokens"]),
        "temperature": req.temperature if req.temperature is not None else gen["temperature"],
        "top_p": req.top_p if req.top_p is not None else gen["top_p"],
        "stream": stream,
    }


def engine_headers() -> dict:
    """Headers for the upstream (external) model endpoint. If the managed endpoint
    requires a token, forward it as a Bearer header. Empty key -> no auth header."""
    key = CFG["model"].get("engine_api_key", "")
    return {"Authorization": f"Bearer {key}"} if key else {}


# --------------------------------------------------------------------------- health
@app.get("/healthz", response_model=HealthResponse)
async def healthz():
    """Liveness: is the PROCESS up? Cheap, no dependencies. K8s restarts on fail."""
    return HealthResponse(status="ok")


@app.get("/readyz", response_model=HealthResponse)
async def readyz():
    """Readiness: is the ENGINE reachable? K8s stops routing traffic on fail.

    Liveness vs readiness matters: a process can be alive but not ready (engine
    still loading weights). Restarting it (liveness) wouldn't help; you just want
    to stop sending it traffic (readiness) until the engine answers.
    """
    url = CFG["model"]["engine_url"].rstrip("/") + "/models"
    try:
        r = await app.state.client.get(url, headers=engine_headers(), timeout=3)
        if r.status_code == 200:
            UPSTREAM_UP.set(1)
            return HealthResponse(status="ready")
        UPSTREAM_UP.set(0)
        return JSONResponse(status_code=503, content={"status": "not_ready", "detail": f"engine {r.status_code}"})
    except Exception as e:  # noqa: BLE001
        UPSTREAM_UP.set(0)
        return JSONResponse(status_code=503, content={"status": "not_ready", "detail": str(e)})


@app.get("/metrics")
async def metrics():
    """Prometheus scrape endpoint. Returns all metrics in text exposition format.
    Prometheus polls this every N seconds; Grafana queries Prometheus for dashboards."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/model-info", response_model=ModelInfo)
async def model_info():
    """Which model + revision is live. Use after a deploy to VERIFY the rollout."""
    m = CFG["model"]
    return ModelInfo(
        repo_id=m["repo_id"],
        revision=m["revision"],
        engine_url=m["engine_url"],
        engine_model_name=m["engine_model_name"],
    )


# --------------------------------------------------------------------------- chat
@app.post("/v1/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
async def chat(req: ChatRequest, request: Request):
    start = time.perf_counter()
    validate_limits(req)

    # --- rate limit (Day 7): per-client quota, 429 when exceeded ---
    client_id = _client_id(request)
    allowed, remaining = await app.state.limiter.check(client_id)
    if not allowed:
        RATE_LIMITED.inc()
        REQUESTS.labels(route="/v1/chat", status="429", variant="none").inc()
        raise HTTPException(status_code=429, detail="rate limit exceeded") from None

    # --- input guardrail (Day 9): block injection / PII BEFORE calling the model ---
    last_user = req.messages[-1].content if req.messages else ""
    ok_in, reason_in = check_input(last_user)
    if not ok_in:
        GUARDRAIL_BLOCKS.labels(stage="input", reason=reason_in).inc()
        REQUESTS.labels(route="/v1/chat", status="400", variant="none").inc()
        log_json(event="guardrail_block", stage="input", reason=reason_in, rid=request.state.rid)
        raise HTTPException(status_code=400, detail=f"request blocked: {reason_in}") from None

    # --- RAG (Day 9): retrieve context from the knowledge base, ground the answer ---
    rag_context = ""
    if os.getenv("DOMAINBOT_RAG_ENABLED", "true").lower() == "true":
        hits = app.state.retriever.retrieve(last_user)
        RAG_RETRIEVALS.labels(result="hit" if hits else "empty").inc()
        rag_context = build_context(hits)

    messages = build_messages(req.messages, extra_system=rag_context)

    # --- A/B routing (Day 7): sticky-per-client canary split ---
    variant = pick_variant(client_id)
    engine_url, revision, model_name = _variant_upstream(variant)
    url = engine_url.rstrip("/") + "/chat/completions"

    # streaming responses bypass the cache (they're consumed incrementally)
    if req.stream:
        return StreamingResponse(
            stream_chat(request, url, engine_payload(req, messages, stream=True, model_name=model_name)),
            media_type="text/event-stream",
        )

    payload = engine_payload(req, messages, stream=False, model_name=model_name)
    temperature = payload["temperature"]

    # --- cache lookup (Day 7): only for deterministic (temp==0) requests ---
    ck = None
    if is_cacheable(temperature):
        ck = cache_key(messages, {k: payload[k] for k in ("max_tokens", "temperature", "top_p")}, revision)
        cached = await app.state.cache.get(ck)
        if cached is not None:
            CACHE_EVENTS.labels(result="hit").inc()
            REQUESTS.labels(route="/v1/chat", status="200", variant=variant).inc()
            LATENCY.labels(route="/v1/chat", variant=variant).observe(time.perf_counter() - start)
            log_json(event="cache_hit", rid=request.state.rid, variant=variant)
            return ChatResponse(**cached)
        CACHE_EVENTS.labels(result="miss").inc()

    try:
        r = await app.state.client.post(url, json=payload, headers=engine_headers())
    except httpx.TimeoutException:
        UPSTREAM_ERRORS.labels(kind="timeout").inc()
        REQUESTS.labels(route="/v1/chat", status="504", variant=variant).inc()
        raise HTTPException(status_code=504, detail="engine timeout") from None
    except httpx.ConnectError:
        UPSTREAM_ERRORS.labels(kind="unavailable").inc()
        REQUESTS.labels(route="/v1/chat", status="503", variant=variant).inc()
        raise HTTPException(status_code=503, detail="engine unavailable") from None
    if r.status_code != 200:
        UPSTREAM_ERRORS.labels(kind="error").inc()
        REQUESTS.labels(route="/v1/chat", status="502", variant=variant).inc()
        raise HTTPException(status_code=502, detail=f"engine error {r.status_code}") from None

    data = r.json()
    choice = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})

    # --- output guardrail (Day 9): block banned content BEFORE returning ---
    # INC-013 regression guard: a <LOCATION>-style placeholder must never ship.
    ok_out, reason_out = check_output(choice)
    if not ok_out:
        GUARDRAIL_BLOCKS.labels(stage="output", reason=reason_out).inc()
        REQUESTS.labels(route="/v1/chat", status="502", variant=variant).inc()
        log_json(event="guardrail_block", stage="output", reason=reason_out, rid=request.state.rid)
        raise HTTPException(status_code=502, detail="response blocked by output guardrail") from None

    result = ChatResponse(
        content=choice,
        model=CFG["model"]["repo_id"],
        revision=revision,
        usage=usage,
    )

    # --- metrics: success, latency, tokens ---
    REQUESTS.labels(route="/v1/chat", status="200", variant=variant).inc()
    LATENCY.labels(route="/v1/chat", variant=variant).observe(time.perf_counter() - start)
    if usage:
        TOKENS.labels(kind="prompt", variant=variant).inc(usage.get("prompt_tokens", 0))
        TOKENS.labels(kind="completion", variant=variant).inc(usage.get("completion_tokens", 0))

    # --- cache store (Day 7) ---
    if ck is not None:
        await app.state.cache.set(ck, result.model_dump())
        log_json(event="cache_miss", rid=request.state.rid, variant=variant)

    return result


def _client_id(request: Request) -> str:
    """Identify the client for rate-limiting + sticky routing.
    Prefer the API key; fall back to source IP."""
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return "key:" + auth[len("Bearer ") :][:16]
    return "ip:" + (request.client.host if request.client else "unknown")


def _variant_upstream(variant: str) -> tuple[str, str, str]:
    """Return (engine_url, revision, model_name) for the chosen A/B variant.
    Each variant can serve under a different model name. Canary uses separate env
    vars; falls back to stable if unset."""
    if variant == "canary":
        url = os.getenv("DOMAINBOT_CANARY_ENGINE_URL", CFG["model"]["engine_url"])
        rev = os.getenv("DOMAINBOT_CANARY_REVISION", CFG["model"]["revision"])
        name = os.getenv("DOMAINBOT_CANARY_MODEL_NAME", CFG["model"]["engine_model_name"])
        return url, rev, name
    name = os.getenv("DOMAINBOT_ENGINE_MODEL_NAME", CFG["model"]["engine_model_name"])
    return CFG["model"]["engine_url"], CFG["model"]["revision"], name


async def stream_chat(request: Request, url: str, payload: dict):
    """Proxy the engine's SSE stream to the client, forwarding token deltas."""
    rid = request.state.rid
    try:
        async with app.state.client.stream("POST", url, json=payload, headers=engine_headers()) as r:
            if r.status_code != 200:
                yield f'data: {{"error": "engine {r.status_code}"}}\n\n'
                return
            async for line in r.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                body = line[len("data: ") :]
                if body.strip() == "[DONE]":
                    yield "data: [DONE]\n\n"
                    return
                try:
                    delta = json.loads(body)["choices"][0]["delta"].get("content", "")
                except (KeyError, IndexError, json.JSONDecodeError):
                    continue
                if delta:
                    yield f"data: {json.dumps({'content': delta})}\n\n"
    except httpx.TimeoutException:
        yield 'data: {"error": "engine timeout"}\n\n'
    except Exception as e:  # noqa: BLE001
        log_json(event="stream_error", rid=rid, error=str(e))
        yield 'data: {"error": "stream failed"}\n\n'
