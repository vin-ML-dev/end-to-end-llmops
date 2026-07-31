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
from fastapi.responses import JSONResponse, StreamingResponse

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
    log_json(event="startup", model=CFG["model"]["repo_id"], revision=CFG["model"]["revision"])
    yield
    await app.state.client.aclose()


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
def build_messages(user_messages: list) -> list[dict]:
    """Prepend the server-side system prompt; enforce history cap.

    The system prompt lives here, not in the client request. This mirrors training
    (the model was fine-tuned with this exact system role) and prevents a client
    from overwriting the persona or safety instructions.
    """
    limits = CFG["limits"]
    trimmed = user_messages[-(limits["max_history_turns"] * 2) :]  # ~turns * (user+assistant)
    out = [{"role": "system", "content": CFG["server"]["system_prompt"].strip()}]
    out += [{"role": m.role, "content": m.content} for m in trimmed]
    return out


def validate_limits(req: ChatRequest) -> ChatRequest:
    """Enforce size caps with production-grade UX.

    1. If the LATEST user message alone exceeds the per-message cap -> reject 413.
       (Pydantic already caps each message at 4000 chars, but we re-check here to
       return an honest 413 tied to the user's actual input, not an opaque 422.)
    2. If the TOTAL conversation history is too long -> trim oldest messages to fit
       (sliding window), always keeping the latest user message. The user's current
       input always goes through; only stale history drops off. This mirrors how
       ChatGPT/Claude handle long conversations.
    """
    limits = CFG["limits"]
    max_msg = limits.get("max_message_chars", 4000)
    max_total = limits["max_prompt_chars"]

    # 1. Latest (current) message — the user's actual input this turn.
    latest = req.messages[-1]
    if len(latest.content) > max_msg:
        raise HTTPException(
            status_code=413,
            detail="your message is too long — please shorten it",
        )

    # 2. Total history — trim oldest first, keep newest, never drop the latest turn.
    total = sum(len(m.content) for m in req.messages)
    if total <= max_total:
        return req  # fits as-is

    kept: list = []
    running = 0
    for m in reversed(req.messages):  # newest first
        if running + len(m.content) > max_total and kept:
            break  # stop before exceeding (but keep at least the latest)
        kept.insert(0, m)  # rebuild in original order
        running += len(m.content)

    req.messages = kept
    return req


def engine_payload(req: ChatRequest, messages: list[dict], stream: bool) -> dict:
    limits, gen = CFG["limits"], CFG["generation"]
    return {
        "model": CFG["model"]["engine_model_name"],
        "messages": messages,
        "max_tokens": min(req.max_tokens or limits["default_new_tokens"], limits["max_new_tokens"]),
        "temperature": req.temperature if req.temperature is not None else gen["temperature"],
        "top_p": req.top_p if req.top_p is not None else gen["top_p"],
        "stream": stream,
    }


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
        r = await app.state.client.get(url, timeout=3)
        if r.status_code == 200:
            return HealthResponse(status="ready")
        return JSONResponse(status_code=503, content={"status": "not_ready", "detail": f"engine {r.status_code}"})
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=503, content={"status": "not_ready", "detail": str(e)})


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
    req = validate_limits(req)
    messages = build_messages(req.messages)
    url = CFG["model"]["engine_url"].rstrip("/") + "/chat/completions"

    if req.stream:
        return StreamingResponse(
            stream_chat(request, url, engine_payload(req, messages, stream=True)),
            media_type="text/event-stream",
        )

    payload = engine_payload(req, messages, stream=False)
    try:
        r = await app.state.client.post(url, json=payload)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="engine timeout") from None
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="engine unavailable") from None
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"engine error {r.status_code}")

    data = r.json()
    choice = data["choices"][0]["message"]["content"]
    return ChatResponse(
        content=choice,
        model=CFG["model"]["repo_id"],
        revision=CFG["model"]["revision"],
        usage=data.get("usage", {}),
    )


async def stream_chat(request: Request, url: str, payload: dict):
    """Proxy the engine's SSE stream to the client, forwarding token deltas."""
    rid = request.state.rid
    try:
        async with app.state.client.stream("POST", url, json=payload) as r:
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
