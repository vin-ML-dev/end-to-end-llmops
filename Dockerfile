# Day 4 — DomainBot gateway image.
# Multi-stage, slim, NON-ROOT. Does NOT bake model weights — the engine pulls
# them from the Hub at a pinned revision (ADR-013). This image is just the
# policy/gateway layer, so it's tiny and rebuilds fast.

# ---- stage 1: build deps into a venv ----
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements-serve.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -U pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements-serve.txt

# ---- stage 2: runtime ----
FROM python:3.12-slim AS runtime
# non-root user (never run a server as root)
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
# only the code the gateway needs
COPY src/serving/ ./src/serving/
COPY src/__init__.py ./src/__init__.py
COPY configs/serving.yaml ./configs/serving.yaml
COPY docs/knowledge_base.jsonl ./docs/knowledge_base.jsonl
USER appuser
EXPOSE 8000
# liveness baked into the image
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8000/healthz').status_code==200 else 1)"
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
