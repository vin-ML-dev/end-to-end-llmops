"""Day 8 — metrics endpoint + instrumentation. Verifies /metrics exposes the
expected series and that the metric objects increment. No Prometheus server needed.
"""

from fastapi.testclient import TestClient

from src.serving.app import app
from src.serving.metrics import CACHE_EVENTS, REQUESTS

client = TestClient(app)


def test_metrics_endpoint_exists_and_is_prometheus_format():
    r = client.get("/metrics")
    assert r.status_code == 200
    # Prometheus text format starts metric families with "# HELP"
    assert "# HELP" in r.text
    assert "domainbot_requests_total" in r.text


def test_expected_metric_families_present():
    r = client.get("/metrics")
    for name in [
        "domainbot_requests_total",
        "domainbot_request_duration_seconds",
        "domainbot_cache_events_total",
        "domainbot_rate_limited_total",
        "domainbot_upstream_errors_total",
        "domainbot_tokens_total",
        "domainbot_upstream_up",
    ]:
        assert name in r.text, f"missing metric: {name}"


def test_counter_increments():
    before = REQUESTS.labels(route="/v1/chat", status="200", variant="stable")._value.get()
    REQUESTS.labels(route="/v1/chat", status="200", variant="stable").inc()
    after = REQUESTS.labels(route="/v1/chat", status="200", variant="stable")._value.get()
    assert after == before + 1


def test_cache_events_labeled_hit_and_miss():
    CACHE_EVENTS.labels(result="hit").inc()
    CACHE_EVENTS.labels(result="miss").inc()
    r = client.get("/metrics")
    assert 'domainbot_cache_events_total{result="hit"}' in r.text
    assert 'domainbot_cache_events_total{result="miss"}' in r.text
