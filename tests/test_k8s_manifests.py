"""Day 5 — validate the k8s manifests (gateway-only, external model endpoint).

This architecture runs ONLY the gateway in Kubernetes; the model is an external
managed endpoint. Tests assert the manifests are self-consistent and that no
engine Deployment leaked back in.
"""

from pathlib import Path

import yaml

K8S = Path(__file__).resolve().parents[1] / "k8s"


def load(name):
    return yaml.safe_load(open(K8S / name))


def test_all_manifests_are_valid_yaml():
    for f in K8S.glob("*.yaml"):
        docs = list(yaml.safe_load_all(open(f)))
        assert docs, f"{f.name} is empty"


def test_no_engine_deployment_in_cluster():
    # the model is EXTERNAL — there must be no engine Deployment/Service here
    names = [f.name for f in K8S.glob("*.yaml")]
    assert not any("engine" in n for n in names), "engine manifests should not exist in this architecture"


def test_gateway_probes_point_at_health_endpoints():
    d = load("20-gateway-deployment.yaml")
    c = d["spec"]["template"]["spec"]["containers"][0]
    assert c["livenessProbe"]["httpGet"]["path"] == "/healthz"  # liveness = process
    assert c["readinessProbe"]["httpGet"]["path"] == "/readyz"  # readiness = external endpoint


def test_gateway_has_resource_requests_for_hpa():
    d = load("20-gateway-deployment.yaml")
    c = d["spec"]["template"]["spec"]["containers"][0]
    assert "cpu" in c["resources"]["requests"]


def test_service_selector_matches_pod_labels():
    dep = load("20-gateway-deployment.yaml")
    svc = load("25-gateway-service.yaml")
    pod_labels = dep["spec"]["template"]["metadata"]["labels"]
    for k, v in svc["spec"]["selector"].items():
        assert pod_labels.get(k) == v


def test_external_endpoint_url_is_in_secret_not_committed_real():
    # the external endpoint URL lives in the Secret and must be a template
    s = load("15-secret.yaml")
    assert "DOMAINBOT_ENGINE_URL" in s["stringData"]
    assert (
        "REPLACE-ME" in s["stringData"]["DOMAINBOT_ENGINE_URL"] or "example" in s["stringData"]["DOMAINBOT_ENGINE_URL"]
    )


def test_hpa_targets_the_gateway_deployment():
    hpa = load("40-hpa.yaml")
    assert hpa["spec"]["scaleTargetRef"]["name"] == "domainbot-gateway"
    assert hpa["spec"]["maxReplicas"] > hpa["spec"]["minReplicas"]


def test_gateway_runs_non_root():
    d = load("20-gateway-deployment.yaml")
    sc = d["spec"]["template"]["spec"]["securityContext"]
    assert sc["runAsNonRoot"] is True


def test_secret_is_template_not_real_values():
    s = load("15-secret.yaml")
    assert "changeme" in s["stringData"]["DOMAINBOT_API_KEY"]
