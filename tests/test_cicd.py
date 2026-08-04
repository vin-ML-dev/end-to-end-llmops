"""Day 6 — validate CI/CD + GitOps config without running them.

Catches the mistakes that otherwise only surface mid-pipeline: a workflow that
doesn't gate deploy on tests, an Argo app pointing at the wrong path, a Helm
chart with a broken image reference.
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(p):
    return yaml.safe_load(open(ROOT / p))


def test_ci_workflow_exists_and_is_valid():
    wf = load(".github/workflows/ci.yaml")
    assert "jobs" in wf
    assert "test" in wf["jobs"]


def test_deploy_is_gated_on_tests():
    # build-push must depend on the test job — never ship untested code
    wf = load(".github/workflows/ci.yaml")
    assert wf["jobs"]["build-push"]["needs"] == "test"


def test_build_only_runs_on_main():
    wf = load(".github/workflows/ci.yaml")
    cond = wf["jobs"]["build-push"]["if"]
    assert "refs/heads/main" in cond


def test_image_tagged_by_git_sha():
    # the immutable deploy artifact must be tagged by SHA, not just :latest
    text = (ROOT / ".github/workflows/ci.yaml").read_text()
    assert "git-${{ github.sha }}" in text


def test_argocd_app_points_at_k8s_path():
    app = load("argocd/application.yaml")
    assert app["spec"]["source"]["path"] == "k8s"
    assert app["spec"]["destination"]["namespace"] == "domainbot"


def test_argocd_selfheal_and_prune_enabled():
    app = load("argocd/application.yaml")
    auto = app["spec"]["syncPolicy"]["automated"]
    assert auto["selfHeal"] is True
    assert auto["prune"] is True


def test_helm_chart_valid():
    c = load("helm/domainbot/Chart.yaml")
    assert c["name"] == "domainbot"
    v = load("helm/domainbot/values.yaml")
    assert "repository" in v["image"]


def test_terraform_secrets_are_variables_not_hardcoded():
    # real secret values must come from variables, never be committed
    text = (ROOT / "terraform/main.tf").read_text()
    assert "var.engine_url" in text
    assert "var.gateway_api_key" in text
