"""Day 10 — orchestration pipeline + drift monitor. Pure logic, no GPU, no cluster."""

from src.monitoring.drift_monitor import DriftSignals, evaluate_drift
from src.pipelines.retrain_pipeline import STAGES, run_pipeline


# ---------------------------------------------------------------- pipeline order
def test_pipeline_stage_order():
    names = [s[0] for s in STAGES]
    assert names == ["retrain", "gate", "register", "deploy", "verify"]


def test_dry_run_completes_all_stages():
    cfg = {"version": "v1.2.0", "candidate_model": "m", "golden_set": "g", "train_config": "c", "run_name": "r"}
    rc = run_pipeline(cfg, start_at="retrain", dry_run=True)
    assert rc == 0


def test_start_at_skips_earlier_stages():
    cfg = {"version": "v1.2.0", "candidate_model": "m", "golden_set": "g", "train_config": "c", "run_name": "r"}
    # starting at gate should still complete (dry run), skipping retrain
    rc = run_pipeline(cfg, start_at="gate", dry_run=True)
    assert rc == 0


# ---------------------------------------------------------------- drift policy
def _healthy() -> DriftSignals:
    return DriftSignals(
        golden_pass_rate=0.92,
        baseline_pass_rate=0.92,
        error_rate=0.01,
        p95_latency_s=1.2,
        new_labelled_examples=0,
        unknown_topic_rate=0.05,
    )


def test_no_retrain_when_healthy():
    d = evaluate_drift(_healthy())
    assert d["retrain"] is False
    assert d["severity"] == "none"


def test_quality_drift_triggers_urgent_retrain():
    s = _healthy()
    s.golden_pass_rate = 0.80  # 12 pts below baseline
    d = evaluate_drift(s)
    assert d["retrain"] is True
    assert d["severity"] == "urgent"
    assert any("quality drift" in r for r in d["reasons"])


def test_high_error_rate_triggers_urgent():
    s = _healthy()
    s.error_rate = 0.10
    d = evaluate_drift(s)
    assert d["retrain"] is True
    assert d["severity"] == "urgent"


def test_new_data_triggers_routine_retrain():
    s = _healthy()
    s.new_labelled_examples = 120
    d = evaluate_drift(s)
    assert d["retrain"] is True
    assert d["severity"] == "routine"


def test_data_drift_flags_kb_update():
    s = _healthy()
    s.unknown_topic_rate = 0.45
    d = evaluate_drift(s)
    assert d["retrain"] is True
    assert any("data drift" in r for r in d["reasons"])
