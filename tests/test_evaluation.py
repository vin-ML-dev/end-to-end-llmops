"""Day 3 tests — the gate logic, no GPU/model needed.

These are the most important tests in the repo: they prove the gate actually
blocks bad models. A gate you haven't tested is not a gate.
"""

from src.evaluation.evaluate import evaluate, gate_decision, judge

GATE = {"min_pass_rate": 0.80, "min_vs_baseline": 0.0, "min_category_pass_rate": 0.50}


# ---------------------------------------------------------------- judge()
def test_judge_pass_when_required_present_and_no_forbidden():
    ok, hits = judge("The average speed is 60 km/h.", ["60"], ["30", "120"])
    assert ok and hits == []


def test_judge_fail_on_forbidden_hit():
    # correct fact present BUT a banned hallucination also present -> FAIL
    ok, hits = judge("It is 60, though some say 120.", ["60"], ["120"])
    assert not ok and hits == ["120"]


def test_judge_fail_when_required_missing():
    ok, hits = judge("I am not sure.", ["60"], [])
    assert not ok


def test_judge_case_insensitive():
    ok, _ = judge("The capital is TOKYO.", ["tokyo"], [])
    assert ok


# ---------------------------------------------------------------- evaluate()
def _golden():
    return [
        {"question": "q1", "must_contain": ["60"], "must_not_contain": [], "category": "cot"},
        {"question": "q2", "must_contain": ["tokyo"], "must_not_contain": [], "category": "flan"},
        {"question": "q3", "must_contain": ["paris"], "must_not_contain": [], "category": "flan"},
    ]


def test_evaluate_counts_and_categories():
    answers = ["it is 60", "tokyo", "wrong"]  # 2/3 pass
    rep = evaluate(answers, _golden())
    assert rep["passed"] == 2 and rep["total"] == 3
    assert rep["pass_rate"] == round(2 / 3, 4)
    assert rep["per_category"]["cot"]["pass_rate"] == 1.0
    assert rep["per_category"]["flan"]["pass_rate"] == 0.5


# ------------------------------------------------------------- gate_decision()
def _report(pass_rate, cats=None):
    return {
        "pass_rate": pass_rate,
        "passed": int(pass_rate * 10),
        "total": 10,
        "per_category": cats or {"a": {"pass_rate": pass_rate, "passed": 5, "total": 10}},
    }


def test_gate_blocks_below_floor():
    d = gate_decision(_report(0.70), GATE)
    assert not d["approved"]
    assert any("floor" in r for r in d["reasons"])


def test_gate_approves_above_floor():
    d = gate_decision(_report(0.90), GATE)
    assert d["approved"]


def test_gate_blocks_regression_vs_baseline():
    # candidate 82% meets the 80% floor but is WORSE than an 88% champion -> BLOCK
    d = gate_decision(_report(0.82), GATE, baseline_report=_report(0.88))
    assert not d["approved"]
    assert any("regression" in r for r in d["reasons"])


def test_gate_blocks_category_collapse():
    # overall high, but one category tanks -> BLOCK (averages hide slice failures)
    cats = {
        "good": {"pass_rate": 1.0, "passed": 8, "total": 8},
        "bad": {"pass_rate": 0.20, "passed": 1, "total": 5},
    }
    d = gate_decision(_report(0.85, cats), GATE)
    assert not d["approved"]
    assert any("bad" in r for r in d["reasons"])


def test_gate_approves_when_beats_baseline_and_all_categories_ok():
    cats = {
        "a": {"pass_rate": 0.9, "passed": 9, "total": 10},
        "b": {"pass_rate": 0.8, "passed": 8, "total": 10},
    }
    d = gate_decision(_report(0.90, cats), GATE, baseline_report=_report(0.85))
    assert d["approved"]
