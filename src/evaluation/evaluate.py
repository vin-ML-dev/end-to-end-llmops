"""Day 3 — golden-set evaluation.

Pure evaluation logic, kept separate from model loading so it's unit-testable
without a GPU. Given a model's answer to a golden question, decide pass/fail:

    PASS  <=>  (at least one must_contain pattern matches)
          AND  (no must_not_contain pattern matches)

The must_not_contain list is what makes this a *regression* suite: it bans the
specific wrong answers we've seen before. A model can be fluent and still fail
by reintroducing a known hallucination.
"""

import json
import re
from collections import defaultdict
from pathlib import Path


def load_golden(path: str) -> list[dict]:
    rows = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            rows.append(json.loads(line))
    return rows


def judge(answer: str, must_contain: list[str], must_not_contain: list[str]) -> tuple[bool, list[str]]:
    """Return (passed, forbidden_hits). Regex, case-insensitive.

    must_contain uses ANY (at least one acceptable phrasing appears).
    must_not_contain uses NONE (no banned phrasing appears).
    """
    ok_must = any(re.search(p, answer, re.I) for p in must_contain) if must_contain else True
    forbidden_hits = [p for p in must_not_contain if re.search(p, answer, re.I)]
    return (ok_must and not forbidden_hits), forbidden_hits


def evaluate(answers: list[str], golden: list[dict]) -> dict:
    """Score a list of answers (aligned to golden rows). No model here — pure logic.

    Returns a report dict with overall + per-category pass rates and per-case detail.
    """
    assert len(answers) == len(golden), "answers must align 1:1 with golden rows"

    per_case = []
    cat_pass: dict[str, int] = defaultdict(int)
    cat_total: dict[str, int] = defaultdict(int)
    passed = 0

    for ans, g in zip(answers, golden, strict=False):
        ok, forbidden = judge(ans, g.get("must_contain", []), g.get("must_not_contain", []))
        passed += ok
        cat = g.get("category", "uncategorized")
        cat_total[cat] += 1
        cat_pass[cat] += ok
        per_case.append(
            {
                "question": g["question"],
                "answer": ans,
                "category": cat,
                "pass": ok,
                "forbidden_hits": forbidden,
                "expected_any": g.get("must_contain", []),
            }
        )

    n = len(golden)
    return {
        "total": n,
        "passed": passed,
        "pass_rate": round(passed / n, 4) if n else 0.0,
        "per_category": {
            c: {"passed": cat_pass[c], "total": cat_total[c], "pass_rate": round(cat_pass[c] / cat_total[c], 4)}
            for c in sorted(cat_total)
        },
        "cases": per_case,
    }


def gate_decision(report: dict, gate_cfg: dict, baseline_report: dict | None = None) -> dict:
    """Apply the three gate conditions. Returns {approved: bool, reasons: [...]}.

    1. absolute floor:   pass_rate >= min_pass_rate
    2. relative:         pass_rate >= baseline_pass_rate + min_vs_baseline (if baseline given)
    3. per-category:     every category >= min_category_pass_rate
    """
    reasons = []
    approved = True

    # 1. absolute floor
    if report["pass_rate"] < gate_cfg["min_pass_rate"]:
        approved = False
        reasons.append(f"pass_rate {report['pass_rate']:.0%} < floor {gate_cfg['min_pass_rate']:.0%}")

    # 2. must be >= currently-deployed model
    if baseline_report is not None:
        needed = baseline_report["pass_rate"] + gate_cfg["min_vs_baseline"]
        if report["pass_rate"] < needed:
            approved = False
            reasons.append(
                f"pass_rate {report['pass_rate']:.0%} < baseline " f"{baseline_report['pass_rate']:.0%} (regression)"
            )

    # 3. no category may collapse
    floor = gate_cfg["min_category_pass_rate"]
    for cat, s in report["per_category"].items():
        if s["pass_rate"] < floor:
            approved = False
            reasons.append(f"category '{cat}' {s['pass_rate']:.0%} < {floor:.0%}")

    if approved:
        reasons.append("all gate conditions satisfied")
    return {"approved": approved, "reasons": reasons}


def write_report(report: dict, decision: dict, out_dir: str, model_name: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"model": model_name, "decision": decision, **report}
    path = out / "eval_report.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path
