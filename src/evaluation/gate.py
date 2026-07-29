"""Day 3 — the quality GATE.

Loads a merged model, asks it every golden question, scores the answers, applies
the gate, and **exits non-zero if the model is not approved**. That non-zero exit
is what makes this a gate rather than a report: CI and the retrain pipeline stop
here, so a regressed model can never be published or deployed.

Run:
    python -m src.evaluation.gate --config configs/eval.yaml
    python -m src.evaluation.gate --model <hf-repo-or-path> --baseline <prod-model>

Exit codes:
    0  -> APPROVED (all gate conditions passed)
    1  -> BLOCKED  (one or more conditions failed)
"""

import argparse
import sys
from pathlib import Path

import yaml

from src.evaluation.evaluate import (
    evaluate,
    gate_decision,
    load_golden,
    write_report,
)

ROOT = Path(__file__).resolve().parents[2]


def pick_dtype():
    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_model(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = pick_dtype()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">> loading {model_id} on {device} ({dtype})")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device if device == "cuda" else None
    )
    model.eval()
    return model, tok


def generate_all(model, tok, golden, gen_cfg) -> list[str]:
    import torch

    system = gen_cfg["system_prompt"].strip()
    answers = []
    for i, g in enumerate(golden, 1):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": g["question"]}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=gen_cfg["max_new_tokens"],
                do_sample=(gen_cfg["temperature"] > 0),
                temperature=gen_cfg["temperature"] if gen_cfg["temperature"] > 0 else None,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        ans = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        answers.append(ans)
        print(f"  [{i}/{len(golden)}] {g['question'][:50]}")
    return answers


def run_on_model(model_id: str, golden, gen_cfg) -> dict:
    model, tok = load_model(model_id)
    answers = generate_all(model, tok, golden, gen_cfg)
    return evaluate(answers, golden)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--model", default=None, help="override the model to gate")
    ap.add_argument("--baseline", default=None, help="override the champion model")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / args.config))
    model_id = args.model or cfg["model"]
    baseline_id = args.baseline if args.baseline is not None else cfg.get("baseline_model")
    golden = load_golden(ROOT / cfg["golden_path"])
    print(f">> {len(golden)} golden cases")

    # candidate
    report = run_on_model(model_id, golden, cfg["generation"])

    # champion (optional)
    baseline_report = None
    if baseline_id:
        print(f">> evaluating baseline (champion): {baseline_id}")
        baseline_report = run_on_model(baseline_id, golden, cfg["generation"])

    decision = gate_decision(report, cfg["gate"], baseline_report)
    path = write_report(report, decision, str(ROOT / cfg["report_dir"]), model_id)

    # ---- human-readable summary ----
    print("\n" + "=" * 64)
    print(f"MODEL: {model_id}")
    print(f"OVERALL: {report['passed']}/{report['total']} passed " f"({report['pass_rate']:.0%})")
    if baseline_report:
        print(f"BASELINE: {baseline_report['pass_rate']:.0%}")
    print("PER-CATEGORY:")
    for cat, s in report["per_category"].items():
        flag = "" if s["pass_rate"] >= cfg["gate"]["min_category_pass_rate"] else "  <-- LOW"
        print(f"   {cat:14} {s['passed']}/{s['total']} ({s['pass_rate']:.0%}){flag}")
    print("\nFAILURES:")
    for c in report["cases"]:
        if not c["pass"]:
            print(f"   [FAIL] {c['question'][:55]}")
            if c["forbidden_hits"]:
                print(f"          forbidden hit: {c['forbidden_hits']}")
            else:
                print(f"          missing any of: {c['expected_any']}")
    print("=" * 64)
    verdict = "APPROVED ✅" if decision["approved"] else "BLOCKED ❌"
    print(f"GATE: {verdict}")
    for r in decision["reasons"]:
        print(f"   - {r}")
    print(f"report: {path}")

    sys.exit(0 if decision["approved"] else 1)


if __name__ == "__main__":
    main()
