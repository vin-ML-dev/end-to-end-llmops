"""Day 3 — register an APPROVED model.

Only run this after the gate passes. It:
  1. re-runs the gate (defense in depth — never register an unvalidated model)
  2. tags the HF model repo with a semantic version (v1.0.0)
  3. writes/updates a model card with the eval results

Semantic versioning policy (documented in README):
  MAJOR = new base model or changed prompt format
  MINOR = new training data
  PATCH = hyperparameter-only change

Run:
    python -m src.evaluation.register --model <hf-repo> --version v1.0.0
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--version", required=True, help="semantic version tag, e.g. v1.0.0")
    ap.add_argument("--skip-gate", action="store_true", help="skip re-running the gate (only if you JUST ran it)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / args.config))
    model_id = args.model or cfg["model"]

    # 1. defense in depth: re-run the gate unless explicitly skipped
    if not args.skip_gate:
        print(">> re-running gate before registering...")
        rc = subprocess.call(
            [sys.executable, "-m", "src.evaluation.gate", "--config", args.config, "--model", model_id],
            cwd=ROOT,
        )
        if rc != 0:
            raise SystemExit("gate FAILED — refusing to register an unapproved model")

    # read the eval report the gate just wrote
    report_path = ROOT / cfg["report_dir"] / "eval_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}

    # 2. tag the HF repo + 3. update the model card
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        api.create_tag(repo_id=model_id, tag=args.version, repo_type="model")
        print(f">> tagged {model_id} @ {args.version}")
    except Exception as e:
        print(f"!! tag failed (may already exist): {e}")

    card = f"""---
license: apache-2.0
tags: [domainbot, qlora, gated]
---

# DomainBot — {args.version}

Fine-tuned assistant. **Passed the golden-fact gate** before registration.

## Evaluation ({args.version})
- Golden pass rate: **{report.get('pass_rate', 0):.0%}** ({report.get('passed','?')}/{report.get('total','?')})
- Gate decision: **{report.get('decision', {}).get('approved', '?')}**

## Versioning
- MAJOR = new base model / prompt format
- MINOR = new training data
- PATCH = hyperparameter change

Do not deploy a version that has not passed the gate.
"""
    import io

    from huggingface_hub import upload_file

    upload_file(
        path_or_fileobj=io.BytesIO(card.encode()),
        path_in_repo="README.md",
        repo_id=model_id,
        repo_type="model",
        commit_message=f"register {args.version}: gate passed",
    )
    print(f">> registered {model_id} @ {args.version}")
    print(f">> https://huggingface.co/{model_id}")


if __name__ == "__main__":
    main()
