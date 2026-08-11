"""Automated retrain → gate → register → deploy pipeline (Day 10 capstone).

This is the whole course, closed into a loop. Every earlier day built one stage;
today they run as ONE orchestrated flow, with the Day 3 doctrine automated:

    (trigger) → retrain from BASE on old+new data
              → evaluate against the golden set
              → GATE (block if it fails — floor / regression / per-category)
              → register (tag the Hub repo) only if approved
              → deploy (point the gateway at the new revision)
              → verify (readiness + a smoke request)

Principles carried from earlier days, now enforced by code:
  - retrain ALWAYS from the base model on old+new data, never from the deployed
    model (Day 2/3 doctrine — no compounding drift across generations).
  - the gate fails the pipeline (non-zero exit): a bad model cannot proceed. The
    gate is never loosened to let a wrong answer through.
  - deploy is a GitOps operation (Day 6): the pipeline bumps the revision in Git and
    lets Argo sync — it does not kubectl-apply directly. Auditable, revertible.
  - every stage logs a structured event; a failure halts the pipeline at that stage.

This module is the ORCHESTRATOR — it shells out to the real stage entrypoints
(train.py, gate.py, register.py) so each stage stays independently runnable and
testable. In a managed setup these same steps become an Argo Workflows / Airflow DAG;
the logic here is identical, just expressed as a DAG.
"""

import argparse
import json
import subprocess
import sys
import time


def log(stage: str, status: str, **extra) -> None:
    print(json.dumps({"ts": round(time.time(), 3), "stage": stage, "status": status, **extra}))


class PipelineError(RuntimeError):
    """Raised when a stage fails; halts the pipeline at that stage."""


def run_stage(name: str, cmd: list[str], dry_run: bool) -> None:
    """Run one stage as a subprocess. Non-zero exit -> halt the whole pipeline."""
    log(name, "start", cmd=" ".join(cmd))
    if dry_run:
        log(name, "skipped_dry_run")
        return
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.returncode != 0:
        log(name, "failed", exit_code=result.returncode, stderr=result.stderr[-500:])
        raise PipelineError(f"stage '{name}' failed with exit code {result.returncode}")
    log(name, "ok")


def stage_retrain(cfg: dict, dry_run: bool) -> None:
    """Retrain from BASE on old+new data. Never from the deployed model."""
    run_stage(
        "retrain",
        [
            sys.executable,
            "-m",
            "src.training.train",
            "--config",
            cfg["train_config"],
            "--run-name",
            cfg["run_name"],
        ],
        dry_run,
    )


def stage_gate(cfg: dict, dry_run: bool) -> None:
    """Evaluate + gate. Exits non-zero (halts pipeline) if the model fails."""
    run_stage(
        "gate",
        [
            sys.executable,
            "-m",
            "src.evaluation.gate",
            "--model",
            cfg["candidate_model"],
            "--golden",
            cfg["golden_set"],
        ],
        dry_run,
    )


def stage_register(cfg: dict, dry_run: bool) -> None:
    """Register (tag the Hub repo) — refuses if the gate didn't pass."""
    run_stage(
        "register",
        [
            sys.executable,
            "-m",
            "src.evaluation.register",
            "--model",
            cfg["candidate_model"],
            "--version",
            cfg["version"],
        ],
        dry_run,
    )


def stage_deploy(cfg: dict, dry_run: bool) -> None:
    """Deploy via GitOps: bump the revision in Git; Argo syncs the cluster.

    We do NOT kubectl-apply here — the pipeline writes the new revision into the
    manifest and commits. Argo CD (Day 6) detects the commit and rolls it out.
    That keeps deploy auditable and revertible (git revert = rollback)."""
    # In a real run this edits k8s/ (the DOMAINBOT_REVISION / canary vars) and commits.
    # Shown as an explicit, logged step rather than a hidden kubectl call.
    run_stage(
        "deploy",
        [
            "bash",
            "-c",
            f"echo 'bump revision -> {cfg['version']} in k8s manifest, commit, push (Argo syncs)'",
        ],
        dry_run,
    )


def stage_verify(cfg: dict, dry_run: bool) -> None:
    """Smoke-check the rollout: readiness + one request against the new revision."""
    run_stage(
        "verify",
        [
            "bash",
            "-c",
            f"echo 'verify /readyz + smoke request; confirm revision == {cfg['version']}'",
        ],
        dry_run,
    )


STAGES = [
    ("retrain", stage_retrain),
    ("gate", stage_gate),
    ("register", stage_register),
    ("deploy", stage_deploy),
    ("verify", stage_verify),
]


def run_pipeline(cfg: dict, start_at: str = "retrain", dry_run: bool = False) -> int:
    """Run the full lifecycle. Halts at the first failing stage (non-zero exit)."""
    log("pipeline", "start", version=cfg["version"], start_at=start_at, dry_run=dry_run)
    started = False
    try:
        for name, fn in STAGES:
            if name == start_at:
                started = True
            if not started:
                log(name, "skipped_before_start")
                continue
            fn(cfg, dry_run)
    except PipelineError as e:
        log("pipeline", "halted", reason=str(e))
        return 1
    log("pipeline", "success", version=cfg["version"])
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Automated retrain->gate->register->deploy pipeline")
    ap.add_argument("--version", required=True, help="target version tag, e.g. v1.2.0")
    ap.add_argument("--candidate-model", default="vinmlops/domainbot-1.5b-rank32")
    ap.add_argument("--golden-set", default="data/golden/golden_set.jsonl")
    ap.add_argument("--train-config", default="configs/train.yaml")
    ap.add_argument("--run-name", default="retrain")
    ap.add_argument(
        "--start-at",
        default="retrain",
        choices=[s[0] for s in STAGES],
        help="skip earlier stages (e.g. --start-at gate to re-gate an existing model)",
    )
    ap.add_argument("--dry-run", action="store_true", help="log the plan without executing stages")
    args = ap.parse_args()

    cfg = {
        "version": args.version,
        "candidate_model": args.candidate_model,
        "golden_set": args.golden_set,
        "train_config": args.train_config,
        "run_name": args.run_name,
    }
    sys.exit(run_pipeline(cfg, start_at=args.start_at, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
