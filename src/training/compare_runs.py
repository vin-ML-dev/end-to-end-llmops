"""Print a comparison table of MLflow runs for the QLoRA experiment.

Usage:
    python -m src.training.compare_runs
    python -m src.training.compare_runs --experiment domainbot-qlora

Reads the local MLflow store (file:outputs/mlruns by default) and shows each
run's key params and eval_loss side by side, so you can pick a winner by
numbers — then confirm with the sample_generations vibe check.
"""

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="domainbot-qlora")
    args = ap.parse_args()

    import mlflow

    os.environ.setdefault("MLFLOW_TRACKING_URI", "file:outputs/mlruns")
    exp = mlflow.get_experiment_by_name(args.experiment)
    if exp is None:
        raise SystemExit(f"No experiment named '{args.experiment}'. Run training first.")

    df = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    if df.empty:
        raise SystemExit("No runs found.")

    cols = {
        "tags.mlflow.runName": "run",
        "params.lora_r": "r",
        "params.epochs": "epochs",
        "params.learning_rate": "lr",
        "params.data_version": "data",
        "params.git_sha": "sha",
        "metrics.eval_loss": "eval_loss",
    }
    have = [c for c in cols if c in df.columns]
    view = df[have].rename(columns=cols)
    if "eval_loss" in view.columns:
        view = view.sort_values("eval_loss")

    print(view.to_string(index=False))
    if "eval_loss" in view.columns and not view["eval_loss"].isna().all():
        best = view.iloc[0]
        print(f"\n>> best by eval_loss: {best.get('run','?')} ({best['eval_loss']:.4f})")
        print(">> now open its sample_generations.json and confirm the answers look right.")


if __name__ == "__main__":
    main()
