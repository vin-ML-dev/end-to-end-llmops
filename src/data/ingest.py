"""
Loads training data from either:
  - OpenOrca on Hugging Face (real data, needs internet), or
  - a local .jsonl file (for offline testing / CI)

Which one is used is controlled by `source_mode` in params.yaml,
or by setting the SOURCE_MODE environment variable to override it, e.g.:

    SOURCE_MODE=sample python ingest_simple.py
"""

import json
import os
from collections import Counter
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Setup: find the project root and load settings from params.yaml
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(PROJECT_ROOT / "params.yaml"))

# Allows overriding the mode from the command line (useful for CI, which
# has no network access and no Hugging Face token).
SOURCE_MODE = os.getenv("SOURCE_MODE", PARAMS["source_mode"])


# ---------------------------------------------------------------------------
# Helper: figure out which "category" a row belongs to
# ---------------------------------------------------------------------------


def get_category(row_id: str) -> str:
    """
    OpenOrca ids look like 'flan.2000000', 't0.1000', 'niv.123', 'cot.456'.
    The part before the dot tells us which task family the row came from.
    We use this as the category, so later on we can make sure every
    train/val/test split contains a mix of all task types.

    Example:
        get_category("flan.2000000") -> "flan"
        get_category("")             -> "unknown"
    """
    if not row_id:
        return "unknown"

    prefix = str(row_id).split(".")[0]
    return prefix.lower()


# ---------------------------------------------------------------------------
# Loaders: one function per data source
# ---------------------------------------------------------------------------


def load_openorca() -> list[dict]:
    """Stream rows directly from the OpenOrca dataset on Hugging Face."""
    from datasets import load_dataset

    settings = PARAMS["openorca"]
    print(
        f"Streaming {settings['repo_id']} (split={settings['split']}, "
        f"max={settings['max_docs']})..."
    )

    dataset = load_dataset(
        settings["repo_id"],
        split=settings["split"],  # "train" = use the training portion of the dataset
        streaming=settings["streaming"],  # True = read row-by-row instead of downloading it all
    )

    # 1. Apply native stream-level filtering first if submixes are restricted
    allowed_categories = set(settings.get("keep_submixes") or [])
    if allowed_categories:
        dataset = dataset.filter(lambda row: get_category(row.get("id", "")) in allowed_categories)

    # 2. Shuffle after filtering so the buffer only fills with targeted categories
    shuffle_buf = settings.get("shuffle_buffer", 0)
    if shuffle_buf > 0:
        dataset = dataset.shuffle(
            seed=settings["seed"],
            buffer_size=shuffle_buf,
        )

    records = []
    # 3. Pull explicitly up to max_docs from the stream
    for row in dataset:
        row_id = row.get("id", "")

        record = {
            "id": row_id,
            "question": str(row.get("question", "")).strip(),
            "answer": str(row.get("response", "")).strip(),
            # OpenOrca ships its own system prompts. We keep the original
            # for reference, but standardize on our own prompt later
            # (see ADR-005).
            "orig_system_prompt": str(row.get("system_prompt", "")).strip(),
            "category": get_category(row_id),
            "source": settings["repo_id"],
        }
        records.append(record)

        if len(records) >= settings["max_docs"]:
            break

    return records


def load_local_jsonl(filename_pattern: str) -> list[dict]:
    """
    Load rows from local .jsonl file(s) in the raw data folder.
    Accepts both OpenOrca-style rows and simple {"question", "answer"} rows.
    """
    raw_dir = PROJECT_ROOT / PARAMS["data"]["raw_dir"]
    records = []

    for file_path in sorted(raw_dir.glob(filename_pattern)):
        lines = file_path.read_text(encoding="utf-8").splitlines()

        for line in lines:
            if not line.strip():
                continue  # skip blank lines

            row = json.loads(line)
            row_id = row.get("id", "")

            # some local files use "answer" instead of "response"
            answer = row.get("response") or row.get("answer", "")

            record = {
                "id": row_id,
                "question": str(row.get("question", "")).strip(),
                "answer": str(answer).strip(),
                "orig_system_prompt": str(row.get("system_prompt", "")).strip(),
                "category": row.get("category") or get_category(row_id),
                "source": file_path.name,
            }
            records.append(record)

    return records


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------


def save_records(records: list[dict]) -> Path:
    """Write all records to a .jsonl file, one JSON object per line."""
    out_dir = PROJECT_ROOT / PARAMS["data"]["interim_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ingested.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return out_path


def print_summary(records: list[dict], mode: str, out_path: Path) -> None:
    """Print how many records were saved, and a breakdown by category."""
    category_counts = Counter(r["category"] for r in records)
    print(f"Ingested {len(records)} records (mode={mode}) -> {out_path}")
    print(f"  submixes: {dict(category_counts)}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    if SOURCE_MODE == "openorca":
        records = load_openorca()
    elif SOURCE_MODE == "sample":
        records = load_local_jsonl("openorca_sample.jsonl")
    elif SOURCE_MODE == "local_kb":
        records = load_local_jsonl("*.jsonl")
    else:
        raise SystemExit(f"Unknown source_mode: {SOURCE_MODE}")

    if not records:
        raise SystemExit(f"No records ingested for source_mode={SOURCE_MODE}")

    out_path = save_records(records)
    print_summary(records, SOURCE_MODE, out_path)


if __name__ == "__main__":
    main()
