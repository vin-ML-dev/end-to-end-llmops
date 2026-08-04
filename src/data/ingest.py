"""Ingest raw instruction pairs into a normalized intermediate JSONL.

Source modes (params.yaml -> source_mode):
    openorca  stream Open-Orca/OpenOrca from the HF Hub (4.2M rows, MIT license)
    local_kb  read your own Q/A JSONL files from data/raw/
    sample    read the bundled OpenOrca-format sample (offline, no network)

Output: data/interim/ingested.jsonl
    {"question": str, "answer": str, "category": str, "source": str, "id": str}

Note on streaming: OpenOrca is ~4.2M rows. `streaming=True` iterates without
downloading the corpus. `shuffle_buffer` matters — taking the head of the stream
gives you one submix in file order, which is a biased sample.
"""

import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(ROOT / "params.yaml"))

# Env override lets CI run the pipeline offline (no network, no HF token):
#   SOURCE_MODE=sample python -m src.data.ingest
SOURCE_MODE = os.getenv("SOURCE_MODE", PARAMS["source_mode"])


def submix_of(row_id: str) -> str:
    """OpenOrca ids look like 'flan.2000000', 't0.1000', 'niv.123', 'cot.456'.

    The prefix is the FLAN submix, which we use as the stratification category
    so every split contains every task family.
    """
    if not row_id:
        return "unknown"
    return str(row_id).split(".", 1)[0].lower()


def load_openorca() -> list[dict]:
    from datasets import load_dataset

    cfg = PARAMS["openorca"]
    print(f"Streaming {cfg['repo_id']} (split={cfg['split']}, max={cfg['max_docs']})...")

    ds = load_dataset(cfg["repo_id"], split=cfg["split"], streaming=cfg["streaming"])
    if cfg.get("shuffle_buffer", 0):
        ds = ds.shuffle(seed=cfg["seed"], buffer_size=cfg["shuffle_buffer"])

    keep = set(cfg.get("keep_submixes") or [])
    records: list[dict] = []
    for row in ds:
        category = submix_of(row.get("id", ""))
        if keep and category not in keep:
            continue
        records.append(
            {
                "id": row.get("id", ""),
                "question": row.get("question", "") or "",
                "answer": row.get("response", "") or "",
                # OpenOrca ships its own varied system prompts; we keep them for
                # analysis but standardize on ours at build time (see ADR-005).
                "orig_system_prompt": row.get("system_prompt", "") or "",
                "category": category,
                "source": cfg["repo_id"],
            }
        )
        if len(records) >= cfg["max_docs"]:
            break
    return records


def load_local_jsonl(pattern: str = "*.jsonl") -> list[dict]:
    records: list[dict] = []
    raw_dir = ROOT / PARAMS["data"]["raw_dir"]
    for path in sorted(raw_dir.glob(pattern)):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            # accept both OpenOrca-style and simple q/a-style rows
            question = row.get("question", "")
            answer = row.get("response", row.get("answer", ""))
            row_id = row.get("id", "")
            records.append(
                {
                    "id": row_id,
                    "question": question,
                    "answer": answer,
                    "orig_system_prompt": row.get("system_prompt", ""),
                    "category": row.get("category") or submix_of(row_id),
                    "source": path.name,
                }
            )
    return records


def main() -> None:
    mode = SOURCE_MODE
    if mode == "openorca":
        records = load_openorca()
    elif mode == "sample":
        records = load_local_jsonl("openorca_sample.jsonl")
    elif mode == "local_kb":
        records = load_local_jsonl("*.jsonl")
    else:
        raise SystemExit(f"Unknown source_mode: {mode}")

    if not records:
        raise SystemExit(f"No records ingested for source_mode={mode}")

    out_dir = ROOT / PARAMS["data"]["interim_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ingested.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter

    by_cat = Counter(r["category"] for r in records)
    print(f"Ingested {len(records)} records (mode={mode}) -> {out_path}")
    print(f"  submixes: {dict(by_cat)}")


if __name__ == "__main__":
    main()
