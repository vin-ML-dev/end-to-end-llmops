"""Validate dataset schema. Exits non-zero on failure -> blocks the DVC pipeline.

This is a GATE, not a report: `sys.exit(1)` is what stops bad data from
reaching training. Same pattern is used for model quality on Day 3.
"""

import json
import sys
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError

ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(ROOT / "params.yaml"))


class Message(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str = Field(min_length=1)


class Example(BaseModel):
    messages: list[Message] = Field(min_length=3)
    category: str = Field(min_length=1)


def main() -> None:
    path = ROOT / PARAMS["data"]["interim_dir"] / "dataset_full.jsonl"
    errors: list[str] = []
    n = 0

    for i, line in enumerate(open(path, encoding="utf-8"), 1):
        if not line.strip():
            continue
        n += 1
        try:
            ex = Example(**json.loads(line))
        except (ValidationError, json.JSONDecodeError) as e:
            errors.append(f"line {i}: {str(e)[:120]}")
            continue

        roles = [m.role for m in ex.messages]
        if roles[0] != "system":
            errors.append(f"line {i}: first message must be system, got {roles[0]}")
        if roles[-1] != "assistant":
            errors.append(f"line {i}: last message must be assistant, got {roles[-1]}")
        if not ex.messages[-1].content.strip():
            errors.append(f"line {i}: empty assistant content")

    print(f"Validated {n} examples, {len(errors)} errors")
    for e in errors[:20]:
        print("  ", e)
    if len(errors) > 20:
        print(f"   ... and {len(errors) - 20} more")

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
