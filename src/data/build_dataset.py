"""Convert curated instruction pairs to chat-format training examples.

Adds:
  - a standardized system prompt (see ADR-005: one persona, injected
    server-side at inference so clients cannot override it)
  - out-of-scope / honesty examples (the model should say "I don't know")

Output: data/interim/dataset_full.jsonl
"""

import json
import random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(ROOT / "params.yaml"))
SYSTEM = PARAMS["system_prompt"].strip()
random.seed(PARAMS["split"]["seed"])

# Honesty / calibration examples. OpenOrca teaches the model to always answer;
# these teach it that "I don't know" is an acceptable answer. Without them a
# fine-tuned model confidently invents facts it was never taught.
HONESTY = [
    (
        "What was my account balance last Tuesday?",
        "I don't have access to your account data, so I can't tell you your balance. "
        "You would need to check your banking app or statement for that information.",
    ),
    (
        "What will the stock market do next week?",
        "I can't predict future market movements — nobody reliably can. I can explain "
        "how markets work or what factors analysts watch, if that would help.",
    ),
    (
        "Who won the football match that finished an hour ago?",
        "I don't have access to live or real-time information, so I can't tell you "
        "that result. A sports site or news app would have it immediately.",
    ),
    (
        "What is the internal employee ID format at a company you've never seen data for?",
        "I don't know that — I have no information about that organization's internal "
        "systems. Guessing a format would be misleading, so I'd rather say I don't know.",
    ),
]


def to_chat(question: str, answer: str, category: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "category": category,
    }


def main() -> None:
    interim = ROOT / PARAMS["data"]["interim_dir"]
    records = [json.loads(line) for line in open(interim / "curated.jsonl", encoding="utf-8")]

    examples = [to_chat(r["question"], r["answer"], r["category"]) for r in records]

    for q, a in HONESTY:
        examples.append(to_chat(q, a, "honesty"))

    random.shuffle(examples)

    out_path = interim / "dataset_full.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Built {len(examples)} chat examples -> {out_path}")
    print(f"  {len(records)} curated pairs + {len(HONESTY)} honesty examples")
    print(f"  system prompt standardized to: {SYSTEM[:70]}...")


if __name__ == "__main__":
    main()
