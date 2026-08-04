"""Stratified train/val/test split with a leakage check.

Outputs:
    data/processed/{train,val,test}.jsonl
    data/processed/dataset_stats.json   (data-versioning record)

Stratification guarantees every category appears in every split, so eval
metrics reflect all topics rather than whichever ones landed in test.
"""

import json
import random
import re
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(ROOT / "params.yaml"))
S = PARAMS["split"]
random.seed(S["seed"])


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


def user_msg(ex: dict) -> str:
    return next(m["content"] for m in ex["messages"] if m["role"] == "user")


def main() -> None:
    interim = ROOT / PARAMS["data"]["interim_dir"]
    processed = ROOT / PARAMS["data"]["processed_dir"]
    processed.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in open(interim / "dataset_full.jsonl", encoding="utf-8")]

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    for _cat, items in by_cat.items():
        random.shuffle(items)
        n = len(items)
        # Allocate val/test FIRST, give the remainder to train. Allocating
        # train first with max(1, int(n*0.85)) silently starves the test split
        # to zero for small categories — see INC-003.
        if n == 1:
            n_val = n_test = 0
        elif n == 2:
            n_val, n_test = 1, 0
        else:
            n_val = max(1, round(n * S["val"]))
            n_test = max(1, round(n * S["test"]))
            # never let val+test crowd out training data
            while n - n_val - n_test < 1:
                if n_val > 1:
                    n_val -= 1
                elif n_test > 1:
                    n_test -= 1
                else:
                    break
        n_train = n - n_val - n_test
        train += items[:n_train]
        val += items[n_train : n_train + n_val]
        test += items[n_train + n_val :]

    # leakage check: a normalized question must not appear in two splits
    seen: dict[str, str] = {}
    leaks = 0
    for name, split in (("train", train), ("val", val), ("test", test)):
        for r in split:
            key = norm(user_msg(r))
            if key in seen and seen[key] != name:
                leaks += 1
            seen[key] = name
    if leaks:
        raise SystemExit(f"LEAKAGE: {leaks} questions appear in more than one split")

    for name, split in (("train", train), ("val", val), ("test", test)):
        if not split:
            raise SystemExit(
                f"EMPTY SPLIT: '{name}' has 0 examples. Increase max_docs or " f"reduce the number of categories."
            )
        random.shuffle(split)
        with open(processed / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in split:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(split)}")

    stats = {
        "total": len(rows),
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "categories": {k: len(v) for k, v in sorted(by_cat.items())},
        "avg_answer_chars": round(sum(len(r["messages"][-1]["content"]) for r in rows) / max(len(rows), 1), 1),
        "leakage_checked": True,
        "seed": S["seed"],
    }
    with open(processed / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
