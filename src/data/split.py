"""
Splits the final dataset into train / val / test files.

Key ideas:
  - Split separately WITHIN each category (so every split gets a mix of
    all task types, not just whichever category happened to shuffle first).
  - Give val/test their share FIRST, then whatever's left goes to train.
    (Doing it the other way around can leave tiny categories with a
    train split but zero val/test rows - see INC-003.)
  - Afterwards, double check no question appears in more than one split
    (that would be "leakage" - the model could partly memorize the answer
    to a question it will later be tested on).
"""

import json
import random
import re
from collections import defaultdict
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(PROJECT_ROOT / "params.yaml"))
SPLIT_SETTINGS = PARAMS["split"]

random.seed(SPLIT_SETTINGS["seed"])  # makes the "random" split reproducible


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse spaces - so two questions that
    only differ by case/punctuation are treated as the same question."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_user_message(example: dict) -> str:
    """Pull out the user's question from a conversation's message list."""
    for message in example["messages"]:
        if message["role"] == "user":
            return message["content"]
    raise ValueError("No user message found in example")


# ---------------------------------------------------------------------------
# Step 1: load data and group it by category
# ---------------------------------------------------------------------------


def load_and_group_by_category(interim_dir: Path) -> dict[str, list[dict]]:
    """Read dataset_full.jsonl and group rows by their category
    (e.g. 'flan', 't0', 'cot', 'niv')."""
    rows = [json.loads(line) for line in open(interim_dir / "dataset_full.jsonl", encoding="utf-8")]

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)

    return rows, grouped


# ---------------------------------------------------------------------------
# Step 2: decide how many rows go to val/test/train, per category
# ---------------------------------------------------------------------------


def decide_split_sizes(n: int, val_fraction: float, test_fraction: float) -> tuple[int, int, int]:
    """Given n items in a category, decide how many go to val and test
    (train gets whatever's left over). Handles small categories carefully
    so tiny categories don't end up with 0 val/test rows."""

    if n == 1:
        return n, 0, 0  # only 1 item: it can only go to train
    if n == 2:
        return n - 1, 1, 0  # 2 items: 1 train, 1 val, 0 test

    n_val = max(1, round(n * val_fraction))
    n_test = max(1, round(n * test_fraction))

    # safety net: never let val+test eat up all the rows, leaving 0 for train
    while n - n_val - n_test < 1:
        if n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break

    n_train = n - n_val - n_test
    return n_train, n_val, n_test


def split_by_category(grouped: dict[str, list[dict]]) -> tuple[list, list, list]:
    """Shuffle each category, then divide it into train/val/test using
    decide_split_sizes(). Doing this per-category (instead of on the whole
    dataset at once) keeps every split balanced across task types."""

    train, val, test = [], [], []

    for _category, items in grouped.items():
        random.shuffle(items)

        n_train, n_val, n_test = decide_split_sizes(
            len(items), SPLIT_SETTINGS["val"], SPLIT_SETTINGS["test"]
        )

        train += items[:n_train]
        val += items[n_train : n_train + n_val]
        test += items[n_train + n_val :]

    return train, val, test


# ---------------------------------------------------------------------------
# Step 3: safety checks
# ---------------------------------------------------------------------------


def check_for_leakage(train: list, val: list, test: list) -> int:
    """Make sure the same question doesn't appear in more than one split.
    Returns how many leaks were found (0 = all good)."""

    seen_in_split: dict[str, str] = {}
    leak_count = 0

    for split_name, split_rows in (("train", train), ("val", val), ("test", test)):
        for row in split_rows:
            key = normalize(get_user_message(row))

            if key in seen_in_split and seen_in_split[key] != split_name:
                leak_count += 1

            seen_in_split[key] = split_name

    return leak_count


def check_no_empty_splits(train: list, val: list, test: list) -> None:
    """Stop early with a clear error if any split ended up with 0 rows."""
    for name, split_rows in (("train", train), ("val", val), ("test", test)):
        if not split_rows:
            raise SystemExit(
                f"EMPTY SPLIT: '{name}' has 0 examples. Increase max_docs or "
                f"reduce the number of categories."
            )


# ---------------------------------------------------------------------------
# Step 4: save results
# ---------------------------------------------------------------------------


def save_splits(train: list, val: list, test: list, processed_dir: Path) -> None:
    """Shuffle and write each split to its own .jsonl file."""
    for name, split_rows in (("train", train), ("val", val), ("test", test)):
        random.shuffle(split_rows)

        with open(processed_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for row in split_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"{name}: {len(split_rows)}")


def build_and_save_stats(
    all_rows: list, grouped: dict, train: list, val: list, test: list, processed_dir: Path
) -> dict:
    """Build a summary of the split (sizes, categories, avg answer length)
    and save it to dataset_stats.json."""

    stats = {
        "total": len(all_rows),
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "categories": {cat: len(items) for cat, items in sorted(grouped.items())},
        "avg_answer_chars": round(
            sum(len(row["messages"][-1]["content"]) for row in all_rows) / max(len(all_rows), 1),
            1,
        ),
        "leakage_checked": True,
        "seed": SPLIT_SETTINGS["seed"],
    }

    with open(processed_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    interim_dir = PROJECT_ROOT / PARAMS["data"]["interim_dir"]
    processed_dir = PROJECT_ROOT / PARAMS["data"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    all_rows, grouped = load_and_group_by_category(interim_dir)
    train, val, test = split_by_category(grouped)

    leak_count = check_for_leakage(train, val, test)
    if leak_count:
        raise SystemExit(f"LEAKAGE: {leak_count} questions appear in more than one split")

    check_no_empty_splits(train, val, test)

    save_splits(train, val, test, processed_dir)
    stats = build_and_save_stats(all_rows, grouped, train, val, test, processed_dir)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
