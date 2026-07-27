"""
Checks that the final training file (dataset_full.jsonl) is correctly formatted
before we actually use it for fine-tuning. Every line must be a valid JSON
"conversation" that follows the rules a chat model expects:
  - starts with a system message
  - ends with an assistant message
  - assistant's reply isn't empty

If anything is wrong, this prints the errors and exits with a failure code
(so it can be used as a CI check that blocks a bad dataset from going further).
"""

import json
import sys
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(PROJECT_ROOT / "params.yaml"))


# ---------------------------------------------------------------------------
# Data shape: what a valid row is supposed to look like
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """One message in a conversation, e.g. {"role": "user", "content": "hi"}"""

    role: str = Field(pattern="^(system|user|assistant)$")  # must be one of these 3
    content: str = Field(min_length=1)  # can't be empty


class Example(BaseModel):
    """One full training row: a list of messages + which category it came from."""

    messages: list[Message] = Field(min_length=3)  # at least system+user+assistant
    category: str = Field(min_length=1)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_line(line: str, line_number: int) -> list[str]:
    """Check one line of the file. Returns a list of error messages (empty if OK)."""
    errors = []

    # Step 1: parse the JSON and check it matches the Example shape above
    try:
        example = Example(**json.loads(line))
    except (ValidationError, json.JSONDecodeError) as e:
        errors.append(f"line {line_number}: {str(e)[:120]}")
        return errors  # can't check further if it didn't even parse

    roles = [message.role for message in example.messages]

    # Step 2: conversation must start with "system"
    if roles[0] != "system":
        errors.append(f"line {line_number}: first message must be system, got {roles[0]}")

    # Step 3: conversation must end with "assistant"
    if roles[-1] != "assistant":
        errors.append(f"line {line_number}: last message must be assistant, got {roles[-1]}")

    # Step 4: the assistant's final reply must not be blank/whitespace-only
    if not example.messages[-1].content.strip():
        errors.append(f"line {line_number}: empty assistant content")

    return errors


def validate_file(path: Path) -> tuple[int, list[str]]:
    """Check every line in the file. Returns (how many lines checked, all errors)."""
    checked_count = 0
    all_errors = []

    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue  # skip blank lines

            checked_count += 1
            all_errors.extend(validate_line(line, line_number))

    return checked_count, all_errors


def print_results(checked_count: int, errors: list[str]) -> None:
    print(f"Validated {checked_count} examples, {len(errors)} errors")

    # only show the first 20 errors so the output doesn't get overwhelming
    for error in errors[:20]:
        print("  ", error)

    if len(errors) > 20:
        print(f"   ... and {len(errors) - 20} more")


def main() -> None:
    path = PROJECT_ROOT / PARAMS["data"]["interim_dir"] / "dataset_full.jsonl"

    checked_count, errors = validate_file(path)
    print_results(checked_count, errors)

    # exit code 1 (failure) if there were any errors, 0 (success) otherwise
    # -> lets this be used as a pass/fail check in a CI pipeline
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
