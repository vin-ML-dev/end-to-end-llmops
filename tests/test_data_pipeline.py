"""Day 1 unit tests — these run in CI from Day 6 onward.

Note: CI runs the pipeline in offline sample mode (SOURCE_MODE=sample) so it
needs no network and no HF token.
"""

import json
from pathlib import Path

import pytest

from src.data.curate import jaccard, normalize, scrub_pii_regex, shingles
from src.data.ingest import submix_of
from src.data.quality_filters import (
    check_pair,
    is_refusal,
    is_truncated,
    question_echo_ratio,
)

ROOT = Path(__file__).resolve().parents[1]

Q = {
    "min_question_words": 3,
    "max_question_words": 1200,
    "min_answer_words": 5,
    "max_answer_words": 800,
    "min_mean_word_len": 2.5,
    "max_mean_word_len": 12.0,
    "min_stopwords": 2,
    "max_top_ngram_ratio": 0.25,
    "max_caps_ratio": 0.40,
    "max_symbol_word_ratio": 0.20,
    "max_question_echo_ratio": 0.90,
    "reject_refusals": True,
    "reject_truncated": True,
    "blocklist_hits_allowed": 0,
}

GOOD_Q = "What is the capital of Japan and what is it known for?"
GOOD_A = (
    "The capital of Japan is Tokyo. It is the most populous metropolitan area in "
    "the world and is known for combining historic temples with modern districts."
)


# ----------------------------------------------------------------- normalize
def test_normalize_strips_case_and_punctuation():
    assert normalize("What is the CAPITAL of Japan?") == normalize("what is the capital of japan")


def test_near_duplicate_detection():
    a = shingles("How many paid leave days do employees get per year")
    b = shingles("How many paid leave days do employees get each year")
    assert 0.4 < jaccard(a, b) < 1.0


# ------------------------------------------------------------------ submixes
@pytest.mark.parametrize(
    "row_id,expected",
    [("flan.2000000", "flan"), ("t0.1000", "t0"), ("niv.123", "niv"), ("cot.456", "cot"), ("", "unknown")],
)
def test_submix_extracted_from_openorca_id(row_id, expected):
    assert submix_of(row_id) == expected


# ------------------------------------------------------------------- filters
def test_good_pair_passes():
    ok, reason = check_pair(GOOD_Q, GOOD_A, Q)
    assert ok and reason == "ok"


def test_refusal_detected():
    assert is_refusal("As an AI language model, I do not have personal opinions.")
    ok, reason = check_pair("What do you think?", "As an AI language model, I cannot provide that opinion.", Q)
    assert not ok and reason == "refusal"


def test_truncated_answer_detected():
    assert is_truncated("The water cycle begins with evaporation, and then it")
    assert not is_truncated("The water cycle begins with evaporation.")


def test_question_echo_detected():
    ratio = question_echo_ratio("What is the capital of France?", "What is the capital of France")
    assert ratio > 0.9


def test_repetition_detected():
    ok, reason = check_pair("List the benefits.", "Exercise is good. " * 12, Q)
    assert not ok and reason == "repetition"


def test_all_caps_detected():
    ok, reason = check_pair(
        "Tell me about this product.",
        "BUY NOW LIMITED TIME OFFER BEST DEAL CLICK HERE FOR FREE SHIPPING TODAY.",
        Q,
    )
    assert not ok and reason == "all_caps"


def test_short_answer_rejected():
    ok, reason = check_pair("Describe a sunset.", "beautiful", Q)
    assert not ok and reason == "answer_length"


# ----------------------------------------------------------------------- PII
def test_pii_regex_scrubs_sensitive_only():
    # After INC-013: scrub email/phone/card/SSN, but NEVER locations or dates.
    text = "Email priya@example.com, card 4532 1122 3344 5566, SSN 123-45-6789"
    scrubbed, found = scrub_pii_regex(text)
    assert "@" not in scrubbed
    assert "EMAIL" in found and "CARD" in found and "SSN" in found


def test_pii_regex_preserves_locations_and_dates():
    # the INC-013 regression guard: "Japan"/"Tokyo"/"3 hours" must survive
    text = "The capital of Japan is Tokyo and it took 3 hours."
    scrubbed, found = scrub_pii_regex(text)
    assert scrubbed == text  # unchanged
    assert found == []


# ------------------------------------------------------------- output shapes
@pytest.mark.parametrize("split_name", ["train", "val", "test"])
def test_processed_splits_are_valid_chat_format(split_name):
    path = ROOT / "data" / "processed" / f"{split_name}.jsonl"
    if not path.exists():
        pytest.skip("run `dvc repro` first")
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    assert rows, f"{split_name} split is empty"  # regression guard for INC-003
    for r in rows:
        roles = [m["role"] for m in r["messages"]]
        assert roles[0] == "system"
        assert roles[-1] == "assistant"
        assert all(m["content"].strip() for m in r["messages"])


def test_golden_set_frozen_and_wellformed():
    rows = [
        json.loads(line)
        for line in open(ROOT / "data" / "golden" / "golden_set.jsonl", encoding="utf-8")
        if line.strip()
    ]
    assert len(rows) >= 20, "golden set should have at least 20 cases"
    for r in rows:
        assert r["must_contain"], f"{r['question']} has no required patterns"
        assert isinstance(r["must_not_contain"], list)
