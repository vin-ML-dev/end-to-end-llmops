"""
Takes the raw ingested data and cleans it up in 4 steps, in this order:
  1. Quality check   -> drop bad question/answer pairs (too short, gibberish, etc.)
  2. PII scrubbing    -> remove emails, phone numbers, names, etc.
  3. Exact dedup      -> drop rows with the exact same question as one we've seen
  4. Near dedup       -> drop rows with a very similar (but not identical) question

Whatever survives all 4 steps gets saved to curated.jsonl, along with a report
showing exactly how many rows were dropped and why.
"""

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from src.data.quality_filters import check_pair

# ---------------------------------------------------------------------------
# Setup: load paths and settings
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(PROJECT_ROOT / "params.yaml"))
CURATION_SETTINGS = PARAMS["curation"]
QUALITY_SETTINGS = PARAMS["quality"]


# ---------------------------------------------------------------------------
# Step 2 helper: PII (personal info) scrubbing
# ---------------------------------------------------------------------------

# Regex patterns as a simple, fast fallback for structured PII
PII_PATTERNS = {
    "EMAIL": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "AADHAAR": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    "PHONE": re.compile(r"(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3,5}\)?[\s-]?)?\d{3}[\s-]?\d{4}\b"),
    "IP": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}

# Presidio (a proper NER-based PII detector) is loaded lazily, only if needed,
# since it's slower to start up than plain regex.
_presidio_analyzer = None
_presidio_anonymizer = None


def scrub_with_regex(text: str) -> tuple[str, list[str]]:
    """Quick fallback: find/replace PII using regex patterns.
    Good at structured stuff (emails, card numbers), misses names."""
    found_types = []
    for label, pattern in PII_PATTERNS.items():
        if pattern.search(text):
            found_types.append(label)
            text = pattern.sub(f"[{label}]", text)
    return text, found_types


def scrub_pii(text: str) -> tuple[str, list[str]]:
    """Remove personal info from text. Tries Presidio first (catches names,
    locations, etc. using NLP) and falls back to regex if Presidio isn't
    available or fails."""
    if not CURATION_SETTINGS["pii_enabled"]:
        return text, []

    global _presidio_analyzer, _presidio_anonymizer
    try:
        if _presidio_analyzer is None:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            _presidio_analyzer = AnalyzerEngine()
            _presidio_anonymizer = AnonymizerEngine()

        results = _presidio_analyzer.analyze(text=text, language="en")
        if not results:
            return text, []

        cleaned = _presidio_anonymizer.anonymize(text=text, analyzer_results=results)
        found_types = sorted({r.entity_type for r in results})
        return cleaned.text, found_types

    except Exception:
        # Presidio isn't installed/working -> fall back to regex.
        # We still report which entity types were found either way, so a
        # missing PERSON count is visible in the report, not silently hidden.
        return scrub_with_regex(text)


# ---------------------------------------------------------------------------
# Step 3 & 4 helpers: deduplication
# ---------------------------------------------------------------------------


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse extra spaces.
    Makes two similar-looking questions easier to compare."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)


def get_shingles(text: str, size: int = 3) -> set[str]:
    """Break text into overlapping chunks of `size` words each.
    Used to measure how similar two pieces of text are."""
    words = normalize(text).split()
    if len(words) < size:
        return {" ".join(words)}
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


def similarity(shingles_a: set[str], shingles_b: set[str]) -> float:
    """Jaccard similarity: overlap between two sets of shingles, 0 to 1."""
    if not (shingles_a | shingles_b):
        return 0.0
    return len(shingles_a & shingles_b) / len(shingles_a | shingles_b)


def get_blocking_key(text: str) -> str:
    """Group similar-looking questions by their first few words.
    We only compare questions within the same group, instead of comparing
    every question to every other question (which gets slow fast as the
    dataset grows)."""
    words = normalize(text).split()
    return " ".join(words[:4]) if words else ""


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def load_ingested_records() -> list[dict]:
    in_path = PROJECT_ROOT / PARAMS["data"]["interim_dir"] / "ingested.jsonl"
    with open(in_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def curate(records: list[dict]) -> tuple[list[dict], dict]:
    """Run all 4 cleaning steps on every record. Returns (kept_records, stats)."""

    kept_records = []
    drop_counts = Counter()  # how many rows dropped, and why
    pii_type_counts = Counter()  # which kinds of PII were found
    drops_by_category = defaultdict(Counter)

    seen_questions = set()  # for exact-dedup check
    question_groups = defaultdict(list)  # for near-dedup check (grouped by blocking key)

    for record in records:
        question = record["question"]
        answer = record["answer"]
        category = record["category"]

        # --- Step 1: quality check ---
        is_ok, drop_reason = check_pair(question, answer, QUALITY_SETTINGS)
        if not is_ok:
            drop_counts[drop_reason] += 1
            drops_by_category[category][drop_reason] += 1
            continue

        # --- Step 2: scrub personal info from both question and answer ---
        found_any_pii = False
        for field in ("question", "answer"):
            record[field], found_types = scrub_pii(record[field])
            for entity_type in found_types:
                pii_type_counts[entity_type] += 1
                found_any_pii = True
        if found_any_pii:
            drop_counts["pii_scrubbed"] += 1  # not a drop, just a counter for the report

        # --- Step 3: exact duplicate check ---
        normalized_question = normalize(record["question"])
        if normalized_question in seen_questions:
            drop_counts["exact_dup"] += 1
            drops_by_category[category]["exact_dup"] += 1
            continue
        seen_questions.add(normalized_question)

        # --- Step 4: near-duplicate check (within the same blocking group) ---
        question_shingles = get_shingles(record["question"])
        group_key = get_blocking_key(record["question"])

        is_near_duplicate = any(
            similarity(question_shingles, other) >= CURATION_SETTINGS["near_dup_threshold"]
            for other in question_groups[group_key]
        )
        if is_near_duplicate:
            drop_counts["near_dup"] += 1
            drops_by_category[category]["near_dup"] += 1
            continue
        question_groups[group_key].append(question_shingles)

        # Passed all checks!
        kept_records.append(record)

    stats = {
        "drop_counts": drop_counts,
        "pii_type_counts": pii_type_counts,
        "drops_by_category": drops_by_category,
    }
    return kept_records, stats


def save_curated_records(records: list[dict]) -> Path:
    out_dir = PROJECT_ROOT / PARAMS["data"]["interim_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "curated.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return out_path


def build_report(all_records: list[dict], kept_records: list[dict], stats: dict) -> dict:
    drop_counts = stats["drop_counts"]

    return {
        "source_mode": os.getenv("SOURCE_MODE", PARAMS["source_mode"]),
        "input_records": len(all_records),
        "kept": len(kept_records),
        "survival_rate": round(len(kept_records) / max(len(all_records), 1), 3),
        "drops": {
            reason: count
            for reason, count in sorted(drop_counts.items())
            if reason not in ("pii_scrubbed",)  # this one isn't a "drop", skip it here
        },
        "records_with_pii_scrubbed": drop_counts["pii_scrubbed"],
        "pii_entities_found": dict(stats["pii_type_counts"]),
        "categories_kept": dict(Counter(r["category"] for r in kept_records)),
        "drops_by_category": {
            cat: dict(reasons) for cat, reasons in sorted(stats["drops_by_category"].items())
        },
    }


def save_report(report: dict) -> None:
    out_dir = PROJECT_ROOT / PARAMS["data"]["interim_dir"]
    with open(out_dir / "curation_report.json", "w") as f:
        json.dump(report, f, indent=2)


def main() -> None:
    records = load_ingested_records()
    print(f"Loaded {len(records)} ingested records")

    kept_records, stats = curate(records)

    save_curated_records(kept_records)
    report = build_report(records, kept_records, stats)
    save_report(report)

    print(json.dumps(report, indent=2))
    print(
        "\n>> REVIEW STEP: sample ~50 rows of data/interim/curated.jsonl and verify the "
        "answers are actually correct. Filters remove junk; only a human confirms truth."
    )


if __name__ == "__main__":
    main()
