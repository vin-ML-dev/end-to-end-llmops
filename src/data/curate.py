"""Curate ingested instruction pairs: quality filter -> PII scrub -> dedup -> report.

Outputs:
    data/interim/curated.jsonl
    data/interim/curation_report.json

Curation is NOT correctness. After this runs, a human must still sample the
survivors and verify the answers are actually right. Never train on unreviewed
machine-generated output — you cement the teacher model's mistakes.
"""

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from src.data.quality_filters import check_pair

ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(ROOT / "params.yaml"))
C = PARAMS["curation"]
Q = PARAMS["quality"]

# --------------------------------------------------------------------- PII
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
CARD = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
AADHAAR = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
PHONE = re.compile(r"(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3,5}\)?[\s-]?)?\d{3}[\s-]?\d{4}\b")
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

_ANALYZER = None
_ANONYMIZER = None


def scrub_pii_regex(text: str) -> tuple[str, list[str]]:
    """Fast regex baseline. Scrubs only structured sensitive PII:
    email, credit card, phone, SSN. Does NOT touch locations, dates, names —
    those are legitimate answer content (INC-013)."""
    found: list[str] = []
    for label, pattern in (
        ("EMAIL", EMAIL),
        ("CARD", CARD),
        ("SSN", SSN),
        ("PHONE", PHONE),
    ):
        if pattern.search(text):
            found.append(label)
            text = pattern.sub(f"[{label}]", text)
    return text, found


# Only scrub GENUINELY SENSITIVE entities. LOCATION, DATE_TIME, and NRP
# (nationality/religion) are legitimate answer content — "Tokyo", "3 hours",
# "Japanese" are not PII. Scrubbing them replaced real words with placeholders
# that leaked into training answers and taught the model to emit "<LOCATION>"
# instead of "Japan" (INC-013). We restrict Presidio to structured, sensitive
# identifiers only.
SENSITIVE_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
]


def scrub_pii(text: str) -> tuple[str, list[str]]:
    """Presidio, restricted to sensitive entities, with automatic regex fallback."""
    if not C["pii_enabled"]:
        return text, []

    global _ANALYZER, _ANONYMIZER
    try:
        if _ANALYZER is None:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            _ANALYZER = AnalyzerEngine()
            _ANONYMIZER = AnonymizerEngine()

        # entities=... is the fix: only look for sensitive types, never LOCATION/DATE_TIME/NRP
        results = _ANALYZER.analyze(text=text, entities=SENSITIVE_ENTITIES, language="en")
        if not results:
            return text, []
        anonymized = _ANONYMIZER.anonymize(text=text, analyzer_results=results)
        return anonymized.text, sorted({r.entity_type for r in results})
    except Exception:
        # Presidio unavailable (no spaCy model, import error) -> regex fallback.
        # Degradation must stay OBSERVABLE: the report shows which entity types
        # were found, so an empty PERSON count is a visible signal, not silence.
        return scrub_pii_regex(text)


# ------------------------------------------------------------------- dedup
def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)


def shingles(text: str, n: int = 3) -> set[str]:
    tokens = normalize(text).split()
    if len(tokens) < n:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0


def blocking_key(text: str) -> str:
    """Cheap LSH-style blocking: only compare pairs sharing their first tokens.

    O(n^2) all-pairs comparison is fine for 5k rows and impossible for 500k.
    This keeps near-dup detection roughly linear as max_docs grows.
    """
    tokens = normalize(text).split()
    return " ".join(tokens[:4]) if tokens else ""


def main() -> None:
    in_path = ROOT / PARAMS["data"]["interim_dir"] / "ingested.jsonl"
    records = [json.loads(line) for line in open(in_path, encoding="utf-8") if line.strip()]
    print(f"Loaded {len(records)} ingested records")

    stats: Counter = Counter()
    pii_types: Counter = Counter()
    per_category_drops: dict[str, Counter] = defaultdict(Counter)
    kept: list[dict] = []
    exact_seen: set[str] = set()
    blocks: dict[str, list[set[str]]] = defaultdict(list)

    for rec in records:
        question, answer = rec["question"], rec["answer"]

        # --- 1. quality gates (the bulk of the removal on a corpus like this)
        ok, reason = check_pair(question, answer, Q)
        if not ok:
            stats[f"dropped_{reason}"] += 1
            per_category_drops[rec["category"]][reason] += 1
            continue

        # --- 2. PII scrubbing
        scrubbed_any = False
        for field in ("question", "answer"):
            rec[field], found = scrub_pii(rec[field])
            for entity in found:
                pii_types[entity] += 1
                scrubbed_any = True
        if scrubbed_any:
            stats["pii_scrubbed"] += 1

        # --- 3. exact dedup on the normalized question
        key = normalize(rec["question"])
        if key in exact_seen:
            stats["dropped_exact_dup"] += 1
            per_category_drops[rec["category"]]["exact_dup"] += 1
            continue
        exact_seen.add(key)

        # --- 4. near dedup within a block
        sh = shingles(rec["question"])
        bkey = blocking_key(rec["question"])
        if any(jaccard(sh, prev) >= C["near_dup_threshold"] for prev in blocks[bkey]):
            stats["dropped_near_dup"] += 1
            per_category_drops[rec["category"]]["near_dup"] += 1
            continue
        blocks[bkey].append(sh)

        kept.append(rec)
        stats["kept"] += 1

    out_dir = ROOT / PARAMS["data"]["interim_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "curated.jsonl", "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "source_mode": os.getenv("SOURCE_MODE", PARAMS["source_mode"]),
        "input_records": len(records),
        "kept": len(kept),
        "survival_rate": round(len(kept) / max(len(records), 1), 3),
        "drops": {k.replace("dropped_", ""): v for k, v in sorted(stats.items()) if k.startswith("dropped_")},
        "records_with_pii_scrubbed": stats["pii_scrubbed"],
        "pii_entities_found": dict(pii_types),
        "categories_kept": dict(Counter(r["category"] for r in kept)),
        "drops_by_category": {k: dict(v) for k, v in sorted(per_category_drops.items())},
    }
    with open(out_dir / "curation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(
        "\n>> REVIEW STEP: sample ~50 rows of data/interim/curated.jsonl and verify the "
        "answers are actually correct. Filters remove junk; only a human confirms truth."
    )


if __name__ == "__main__":
    main()
