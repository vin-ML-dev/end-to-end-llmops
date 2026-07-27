"""Quality heuristics for machine-generated instruction pairs (OpenOrca-style).

Why these exist: OpenOrca is ~4.2M GPT-3.5/GPT-4 completions over FLAN prompts.
Generated at that scale, a meaningful slice is unusable:

  - assistant refusals ("As an AI language model, I cannot...")  -> teaches refusal
  - truncated answers cut off by max_tokens                      -> teaches stopping mid-sentence
  - answers that just echo the question                          -> teaches nothing
  - degenerate repetition loops                                  -> teaches looping
  - template / format artifacts from the FLAN submixes           -> teaches noise

Training on these doesn't just waste compute — the model learns the defect.

Each check returns a reason string so `curation_report.json` shows exactly what
your thresholds removed. Tune from the report, not by feel.

Reference: Gopher (Rae et al. 2021) and C4 (Raffel et al. 2020) heuristics,
adapted from documents to instruction pairs.
"""

import re
from collections import Counter

STOPWORDS = {
    "the",
    "be",
    "to",
    "of",
    "and",
    "a",
    "in",
    "that",
    "have",
    "it",
    "for",
    "not",
    "on",
    "with",
    "he",
    "as",
    "you",
    "do",
    "at",
    "this",
    "but",
    "his",
    "by",
    "from",
    "they",
    "we",
    "say",
    "her",
    "she",
    "or",
    "an",
    "will",
    "my",
    "one",
    "all",
    "would",
    "there",
    "their",
    "is",
    "are",
    "was",
    "were",
    "can",
}

# Model-refusal and assistant boilerplate. Correct behavior for the teacher
# model, but poison for a fine-tune: the student learns to refuse.
REFUSAL_MARKERS = (
    "as an ai language model",
    "as an ai assistant",
    "i cannot fulfill",
    "i can't fulfill",
    "i cannot provide",
    "i'm unable to provide",
    "i am unable to provide",
    "i'm sorry, but i cannot",
    "i am sorry, but i cannot",
    "it is not appropriate for me",
    "i do not have personal opinions",
    "i don't have personal opinions",
    "i do not have access to real-time",
)

BLOCKLIST = ("xxx", "porn", "viagra", "casino bonus", "free download crack")

# Sentence-final punctuation; anything else at the end suggests truncation.
SENTENCE_END = re.compile(r'[.!?"\')\]}:;]\s*$|```\s*$|\d\s*$')
WORD = re.compile(r"\S+")


def _words(text: str) -> list[str]:
    return WORD.findall(text)


def _top_ngram_ratio(words: list[str], n: int = 3) -> float:
    """Share of the text occupied by its single most frequent n-gram."""
    if len(words) < n * 3:
        return 0.0
    grams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return 0.0
    top = Counter(grams).most_common(1)[0][1]
    return (top * n) / len(words)


def _token_set(text: str) -> set[str]:
    return {w.lower().strip(".,!?;:\"'()[]") for w in _words(text)}


def looks_english(text: str, min_stopwords: int) -> bool:
    """Language check: langdetect when available, stopword heuristic otherwise.

    The stopword fallback is weak — Spanish/Portuguese text with incidental
    matches can slip through. Install `langdetect` (or fastText lid.176) for a
    real check. Same pattern as Presidio: optional lib, graceful fallback,
    and the limitation documented rather than hidden.
    """
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        return detect(text) == "en"
    except Exception:
        words = _words(text)
        if not words:
            return False
        hits = sum(1 for w in words if w.lower().strip(".,!?;:\"'()") in STOPWORDS)
        return hits >= min_stopwords


def is_refusal(answer: str) -> bool:
    low = answer.lower()
    return any(marker in low for marker in REFUSAL_MARKERS)


def is_truncated(answer: str) -> bool:
    """Answer appears cut off mid-sentence (hit a max_tokens ceiling)."""
    stripped = answer.rstrip()
    if not stripped:
        return True
    return not bool(SENTENCE_END.search(stripped))


def question_echo_ratio(question: str, answer: str) -> float:
    """Fraction of the answer's vocabulary that is just the question's."""
    q_tokens, a_tokens = _token_set(question), _token_set(answer)
    if not a_tokens:
        return 1.0
    return len(a_tokens & q_tokens) / len(a_tokens)


def check_pair(question: str, answer: str, q: dict) -> tuple[bool, str]:
    """Return (passed, reason). `q` is the `quality` block from params.yaml."""
    if not question or not question.strip():
        return False, "empty_question"
    if not answer or not answer.strip():
        return False, "empty_answer"

    q_words, a_words = _words(question), _words(answer)

    if not (q["min_question_words"] <= len(q_words) <= q["max_question_words"]):
        return False, "question_length"
    if not (q["min_answer_words"] <= len(a_words) <= q["max_answer_words"]):
        return False, "answer_length"

    # Ordering matters: specific reasons before generic ones, so the report
    # tells you WHY data was removed rather than lumping everything into one
    # bucket. `truncated` is checked late because a malformed answer often
    # also lacks terminal punctuation and would mask the real cause.
    if q["reject_refusals"] and is_refusal(answer):
        return False, "refusal"

    if question_echo_ratio(question, answer) > q["max_question_echo_ratio"]:
        return False, "question_echo"

    if _top_ngram_ratio(a_words) > q["max_top_ngram_ratio"]:
        return False, "repetition"

    caps = sum(1 for w in a_words if len(w) > 2 and w.isupper())
    if caps / len(a_words) > q["max_caps_ratio"]:
        return False, "all_caps"

    symbols = answer.count("#") + answer.count("...") + answer.count("\u2026")
    if symbols / len(a_words) > q["max_symbol_word_ratio"]:
        return False, "symbol_ratio"

    mean_len = sum(len(w) for w in a_words) / len(a_words)
    if not (q["min_mean_word_len"] <= mean_len <= q["max_mean_word_len"]):
        return False, "mean_word_length"

    if not looks_english(answer, q["min_stopwords"]):
        return False, "not_english"

    if q["reject_truncated"] and is_truncated(answer):
        return False, "truncated"

    combined = (question + " " + answer).lower()
    if sum(1 for term in BLOCKLIST if term in combined) > q["blocklist_hits_allowed"]:
        return False, "blocklist"

    return True, "ok"
