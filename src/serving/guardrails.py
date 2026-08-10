"""Input + output guardrails (Day 9).

A production LLM gateway is the trust boundary. Two failure modes to defend:

  INPUT  — a user sends something we must not process: a prompt-injection attempt
           ("ignore your instructions..."), a request for disallowed content, or
           PII we don't want to forward to a third-party endpoint.
  OUTPUT — the model returns something we must not ship: leaked secrets, the banned
           <LOCATION>-style placeholder from INC-013, or unsafe content.

Design principles:
  - guardrails are DETERMINISTIC and cheap (regex/keyword) at this layer — fast, no
    extra model call. A heavier LLM-based classifier can sit behind this, but the
    first line is cheap rules that fail fast.
  - input guard runs BEFORE the model (block early, save the upstream call).
  - output guard runs AFTER the model, BEFORE returning (catch what slipped through).
  - each block is logged + counted (metrics) so you can SEE what's being blocked.
  - fail CLOSED on a real safety hit (block), but never crash — a guard error must
    not take down serving.
"""

import re

# --- INPUT: prompt-injection / jailbreak patterns -------------------------------
# Not exhaustive — the point is the PATTERN of defense, layered before the model.
INJECTION_PATTERNS = [
    r"ignore (all |your |previous )?(instructions|prompts?|rules)",
    r"disregard (the |your )?(above|previous|system)",
    r"you are now (a |an )?\w+",  # role reassignment
    r"pretend (you are|to be)",
    r"reveal (your |the )?(system )?(prompt|instructions)",
    r"what (is|are) your (system )?(prompt|instructions)",
]

# --- INPUT: PII we won't forward to a third-party endpoint ----------------------
PII_PATTERNS = {
    "email": r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
    "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
}

# --- OUTPUT: things that must never reach the user ------------------------------
# INC-013 regression guard: the training-data scrub placeholders must NEVER appear
# in a served answer. If they do, block — a defect got past the gate.
OUTPUT_BANNED = [
    r"<LOCATION>",
    r"<PERSON>",
    r"<DATE_TIME>",
    r"<NRP>",
    r"<EMAIL_ADDRESS>",
    r"<PHONE_NUMBER>",
    r"<CREDIT_CARD>",
    r"<US_SSN>",
    r"<IP_ADDRESS>",
]


def check_input(text: str) -> tuple[bool, str]:
    """Returns (allowed, reason). allowed=False -> block before calling the model."""
    low = text.lower()
    for pat in INJECTION_PATTERNS:
        if re.search(pat, low):
            return False, "prompt_injection"
    for kind, pat in PII_PATTERNS.items():
        if re.search(pat, text):
            return False, f"pii_{kind}"
    return True, ""


def check_output(text: str) -> tuple[bool, str]:
    """Returns (allowed, reason). allowed=False -> do not return this to the user."""
    for pat in OUTPUT_BANNED:
        if re.search(pat, text):
            return False, "banned_placeholder"
    return True, ""
