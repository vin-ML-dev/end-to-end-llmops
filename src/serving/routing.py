"""A/B traffic splitting between two model versions (Day 7).

Canary/A-B rollout: send a fraction of traffic to a NEW model version while most
stays on the stable one. Lets you validate a new model on real traffic before a
full switch — and roll back instantly by setting the split to 0.

Design:
  - two upstreams: 'stable' (current) and 'canary' (new). Config gives each a URL,
    revision, and the canary's traffic percentage.
  - assignment is STICKY per client (hash of client id), so a given user always
    hits the same variant within a rollout — consistent experience, and clean A/B
    measurement. Not random-per-request (which would flip a user between versions).
  - split=0 -> everyone on stable (canary off). split=100 -> full cutover.
"""

import hashlib
import os


def canary_percent() -> int:
    """0-100. How much traffic goes to the canary model."""
    try:
        return max(0, min(100, int(os.getenv("DOMAINBOT_CANARY_PERCENT", "0"))))
    except ValueError:
        return 0


def pick_variant(client_id: str) -> str:
    """Sticky assignment: 'canary' or 'stable'. Same client -> same variant."""
    pct = canary_percent()
    if pct <= 0:
        return "stable"
    if pct >= 100:
        return "canary"
    # hash client id to a 0-99 bucket; stable mapping, no per-request randomness
    bucket = int(hashlib.sha256(client_id.encode()).hexdigest(), 16) % 100
    return "canary" if bucket < pct else "stable"
