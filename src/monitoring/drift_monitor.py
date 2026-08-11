"""Drift + performance monitor — decides WHEN to trigger a retrain (Day 10).

The retrain pipeline answers "how do we ship a new model safely." This answers the
other half: "when should we?" A production model degrades for concrete, detectable
reasons, and each maps to a signal you already collect:

  - QUALITY drift   — golden-set pass rate on live traffic drops (new failure modes).
                      Source: periodic re-eval + user thumbs-down / escalations.
  - DATA drift      — the distribution of incoming prompts shifts away from training
                      (new topics, new phrasing). Source: prompt category counts.
  - PERFORMANCE     — latency / error / cache-hit regressions. Source: Day 8 metrics.
  - VOLUME of new   — enough NEW labelled data (from prod failures added to the golden
    ground truth      set / training pool) to be worth a retrain.

The monitor is deliberately a POLICY over signals, not a magic detector: it reads
metrics + counts, applies thresholds, and emits a single decision — retrain or not,
and why. That decision can page a human OR kick off retrain_pipeline automatically.

Conservative by design: retraining is expensive and gated, so we trigger on a
sustained, meaningful signal — not a single bad day. False "retrain now" alarms cost
money and risk; a missed week of gradual drift is cheaper to catch on the next poll.
"""

from dataclasses import dataclass


@dataclass
class DriftSignals:
    """The inputs a real monitor would pull from Prometheus + the eval store."""

    golden_pass_rate: float  # latest re-eval on the golden set (0..1)
    baseline_pass_rate: float  # the deployed model's pass rate at ship time
    error_rate: float  # 5xx fraction from Day 8 metrics (0..1)
    p95_latency_s: float  # p95 from the latency histogram
    new_labelled_examples: int  # new golden/training rows since last train
    unknown_topic_rate: float  # fraction of prompts we retrieve no context for


# Thresholds — the retrain POLICY. Tuned conservatively; each is defensible.
QUALITY_DROP = 0.05  # >5 pts below baseline pass rate -> quality drift
ERROR_CEILING = 0.05  # >5% errors sustained -> something's wrong
LATENCY_CEILING_S = 3.0  # p95 above SLO
MIN_NEW_EXAMPLES = 50  # enough new ground truth to justify a retrain
UNKNOWN_TOPIC_CEILING = 0.30  # >30% of prompts hit no KB context -> data drift


def evaluate_drift(s: DriftSignals) -> dict:
    """Apply the policy. Returns {retrain: bool, reasons: [...], severity}."""
    reasons = []

    if s.baseline_pass_rate - s.golden_pass_rate > QUALITY_DROP:
        reasons.append(
            f"quality drift: pass rate {s.golden_pass_rate:.0%} is "
            f"{s.baseline_pass_rate - s.golden_pass_rate:.0%} below baseline"
        )
    if s.error_rate > ERROR_CEILING:
        reasons.append(f"error rate {s.error_rate:.0%} over ceiling {ERROR_CEILING:.0%}")
    if s.p95_latency_s > LATENCY_CEILING_S:
        reasons.append(f"p95 latency {s.p95_latency_s:.1f}s over SLO {LATENCY_CEILING_S:.0f}s")
    if s.unknown_topic_rate > UNKNOWN_TOPIC_CEILING:
        reasons.append(
            f"data drift: {s.unknown_topic_rate:.0%} of prompts hit no KB context "
            "(consider updating the knowledge base, not just retraining)"
        )

    # New-data trigger is separate: enough new ground truth is a REASON to retrain
    # even without a regression — the model can get better, and the golden set grew.
    has_new_data = s.new_labelled_examples >= MIN_NEW_EXAMPLES

    # Decide. A quality/error/latency regression is urgent; new-data alone is routine.
    urgent = any("drift" in r or "error" in r or "latency" in r for r in reasons)
    retrain = urgent or has_new_data
    if has_new_data and not urgent:
        reasons.append(f"{s.new_labelled_examples} new labelled examples (routine refresh)")

    return {
        "retrain": retrain,
        "severity": "urgent" if urgent else ("routine" if retrain else "none"),
        "reasons": reasons or ["all signals within thresholds"],
    }
