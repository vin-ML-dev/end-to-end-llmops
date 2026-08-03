# DomainBot — Production LLMOps Platform

> Fine-tune → gate → serve → deploy → observe → retrain.
> A complete LLM lifecycle on Kubernetes with CI/CD, caching, A/B rollout,
> observability, and automated retraining.

**Status:** 🚧 Day 4/10 — serving live (vLLM + FastAPI gateway + Docker); model `v1.1.0` deployed
**Dataset:** [`Open-Orca/OpenOrca`](https://huggingface.co/datasets/Open-Orca/OpenOrca) — ~4.2M GPT-3.5/GPT-4 instruction pairs over FLAN prompts (MIT)
**Base model:** `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0)

---

## Architecture

_(diagram lands Day 5)_

```
OpenOrca (stream) → quality filters → PII scrub → dedup → chat format → train/val/test
                          ↓                                                    ↓
                  curation_report.json                          QLoRA fine-tune (MLflow)
                                                                             ↓
                              merge adapter → push to HF Hub → GATE → register v1.1.0
                                                                             ↓
                                          vLLM engine ← FastAPI gateway ← client
```

## Quickstart

```bash
make setup                      # deps + hooks + spaCy model
make data                       # dvc repro — streams OpenOrca and builds the dataset
SOURCE_MODE=sample make data    # offline: use the bundled sample, no network needed
make test
```

## Why OpenOrca

It is **machine-generated at scale**, which makes it a realistic curation exercise:
refusals, truncated completions, question echoes, repetition loops and template
artifacts all appear in the raw stream. That is the point — the filtering work here
is the same work you'd do on production logs.

Its `id` prefixes (`flan.`, `t0.`, `niv.`, `cot.`) are the FLAN submixes, which we
use as **stratification categories** so every split contains every task family.

## Results (sample mode, 24 rows)

| Metric | Value |
|---|---|
| Raw rows ingested | 24 |
| After curation | 13 (**54% survival**) |
| Removed | refusal 1 · truncated 1 · question-echo 1 · repetition 2 · all-caps 1 · non-English 1 · length 2 · exact-dup 2 |
| PII scrubbed | 1 record (EMAIL, CARD, IP) |
| Chat examples built | 17 (13 curated + 4 honesty) |
| Train / val / test | 7 / 5 / 5 |
| Golden set (frozen) | 22 cases |

> Run against real OpenOrca (`max_docs: 5000`) and expect a different mix —
> read `data/interim/curation_report.json` and tune thresholds from it, not by guessing.

## Dataset design notes

- **Chat format** (`system`/`user`/`assistant`) so TRL's `SFTTrainer` applies the
  model's own chat template — the same template must be applied at inference (Day 4).
- **One standardized system prompt**, not OpenOrca's dozens (ADR-005): a product has
  one persona, and Day 4 injects it server-side so clients cannot override it.
- **Honesty examples** teach the model to say "I don't know". OpenOrca trains it to
  always answer; without a counterweight a fine-tune confidently invents facts.
- **Golden set is frozen** — never trained on, never edited to make a model pass,
  only ever grows as production bugs are found.
- **PII scrubbing is narrow (INC-013):** scrub only email/phone/card/SSN — NOT locations,
  dates, or names, which are legitimate answer content. Over-scrubbing them once leaked
  `<LOCATION>` placeholders into training answers; the fix + full trace is in INCIDENTS.md.

## Docs

- [Decisions (ADRs)](DECISIONS.md) — why each tool was chosen, and the tradeoffs
- [Incidents](INCIDENTS.md) — failures, root causes, preventions
- [Runbook](RUNBOOK.md) · [Costs](COSTS.md) · [Day 1 notes](docs/theory-day1.md)
- [COMMANDS.txt](COMMANDS.txt) · [COMMANDS-day2.txt](COMMANDS-day2.txt) · [COMMANDS-day3.txt](COMMANDS-day3.txt) · [COMMANDS-day4.txt](COMMANDS-day4.txt)



---

## Day 2 — QLoRA fine-tuning + MLflow

Fine-tune `Qwen2.5-1.5B-Instruct` on the Day 1 dataset using **QLoRA** (4-bit NF4
base + LoRA adapters, ~1-3% of params trainable), tracked in **MLflow**.

- `configs/train.yaml` — every hyperparameter; run experiments by editing this, not code
- `src/training/train.py` — 4-bit load → LoRA → SFT → early stopping → save adapter + sample generations → log to MLflow with **data-version + git-SHA lineage**
- `src/training/compare_runs.py` — comparison table across runs

Run (on a GPU box):
```bash
make train-setup
MLFLOW_TRACKING_URI=file:outputs/mlruns python -m src.training.train --run-name baseline
make mlflow-ui        # http://localhost:5000
```

**Three deliberate experiments** (simulating future retrains): baseline · more epochs +
lower LR · higher LoRA rank. Winner chosen by eval_loss **and** a read of
`sample_generations.json` — eval_loss alone never tells you if answers are actually good.

| Run | lora_r | epochs | lr | eval_loss | notes |
|---|---|---|---|---|---|
| baseline | 16 | 3 | 2e-4 | 1.3181 | tied on loss |
| more_epochs | 16 | 5 | 1e-4 | 1.3174 | overfit after epoch 2 (early-stopped) |
| **rank32** ✅ | 32 | 3 | 2e-4 | **1.3156** | winner: lowest loss + cleanest generations |

All three landed within **0.003 eval_loss** — a tie. Generations broke it, and later a full
retrain on PII-corrected data produced the deployed **v1.1.0** (see Day 3/Day 4).

> Compute dtype is auto-detected: **bf16** on Ampere+ (RTX 30xx/A100), **fp16** on T4.
> Hardcoding it is the #1 QLoRA portability crash (INC-008).


## Day 3 — Evaluation, quality gate, regression suite, registry

The heart of production ML: a gate that can **block a bad model from shipping**.

- `src/evaluation/evaluate.py` — pure scoring logic (GPU-free, unit-tested): each golden case
  passes iff a `must_contain` pattern matches AND no `must_not_contain` pattern does. The
  `must_not_contain` bans make it a **regression suite** — every past hallucination stays banned.
- `src/evaluation/gate.py` — loads the model, runs all golden cases, applies the gate, and
  **exits non-zero if blocked** (so CI / the retrain pipeline halts). Three conditions, all required:
  1. absolute floor (≥80% pass)
  2. ≥ the currently-deployed model (no regressions)
  3. no per-category collapse (averages hide slice failures)
- `src/evaluation/register.py` — re-runs the gate, then tags the HF repo `v1.0.0` and updates
  the model card. Refuses to register an unapproved model.

```bash
make gate-test                         # prove the gate BLOCKS (10 tests, no GPU)
make gate                              # run it against your model (GPU)
make register VERSION=v1.0.0           # register only if it passed
```

> **The gate + a golden-set fix, both directions.** Run 1 **blocked** at 82% — the per-category
> condition caught `honesty` collapsing to 33% that a floor-only gate would miss. Investigation
> showed the model's refusals were *correct* but the golden patterns too narrow (false-positive
> block); broadening them → **100%, approved**. Separately, INC-013 (PII placeholder leak) was
> traced end-to-end and fixed at the data source; the golden set now bans `<LOCATION>`/`<PERSON>`
> tags so the bug can never return. Retrained clean model registered as **v1.1.0**. The golden set
> is never edited to force a *bad* model through — only fixed when it wrongly rejects a *good* one.


## Day 4 — Serving: vLLM + FastAPI gateway + Docker

The registered `v1.1.0` model becomes a production API. Two processes: an OpenAI-compatible
**inference engine** (vLLM on GPU, or llama.cpp on CPU) behind a thin **FastAPI gateway** that
owns everything that is policy rather than inference.

```
[client] → [FastAPI gateway :8000] → [vLLM / llama.cpp engine :8001]
             auth · limits · system-prompt · health · errors      inference
```

- `src/serving/app.py` — the gateway: Pydantic validation, **server-side system prompt**,
  prompt-injection defense (**client `role: system` is rejected**), SSE streaming, API-key auth,
  `/healthz` (liveness) vs `/readyz` (readiness — pings the engine), `/v1/model-info` (verify
  rollouts), and error mapping (timeout→504, engine down→503, engine error→502).
- `src/serving/schemas.py` — request/response models; the validation *is* the security boundary.
- `Dockerfile` — multi-stage, slim, **non-root**, `HEALTHCHECK`; **no weights baked in**.

```bash
make serve-test                        # 10 gateway tests, no GPU
vllm serve vinmlops/domainbot-1.5b-rank32 --revision v1.1.0 --served-model-name domainbot --host 0.0.0.0 --port 8001
make serve                             # gateway on :8000
```

**Two design decisions carry the day:**
- **Gateway in front of the engine (ADR-012)** — the engine does inference; it doesn't know your
  auth, limits, persona, or health semantics. Keeping those in a thin gateway makes the engine
  swappable (vLLM ↔ llama.cpp) with zero client changes.
- **Pull weights at a pinned revision, never bake them (ADR-013)** — the image stays tiny (code
  only), and the model version is switchable by one env var (`DOMAINBOT_REVISION`) with no rebuild
  — the basis for Day 5 rollback. "Same tag, new weights" is a silent prod killer, so pin.

> The **server-side system prompt + `apply_chat_template`** here is the same one used in training
> (Day 2) — inference must mirror training. And a client cannot override it: `role: system` from a
> client is rejected (422), closing a prompt-injection vector.

**Verified end-to-end:** vLLM served `v1.1.0` on a remote GPU (RunPod), the gateway ran locally
pointed at it via `DOMAINBOT_ENGINE_URL`, and `POST /v1/chat` returned the clean answer
*"The capital of Japan is Tokyo."* — no `<LOCATION>` placeholder, confirming the INC-013 fix in
production. Remote engine reached from the laptop via RunPod's port proxy; `vllm` bound to
`--host 0.0.0.0` so the proxy can route to it.

## What I'd do differently at 100× scale

_(written Day 10)_

## Tech stack

`Python` `HF datasets (streaming)` `DVC` `Presidio` `pydantic` `pytest` `ruff` `pre-commit`
`PEFT/QLoRA` `TRL` `bitsandbytes` `MLflow` `Hugging Face Hub (registry + tags)`
`vLLM` `FastAPI` `Docker` `GHCR`
_(growing daily: Kubernetes, Helm, Argo CD, Terraform, Redis, Prometheus, Grafana)_
