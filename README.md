# DomainBot — Production LLMOps Platform

> Fine-tune → gate → serve → deploy → observe → retrain.
> A complete LLM lifecycle on Kubernetes with CI/CD, caching, A/B rollout,
> observability, and automated retraining.

**Status:** 🚧 Day 3/10 — evaluation gate passed, model registered `v1.0.0`
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
                                              merge adapter → push to HF Hub → GATE → register v1.0.0
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

## Results — Day 1 data pipeline (sample mode, 24 rows)

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
- **Known issue (INC-013):** Presidio over-scrubbed non-sensitive entities — city names
  (LOCATION) and durations (DATE_TIME) were replaced with placeholders that leaked into
  training answers. This directly caused a model failure (see Day 2 finding below). Fix
  in a future data iteration: scrub only EMAIL/PHONE/CARD/PERSON, not LOCATION/DATE_TIME.



---

## Day 2 — QLoRA fine-tuning + MLflow

Fine-tune `Qwen2.5-1.5B-Instruct` on the Day 1 dataset using **QLoRA** (4-bit NF4
base + LoRA adapters, ~1-3% of params trainable), tracked in **MLflow**.

- `configs/train.yaml` — every hyperparameter; run experiments by editing this, not code
- `src/training/train.py` — 4-bit load → LoRA → SFT → early stopping → save adapter + sample generations → log to MLflow with **data-version + git-SHA lineage**
- `src/training/compare_runs.py` — comparison table across runs
- `notebooks/v2_qlora_finetune_runpod_final.ipynb` — interactive RunPod version (train → merge → push to HF Hub)

Run (on a GPU box):
```bash
make train-setup
MLFLOW_TRACKING_URI=file:outputs/mlruns python -m src.training.train --run-name baseline
make mlflow-ui        # http://localhost:5000
```

**Three deliberate experiments** (simulating future retrains): baseline · more epochs +
lower LR · higher LoRA rank. Winner chosen by eval_loss **and** a read of the actual
generations — eval_loss alone never tells you if answers are good.

| Run | lora_r | epochs | lr | eval_loss | notes |
|---|---|---|---|---|---|
| baseline | 16 | 3 | 2e-4 | 1.3181 | got "capital of Japan" **wrong** (emitted a placeholder) |
| more_epochs | 16 | 5 | 1e-4 | 1.3174 | early-stopped at epoch 4; best at epoch 2 (overfitting) |
| **rank32** ✅ | 32 | 3 | 2e-4 | **1.3156** | correct on all prompts; cleanest honesty answer |

**Key finding:** all three eval_losses landed within **0.003** — statistically tied. The
generations broke the tie: **baseline was silently broken** on a basic fact ("capital of
Japan"), which eval_loss never revealed. Selected **rank32** (lowest loss *and* correct on
every prompt). Lesson: *eval_loss ranks token-prediction, not correctness — always read the
generations before choosing a model.*

**Overfitting observed:** the `more_epochs` run's validation loss bottomed at epoch 2 then
rose, so early stopping (patience 2) halted it at epoch 4 and kept the epoch-2 weights.
More epochs did not help on a dataset this small — a real, defensible finding for the ADR.

> Compute dtype is auto-detected: **bf16** on Ampere+ (RTX A4500/30xx/A100), **fp16** on T4.
> Hardcoding it is the #1 QLoRA portability crash (INC-008).
>
> **Environment note:** on managed GPU pods, `torch` is pinned to the driver's CUDA version —
> don't reinstall it. torch + transformers + tokenizers + training libs are one matched set,
> pinned together (INC-009…012). `notebooks/requirements-lock.txt` freezes the working environment
> so a fresh pod installs in one line.

## Day 3 — Evaluation, quality gate, regression suite, registry

The heart of production ML: a gate that can **block a bad model from shipping**, and a model
registry with versioned, rollback-able releases.

- `src/evaluation/evaluate.py` — pure scoring logic (GPU-free, unit-tested): a golden case passes
  iff a `must_contain` pattern matches **and** no `must_not_contain` pattern does. The bans make it
  a **regression suite** — every past hallucination stays permanently forbidden.
- `src/evaluation/gate.py` — loads the model, runs all golden cases, applies the gate, and
  **exits non-zero if blocked** (so CI / the retrain pipeline halts). Three conditions, all required:
  1. **absolute floor** — ≥ 80% pass
  2. **no regression** — ≥ the currently-deployed model
  3. **no slice collapse** — no category below 50%
- `src/evaluation/register.py` — re-runs the gate, then tags the HF repo `v1.0.0` and writes the
  model card. Refuses to register an unapproved model.
- `notebooks/day3_evaluation_gate_runpod.ipynb` — interactive RunPod version.

```bash
make gate-test                         # prove the gate BLOCKS (10 tests, no GPU)
make gate                              # run it against your model (GPU)
make register VERSION=v1.0.0           # register only if it passed
```

### The gate did its job — twice

**Run 1 — BLOCKED at 82%.** Overall pass rate cleared the 80% floor, but the **per-category
condition caught that `honesty` had collapsed to 33%** (2/3 fail) — exactly the slice failure an
aggregate-only threshold would have missed.

| category | run 1 | run 2 |
|---|---|---|
| cot | 4/5 (80%) | 5/5 |
| flan | 6/7 (86%) | 7/7 |
| **honesty** | **1/3 (33%)** ❌ | 3/3 ✅ |
| niv | 2/2 | 2/2 |
| t0 | 3/3 | 3/3 |
| quality | 2/2 | 2/2 |
| **overall** | 18/22 (82%) **BLOCKED** | **22/22 (100%) APPROVED** |

**The investigation is the lesson.** Reading the failures showed the model's answers were
*actually correct* — it refused appropriately ("I cannot provide information about your specific
bank account balance...") — but the golden `must_contain` patterns were too narrow to recognize
those valid refusals. A few `must_not_contain` bans were also too crude (banning the bare digit
`"15"`, which appears in the question "15% of 200").

These were **false-positive blocks**: the gate wrongly rejecting a good model, not a bad model
slipping through. The fix was to **broaden `must_contain` to match semantically-valid refusals**
and **anchor `must_not_contain` to ban only actual wrong answers** (`"is 30"`, not bare `"30"`).
Re-ran → **100%, APPROVED legitimately**, and registered `v1.0.0`.

> **This is distinct from editing the golden set to pass a bad model.** Here the model was right
> and the *tests* were broken. The rule holds: never loosen a test to pass a wrong answer — but
> fixing a brittle test that rejects a correct answer is correct. The check: *would a human reading
> this answer call it right?* For the refusals, yes.
>
> **Takeaway:** a gate can wrongly **block** good models as well as wrongly **pass** bad ones.
> Golden-set quality is as important as model quality, and tuning it is real work.

**Registry:** the approved model is tagged `v1.0.0` on the Hub (semantic versioning — MAJOR = new
base/prompt, MINOR = new data, PATCH = hyperparameters). Rollback later = point serving at a prior
tag; one line, no retraining.

## What I'd do differently at 100× scale

_(written Day 10)_

## Tech stack

`Python` `HF datasets (streaming)` `DVC` `Presidio` `langdetect` `pydantic` `pytest` `ruff` `pre-commit`
`PEFT/QLoRA` `TRL` `bitsandbytes` `MLflow` `Hugging Face Hub (registry + tags)`
_(growing daily: vLLM, FastAPI, Docker, Kubernetes, Helm, Argo CD, Terraform, Redis,
Prometheus, Grafana)_
