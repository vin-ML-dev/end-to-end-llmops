# DomainBot — Production LLMOps Platform

> Fine-tune → gate → serve → deploy → observe → retrain.
> A complete LLM lifecycle on Kubernetes with CI/CD, caching, A/B rollout,
> observability, and automated retraining.

**Status:** 🚧 Day 2/10 — QLoRA fine-tuning complete
**Dataset:** [`Open-Orca/OpenOrca`](https://huggingface.co/datasets/Open-Orca/OpenOrca) — ~4.2M GPT-3.5/GPT-4 instruction pairs over FLAN prompts (MIT)

---

## Architecture

_(diagram lands Day 5)_

```
OpenOrca (stream) → quality filters → PII scrub → dedup → chat format → train/val/test
                          ↓                                                    ↓
                  curation_report.json                            (Day 2: QLoRA fine-tune)
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

## Docs

- [Decisions (ADRs)](DECISIONS.md) — why each tool was chosen, and the tradeoffs
- [Incidents](INCIDENTS.md) — failures, root causes, preventions
- [Runbook](RUNBOOK.md) · [Costs](COSTS.md) · [Day 1 notes](docs/theory-day1.md)
- [COMMANDS.txt](COMMANDS.txt) — every command for Day 1, in order



---

## Day 2 — QLoRA fine-tuning + MLflow

Fine-tune `Qwen2.5-0.5B-Instruct` on the Day 1 dataset using **QLoRA** (4-bit NF4
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

| Run | eval_loss | notes |
|---|---|---|
| baseline | _fill in_ | |
| more_epochs | _fill in_ | |
| rank32 | _fill in_ | |

> Compute dtype is auto-detected: **bf16** on Ampere+ (RTX 30xx/A100), **fp16** on T4.
> Hardcoding it is the #1 QLoRA portability crash (INC-008).

## What I'd do differently at 100× scale

_(written Day 10)_

## Tech stack

`Python` `HF datasets (streaming)` `DVC` `Presidio` `langdetect` `pydantic` `pytest` `ruff` `pre-commit`
_(growing daily: PEFT/QLoRA, MLflow, vLLM, FastAPI, Docker, Kubernetes, Helm, Argo CD,
Terraform, Redis, Prometheus, Grafana)_
