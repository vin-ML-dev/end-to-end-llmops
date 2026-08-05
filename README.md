# DomainBot — Production LLMOps Platform

> Fine-tune → gate → serve → deploy → observe → retrain.
> A complete LLM lifecycle on Kubernetes with CI/CD, caching, A/B rollout,
> observability, and automated retraining.

**Status:** ✅ Day 6/10 — automated CI/CD (GitHub Actions → GHCR → Argo CD GitOps); gateway live on Kubernetes
**Dataset:** [`Open-Orca/OpenOrca`](https://huggingface.co/datasets/Open-Orca/OpenOrca) — ~4.2M GPT-3.5/GPT-4 instruction pairs over FLAN prompts (MIT)

---

## Architecture

```
OpenOrca → curate (quality · PII · dedup) → chat format → train/val/test
                                                              ↓
                                              QLoRA fine-tune (MLflow)
                                                              ↓
                              merge → push to HF Hub → GATE → register v1.1.0
                                                              ↓
        push to main → CI (test·build·push git-sha) → Argo CD (GitOps) → K8s
                                                              ↓
        [user] → [K8s: FastAPI gateway] ──HTTP──▶ [external model endpoint (RunPod)]
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
| PII scrubbed | email · phone · card · SSN only (locations/dates preserved — INC-013) |
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
- **PII scrubbing is narrow (INC-013):** only email/phone/card/SSN — never locations, dates, or
  names (legitimate content). Over-scrubbing once leaked `<LOCATION>` into training answers.

## Docs

- [Decisions (ADRs)](DECISIONS.md) — why each tool was chosen, and the tradeoffs
- [Incidents](INCIDENTS.md) — failures, root causes, preventions
- [Runbook](RUNBOOK.md) · [Costs](COSTS.md) · [Day 1 notes](docs/theory-day1.md)
- [COMMANDS.txt](COMMANDS.txt) · day2 · day3 · day4 · day5 · [day6](COMMANDS-day6.txt) — every command, in order



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

| Run | lora_r | epochs | eval_loss | notes |
|---|---|---|---|---|
| baseline | 16 | 3 | 1.3181 | tied; emitted a placeholder on a fact |
| more_epochs | 16 | 5 | 1.3174 | overfit after epoch 2 (early-stopped) |
| **rank32** ✅ | 32 | 3 | **1.3156** | winner: lowest loss + cleanest generations |

All within **0.003 eval_loss** — a tie broken by reading generations. Later retrained on
PII-corrected data → deployed **v1.1.0**.

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

> **The gate worked in both directions.** Run 1 blocked at 82% — the per-category condition caught
> `honesty` at 33% (a floor-only gate would miss it). The model was actually *right*; the golden
> patterns were too narrow (false-positive block) → broadened → 100%, approved. Separately, INC-013
> (PII `<LOCATION>` leak) was traced data→weights→gate→production and fixed at the source; the golden
> set now bans placeholder tags. Clean retrain → **v1.1.0**. The set is never loosened to pass a
> *bad* model — only fixed when it wrongly rejects a *good* one.


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

## Day 5 — Kubernetes (gateway-only) → external model endpoint

The model runs on an **external cloud platform**; we have only its API endpoint. Kubernetes runs
**only the FastAPI gateway**, which calls out to that endpoint and returns responses to users.
No GPU, no engine, in our cluster — a common, cost-effective production pattern (managed inference).

```
[user] → [K8s: FastAPI gateway] ──HTTP──▶ [external cloud model endpoint /v1]
              probes · auth · limits            (managed vLLM — we just call it)
```

- **Gateway-only in K8s (ADR-016)** — the platform owns GPU scheduling, weights, and engine
  scaling; we own auth, validation, the system prompt, limits, health, and logging. The gateway
  forwards an optional Bearer token upstream, so it works with managed APIs that require auth.
- **Probes** — `livenessProbe → /healthz` (process; fail = restart); `readinessProbe → /readyz`
  (pings the **external endpoint**; fail = stop routing traffic, no restart). If the endpoint is
  down, pods go NotReady but aren't restarted — correct handling of an upstream outage.
- **HPA** autoscales the (CPU-only) gateway 2→10 on load. **PDB** keeps ≥1 during disruptions.
- **Rollback** — `rollout undo` for gateway code; a ConfigMap change for the reported model version.

```bash
make k8s-validate                                  # 9 manifest tests, no cluster
kubectl -n domainbot create secret generic domainbot-secrets \
  --from-literal=DOMAINBOT_ENGINE_URL="https://your-endpoint/v1" \
  --from-literal=DOMAINBOT_API_KEY="$(openssl rand -hex 16)" --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -k k8s/                              # deploy the gateway
kubectl -n domainbot port-forward svc/domainbot-gateway 8000:8000
```

**Two keys, two doors:** `DOMAINBOT_API_KEY` guards the gateway (client → gateway);
`DOMAINBOT_ENGINE_API_KEY` authenticates the gateway to the external endpoint (gateway → RunPod).
`api_key_env` in `serving.yaml` must point at the *gateway's* key — crossing them makes the gateway
demand the upstream key from clients (a real bug hit and fixed). In K8s, both keys come from the
Secret, not `.env`.

> **Verified end-to-end:** gateway on a kind cluster → RunPod Serverless vLLM (OpenAI `/openai/v1`)
> → clean *"The capital of Japan is Tokyo."* Readiness correctly held traffic (0/1, no restart) when
> the endpoint was unreachable. Swap `DOMAINBOT_ENGINE_URL` to move providers with zero code change.
> This gateway-to-external-endpoint architecture is the standing setup for Days 6–10.

## Day 6 — CI/CD: GitHub Actions + Argo CD (GitOps) + Terraform + Helm

Every push to main is tested, built, and rolled out automatically. Deploy = merge;
rollback = `git revert`. The gateway ships through the pipeline; the model stays external.

```
push to main → CI: lint → test → build+push image(git-<sha>) → bump kustomization.yaml
                                                                       ↓ (git commit)
                                              Argo CD (in-cluster) detects → syncs cluster
```

- **CI (`.github/workflows/ci.yaml`)** — lint (ruff) + pytest + validate k8s manifests, then
  build and push `ghcr.io/.../domainbot-gateway:git-<sha>`. **Tests gate the build**
  (`build-push needs: test`) — untested code can never ship (ADR-017).
- **GitOps (Argo CD, `argocd/application.yaml`)** — the cluster *pulls* desired state from Git.
  CI writes the new image tag into `k8s/kustomization.yaml`; Argo detects the commit and syncs.
  `selfHeal` reverts manual drift; `prune` removes deleted resources. Rollback = `git revert`
  (ADR-018). No CI system holds cluster credentials.
- **Terraform (`terraform/`)** — declaratively provisions the namespace + secrets (and, in a cloud
  setup, the cluster itself). Infra layer, separate from app deploy (ADR-019).
- **Helm (`helm/domainbot/`)** — values-based packaging as an alternative to Kustomize.

```bash
make cicd-test                         # 8 config tests, no pipeline run
kubectl apply -f argocd/application.yaml    # register the Argo app
git push                               # → CI builds → Argo deploys
```

> **Deploy and rollback are ordinary Git operations** — auditable, reviewable, revertible. The
> cluster state always equals what's in Git. This pipeline carries only the gateway; the model
> endpoint is managed externally (Day 5), so retrains publish to the Hub and the gateway just
> points at the new revision.

**GHCR gotcha (learned the hard way):** image references must be **lowercase** — a `vin-ML-dev`
namespace fails to parse (`repository name must be lowercase`); GHCR stores it as `vin-ml-dev`. And
the kustomization `images.name` must match the deployment image *exactly* (a one-character typo
silently drops the tag → `InvalidImageName`). CI now lowercases the owner before tagging.

## What I'd do differently at 100× scale

_(written Day 10)_

## Tech stack

`Python` `HF datasets (streaming)` `DVC` `Presidio` `pydantic` `pytest` `ruff` `pre-commit`
`PEFT/QLoRA` `TRL` `bitsandbytes` `MLflow` `Hugging Face Hub`
`vLLM` `FastAPI` `Docker` `Kubernetes` `Kustomize` `Helm`
`GitHub Actions` `GHCR` `Argo CD` `Terraform`
_(growing daily: Redis, Prometheus, Grafana)_
