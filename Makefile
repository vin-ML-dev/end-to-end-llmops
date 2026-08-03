.PHONY: help setup data validate test lint clean

help:  ## show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## install deps + pre-commit hooks
	pip install -e ".[dev]"
	pre-commit install
	python -m spacy download en_core_web_sm

data:  ## rebuild the dataset pipeline (dvc)
	dvc repro

validate:  ## validate the processed dataset schema
	python -m src.data.validate

test:  ## run unit tests
	pytest tests/ -v

lint:  ## lint + format
	ruff check --fix .
	ruff format .

clean:  ## remove interim artifacts and caches
	rm -rf data/interim/* .pytest_cache .ruff_cache

# --- Day 2: training (run on a GPU box) ---
train-setup:  ## install GPU training deps
	pip install -r requirements-train.txt

train:  ## run the baseline QLoRA fine-tune
	MLFLOW_TRACKING_URI=file:outputs/mlruns python -m src.training.train --config configs/train.yaml

mlflow-ui:  ## open the MLflow UI on the local run store
	mlflow ui --backend-store-uri file:outputs/mlruns --port 5000

compare:  ## compare tracked runs
	MLFLOW_TRACKING_URI=file:outputs/mlruns python -m src.training.compare_runs

# --- Day 3: evaluation, gate, registry ---
gate:  ## run the golden-fact gate against the model in configs/eval.yaml
	python -m src.evaluation.gate --config configs/eval.yaml

gate-test:  ## prove the gate blocks (unit tests, no GPU)
	pytest tests/test_evaluation.py -v

register:  ## register an approved model (usage: make register VERSION=v1.0.0)
	python -m src.evaluation.register --config configs/eval.yaml --version $(VERSION)

# --- Day 4: serving (gateway) ---
serve-setup:  ## install gateway deps
	pip install -r requirements-serve.txt

serve:  ## run the gateway locally (engine must be running separately)
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

serve-test:  ## test the gateway (no engine/GPU needed)
	pytest tests/test_serving.py -v

docker-build:  ## build the gateway image (tag with git sha)
	docker build -t domainbot-gateway:$(shell git rev-parse --short HEAD) -t domainbot-gateway:local .

docker-run:  ## run the gateway container
	docker run --rm -p 8000:8000 \
	  -e DOMAINBOT_ENGINE_URL=http://host.docker.internal:8001/v1 \
	  domainbot-gateway:local

# --- Day 5: Kubernetes ---
k8s-validate:  ## validate manifests without a cluster
	pytest tests/test_k8s_manifests.py -v
	kubectl apply -k k8s/ --dry-run=client >/dev/null && echo "kustomize OK"

k8s-deploy:  ## apply everything to the current cluster
	kubectl apply -k k8s/

k8s-status:  ## see pods, services, hpa
	kubectl -n domainbot get pods,svc,hpa

k8s-rollback:  ## undo last gateway rollout
	./scripts/rollback.sh deploy

k8s-logs:  ## tail gateway logs
	kubectl -n domainbot logs -l component=gateway -f
