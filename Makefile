.PHONY: help setup data validate test lint clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:        ## install deps + hooks
	pip install -e ".[dev]" && pre-commit install

data:         ## rebuild the dataset pipeline
	dvc repro

validate:     ## validate processed data
	python -m src.data.validate

test:         ## run unit tests
	pytest tests/ -v

lint:         ## lint + format
	ruff check --fix . && ruff format .

clean:        ## remove interim artifacts
	rm -rf data/interim/* .pytest_cache .ruff_cache
