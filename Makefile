# Lifecycle shortcuts. Windows users: run the `newsclf ...` commands directly
# (see README), or use Git Bash / WSL for make.
PY ?= python
SERVING_URL ?= http://localhost:8501

.PHONY: help setup nltk validate data baseline train-all benchmark leaderboard deploy serve \
        canary promote route monitor test lint format pipeline tf-serving-up tf-serving-down mlflow-ui clean

help:            ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

setup:           ## install the package with dev + augmentation extras
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev,augment]"
	$(MAKE) nltk

nltk:            ## download WordNet for EDA synonym augmentation (optional)
	$(PY) -c "import nltk; nltk.download('wordnet', quiet=True)" || echo "WordNet unavailable: EDA falls back to swap/delete"

validate:        ## data validation report on the raw files
	newsclf validate

data:            ## build versioned splits for E2 and E3+
	newsclf prepare -e E2 E3

baseline:        ## E1: re-measure the inherited prototype
	newsclf baseline

train-all:       ## E2..E8
	newsclf train -e E2 E3 E4 E5 E6 E7 E8

benchmark:       ## B1: TF-IDF + logistic regression
	newsclf benchmark

leaderboard:     ## rank runs, select champion
	newsclf leaderboard

deploy:          ## legacy -> stable v1, champion -> canary v2
	newsclf deploy

serve:           ## local TF-Serving-compatible server on :8501
	newsclf serve

canary:          ## canary vs stable on held-out data (in-process)
	newsclf canary-check

promote:         ## canary -> stable (previous stable -> deprecated)
	newsclf promote

route:           ## human-in-the-loop split of the dev set into auto/manual csv
	newsclf route --input $$(python -c "import json;m=json.load(open('artifacts/'+json.load(open('reports/champion.json'))['experiment']+'/metrics.json'));print(m['data_dir']+'/dev.csv')")

monitor:         ## input/output monitoring on the production log
	newsclf monitor

pipeline:        ## full lifecycle end to end
	newsclf validate && newsclf run-all && newsclf deploy && newsclf canary-check --promote && $(MAKE) route && newsclf monitor

tf-serving-up:   ## TF Serving in Docker (needs `make deploy` first)
	docker compose -f serving/docker-compose.yml up -d

tf-serving-down:
	docker compose -f serving/docker-compose.yml down

mlflow-ui:       ## browse runs at http://127.0.0.1:5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db

test:            ## unit + end-to-end tests
	$(PY) -m pytest

lint:
	ruff check src tests

format:
	ruff check --fix src tests && ruff format src tests

clean:           ## delete generated artifacts (keeps data/raw and legacy model)
	rm -rf data/processed artifacts serving/models serving/registry.json mlruns mlflow.db reports/routing reports/monitoring
