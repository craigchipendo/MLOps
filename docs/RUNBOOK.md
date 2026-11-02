# Operations Runbook

Procedures for running, releasing, rolling back and monitoring the news topic classifier. Every command runs from the project root with the virtual environment active.

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev,augment]"
python -c "import nltk; nltk.download('wordnet')"   # optional, enables EDA synonym ops
pytest                                # 20 tests, about 15 s
```

Python 3.10 to 3.12 is recommended (3.13 works with TensorFlow 2.20). On Windows, TensorFlow runs CPU-only, which is fine for this project.

## 2. Release a new model

| Step | Command | Gate |
|---|---|---|
| Validate raw data | `newsclf validate` | Hard failures raise (schema, unknown topics) |
| Build splits | `newsclf prepare -e E3` | Manifest records hashes, repair and leakage reports |
| Train | `newsclf train -e E4` | Logged to `reports/experiment_tracker.csv` and MLflow |
| Compare | `newsclf leaderboard` | Champion = best dev macro-F1 among promotable runs |
| Export as canary | `newsclf export -e E4 --label canary` | Only promotable runs (data-quality gate passed) can be exported |
| Canary check | `newsclf canary-check [--url http://localhost:8501]` | Macro-F1 gain >= 0 and no topic drops > 0.05 F1, measured on titles neither model trained on |
| Promote | `newsclf promote` or `newsclf canary-check --promote` | Previous stable becomes `deprecated` and stays loaded |

`newsclf deploy` is the bootstrap shortcut: it wipes `serving/models/`, registers the legacy prototype as `stable` (v1) and exports the champion as `canary` (v2).

## 3. Roll back

```bash
newsclf registry                 # see versions and labels
newsclf promote --version 1      # make version 1 stable again
```

TF Serving re-reads `models.docker.config` every 10 seconds and the local server re-reads `serving/registry.json` per request, so no restart is needed.

## 4. Serving

| Option | Command | When |
|---|---|---|
| TF Serving (Docker) | `docker compose -f serving/docker-compose.yml up -d` | Default, any OS with Docker |
| TF Serving (native) | `bash serving/run_tf_serving_native.sh` | Linux with `tensorflow-model-server` from APT |
| Local compatible server | `newsclf serve` | No Docker; same URL scheme on :8501, plus `/v1/classify` and `/docs` |

Smoke test:

```bash
curl http://localhost:8501/v1/models/newsapp_model
curl -d '{"instances": ["NASA launches a new space telescope"]}' \
     -X POST http://localhost:8501/v1/models/newsapp_model/labels/stable:predict
```

Version 1 (legacy) expects 20 integer token ids, not strings. A string request to it returns an error by design; it shows why every new export takes raw text.

## 5. Monitoring and alarms

Run `newsclf monitor` on each batch (the production log is `data/raw/news_articles.db`). Outputs land in `reports/monitoring/`.

| Alarm | Likely cause | Action |
|---|---|---|
| Rolling OOV rate above 25% | New source, new language, new named entities | Check `flagged_inputs.csv` by domain; if a single domain, filter or tag the feed at ingestion |
| Titles flagged non-English | Source publishing in an unsupported language | Route to manual (automatic); raise with ingestion; do not auto-publish |
| PSI on title length > 0.25 | Source format change (e.g. "Live updates" headlines) | Inspect samples; re-train if the change is permanent |
| Topic mix shifts sharply | News cycle (real) or upstream bug (fake) | Compare with editors' manual labels from the review queue |
| Share below threshold rising | Concept drift; model is less sure | Pull the manual queue's labels into the next training set and re-train |

## 6. Re-training cadence

1. Export the editors' manual decisions from the review queue as labelled rows.
2. Append them to the raw data (or a new raw file listed in `data.raw_files`).
3. `newsclf validate && newsclf train -e E4 && newsclf export -e E4 --label canary && newsclf canary-check --promote`.

## 7. Reset everything

```bash
make clean        # removes processed data, artifacts, serving folder, MLflow store, generated reports
make pipeline     # rebuilds all of it
```
