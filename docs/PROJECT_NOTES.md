# Project Notes

Working log for the news topic classifier: what was found, what was decided and why, and the exact procedure followed at each stage. Numbers come from the committed `reports/` and `artifacts/*/metrics.json` produced by `make pipeline`.

---

## 0. Starting point

**Inherited:** one experiment folder (`E1`) with train/test CSVs, a Keras 2 SavedModel, a vocabulary and label file, a helper module (`lab_utils.py`), TF Serving config stubs, and a SQLite log of production predictions. No metrics, no dev split, no tests, no record of how the vocabulary was built.

**How each piece was handled:**

| Inherited | Now |
|---|---|
| `E1/data/*.csv` | `data/raw/` (immutable inputs) |
| `E1/model`, `E1/vocab` | `models/legacy/e1_baseline/` (baseline + rollback target, served as v1) |
| `news_articles.db` | `data/raw/news_articles.db` (monitoring input) |
| `experiment-tracker.csv` | `reports/experiment_tracker.csv` (auto-appended) + MLflow |
| `serving/models.config`, `models.config-docker` | Generated from `serving/registry.json` into `serving/models/` |
| `lab_utils.df_to_tfdata` | `models/train.py::encode` + `tf.data` pipeline |
| `lab_utils.model_reset_weights` | Not needed: each run builds a fresh model from config |
| `lab_utils.get_errors` | `evaluation/error_analysis.py::errors_for`, `confusion_pairs` |
| `lab_utils.print_metric_per_topic` | `evaluation/metrics.py::compute_metrics` (per-topic P/R/F1/top-2) |
| `lab_utils.save_data/save_vocab/save_labels/set_experiment_dirs` | `data/prepare.py` (versioned splits), `features/text.py`, `models/train.py::experiment_dirs` |
| Manual notebook cells | `newsclf` CLI + `pipelines.py` (same code from CLI, notebook, CI) |

**Environment decision:** the SavedModel was written by Keras 2 and does not load with `keras.models.load_model` under Keras 3 (TF >= 2.16). Rather than an old TensorFlow being pinned, it is loaded through its serving signature (`tf.saved_model.load(...).signatures["serving_default"]`, float32 input `embedding_1_input`). The vocabulary file starts with an empty padding line, so it is loaded as a list (`TextVectorization(vocabulary=[...])`) instead of a file path. Result: E1 reproduces at 0.7778 accuracy.

---

## 1. Scoping

**Problem:** file incoming articles under 8 topics from the title, fast enough to keep up with new sources. Manual sorting does not scale; rules did not work.

**Success criteria (set before modeling):**

* Business: share of articles filed automatically at >= 95% precision.
* Model: macro-F1 and per-topic F1 (accuracy hides SCIENCE at ~3%); top-2 accuracy as an ambiguity indicator.
* Operations: identical preprocessing in training and serving, one-command rollback, drift alarms.

**Questions raised with stakeholders:** labelling process and inter-rater agreement; topic definitions and tie-break rules; whether an article may appear under two topics; publisher agreements and click-through; policy on misinformation and unsupported languages.

**Constraint kept for this phase:** title-only input, CPU-friendly models.

**Plan:** week 1 baseline + audit; week 2 data fixes + tracked experiments; week 3 deployment, routing, monitoring.

---

## 2. Data

### 2.1 Procedure

```bash
newsclf validate            # raw audit -> reports/data_validation_raw.json
newsclf prepare -e E2 E3    # versioned splits -> data/processed/<hash>/
```

### 2.2 Findings

| # | Finding | Evidence | Decision |
|---|---|---|---|
| D1 | BUSINESS collapsed to one title | 15,000 rows, 1 unique title | Repair (D1 fix below); add `collapsed_classes` check to validation |
| D2 | Duplicate titles | 21,329 duplicates | De-duplicate on normalised title before splitting |
| D3 | Conflicting labels | 348 titles (407 after repair) with >1 label | Majority vote; report to labelling team |
| D4 | Train/test skew | SCIENCE 0.7% of train vs 13.2% of test | Pool and re-split, stratified |
| D5 | Minority class | SCIENCE ~3% after pooling | Class weights, per-topic metrics, augmentation experiment |
| D6 | Ambiguous labels | Hand-labelled sample (rows 30-39): several titles fit two topics | Top-2 metric, merged-taxonomy study, guideline questions |

### 2.3 D1 fix: title repair

The `.backup.csv` mentioned in the original hand-over was not included in the files that were received. The links survived, and news URLs nearly always carry the headline as a hyphenated slug, so `data/repair.py`:

1. Detects collapsed classes (unique titles / rows < 5%).
2. Restores titles from `data.reference_titles_file` if configured (joined on `link`; supports the course backup or the Kaggle export).
3. Otherwise rebuilds them from the URL slug: picks the path segment with the most words, strips file extensions, trailing IDs (`idUSKCN25A04M`, `1203007840`, `p55kum`) and trailing date runs.
4. Drops rows with no usable slug and records `title_source` per row.

Result: 14,009 recovered, 991 dropped. Trade-off: slug titles lose punctuation and casing (the vectorizer strips these anyway) and sometimes stop words.

### 2.4 Splits and versioning

* Stratified 60/20/20 with `random_state = seed`.
* Folder name = hash of the data config + seed, so identical settings share files and different settings never overwrite.
* `manifest.json`: data config, labels, row counts, SHA-256 per file, raw and processed validation reports, repair and de-duplication reports, cross-split title overlap (must be 0).

---

## 3. Modeling

### 3.1 Procedure

```bash
newsclf baseline            # E1
newsclf train -e E2 ... E8  # or: newsclf run-all
newsclf benchmark           # B1
newsclf leaderboard         # reports/leaderboard.md + reports/champion.json
```

### 3.2 Experiment log

| Exp | Hypothesis | Result (test macro-F1) | Conclusion |
|---|---|---:|---|
| E1 | Re-measure prototype | 0.768 (acc 0.778) | BUSINESS F1 = 1.00 is a red flag |
| E2 | A balanced split helps | 0.813 | Inflated: collapsed BUSINESS + duplicates leak across splits. Non-promotable by gate |
| E3 | Clean data, same model | 0.753 | First honest number |
| E4 | Pooling + regularisation + class weights | **0.794** | Champion (dev macro-F1 0.794) |
| E5 | EDA augmentation of SCIENCE | 0.790 | SCIENCE F1 up (0.804 -> 0.825), overall slightly down. Kept as an option |
| E6 | BiLSTM | 0.774 | Word order adds little on short headlines |
| E7 | CNN + stop-word removal | 0.772 | No gain |
| E8 | Merge WORLD+NATION, SCIENCE+TECHNOLOGY | 0.845 (6 classes) | Quantifies the taxonomy cost; product decision |
| B1 | TF-IDF + logistic regression | 0.821 | Best overall; not servable on TF Serving yet |

Also tried outside the tracked set (scratch runs, not committed): bigram tokens in the pooled model (no gain), wider embeddings and alternative dropout settings (within +-0.5 points of E4), and a dense TF-IDF MLP in Keras (out of memory at 50k features; sparse TF-IDF output in Keras 3 failed). Recorded so nobody repeats them blind.

### 3.3 Error analysis

* Top confusions (E3 dev): WORLD -> NATION (11% of errors), NATION -> WORLD (9%), NATION -> HEALTH, WORLD -> HEALTH.
* Top-2 recall gain: WORLD +24 points, NATION +21.5 points. The model usually has the right answer second.
* Many "errors" are defensible either way, so guideline work beats model work here.

### 3.4 Selection rule and gates

* Champion = best **dev** macro-F1 among promotable runs. Test is reported, never used to choose.
* Promotable = config allows it (E8 does not: different label space) **and** the dataset passed the data-quality gate (no collapsed classes, zero titles shared across splits). This is what stops E2.

---

## 4. Deployment

### 4.1 Procedure

```bash
newsclf deploy                         # reset serving; v1 = legacy (stable), v2 = champion (canary)
newsclf serve                          # or: docker compose -f serving/docker-compose.yml up -d
newsclf canary-check --url http://localhost:8501
newsclf promote                        # canary -> stable, stable -> deprecated
```

### 4.2 Decisions

* **Preprocessing inside the model.** Demonstrated risk: loading the wrong vocabulary into the prototype drops accuracy from 77.8% to 13.2% silently. All exports take strings.
* **Registry as source of truth.** Both TF Serving configs are rendered from `serving/registry.json`, so they cannot drift apart. `assets.extra/` in each version holds labels, vocabulary and a model card.
* **Legacy stays served as v1** so the comparison is real and rollback is instant. It still needs client-side tokenization; `TopicClassifier` handles that for it and nothing else.
* **Canary contamination control.** The prototype had trained on 13,780 of the 20,184 test titles. Comparing on all of them would favour memorisation (an early run with the full test set failed the gate on WORLD/NATION for exactly this reason). The check now drops titles either model trained on, then requires macro-F1 gain >= 0 and no topic losing > 0.05 F1. On 6,404 unseen titles: canary 0.770 vs stable 0.467 macro-F1.
* **Local TF-Serving-compatible server** (`newsclf serve`, FastAPI) exists so the full REST workflow runs on machines without Docker (common on Windows), and so the tests can exercise the HTTP contract. It also offers `/v1/classify`, the product-facing call that returns topic, routing decision and guard flags.

### 4.3 Items from the original notebook's local-machine notes, and how they are handled

| Original note | Implementation |
|---|---|
| Install TF Serving locally (APT) | `serving/run_tf_serving_native.sh` + generated `serving/models/models.config` with this machine's absolute path |
| Use Docker instead | `serving/docker-compose.yml`, mounts `serving/models` as `/models`, uses `models.docker.config`, polls config every 10 s, `--allow_version_labels_for_unavailable_models` |
| Delete old models in the serving dir before re-running | `newsclf deploy` resets the serving folder and registry; `make clean` resets everything |
| Stop the server (`docker stop` / `kill`) | `docker compose ... down`; the notebook terminates its own server process |
| Exercise: deploy E3 as "canary" | Generalised: `newsclf export -e <exp> --label canary` for any promotable run |
| Exercise: own filter -> `auto.csv` / `manual.csv` | `newsclf route`: dev-calibrated confidence threshold + OOV guard + language guard |
| Model expects strings, error on old model | Reproduced on purpose: v1 rejects strings exactly as TF Serving does |

### 4.4 Human-in-the-loop routing

* Threshold = lowest confidence where the dev auto-filed bucket reaches 95% accuracy (E4: 0.90), stored in `artifacts/E4/routing_policy.json` and the version's model card.
* On test: 55.1% auto-filed at 95.2% accuracy. On the dev routing run: 9,078 titles to editors, of which only 60% would have been right if automated.

---

## 5. Monitoring

### 5.1 Procedure

```bash
newsclf monitor             # reports/monitoring/{monitoring_report.json, oov_trend.png, flagged_inputs.csv}
```

### 5.2 Findings on the production log

* Rolling OOV alarm from article 102; 5 titles flagged non-English, all from `corriere.it` (Italian).
* Language guard tuning: single-letter words ("a", "i") appear in Italian, so they are excluded from the English function-word signal; titles under 5 tokens are not language-checked (short names and places produced false alarms).
* Output side: 70% agreement between the logged prototype predictions and the new model; topic mix and confidence recorded as the reference for the next batch.

### 5.3 Alarm responses

See [RUNBOOK.md](RUNBOOK.md#5-monitoring-and-alarms).

---

## 6. Recommendations handed to the product owner

1. Fix labelling guidelines for WORLD/NATION and SCIENCE/TECHNOLOGY, or merge them (E8: +5 points), or move to multi-label annotation; measure inter-rater agreement.
2. Let articles appear under two topics; top-2 accuracy is ~91%.
3. Recover the original BUSINESS headlines.
4. Source more SCIENCE articles.
5. Run with editors in the loop at the calibrated threshold; track queue size as a KPI.
6. Filter or tag unsupported-language feeds at ingestion.

## 7. Next iteration

* Add the article description as a second input.
* Serve the TF-IDF linear model (port into the TF graph).
* Small pre-trained transformer on GPU.
* Scheduled re-training (weekly) with the canary gate in CI, fed by editor decisions from the review queue.
