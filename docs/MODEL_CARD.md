# Model Card: `newsapp_model` v2 (experiment E4)

## Overview

| | |
|---|---|
| Task | Single-label topic classification of English news headlines into 8 topics |
| Architecture | `TextVectorization` (20k vocab, 24 tokens, lowercase + strip punctuation) -> Embedding(64) -> SpatialDropout(0.2) -> [GlobalAvgPool, GlobalMaxPool] -> Dense(64, ReLU) -> Dropout(0.4) -> Softmax(8) |
| Parameters | 1.29M |
| Training | Adam (1e-3), batch 64, balanced class weights, early stopping on dev loss (6 epochs run, best weights restored), deterministic ops, seed 42 |
| Serving signature | `serving_default(title: string[N]) -> probabilities: float[N, 8]` |
| Label order | BUSINESS, ENTERTAINMENT, HEALTH, NATION, SCIENCE, SPORTS, TECHNOLOGY, WORLD (`assets.extra/labels.txt`) |
| Owner | Craig Tanaka Chipendo |

## Intended use

* **In scope:** filing English news articles under the app's 8 topics from their headline, in hourly batches, with a human reviewer for low-confidence items.
* **Out of scope:** non-English text, article bodies, misinformation or credibility judgement, multi-label tagging, anything with legal or safety consequences.
* **Operating mode:** human in the loop. Predictions with confidence >= 0.90 (calibrated on dev for 95% precision) are published automatically; everything else, plus any title with > 50% unknown tokens or that looks non-English, goes to the editor queue.

## Data

* Source: the inherited prototype's train and test exports, a revised version of the Topic Labeled News Dataset (headlines from mid-2020, a few older).
* Processing: BUSINESS titles reconstructed from URL slugs (14,009 rows; 991 rows with no slug dropped) because the stored titles had been overwritten; de-duplicated on normalised title (107,783 -> 100,928 rows; 407 label conflicts resolved by majority vote); titles with fewer than 2 tokens dropped.
* Split: stratified 60/20/20 (60,552 / 20,184 / 20,184), zero titles shared across splits. Data version `70d8866e05` (see its `manifest.json`).
* Class balance: SCIENCE is about 3% of rows; the other topics about 13.8% each.

## Performance (test split, 20,184 titles)

| Metric | Value |
|---|---:|
| Accuracy | 0.794 |
| Macro-F1 | 0.794 |
| Top-2 accuracy | 0.908 |
| Expected calibration error | 0.040 |
| Auto-filed share at threshold 0.90 | 55.1% (accuracy 95.2%) |

| Topic | Precision | Recall | F1 | Top-2 recall |
|---|---:|---:|---:|---:|
| BUSINESS | 0.783 | 0.823 | 0.802 | 0.901 |
| ENTERTAINMENT | 0.841 | 0.879 | 0.860 | 0.935 |
| HEALTH | 0.763 | 0.781 | 0.772 | 0.890 |
| NATION | 0.671 | 0.644 | 0.657 | 0.871 |
| SCIENCE | 0.777 | 0.833 | 0.804 | 0.895 |
| SPORTS | 0.926 | 0.905 | 0.916 | 0.940 |
| TECHNOLOGY | 0.884 | 0.874 | 0.879 | 0.934 |
| WORLD | 0.680 | 0.641 | 0.660 | 0.882 |

Against the inherited prototype on 6,404 test titles that neither model trained on: macro-F1 0.770 vs 0.467, accuracy 0.806 vs 0.419.

## Limitations and known failure modes

* **WORLD vs NATION and HEALTH vs NATION** are the main confusions. Many are genuinely ambiguous (a COVID update from a US county), so part of this error is label noise, not model error.
* **BUSINESS training titles come from URL slugs**, which lack punctuation and sometimes drop stop words; live BUSINESS headlines look slightly different. Monitor BUSINESS precision in the editor queue.
* **Temporal drift:** training data is centred on 2020 (COVID-heavy). New entities and topics will raise the OOV rate; the monitoring alarm and re-training procedure exist for this.
* **Non-English input** gets confident but meaningless predictions from the network alone; the routing guard catches it.
* A simple TF-IDF + logistic regression benchmark scores about 2.5 points higher macro-F1; it is not served yet because TF Serving needs a TensorFlow graph.

## Ethical considerations

* Publisher agreements and click-through to the source site were raised with the product owner during scoping.
* The model does not assess truthfulness; misinformation handling is a separate product policy.
* Automatic filing is restricted to high-confidence, in-distribution English headlines; humans review the rest.

## Versioning

| Version | Source | Input | Label |
|---|---|---|---|
| 1 | E1 inherited prototype | 20 token ids (client-side tokenization) | deprecated |
| 2 | E4 | raw strings | stable |

Release history is in `serving/registry.json` (`history`), and the canary evidence is in `reports/canary_report.json`.
