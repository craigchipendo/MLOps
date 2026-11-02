"""Classical benchmark: TF-IDF (word + char n-grams) + logistic regression.

A strong linear baseline answers "is the neural model earning its complexity?"
It is trained on the same processed split as the neural experiments.
"""

from __future__ import annotations

import shutil

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from newsclf.config import resolve, save_config
from newsclf.data.prepare import load_manifest, load_splits, prepare_dataset
from newsclf.evaluation.metrics import compute_metrics, per_class_frame
from newsclf.tracking import log_run
from newsclf.utils import get_logger, set_seed, write_json

log = get_logger(__name__)


def run_benchmark(cfg: dict, track: bool = True) -> dict:
    set_seed(cfg["seed"])
    cfg = {
        **cfg,
        "name": "B1",
        "description": "TF-IDF word(1-2)+char(3-5) n-grams + logistic regression",
        "promotable": False,
    }
    out_dir = resolve(cfg["paths"]["artifacts_dir"]) / "B1"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    data_dir = prepare_dataset(cfg)
    manifest = load_manifest(data_dir)
    labels = manifest["labels"]
    splits = load_splits(data_dir)
    index = {lab: i for i, lab in enumerate(labels)}

    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=200_000, sublinear_tf=True)
    train_text = splits["train"]["title"].astype(str)
    x_train = hstack([word.fit_transform(train_text), char.fit_transform(train_text)]).tocsr()

    def transform(df):
        text = df["title"].astype(str)
        return hstack([word.transform(text), char.transform(text)]).tocsr()

    clf = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced")
    clf.fit(x_train, splits["train"]["topic"].map(index))

    result = {
        "experiment": "B1",
        "description": cfg["description"],
        "model_type": "tfidf_logreg",
        "data_version": manifest["data_version"],
        "labels": labels,
        "promotable": False,
        "params": int(np.prod(clf.coef_.shape)),
    }
    for split in ("dev", "test"):
        probs = clf.predict_proba(transform(splits[split]))
        result[split] = compute_metrics(
            splits[split]["topic"].map(index).to_numpy(), probs, labels, cfg["evaluation"]["top_k"]
        )
        per_class_frame(result[split]).to_csv(out_dir / f"per_class_{split}.csv")
    joblib.dump({"word": word, "char": char, "clf": clf, "labels": labels}, out_dir / "model.joblib")
    write_json(result, out_dir / "metrics.json")
    save_config(cfg, out_dir / "config.yaml")
    log.info(
        "B1 dev acc=%.4f macroF1=%.4f | test acc=%.4f macroF1=%.4f",
        result["dev"]["accuracy"],
        result["dev"]["macro_f1"],
        result["test"]["accuracy"],
        result["test"]["macro_f1"],
    )
    if track:
        log_run(cfg, result, out_dir, notes="classical benchmark, not served")
    return result
