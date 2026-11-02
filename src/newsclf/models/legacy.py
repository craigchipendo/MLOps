"""E1: re-measure the inherited prototype.

The previous developer left no metrics, so the baseline is recomputed from the
SavedModel and vocabulary they shipped, on the test file they shipped. The
SavedModel predates Keras 3, so it is loaded through `tf.saved_model.load` and
its serving signature (float32 token ids) rather than `keras.models.load_model`.
"""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import tensorflow as tf

from newsclf.config import resolve, save_config
from newsclf.data.validate import validate_frame
from newsclf.evaluation.error_analysis import confusion_pairs, prediction_frame
from newsclf.evaluation.metrics import compute_metrics, per_class_frame, plot_confusion_matrix
from newsclf.features.text import build_vectorizer, load_vocabulary
from newsclf.tracking import log_run
from newsclf.utils import get_logger, write_json

log = get_logger(__name__)
LEGACY_FEATURES = {"max_tokens": 10000, "sequence_length": 20, "remove_stopwords": False}


class LegacyModel:
    def __init__(self, model_dir):
        model_dir = resolve(model_dir)
        self.signature = tf.saved_model.load(str(model_dir / "saved_model")).signatures["serving_default"]
        self.input_name = next(iter(self.signature.structured_input_signature[1]))
        self.output_name = next(iter(self.signature.structured_outputs))
        self.vectorizer = build_vectorizer(LEGACY_FEATURES, load_vocabulary(model_dir / "vocab" / "vocabulary.txt"))
        self.labels = load_vocabulary(model_dir / "vocab" / "labels.txt")

    def predict_proba(self, titles, batch_size: int = 4096) -> np.ndarray:
        out = []
        titles = list(map(str, titles))
        for i in range(0, len(titles), batch_size):
            seq = tf.cast(self.vectorizer(tf.constant(titles[i : i + batch_size])), tf.float32)
            out.append(self.signature(**{self.input_name: seq})[self.output_name].numpy())
        return np.concatenate(out)


def evaluate_legacy(cfg: dict, track: bool = True) -> dict:
    cfg = {
        **cfg,
        "name": "E1",
        "description": "Inherited prototype re-measured on its own test file",
        "promotable": False,
    }
    out_dir = resolve(cfg["paths"]["artifacts_dir"]) / "E1"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "reports").mkdir(parents=True)

    model = LegacyModel(cfg["paths"]["legacy_model_dir"])
    test_df = pd.read_csv(resolve(cfg["paths"]["raw_dir"]) / "test_data.csv")
    train_df = pd.read_csv(resolve(cfg["paths"]["raw_dir"]) / "train_data.csv")
    probs = model.predict_proba(test_df["title"])
    y = test_df["topic"].map({lab: i for i, lab in enumerate(model.labels)}).to_numpy()
    metrics = compute_metrics(y, probs, model.labels, cfg["evaluation"]["top_k"])

    per_class_frame(metrics).to_csv(out_dir / "reports" / "per_class_test.csv")
    pred = prediction_frame(test_df, probs, model.labels)
    confusion_pairs(pred).to_csv(out_dir / "reports" / "top_confusions_test.csv", index=False)
    plot_confusion_matrix(
        y,
        probs.argmax(1),
        model.labels,
        out_dir / "reports" / "confusion_matrix_test.png",
        "E1 prototype - legacy test (row-normalised)",
    )
    result = {
        "experiment": "E1",
        "description": cfg["description"],
        "model_type": "legacy_dense (inherited SavedModel)",
        "data_version": "legacy",
        "labels": model.labels,
        "promotable": False,
        "dev": {},
        "test": metrics,
        "legacy_data_audit": {
            "train": validate_frame(train_df),
            "test": validate_frame(test_df),
        },
    }
    write_json(result, out_dir / "metrics.json")
    save_config(cfg, out_dir / "config.yaml")
    log.info(
        "E1 legacy test accuracy=%.4f top2=%.4f macroF1=%.4f",
        metrics["accuracy"],
        metrics["top2_accuracy"],
        metrics["macro_f1"],
    )
    if track:
        log_run(cfg, result, out_dir, notes="baseline; no dev split existed")
    return result
