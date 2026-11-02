"""Training stage for one experiment."""

from __future__ import annotations

import shutil
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from newsclf.config import relpath, resolve, save_config
from newsclf.data.augment import augment_frame
from newsclf.data.prepare import load_manifest, load_splits, prepare_dataset
from newsclf.evaluation.error_analysis import confusion_pairs, prediction_frame
from newsclf.evaluation.metrics import (
    choose_threshold,
    compute_metrics,
    per_class_frame,
    plot_confusion_matrix,
    plot_history,
    routing_curve,
)
from newsclf.features.text import fit_vectorizer, save_vocabulary
from newsclf.models.architectures import build_model
from newsclf.tracking import log_run
from newsclf.utils import get_logger, set_seed, write_json

log = get_logger(__name__)


def experiment_dirs(cfg: dict) -> dict[str, Path]:
    base = resolve(cfg["paths"]["artifacts_dir"]) / cfg["name"]
    return {"base": base, "model": base / "model", "vocab": base / "vocab", "reports": base / "reports"}


def encode(vectorizer, df: pd.DataFrame, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    index = {label: i for i, label in enumerate(labels)}
    x = vectorizer(tf.constant(df["title"].astype(str).tolist())).numpy()
    y = df["topic"].map(index).to_numpy()
    if np.isnan(y.astype(float)).any():
        raise ValueError("Found labels outside the label vocabulary")
    return x, y.astype("int64")


def balanced_class_weights(y: np.ndarray, n_classes: int) -> dict[int, float]:
    counts = np.bincount(y, minlength=n_classes)
    return {i: float(len(y) / (n_classes * max(c, 1))) for i, c in enumerate(counts)}


def run_experiment(cfg: dict, track: bool = True) -> dict:
    """Prepare data, train, evaluate, save artifacts and log one experiment."""
    set_seed(cfg["seed"])
    dirs = experiment_dirs(cfg)
    if dirs["base"].exists():
        shutil.rmtree(dirs["base"])
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    data_dir = prepare_dataset(cfg)
    manifest = load_manifest(data_dir)
    labels = manifest["labels"]
    splits = load_splits(data_dir)

    train_df = splits["train"]
    acfg = cfg["augmentation"]
    if acfg["enabled"]:
        before = len(train_df)
        train_df = augment_frame(train_df, acfg["target_classes"], acfg["n_aug_per_sample"], acfg["alpha"], cfg["seed"])
        log.info("Augmented training split: %d -> %d rows", before, len(train_df))

    vectorizer = fit_vectorizer(cfg["features"], train_df["title"].astype(str))
    save_vocabulary(vectorizer, dirs["vocab"] / "vocabulary.txt")
    (dirs["vocab"] / "labels.txt").write_text("\n".join(labels), encoding="utf-8")

    x_train, y_train = encode(vectorizer, train_df, labels)
    x_dev, y_dev = encode(vectorizer, splits["dev"], labels)
    x_test, y_test = encode(vectorizer, splits["test"], labels)

    tcfg = cfg["training"]
    model = build_model(
        vocab_size=vectorizer.vocabulary_size(),
        seq_len=cfg["features"]["sequence_length"],
        n_classes=len(labels),
        model_cfg=cfg["model"],
        learning_rate=tcfg["learning_rate"],
        top_k=cfg["evaluation"]["top_k"],
    )
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(tcfg["shuffle_buffer"], seed=cfg["seed"])
        .batch(tcfg["batch_size"])
        .prefetch(tf.data.AUTOTUNE)
    )
    dev_ds = tf.data.Dataset.from_tensor_slices((x_dev, y_dev)).batch(256)

    callbacks = [keras.callbacks.CSVLogger(str(dirs["reports"] / "training_log.csv"))]
    if tcfg.get("early_stopping_patience"):
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=tcfg["early_stopping_patience"], restore_best_weights=True
            )
        )
    class_weight = balanced_class_weights(y_train, len(labels)) if tcfg.get("class_weight") == "balanced" else None

    log.info("Training %s (%s) on %d rows", cfg["name"], cfg["model"]["architecture"], len(y_train))
    history = model.fit(
        train_ds,
        validation_data=dev_ds,
        epochs=tcfg["epochs"],
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )
    history = {k: [float(v) for v in vals] for k, vals in history.history.items()}

    model.save(dirs["model"] / "model.keras")

    top_k = cfg["evaluation"]["top_k"]
    p_dev = model.predict(x_dev, batch_size=1024, verbose=0)
    p_test = model.predict(x_test, batch_size=1024, verbose=0)
    dev_metrics = compute_metrics(y_dev, p_dev, labels, top_k)
    test_metrics = compute_metrics(y_test, p_test, labels, top_k)

    # Human-in-the-loop routing policy is calibrated on DEV, reported on TEST.
    curve = routing_curve(y_dev, p_dev)
    policy = choose_threshold(
        curve, cfg["evaluation"]["routing_target_accuracy"], cfg["deployment"]["default_confidence_threshold"]
    )
    test_auto = p_test.max(1) >= policy["confidence_threshold"]
    policy["test_coverage"] = float(test_auto.mean())
    policy["test_auto_accuracy"] = float((p_test.argmax(1) == y_test)[test_auto].mean()) if test_auto.any() else None
    curve.to_csv(dirs["reports"] / "routing_curve_dev.csv", index=False)
    write_json(policy, dirs["base"] / "routing_policy.json")

    pred_dev = prediction_frame(splits["dev"], p_dev, labels)
    confusion_pairs(pred_dev).to_csv(dirs["reports"] / "top_confusions_dev.csv", index=False)
    pred_dev[~pred_dev["correct"]].head(500).to_csv(dirs["reports"] / "errors_dev_sample.csv", index=False)
    per_class_frame(dev_metrics).to_csv(dirs["reports"] / "per_class_dev.csv")
    per_class_frame(test_metrics).to_csv(dirs["reports"] / "per_class_test.csv")
    plot_confusion_matrix(
        y_test,
        p_test.argmax(1),
        labels,
        dirs["reports"] / "confusion_matrix_test.png",
        f"{cfg['name']} - test (row-normalised)",
    )
    plot_history(history, dirs["reports"] / "training_curves.png", f"{cfg['name']} training")

    # Quality gate: a model trained or evaluated on data that failed validation
    # (collapsed classes, titles shared across splits) can never be promoted,
    # however good its numbers look.
    leakage = sum(manifest["split_validation"]["title_overlap"].values())
    collapsed = list(manifest["processed_validation"]["collapsed_classes"])
    data_quality = {
        "collapsed_classes": collapsed,
        "split_title_overlap": int(leakage),
        "passed": not collapsed and leakage == 0,
    }

    result = {
        "experiment": cfg["name"],
        "description": cfg.get("description", ""),
        "model_type": cfg["model"]["architecture"],
        "data_version": manifest["data_version"],
        "data_dir": relpath(data_dir),
        "labels": labels,
        "promotable": bool(cfg.get("promotable", True)) and data_quality["passed"],
        "data_quality": data_quality,
        "params": int(model.count_params()),
        "epochs_trained": len(history.get("loss", [])),
        "train_rows": int(len(y_train)),
        "dev": dev_metrics,
        "test": test_metrics,
        "routing_policy": policy,
        "history": history,
    }
    save_config(cfg, dirs["base"] / "config.yaml")
    write_json(result, dirs["base"] / "metrics.json")
    log.info(
        "%s dev acc=%.4f macroF1=%.4f | test acc=%.4f macroF1=%.4f",
        cfg["name"],
        dev_metrics["accuracy"],
        dev_metrics["macro_f1"],
        test_metrics["accuracy"],
        test_metrics["macro_f1"],
    )
    if track:
        log_run(cfg, result, dirs["base"])
    return result
