"""Export trained experiments as TF Serving SavedModel versions.

The vectorizer is attached in front of the trained network before export, so
the served model takes raw title strings. This removes the training/serving
skew risk of clients tokenizing with the wrong vocabulary file.

Each version folder carries `assets.extra/` (the TF Serving convention for
side-car files) with the label list and the model's metadata card.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import keras
import tensorflow as tf

from newsclf.config import load_yaml, relpath, resolve
from newsclf.features.text import load_vectorizer
from newsclf.serving.registry import Registry
from newsclf.utils import get_logger, read_json, write_json

log = get_logger(__name__)
tf.get_logger().setLevel("ERROR")  # keep absolute "Assets written to" paths out of logs


def build_serving_model(exp_dir: Path) -> tuple[keras.Model, list[str], dict]:
    cfg = load_yaml(exp_dir / "config.yaml")
    vectorizer = load_vectorizer(cfg["features"], exp_dir / "vocab" / "vocabulary.txt")
    labels = (exp_dir / "vocab" / "labels.txt").read_text(encoding="utf-8").split("\n")
    trained = keras.models.load_model(exp_dir / "model" / "model.keras")
    title = keras.Input(shape=(), dtype="string", name="title")
    probs = trained(vectorizer(title))
    return keras.Model(title, probs, name=f"{cfg['name']}_serving"), labels, cfg


def export_experiment(cfg: dict, experiment: str, label: str | None = None) -> int:
    exp_dir = resolve(cfg["paths"]["artifacts_dir"]) / experiment
    if not (exp_dir / "model" / "model.keras").exists():
        raise FileNotFoundError(f"No trained model for {experiment}; run `newsclf train -e {experiment}`")
    metrics = read_json(exp_dir / "metrics.json")
    if not metrics.get("promotable", True):
        raise ValueError(f"{experiment} is marked non-promotable (different label space or study run)")

    registry = Registry(cfg)
    version = registry.next_version()
    out = registry.version_dir(version)
    if out.exists():
        shutil.rmtree(out)

    model, labels, exp_cfg = build_serving_model(exp_dir)
    archive = keras.export.ExportArchive()
    archive.track(model)
    archive.add_endpoint(
        name="serving_default",
        fn=lambda titles: {"probabilities": model(titles, training=False)},
        input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name="title")],
    )
    archive.write_out(str(out), verbose=False)

    card = {
        "source": experiment,
        "description": metrics.get("description"),
        "input_type": "text",
        "labels": labels,
        "data_version": metrics.get("data_version"),
        "train_data": f"{metrics['data_dir']}/train.csv",
        "dev_macro_f1": metrics["dev"]["macro_f1"],
        "test_accuracy": metrics["test"]["accuracy"],
        "test_macro_f1": metrics["test"]["macro_f1"],
        "routing_policy": metrics.get("routing_policy"),
        "features": exp_cfg["features"],
    }
    _write_assets_extra(out, labels, card, exp_dir / "vocab" / "vocabulary.txt")
    registry.register(version, card)
    if label:
        registry.promote(version) if label == "stable" else registry.set_label(label, version)
    registry.save()
    log.info("Exported %s as %s version %d (%s)", experiment, registry.model_name, version, label or "unlabelled")
    return version


def export_legacy(cfg: dict, label: str | None = "stable") -> int:
    """Register the inherited prototype (integer-token input) as a served version."""
    registry = Registry(cfg)
    legacy_dir = resolve(cfg["paths"]["legacy_model_dir"])
    version = registry.next_version()
    out = registry.version_dir(version)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(legacy_dir / "saved_model", out)
    labels = (legacy_dir / "vocab" / "labels.txt").read_text(encoding="utf-8").split("\n")
    card = {
        "source": "E1",
        "description": "Inherited prototype; expects 20 token ids, tokenized client-side",
        "input_type": "token_ids",
        "train_data": relpath(resolve(cfg["paths"]["raw_dir"]) / "train_data.csv"),
        "vocab_path": relpath(legacy_dir / "vocab" / "vocabulary.txt"),
        "labels": labels,
        "features": {"max_tokens": 10000, "sequence_length": 20, "remove_stopwords": False},
    }
    _write_assets_extra(out, labels, card, legacy_dir / "vocab" / "vocabulary.txt")
    registry.register(version, card)
    if label:
        registry.promote(version) if label == "stable" else registry.set_label(label, version)
    registry.save()
    log.info("Registered legacy prototype as version %d (%s)", version, label)
    return version


def _write_assets_extra(version_dir: Path, labels: list[str], card: dict, vocab_file: Path) -> None:
    extra = version_dir / "assets.extra"
    extra.mkdir(parents=True, exist_ok=True)
    shutil.copy(vocab_file, extra / "vocabulary.txt")
    (extra / "labels.txt").write_text("\n".join(labels), encoding="utf-8")
    write_json(card, extra / "model_card.json")


def reset_serving(cfg: dict) -> None:
    """Delete every served version and the registry (start from a clean slate)."""
    registry = Registry(cfg)
    if registry.model_dir.exists():
        shutil.rmtree(registry.model_dir)
    if registry.path.exists():
        registry.path.unlink()
    log.info("Serving directory and registry reset")
