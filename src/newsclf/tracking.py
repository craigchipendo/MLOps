"""Experiment tracking.

Two sinks, on purpose:
* `reports/experiment_tracker.csv` - a flat, git-diffable ledger of every run
  (the evolution of the prototype's `experiment-tracker.csv`).
* MLflow (local SQLite backend) - params, metrics and artifacts for drill-down
  in the MLflow UI (`mlflow ui --backend-store-uri sqlite:///mlflow.db`).
MLflow is optional at runtime: if it is missing or fails, the CSV ledger still
records the run.
"""

from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path

import pandas as pd

from newsclf.config import relpath, resolve
from newsclf.utils import get_logger, git_commit

log = get_logger(__name__)

LEDGER_COLUMNS = [
    "run_id",
    "timestamp",
    "experiment",
    "description",
    "model_type",
    "data_version",
    "git_commit",
    "dev_accuracy",
    "dev_top2_accuracy",
    "dev_macro_f1",
    "test_accuracy",
    "test_top2_accuracy",
    "test_macro_f1",
    "test_science_f1",
    "epochs_trained",
    "params",
    "artifacts",
    "notes",
]


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def log_run(cfg: dict, result: dict, artifacts_dir: Path | None, notes: str = "") -> str:
    run_id = uuid.uuid4().hex[:8]
    dev, test = result.get("dev", {}), result.get("test", {})
    science = test.get("per_class", {}).get("SCIENCE", {}).get("f1")
    row = {
        "run_id": run_id,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "experiment": cfg["name"],
        "description": cfg.get("description", ""),
        "model_type": result.get("model_type", cfg.get("model", {}).get("architecture")),
        "data_version": result.get("data_version", ""),
        "git_commit": git_commit(),
        "dev_accuracy": dev.get("accuracy"),
        "dev_top2_accuracy": dev.get("top2_accuracy"),
        "dev_macro_f1": dev.get("macro_f1"),
        "test_accuracy": test.get("accuracy"),
        "test_top2_accuracy": test.get("top2_accuracy"),
        "test_macro_f1": test.get("macro_f1"),
        "test_science_f1": science,
        "epochs_trained": result.get("epochs_trained"),
        "params": result.get("params"),
        "artifacts": relpath(artifacts_dir) if artifacts_dir else "",
        "notes": notes,
    }
    ledger = resolve(cfg["tracking"]["csv"])
    ledger.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row], columns=LEDGER_COLUMNS)
    frame.to_csv(ledger, mode="a", header=not ledger.exists(), index=False)
    log.info("Logged run %s (%s) to %s", run_id, cfg["name"], relpath(ledger))

    if cfg["tracking"]["mlflow"].get("enabled"):
        _log_mlflow(cfg, result, artifacts_dir, run_id)
    return run_id


def _log_mlflow(cfg: dict, result: dict, artifacts_dir: Path | None, run_id: str) -> None:
    try:
        import mlflow
    except ImportError:
        log.warning("mlflow not installed; skipping MLflow logging")
        return
    try:
        mcfg = cfg["tracking"]["mlflow"]
        uri = mcfg["tracking_uri"]
        if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
            uri = f"sqlite:///{resolve(uri.removeprefix('sqlite:///')).as_posix()}"
        mlflow.set_tracking_uri(uri)
        if mlflow.get_experiment_by_name(mcfg["experiment_name"]) is None:
            mlflow.create_experiment(
                mcfg["experiment_name"], artifact_location=resolve(mcfg["artifact_location"]).as_uri()
            )
        mlflow.set_experiment(mcfg["experiment_name"])
        with mlflow.start_run(run_name=f"{cfg['name']}-{run_id}"):
            mlflow.set_tags(
                {
                    "experiment": cfg["name"],
                    "description": cfg.get("description", ""),
                    "git_commit": git_commit(),
                    "data_version": result.get("data_version", ""),
                }
            )
            params = {
                k: str(v)[:250]
                for k, v in _flatten(
                    {k: cfg[k] for k in ("data", "features", "model", "training", "augmentation") if k in cfg}
                ).items()
            }
            mlflow.log_params(params)
            for split in ("dev", "test"):
                for key, value in result.get(split, {}).items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"{split}_{key}", float(value))
                for label, vals in result.get(split, {}).get("per_class", {}).items():
                    mlflow.log_metric(f"{split}_f1_{label}", float(vals["f1"]))
            for epoch, values in enumerate(zip(*result.get("history", {}).values(), strict=False)):
                for key, value in zip(result["history"].keys(), values, strict=False):
                    mlflow.log_metric(f"epoch_{key}", float(value), step=epoch)
            if artifacts_dir and artifacts_dir.exists():
                for path in artifacts_dir.iterdir():
                    if path.is_file():
                        mlflow.log_artifact(str(path))
    except Exception as exc:  # tracking must never break training
        log.warning("MLflow logging failed (%s); CSV ledger still updated", exc)
