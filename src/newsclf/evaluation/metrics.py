"""Evaluation: overall metrics, per-slice metrics, calibration and routing policy."""

from __future__ import annotations

from pathlib import Path

# The backend is not forced here: plots are written with savefig, which works on any
# backend, and forcing "Agg" at import would stop notebooks from displaying figures.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support  # noqa: E402


def top_k_accuracy(y_true: np.ndarray, probs: np.ndarray, k: int = 2) -> float:
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float(np.mean([y in row for y, row in zip(y_true, topk, strict=True)]))


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(1)
    correct = probs.argmax(1) == y_true
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, labels: list[str], top_k: int = 2) -> dict:
    y_pred = probs.argmax(1)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(len(labels)), zero_division=0)
    per_class = {
        label: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(s[i]),
            f"top{top_k}_recall": top_k_accuracy(y_true[y_true == i], probs[y_true == i], top_k)
            if (y_true == i).any()
            else float("nan"),
        }
        for i, label in enumerate(labels)
    }
    return {
        "accuracy": float((y_pred == y_true).mean()),
        f"top{top_k}_accuracy": top_k_accuracy(y_true, probs, top_k),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "ece": expected_calibration_error(y_true, probs),
        "n": int(len(y_true)),
        "per_class": per_class,
    }


def per_class_frame(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame(metrics["per_class"]).T.round(4)


def routing_curve(y_true: np.ndarray, probs: np.ndarray, thresholds=None) -> pd.DataFrame:
    """Coverage vs. accuracy of the auto-routed bucket for each confidence threshold."""
    thresholds = np.round(np.arange(0.30, 0.99, 0.01), 2) if thresholds is None else thresholds
    conf = probs.max(1)
    correct = probs.argmax(1) == y_true
    rows = []
    for t in thresholds:
        auto = conf >= t
        rows.append(
            {
                "threshold": float(t),
                "coverage": float(auto.mean()),
                "auto_accuracy": float(correct[auto].mean()) if auto.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(curve: pd.DataFrame, target_accuracy: float, fallback: float = 0.6) -> dict:
    """Smallest threshold whose auto-routed bucket meets the target accuracy on dev."""
    ok = curve[curve["auto_accuracy"] >= target_accuracy]
    if ok.empty:
        row = curve.loc[(curve["threshold"] - fallback).abs().idxmin()]
        reason = f"target {target_accuracy:.0%} unreachable, using fallback"
    else:
        row = ok.iloc[0]
        reason = f"lowest threshold with dev auto-accuracy >= {target_accuracy:.0%}"
    return {
        "confidence_threshold": float(row["threshold"]),
        "expected_coverage": float(row["coverage"]),
        "expected_auto_accuracy": float(row["auto_accuracy"]),
        "target_accuracy": target_accuracy,
        "reason": reason,
    }


def plot_confusion_matrix(y_true, y_pred, labels: list[str], path: str | Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)), normalize="true")
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{cm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > 0.5 else "black",
                fontsize=8,
            )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_history(history: dict, path: str | Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["loss", "accuracy"], strict=True):
        ax.plot(history.get(metric, []), label="train")
        ax.plot(history.get(f"val_{metric}", []), label="dev")
        ax.set_title(metric)
        ax.set_xlabel("epoch")
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
