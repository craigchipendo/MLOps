"""Error analysis helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def prediction_frame(df: pd.DataFrame, probs: np.ndarray, labels: list[str]) -> pd.DataFrame:
    order = np.argsort(-probs, axis=1)
    out = df[["title", "topic"]].reset_index(drop=True).copy()
    out["predicted"] = [labels[i] for i in order[:, 0]]
    out["second_guess"] = [labels[i] for i in order[:, 1]]
    out["confidence"] = probs.max(1)
    out["margin"] = probs[np.arange(len(probs)), order[:, 0]] - probs[np.arange(len(probs)), order[:, 1]]
    out["correct"] = out["predicted"] == out["topic"]
    return out


def errors_for(pred_df: pd.DataFrame, predicted_topic: str, n: int = 20) -> pd.DataFrame:
    """Rows the model predicted as `predicted_topic` whose label is different."""
    mask = (pred_df["predicted"] == predicted_topic) & ~pred_df["correct"]
    return pred_df[mask].head(n)


def confusion_pairs(pred_df: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    wrong = pred_df[~pred_df["correct"]]
    pairs = wrong.groupby(["topic", "predicted"]).size().rename("count").reset_index()
    pairs["share_of_errors"] = (pairs["count"] / max(len(wrong), 1)).round(4)
    return pairs.sort_values("count", ascending=False).head(top).reset_index(drop=True)
