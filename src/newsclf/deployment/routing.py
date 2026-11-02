"""Degrees of automation: human-in-the-loop routing.

A prediction is published automatically only when the model is confident AND
the input looks like the traffic it was trained on. Everything else goes to the
manual review queue. The confidence threshold is not hand-picked: it is
calibrated per model on the dev split (see `evaluation.metrics.choose_threshold`)
and shipped in the version's model card.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class RoutingPolicy:
    confidence_threshold: float = 0.6
    margin_threshold: float = 0.0
    max_oov_rate: float = 0.5
    block_non_english: bool = True

    @classmethod
    def from_version(cls, cfg: dict, info: dict) -> RoutingPolicy:
        calibrated = (info.get("routing_policy") or {}).get("confidence_threshold")
        return cls(
            confidence_threshold=calibrated or cfg["deployment"]["default_confidence_threshold"],
            margin_threshold=cfg["deployment"]["default_margin_threshold"],
            max_oov_rate=cfg["monitoring"]["oov_rate_alert"],
        )

    def as_dict(self) -> dict:
        return asdict(self)


def route_predictions(
    titles: list[str], probs: np.ndarray, labels: list[str], policy: RoutingPolicy, profile: pd.DataFrame | None = None
) -> pd.DataFrame:
    order = np.argsort(-probs, axis=1)
    idx = np.arange(len(probs))
    out = pd.DataFrame(
        {
            "title": list(titles),
            "predicted_topic": [labels[i] for i in order[:, 0]],
            "confidence": probs[idx, order[:, 0]].round(4),
            "second_guess": [labels[i] for i in order[:, 1]],
            "margin": (probs[idx, order[:, 0]] - probs[idx, order[:, 1]]).round(4),
        }
    )
    if profile is not None:
        out["oov_rate"] = profile["oov_rate"].round(4).to_numpy()
        out["likely_non_english"] = profile["likely_non_english"].to_numpy()
    else:
        out["oov_rate"] = np.nan
        out["likely_non_english"] = False

    reasons = []
    for row in out.itertuples():
        r = []
        if policy.block_non_english and row.likely_non_english:
            r.append("input_not_english")
        if not np.isnan(row.oov_rate) and row.oov_rate > policy.max_oov_rate:
            r.append("high_oov_rate")
        if row.confidence < policy.confidence_threshold:
            r.append("low_confidence")
        if row.margin < policy.margin_threshold:
            r.append("ambiguous_top2")
        reasons.append(";".join(r))
    out["reason"] = reasons
    out["route"] = np.where(out["reason"] == "", "auto", "manual")
    return out


def write_routing_outputs(decisions: pd.DataFrame, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    auto = decisions[decisions["route"] == "auto"]
    manual = decisions[decisions["route"] == "manual"]
    auto.to_csv(out_dir / "auto.csv", index=False)
    manual.to_csv(out_dir / "manual.csv", index=False)
    summary = {
        "total": int(len(decisions)),
        "auto": int(len(auto)),
        "manual": int(len(manual)),
        "automation_rate": round(len(auto) / max(len(decisions), 1), 4),
        "manual_reasons": manual["reason"].str.split(";").explode().value_counts().to_dict(),
    }
    if "topic" in decisions.columns:
        summary["auto_accuracy"] = (
            round(float((auto["predicted_topic"] == auto["topic"]).mean()), 4) if len(auto) else None
        )
        summary["manual_accuracy_if_automated"] = (
            round(float((manual["predicted_topic"] == manual["topic"]).mean()), 4) if len(manual) else None
        )
    return summary
