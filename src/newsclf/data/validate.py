"""Data validation.

Checks run on every dataset before it is allowed into training. Hard failures
raise; soft findings are written to a JSON report so they are reviewed, not lost.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from newsclf.data.ingest import REQUIRED_COLUMNS
from newsclf.data.repair import find_collapsed_classes

EXPECTED_TOPICS = {
    "BUSINESS",
    "ENTERTAINMENT",
    "HEALTH",
    "NATION",
    "SCIENCE",
    "SPORTS",
    "TECHNOLOGY",
    "WORLD",
}


class DataValidationError(ValueError):
    pass


def normalize_title(title: str) -> str:
    title = str(title).lower()
    title = re.sub(r"[^\w\s]", " ", title)
    return re.sub(r"\s+", " ", title).strip()


def class_distribution(series: pd.Series) -> dict[str, float]:
    return series.value_counts(normalize=True).sort_index().round(4).to_dict()


def total_variation(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    return round(0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in keys), 4)


def validate_frame(df: pd.DataFrame, expected_topics: set[str] | None = None, collapse_ratio: float = 0.05) -> dict:
    """Validate one dataframe and return a findings report."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    expected_topics = expected_topics or EXPECTED_TOPICS
    unknown = sorted(set(df["topic"].dropna()) - expected_topics)
    if unknown:
        raise DataValidationError(f"Unknown topics found: {unknown}")

    norm = df["title"].fillna("").map(normalize_title)
    conflicts = df.assign(_n=norm).groupby("_n")["topic"].nunique()
    non_ascii = df["title"].fillna("").map(lambda t: sum(ord(c) > 127 for c in t) / max(len(t), 1))

    report = {
        "rows": int(len(df)),
        "nulls": {k: int(v) for k, v in df[REQUIRED_COLUMNS].isna().sum().items() if v},
        "empty_titles": int((norm == "").sum()),
        "class_distribution": class_distribution(df["topic"]),
        "class_counts": {k: int(v) for k, v in df["topic"].value_counts().sort_index().items()},
        "duplicate_titles": int(norm.duplicated().sum()),
        "titles_with_conflicting_labels": int((conflicts > 1).sum()),
        "collapsed_classes": find_collapsed_classes(df, collapse_ratio),
        "title_tokens_p50": float(norm.str.split().str.len().median()),
        "title_tokens_p99": float(np.percentile(norm.str.split().str.len(), 99)),
        "mostly_non_ascii_titles": int((non_ascii > 0.3).sum()),
    }
    warnings = []
    if report["collapsed_classes"]:
        warnings.append(
            f"Collapsed classes (near-constant title): {list(report['collapsed_classes'])}. "
            "Model would learn a single string for these topics."
        )
    if report["duplicate_titles"]:
        warnings.append(f"{report['duplicate_titles']} duplicate titles: leakage risk across splits.")
    if report["titles_with_conflicting_labels"]:
        warnings.append(f"{report['titles_with_conflicting_labels']} titles carry more than one label (label noise).")
    smallest = min(report["class_distribution"].values()) if report["class_distribution"] else 0
    if smallest < 0.05:
        warnings.append(f"Minority class below 5% of rows ({smallest:.1%}): monitor per-class metrics.")
    report["warnings"] = warnings
    return report


def validate_splits(splits: dict[str, pd.DataFrame]) -> dict:
    """Check class balance and leakage across splits."""
    dists = {name: class_distribution(df["topic"]) for name, df in splits.items()}
    names = list(splits)
    tv = {f"{a}_vs_{b}": total_variation(dists[a], dists[b]) for i, a in enumerate(names) for b in names[i + 1 :]}
    norm = {n: set(df["title"].map(normalize_title)) for n, df in splits.items()}
    leakage = {f"{a}_vs_{b}": len(norm[a] & norm[b]) for i, a in enumerate(names) for b in names[i + 1 :]}
    return {"class_distribution": dists, "total_variation": tv, "title_overlap": leakage}
