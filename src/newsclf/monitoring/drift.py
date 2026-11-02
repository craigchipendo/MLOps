"""Production monitoring: input metrics, output metrics and alarms.

Input metrics  - token count, out-of-vocabulary (OOV) rate, English stop-word
                 ratio, non-ASCII ratio, language guard.
Output metrics - predicted-topic mix, mean confidence, share below the routing
                 threshold, agreement between the logged legacy scores and the
                 current stable model.
Alarms         - batch OOV rate, rolling-window OOV spike, length-distribution
                 shift (PSI) versus the training data.
"""

from __future__ import annotations

from pathlib import Path

# The backend is not forced here: plots are written with savefig, which works on any
# backend, and forcing "Agg" at import would stop notebooks from displaying figures.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from newsclf.data.augment import STOPWORDS  # noqa: E402
from newsclf.data.validate import normalize_title  # noqa: E402
from newsclf.features.text import build_vectorizer, load_vocabulary, oov_rate  # noqa: E402

# Single letters ("a", "i", "s", "t") also occur in Romance languages, so they
# are excluded from the English function-word signal.
ENGLISH_MARKERS = frozenset(w for w in STOPWORDS if len(w) > 1)


class InputProfiler:
    def __init__(self, vocabulary: list[str], features: dict, monitoring_cfg: dict):
        self.vectorizer = build_vectorizer(features, vocabulary)
        self.mcfg = monitoring_cfg

    @classmethod
    def for_version(cls, cfg: dict, info: dict, version_dir: Path) -> InputProfiler:
        """Profile inputs against the vocabulary of a served version."""
        vocab = load_vocabulary(Path(version_dir) / "assets.extra" / "vocabulary.txt")
        return cls(vocab, info["features"], cfg["monitoring"])

    def profile(self, titles: list[str]) -> pd.DataFrame:
        titles = [str(t) for t in titles]
        tokens = [normalize_title(t).split() for t in titles]
        df = pd.DataFrame(
            {
                "title": titles,
                "n_tokens": [len(t) for t in tokens],
                "oov_rate": oov_rate(self.vectorizer, titles).numpy(),
                "stopword_ratio": [sum(w in ENGLISH_MARKERS for w in t) / max(len(t), 1) for t in tokens],
                "non_ascii_ratio": [sum(ord(c) > 127 for c in t) / max(len(t), 1) for t in titles],
            }
        )
        long_enough = df["n_tokens"] >= self.mcfg["min_tokens_for_language_check"]
        df["likely_non_english"] = (
            long_enough
            & (df["oov_rate"] >= self.mcfg["oov_rate_alert"])
            & (df["stopword_ratio"] < self.mcfg["min_english_stopword_ratio"])
        ) | (df["non_ascii_ratio"] > 0.3)
        return df


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population stability index. <0.1 stable, 0.1-0.25 moderate, >0.25 major shift."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(current, edges)[0] / max(len(current), 1)
    ref, cur = np.clip(ref, 1e-4, None), np.clip(cur, 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def detect_alarms(profile: pd.DataFrame, mcfg: dict, reference_profile: pd.DataFrame | None = None) -> dict:
    window = mcfg["rolling_window"]
    rolling = profile["oov_rate"].rolling(window, min_periods=window).mean()
    breach = rolling[rolling > mcfg["batch_oov_alert"]]
    alarms = []
    batch_oov = float(profile["oov_rate"].mean())
    if batch_oov > mcfg["batch_oov_alert"]:
        alarms.append(f"Batch mean OOV rate {batch_oov:.1%} exceeds {mcfg['batch_oov_alert']:.0%}")
    if not breach.empty:
        alarms.append(f"Rolling OOV rate (window={window}) breached threshold from row {int(breach.index[0])}")
    n_non_en = int(profile["likely_non_english"].sum())
    if n_non_en:
        alarms.append(f"{n_non_en} titles look non-English - check the source feed")
    result = {
        "batch_oov_rate": round(batch_oov, 4),
        "rolling_breach_start_row": int(breach.index[0]) if not breach.empty else None,
        "non_english_titles": n_non_en,
        "alarms": alarms,
    }
    if reference_profile is not None:
        result["psi_n_tokens"] = round(psi(reference_profile["n_tokens"].to_numpy(), profile["n_tokens"].to_numpy()), 4)
        result["psi_oov_rate"] = round(psi(reference_profile["oov_rate"].to_numpy(), profile["oov_rate"].to_numpy()), 4)
        if result["psi_n_tokens"] > 0.25:
            alarms.append(f"Title-length distribution shifted (PSI={result['psi_n_tokens']:.2f})")
    return result


def output_metrics(probs: np.ndarray, labels: list[str], threshold: float) -> dict:
    pred = probs.argmax(1)
    conf = probs.max(1)
    return {
        "predicted_topic_share": {labels[i]: round(float((pred == i).mean()), 4) for i in range(len(labels))},
        "mean_confidence": round(float(conf.mean()), 4),
        "share_below_threshold": round(float((conf < threshold).mean()), 4),
    }


def plot_oov_trend(profile: pd.DataFrame, ids, mcfg: dict, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ids, profile["oov_rate"], lw=1, alpha=0.6, label="OOV rate per title")
    ax.plot(
        ids,
        profile["oov_rate"].rolling(mcfg["rolling_window"], min_periods=1).mean(),
        lw=2,
        label=f"rolling mean (window={mcfg['rolling_window']})",
    )
    ax.axhline(mcfg["batch_oov_alert"], color="red", ls="--", label="alarm threshold")
    flagged = profile["likely_non_english"].to_numpy()
    ax.scatter(
        np.asarray(ids)[flagged], profile["oov_rate"][flagged], color="red", s=18, zorder=3, label="likely non-English"
    )
    ax.set_xlabel("article id")
    ax.set_ylabel("share of unknown tokens")
    ax.set_title("Production input monitor: out-of-vocabulary rate")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
