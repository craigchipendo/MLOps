"""Data repair.

The prototype's BUSINESS rows all carry the same title ("Why iPhone 12 Will Be
Another 'Defining Chapter' In Apple's Growth Story") because a character-encoding
fix overwrote the column. The links survived, so titles can be restored:

1. From a trusted reference file keyed by `link`, when one is available.
2. Otherwise, from the URL slug, which for news sites is almost always a
   hyphenated copy of the headline.
"""

from __future__ import annotations

import re
from urllib.parse import unquote, urlparse

import pandas as pd

from newsclf.utils import get_logger

log = get_logger(__name__)

_EXTENSIONS = re.compile(r"\.(s?html?|aspx?|php|cms|stm|ece|jsp)$", re.IGNORECASE)
_WORD = re.compile(r"^[A-Za-z][A-Za-z']*$")
_NUMERIC = re.compile(r"^[\d.]+$")


def _is_identifier(token: str) -> bool:
    """Tokens like `idUSKCN25A04M`, `20200812`, `p55kum`, `1203007840`."""
    if not token:
        return True
    if _NUMERIC.match(token):
        return len(token) >= 5
    has_digit = any(c.isdigit() for c in token)
    has_alpha = any(c.isalpha() for c in token)
    return has_digit and has_alpha and len(token) >= 6


def title_from_url(url: str, min_words: int = 3) -> str | None:
    """Reconstruct a headline from an article URL slug. Returns None if no slug."""
    if not isinstance(url, str) or not url:
        return None
    path = unquote(urlparse(url).path)
    candidates: list[list[str]] = []
    for segment in (s for s in path.split("/") if s):
        segment = _EXTENSIONS.sub("", segment)
        tokens = [t for t in re.split(r"[-_+]+", segment) if t]
        if sum(bool(_WORD.match(t)) for t in tokens) >= min_words:
            candidates.append(tokens)
    if not candidates:
        return None
    tokens = max(candidates, key=lambda ts: sum(bool(_WORD.match(t)) for t in ts))
    while tokens and _is_identifier(tokens[-1]):
        tokens = tokens[:-1]
    # Trailing dates such as "09-08-2020" / "2020-08-07" become runs of numbers.
    run = 0
    while run < len(tokens) and _NUMERIC.match(tokens[-1 - run]):
        run += 1
    if run >= 2:
        tokens = tokens[:-run]
    while tokens and _is_identifier(tokens[0]):
        tokens = tokens[1:]
    if sum(bool(_WORD.match(t)) for t in tokens) < min_words:
        return None
    return " ".join(tokens)


def find_collapsed_classes(df: pd.DataFrame, max_unique_ratio: float = 0.05) -> dict[str, str]:
    """Return {topic: dominant_title} for topics whose titles have collapsed."""
    collapsed = {}
    for topic, group in df.groupby("topic"):
        ratio = group["title"].nunique() / max(len(group), 1)
        if len(group) > 50 and ratio < max_unique_ratio:
            collapsed[topic] = group["title"].mode().iloc[0]
    return collapsed


def repair_collapsed_titles(
    df: pd.DataFrame,
    max_unique_ratio: float = 0.05,
    reference: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Restore titles for rows of collapsed classes. Unrecoverable rows are dropped."""
    df = df.copy()
    collapsed = find_collapsed_classes(df, max_unique_ratio)
    report = {"collapsed_classes": list(collapsed), "from_reference": 0, "from_slug": 0, "dropped": 0}
    if not collapsed:
        return df, report

    mask = pd.Series(False, index=df.index)
    for topic, bad_title in collapsed.items():
        mask |= (df["topic"] == topic) & (df["title"] == bad_title)
    df["title_source"] = "original"

    if reference is not None:
        lookup = reference.set_index("link")["title"]
        restored = df.loc[mask, "link"].map(lookup)
        hit = restored.notna()
        df.loc[restored[hit].index, "title"] = restored[hit]
        df.loc[restored[hit].index, "title_source"] = "reference"
        report["from_reference"] = int(hit.sum())
        mask &= df["title_source"] == "original"

    slugs = df.loc[mask, "link"].map(title_from_url)
    ok = slugs.notna()
    df.loc[slugs[ok].index, "title"] = slugs[ok]
    df.loc[slugs[ok].index, "title_source"] = "url_slug"
    report["from_slug"] = int(ok.sum())

    unrecoverable = slugs[~ok].index
    report["dropped"] = int(len(unrecoverable))
    df = df.drop(index=unrecoverable)
    log.info(
        "Repaired collapsed classes %s: %d from reference, %d from URL slug, %d dropped",
        list(collapsed),
        report["from_reference"],
        report["from_slug"],
        report["dropped"],
    )
    return df, report
