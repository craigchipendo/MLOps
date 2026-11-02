"""EDA text augmentation (Wei & Zou, 2019, https://arxiv.org/abs/1901.11196).

Four operations: synonym replacement (SR), random insertion (RI), random swap
(RS) and random deletion (RD). SR and RI need WordNet; when NLTK's WordNet
corpus is not installed the augmenter falls back to RS + RD only, so the
pipeline never breaks on a machine without the corpus.

Augmentation is only ever applied to the TRAINING split. Dev and test must stay
a faithful picture of production traffic.
"""

from __future__ import annotations

import random
from collections.abc import Callable

import pandas as pd

from newsclf.utils import get_logger

log = get_logger(__name__)

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "s",
        "t",
        "can",
        "will",
        "just",
        "don",
        "should",
        "now",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "him",
        "his",
        "she",
        "her",
        "it",
        "its",
        "they",
        "them",
        "their",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "am",
        "as",
        "until",
        "while",
    ]
)

SynonymFn = Callable[[str], list[str]]


def wordnet_synonyms() -> SynonymFn | None:
    """Return a WordNet synonym lookup, or None if the corpus is unavailable."""
    try:
        from nltk.corpus import wordnet

        wordnet.synsets("test")
    except Exception:
        return None

    def lookup(word: str) -> list[str]:
        names = {lemma.name().replace("_", " ").lower() for syn in wordnet.synsets(word) for lemma in syn.lemmas()}
        names.discard(word.lower())
        return sorted(n for n in names if n.replace(" ", "").isalpha())

    return lookup


class EDAAugmenter:
    def __init__(self, alpha: float = 0.1, synonyms: SynonymFn | None = None, seed: int = 42):
        self.alpha = alpha
        self.synonyms = synonyms
        self.rng = random.Random(seed)
        ops = [self.random_swap, self.random_deletion]
        if synonyms is not None:
            ops = [self.synonym_replacement, self.random_insertion, *ops]
        self.ops = ops

    def _n(self, words: list[str]) -> int:
        return max(1, int(self.alpha * len(words)))

    def synonym_replacement(self, words: list[str]) -> list[str]:
        words = words.copy()
        candidates = [i for i, w in enumerate(words) if w.lower() not in STOPWORDS]
        self.rng.shuffle(candidates)
        replaced = 0
        for i in candidates:
            options = self.synonyms(words[i]) if self.synonyms else []
            if options:
                words[i] = self.rng.choice(options)
                replaced += 1
            if replaced >= self._n(words):
                break
        return words

    def random_insertion(self, words: list[str]) -> list[str]:
        words = words.copy()
        for _ in range(self._n(words)):
            content = [w for w in words if w.lower() not in STOPWORDS] or words
            options = self.synonyms(self.rng.choice(content)) if self.synonyms else []
            if options:
                words.insert(self.rng.randint(0, len(words)), self.rng.choice(options))
        return words

    def random_swap(self, words: list[str]) -> list[str]:
        words = words.copy()
        if len(words) < 2:
            return words
        for _ in range(self._n(words)):
            i, j = self.rng.sample(range(len(words)), 2)
            words[i], words[j] = words[j], words[i]
        return words

    def random_deletion(self, words: list[str]) -> list[str]:
        if len(words) <= 1:
            return words
        kept = [w for w in words if self.rng.random() > self.alpha]
        return kept or [self.rng.choice(words)]

    def augment(self, text: str, n: int) -> list[str]:
        words = str(text).split()
        out = []
        for k in range(n):
            op = self.ops[k % len(self.ops)]
            candidate = " ".join(op(words))
            if candidate and candidate != text:
                out.append(candidate)
        return out


def augment_frame(
    df: pd.DataFrame, target_classes: list[str], n_per_sample: int, alpha: float, seed: int
) -> pd.DataFrame:
    """Append augmented copies of rows belonging to `target_classes`."""
    synonyms = wordnet_synonyms()
    augmenter = EDAAugmenter(alpha=alpha, synonyms=synonyms, seed=seed)
    log.info(
        "EDA augmentation on %s (ops: %s)",
        target_classes,
        [op.__name__ for op in augmenter.ops],
    )
    rows = []
    for _, row in df[df["topic"].isin(target_classes)].iterrows():
        for text in augmenter.augment(row["title"], n_per_sample):
            new = row.copy()
            new["title"] = text
            new["is_augmented"] = True
            rows.append(new)
    base = df.assign(is_augmented=False)
    if not rows:
        return base
    out = pd.concat([base, pd.DataFrame(rows)], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
