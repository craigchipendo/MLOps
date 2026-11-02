"""Text vectorization.

The vectorizer is fitted on the TRAINING split only, saved as a plain vocabulary
file next to the model, and re-attached to the model at export time so serving
always uses the exact preprocessing the model was trained with.
"""

from __future__ import annotations

import re
from pathlib import Path

import keras
import tensorflow as tf

from newsclf.data.augment import STOPWORDS

_STOPWORD_PATTERN = r"\b(" + "|".join(sorted(map(re.escape, STOPWORDS))) + r")\b"
PUNCTUATION = r'[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']'
PAD_TOKEN, OOV_TOKEN = "", "[UNK]"


@keras.saving.register_keras_serializable(package="newsclf")
def standardize_remove_stopwords(text: tf.Tensor) -> tf.Tensor:
    """Lowercase, strip punctuation, drop English stop words."""
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, PUNCTUATION, " ")
    text = tf.strings.regex_replace(text, _STOPWORD_PATTERN, " ")
    return tf.strings.regex_replace(text, r"\s+", " ")


def build_vectorizer(features_cfg: dict, vocabulary: list[str] | None = None) -> keras.layers.TextVectorization:
    standardize = (
        standardize_remove_stopwords if features_cfg.get("remove_stopwords") else "lower_and_strip_punctuation"
    )
    return keras.layers.TextVectorization(
        max_tokens=features_cfg["max_tokens"],
        output_sequence_length=features_cfg["sequence_length"],
        standardize=standardize,
        vocabulary=vocabulary,
        name="title_vectorizer",
    )


def fit_vectorizer(features_cfg: dict, texts) -> keras.layers.TextVectorization:
    vectorizer = build_vectorizer(features_cfg)
    vectorizer.adapt(tf.constant(list(texts)))
    return vectorizer


def save_vocabulary(vectorizer: keras.layers.TextVectorization, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(vectorizer.get_vocabulary()), encoding="utf-8")


def load_vocabulary(path: str | Path) -> list[str]:
    return Path(path).read_text(encoding="utf-8").split("\n")


def load_vectorizer(features_cfg: dict, vocab_path: str | Path) -> keras.layers.TextVectorization:
    return build_vectorizer(features_cfg, vocabulary=load_vocabulary(vocab_path))


def oov_rate(vectorizer: keras.layers.TextVectorization, texts) -> tf.Tensor:
    """Share of non-padding tokens that map to [UNK] (index 1), per text."""
    seq = vectorizer(tf.constant(list(texts)))
    tokens = tf.cast(seq > 0, tf.float32)
    unk = tf.cast(seq == 1, tf.float32)
    return tf.math.divide_no_nan(tf.reduce_sum(unk, 1), tf.reduce_sum(tokens, 1))
