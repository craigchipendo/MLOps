"""Model architectures. All take integer token sequences and output softmax scores."""

from __future__ import annotations

import keras
from keras import layers


def legacy_dense(vocab_size: int, seq_len: int, n_classes: int, cfg: dict) -> keras.Model:
    """Exact replica of the prototype: Embedding -> Dense(per token) -> Flatten -> Softmax."""
    inputs = keras.Input(shape=(seq_len,), dtype="int64", name="tokens")
    x = layers.Embedding(vocab_size, cfg["embedding_dim"])(inputs)
    x = layers.Dense(cfg["dense_dim"], activation="relu")(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="probabilities")(x)
    return keras.Model(inputs, outputs, name="legacy_dense")


def pooled(vocab_size: int, seq_len: int, n_classes: int, cfg: dict) -> keras.Model:
    """Order-invariant bag-of-embeddings with average + max pooling.

    Flatten in the prototype ties each weight to a token *position*, so "Apple"
    in slot 1 and slot 5 are unrelated features. Pooling removes that, which
    matters for short, variable-length headlines.
    """
    inputs = keras.Input(shape=(seq_len,), dtype="int64", name="tokens")
    x = layers.Embedding(vocab_size, cfg["embedding_dim"])(inputs)
    x = layers.SpatialDropout1D(cfg["spatial_dropout"])(x)
    x = layers.Concatenate()([layers.GlobalAveragePooling1D()(x), layers.GlobalMaxPooling1D()(x)])
    x = layers.Dense(cfg["dense_dim"], activation="relu")(x)
    x = layers.Dropout(cfg["dropout"])(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="probabilities")(x)
    return keras.Model(inputs, outputs, name="pooled")


def bilstm(vocab_size: int, seq_len: int, n_classes: int, cfg: dict) -> keras.Model:
    inputs = keras.Input(shape=(seq_len,), dtype="int64", name="tokens")
    x = layers.Embedding(vocab_size, cfg["embedding_dim"], mask_zero=True)(inputs)
    x = layers.SpatialDropout1D(cfg["spatial_dropout"])(x)
    x = layers.Bidirectional(layers.LSTM(cfg["rnn_units"]))(x)
    x = layers.Dense(cfg["dense_dim"], activation="relu")(x)
    x = layers.Dropout(cfg["dropout"])(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="probabilities")(x)
    return keras.Model(inputs, outputs, name="bilstm")


def cnn(vocab_size: int, seq_len: int, n_classes: int, cfg: dict) -> keras.Model:
    inputs = keras.Input(shape=(seq_len,), dtype="int64", name="tokens")
    x = layers.Embedding(vocab_size, cfg["embedding_dim"])(inputs)
    x = layers.SpatialDropout1D(cfg["spatial_dropout"])(x)
    x = layers.Conv1D(cfg["conv_filters"], cfg["kernel_size"], padding="same", activation="relu")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(cfg["dense_dim"], activation="relu")(x)
    x = layers.Dropout(cfg["dropout"])(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="probabilities")(x)
    return keras.Model(inputs, outputs, name="cnn")


ARCHITECTURES = {"legacy_dense": legacy_dense, "pooled": pooled, "bilstm": bilstm, "cnn": cnn}


def build_model(
    vocab_size: int, seq_len: int, n_classes: int, model_cfg: dict, learning_rate: float, top_k: int = 2
) -> keras.Model:
    arch = model_cfg["architecture"]
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{arch}'. Choose from {list(ARCHITECTURES)}")
    model = ARCHITECTURES[arch](vocab_size, seq_len, n_classes, model_cfg)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(k=top_k, name=f"top{top_k}_accuracy"),
        ],
    )
    return model
