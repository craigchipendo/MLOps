"""Prediction backends.

* `LocalBackend` loads the served SavedModel versions in-process.
* `RestBackend` talks to TF Serving (Docker or native) or to the local
  TF-Serving-compatible server, over the standard REST predict API.

`TopicClassifier` sits on top of either and turns raw titles into labelled
predictions. For a version that takes token ids (the legacy prototype) it does
the client-side tokenization itself - the exact pattern the string-input export
was designed to eliminate.
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import requests
import tensorflow as tf

from newsclf.config import resolve
from newsclf.features.text import build_vectorizer, load_vocabulary
from newsclf.serving.registry import Registry


class LocalBackend:
    def __init__(self, registry: Registry):
        self.registry = registry
        self._signatures: dict[int, object] = {}

    def signature(self, version: int):
        if version not in self._signatures:
            loaded = tf.saved_model.load(str(self.registry.version_dir(version)))
            self._signatures[version] = (loaded, loaded.signatures["serving_default"])
        return self._signatures[version][1]

    def predict(self, instances: list, version: int | None = None, label: str | None = None) -> list:
        version = self.registry.version_for(label, version)
        sig = self.signature(version)
        ((name, spec),) = sig.structured_input_signature[1].items()
        tensor = tf.constant(np.asarray(instances), dtype=spec.dtype)
        outputs = sig(**{name: tensor})
        if len(outputs) == 1:
            return next(iter(outputs.values())).numpy().tolist()
        n = len(instances)
        return [{k: v.numpy()[i].tolist() for k, v in outputs.items()} for i in range(n)]


class RestBackend:
    def __init__(self, base_url: str, model_name: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout

    def url(self, version: int | None = None, label: str | None = None) -> str:
        path = f"{self.base_url}/v1/models/{self.model_name}"
        if version is not None:
            path += f"/versions/{version}"
        elif label is not None:
            path += f"/labels/{label}"
        return path

    def predict(self, instances: list, version: int | None = None, label: str | None = None) -> list:
        resp = requests.post(
            self.url(version, label) + ":predict",
            data=json.dumps({"instances": instances}),
            headers={"content-type": "application/json"},
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Serving error {resp.status_code}: {resp.text}")
        return resp.json()["predictions"]

    def status(self) -> dict:
        return requests.get(self.url(), timeout=self.timeout).json()


@lru_cache(maxsize=8)
def _legacy_vectorizer(vocab_path: str, max_tokens: int, seq_len: int):
    return build_vectorizer(
        {"max_tokens": max_tokens, "sequence_length": seq_len}, load_vocabulary(resolve(vocab_path))
    )


class TopicClassifier:
    def __init__(self, registry: Registry, backend):
        self.registry = registry
        self.backend = backend

    def predict_proba(
        self, titles: list[str], version: int | None = None, label: str | None = None, batch_size: int = 512
    ) -> tuple[np.ndarray, list[str], int]:
        version = self.registry.version_for(label, version)
        info = self.registry.info(version)
        out = []
        for i in range(0, len(titles), batch_size):
            chunk = list(map(str, titles[i : i + batch_size]))
            if info["input_type"] == "token_ids":
                feats = info["features"]
                vec = _legacy_vectorizer(info["vocab_path"], feats["max_tokens"], feats["sequence_length"])
                chunk = vec(tf.constant(chunk)).numpy().tolist()
            out.extend(self.backend.predict(chunk, version=version))
        return np.asarray(out, dtype="float32"), info["labels"], version

    def classify(self, titles: list[str], **kw) -> list[dict]:
        probs, labels, version = self.predict_proba(titles, **kw)
        order = np.argsort(-probs, axis=1)
        return [
            {
                "title": t,
                "topic": labels[o[0]],
                "confidence": float(p[o[0]]),
                "second_guess": labels[o[1]],
                "margin": float(p[o[0]] - p[o[1]]),
                "probabilities": dict(zip(labels, map(float, p), strict=True)),
                "model_version": version,
            }
            for t, p, o in zip(titles, probs, order, strict=True)
        ]
