"""Train -> export -> serve (TF-Serving-compatible API) -> route, on synthetic data."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from newsclf.models.train import run_experiment
from newsclf.serving.export import export_experiment
from newsclf.serving.local_server import create_app
from newsclf.serving.predictor import LocalBackend, TopicClassifier
from newsclf.serving.registry import Registry


@pytest.mark.slow
def test_train_export_serve(tmp_project):
    result = run_experiment(tmp_project, track=True)
    assert result["test"]["accuracy"] > 0.5
    assert result["routing_policy"]["confidence_threshold"] > 0

    version = export_experiment(tmp_project, "T1", label="stable")
    registry = Registry(tmp_project)
    assert registry.data["labels"]["stable"] == version
    assert (registry.version_dir(version) / "assets.extra" / "labels.txt").exists()

    clf = TopicClassifier(registry, LocalBackend(registry))
    out = clf.classify(["nasa telescope planet fossil species"], label="stable")
    assert out[0]["topic"] == "SCIENCE"

    client = TestClient(create_app(tmp_project))
    assert client.get("/health").status_code == 200
    status = client.get("/v1/models/newsapp_model").json()
    assert status["model_version_status"][0]["state"] == "AVAILABLE"

    resp = client.post(
        "/v1/models/newsapp_model/labels/stable:predict",
        json={"instances": ["football match league coach", "covid vaccine hospital"]},
    )
    preds = np.array(resp.json()["predictions"])
    assert preds.shape == (2, len(result["labels"]))
    assert np.allclose(preds.sum(1), 1, atol=1e-4)

    resp = client.post(f"/v1/models/newsapp_model/versions/{version}:predict", json={"instances": ["film actor"]})
    assert resp.status_code == 200
    assert client.post("/v1/models/newsapp_model/labels/nope:predict", json={"instances": ["x"]}).status_code == 404

    body = client.post(
        "/v1/classify",
        json={"titles": ["Una folla oceanica di oltre 2 milioni di persone", "football match league coach team"]},
    ).json()
    routes = [r["route"] for r in body["results"]]
    assert routes[0] == "manual"
    assert body["results"][1]["predicted_topic"] == "SPORTS"
