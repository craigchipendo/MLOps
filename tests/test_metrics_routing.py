import numpy as np
import pandas as pd

from newsclf.deployment.routing import RoutingPolicy, route_predictions, write_routing_outputs
from newsclf.evaluation.metrics import choose_threshold, compute_metrics, routing_curve, top_k_accuracy
from newsclf.monitoring.drift import psi

LABELS = ["A", "B", "C"]
PROBS = np.array([[0.9, 0.05, 0.05], [0.4, 0.35, 0.25], [0.1, 0.2, 0.7], [0.3, 0.6, 0.1]])
Y = np.array([0, 1, 2, 1])


def test_metrics():
    m = compute_metrics(Y, PROBS, LABELS)
    assert m["accuracy"] == 0.75
    assert m["top2_accuracy"] == 1.0
    assert set(m["per_class"]) == set(LABELS)
    assert top_k_accuracy(Y, PROBS, 1) == 0.75


def test_threshold_selection():
    curve = routing_curve(Y, PROBS, thresholds=[0.3, 0.5, 0.65])
    policy = choose_threshold(curve, target_accuracy=1.0)
    assert policy["confidence_threshold"] == 0.5
    assert policy["expected_auto_accuracy"] == 1.0


def test_routing_rules(tmp_path):
    profile = pd.DataFrame({"oov_rate": [0.0, 0.0, 0.9, 0.0], "likely_non_english": [False, False, True, False]})
    policy = RoutingPolicy(confidence_threshold=0.6, margin_threshold=0.1)
    out = route_predictions(["t1", "t2", "t3", "t4"], PROBS, LABELS, policy, profile)
    assert out.route.tolist() == ["auto", "manual", "manual", "auto"]
    assert "input_not_english" in out.loc[2, "reason"]
    out.insert(1, "topic", ["A", "B", "C", "B"])
    summary = write_routing_outputs(out, tmp_path)
    assert (tmp_path / "auto.csv").exists() and (tmp_path / "manual.csv").exists()
    assert summary["auto"] == 2 and summary["auto_accuracy"] == 1.0


def test_psi():
    rng = np.random.default_rng(0)
    ref = rng.normal(10, 2, 5000)
    assert psi(ref, rng.normal(10, 2, 5000)) < 0.1
    assert psi(ref, rng.normal(16, 2, 5000)) > 0.25
