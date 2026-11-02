"""Pipeline orchestration: the lifecycle stages wired together.

Each function is one stage and is exposed through the CLI (`newsclf <stage>`),
the Makefile and the walkthrough notebook, so every entry point runs the same
code path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from newsclf.config import list_experiments, load_config, resolve
from newsclf.data.ingest import load_production_db, load_raw
from newsclf.data.prepare import load_splits
from newsclf.data.validate import normalize_title, validate_frame
from newsclf.deployment.routing import RoutingPolicy, route_predictions, write_routing_outputs
from newsclf.evaluation.metrics import compute_metrics
from newsclf.monitoring.drift import InputProfiler, detect_alarms, output_metrics, plot_oov_trend
from newsclf.serving.predictor import LocalBackend, RestBackend, TopicClassifier
from newsclf.serving.registry import Registry
from newsclf.utils import get_logger, read_json, write_json

log = get_logger(__name__)


# --------------------------------------------------------------------- data
def validate_raw(cfg: dict) -> dict:
    report = validate_frame(load_raw(cfg), collapse_ratio=cfg["data"]["collapse_unique_ratio"])
    write_json(report, resolve(cfg["paths"]["reports_dir"]) / "data_validation_raw.json")
    for w in report["warnings"]:
        log.warning(w)
    return report


# ------------------------------------------------------------- experiments
def run_all(cfg_base: dict, experiments: list[str] | None = None, include_baselines: bool = True) -> pd.DataFrame:
    from newsclf.models.benchmark import run_benchmark
    from newsclf.models.legacy import evaluate_legacy
    from newsclf.models.train import run_experiment

    if include_baselines:
        evaluate_legacy(cfg_base)
    for name in experiments or list_experiments():
        run_experiment(load_config(name))
    if include_baselines:
        run_benchmark(load_config("E3"))
    board = leaderboard(cfg_base)
    select_champion(cfg_base)
    return board


def leaderboard(cfg: dict) -> pd.DataFrame:
    rows = []
    for path in sorted(resolve(cfg["paths"]["artifacts_dir"]).glob("*/metrics.json")):
        m = read_json(path)
        dev, test = m.get("dev", {}), m.get("test", {})
        rows.append(
            {
                "experiment": m["experiment"],
                "model": m.get("model_type"),
                "classes": len(m.get("labels", [])),
                "dev_acc": dev.get("accuracy"),
                "dev_macro_f1": dev.get("macro_f1"),
                "test_acc": test.get("accuracy"),
                "test_top2": test.get("top2_accuracy"),
                "test_macro_f1": test.get("macro_f1"),
                "test_SCIENCE_f1": test.get("per_class", {}).get("SCIENCE", {}).get("f1"),
                "test_BUSINESS_f1": test.get("per_class", {}).get("BUSINESS", {}).get("f1"),
                "auto_coverage@95%": (m.get("routing_policy") or {}).get("test_coverage"),
                "params": m.get("params"),
                "data_quality_passed": (m.get("data_quality") or {}).get("passed"),
                "promotable": m.get("promotable", False),
                "description": m.get("description", ""),
            }
        )
    board = pd.DataFrame(rows)
    if board.empty:
        return board
    out = resolve(cfg["paths"]["reports_dir"])
    out.mkdir(parents=True, exist_ok=True)
    board.to_csv(out / "leaderboard.csv", index=False)
    (out / "leaderboard.md").write_text(board.round(4).to_markdown(index=False), encoding="utf-8")
    return board


def select_champion(cfg: dict) -> str:
    """Champion = best DEV macro-F1 among promotable experiments (test is reported, not used to choose)."""
    board = leaderboard(cfg)
    eligible = board[board["promotable"] & board["dev_macro_f1"].notna()]
    if eligible.empty:
        raise RuntimeError("No promotable experiments with dev metrics; run `newsclf run-all` first")
    best = eligible.sort_values("dev_macro_f1", ascending=False).iloc[0]
    champion = {
        "experiment": best["experiment"],
        "selection_metric": "dev_macro_f1",
        "dev_macro_f1": float(best["dev_macro_f1"]),
        "test_macro_f1": float(best["test_macro_f1"]),
        "test_accuracy": float(best["test_acc"]),
    }
    write_json(champion, resolve(cfg["paths"]["reports_dir"]) / "champion.json")
    log.info("Champion: %s (dev macro-F1 %.4f)", champion["experiment"], champion["dev_macro_f1"])
    return champion["experiment"]


# -------------------------------------------------------------- deployment
def make_backend(cfg: dict, url: str | None = None):
    registry = Registry(cfg)
    backend = RestBackend(url, registry.model_name) if url else LocalBackend(registry)
    return registry, backend


def deploy(cfg: dict, candidate: str | None = None, reset: bool = True) -> dict:
    """Bootstrap serving: legacy prototype as `stable` (v1), champion as `canary` (v2)."""
    from newsclf.serving.export import export_experiment, export_legacy, reset_serving

    if reset:
        reset_serving(cfg)
    candidate = candidate or read_json(resolve(cfg["paths"]["reports_dir"]) / "champion.json")["experiment"]
    v_legacy = export_legacy(cfg, label="stable")
    v_candidate = export_experiment(cfg, candidate, label="canary")
    reg = Registry(cfg)
    return {"labels": reg.data["labels"], "versions": {v_legacy: "E1", v_candidate: candidate}}


def canary_check(cfg: dict, url: str | None = None, promote: bool = False, sample: int | None = None) -> dict:
    """Compare canary vs stable on the candidate's held-out test split, optionally promote."""
    registry, backend = make_backend(cfg, url)
    clf = TopicClassifier(registry, backend)
    canary_v, stable_v = registry.version_for("canary"), registry.version_for("stable")
    canary_info = registry.info(canary_v)
    exp_metrics = read_json(resolve(cfg["paths"]["artifacts_dir"]) / canary_info["source"] / "metrics.json")
    test = load_splits(resolve(exp_metrics["data_dir"]))["test"]
    # Contamination control: drop test titles that either model saw during training,
    # otherwise the incumbent is rewarded for memorisation.
    seen: set[str] = set()
    for v in (canary_v, stable_v):
        train_ref = registry.info(v).get("train_data")
        if train_ref and resolve(train_ref).exists():
            seen |= set(pd.read_csv(resolve(train_ref), usecols=["title"])["title"].map(normalize_title))
    rows_before = len(test)
    test = test[~test["title"].map(normalize_title).isin(seen)]
    if sample:
        test = test.sample(min(sample, len(test)), random_state=cfg["seed"])
    titles = test["title"].astype(str).tolist()

    report = {
        "canary_version": canary_v,
        "stable_version": stable_v,
        "rows": len(titles),
        "rows_removed_as_seen_in_training": rows_before - len(test) if not sample else None,
    }
    for name, version in (("stable", stable_v), ("canary", canary_v)):
        probs, labels, _ = clf.predict_proba(titles, version=version)
        y = test["topic"].map({lab: i for i, lab in enumerate(labels)}).to_numpy()
        report[name] = compute_metrics(y, probs, labels)

    s, c = report["stable"], report["canary"]
    gain = c["macro_f1"] - s["macro_f1"]
    drops = {k: round(s["per_class"][k]["f1"] - c["per_class"][k]["f1"], 4) for k in c["per_class"]}
    worst = max(drops.values())
    dcfg = cfg["deployment"]
    passed = gain >= dcfg["canary_min_macro_f1_gain"] and worst <= dcfg["canary_max_class_f1_drop"]
    report["decision"] = {
        "macro_f1_gain": round(gain, 4),
        "per_class_f1_drop": drops,
        "worst_class_drop": worst,
        "passed": bool(passed),
        "promoted": False,
    }
    if promote and passed:
        registry.promote(canary_v)
        registry.save()
        report["decision"]["promoted"] = True
    write_json(report, resolve(cfg["paths"]["reports_dir"]) / "canary_report.json")
    log.info(
        "Canary v%s vs stable v%s: macro-F1 %+.4f, worst class drop %.4f -> %s",
        canary_v,
        stable_v,
        gain,
        worst,
        "PASS" if passed else "FAIL",
    )
    return report


def route(
    cfg: dict,
    input_csv: str | Path,
    url: str | None = None,
    label: str = "stable",
    out_dir: str | Path | None = None,
    limit: int | None = None,
) -> dict:
    """Score a CSV of titles and split it into auto.csv / manual.csv."""
    registry, backend = make_backend(cfg, url)
    df = pd.read_csv(input_csv)
    if limit:
        df = df.head(limit)
    clf = TopicClassifier(registry, backend)
    version = registry.version_for(label)
    info = registry.info(version)
    probs, labels, _ = clf.predict_proba(df["title"].astype(str).tolist(), version=version)
    profile = InputProfiler.for_version(cfg, info, registry.version_dir(version)).profile(df["title"].tolist())
    policy = RoutingPolicy.from_version(cfg, info)
    decisions = route_predictions(df["title"].tolist(), probs, labels, policy, profile)
    if "topic" in df.columns:
        decisions.insert(1, "topic", df["topic"].to_numpy())
    out_dir = Path(out_dir) if out_dir else resolve(cfg["paths"]["reports_dir"]) / "routing"
    summary = write_routing_outputs(decisions, out_dir)
    summary.update({"model_version": version, "policy": policy.as_dict()})
    write_json(summary, out_dir / "routing_summary.json")
    log.info("Routed %d titles: %.1f%% automated", summary["total"], 100 * summary["automation_rate"])
    return summary


# -------------------------------------------------------------- monitoring
def monitor(cfg: dict, url: str | None = None, label: str = "stable", db: str | Path | None = None) -> dict:
    registry, backend = make_backend(cfg, url)
    version = registry.version_for(label)
    info = registry.info(version)
    db = db or resolve(cfg["paths"]["raw_dir"]) / cfg["data"]["production_db"]
    prod = load_production_db(db)
    titles = prod["title"].astype(str).tolist()

    profiler = InputProfiler.for_version(cfg, info, registry.version_dir(version))
    profile = profiler.profile(titles)
    reference = None
    if info.get("data_version"):
        ref_path = resolve(cfg["paths"]["processed_dir"]) / info["data_version"] / "train.csv"
        if ref_path.exists():
            reference = profiler.profile(pd.read_csv(ref_path)["title"].sample(5000, random_state=0).tolist())
    alarms = detect_alarms(profile.set_index(prod["id"]), cfg["monitoring"], reference)

    policy = RoutingPolicy.from_version(cfg, info)
    clf = TopicClassifier(registry, backend)
    probs, labels, _ = clf.predict_proba(titles, version=version)
    report = {
        "model_version": version,
        "source": info["source"],
        "rows": len(prod),
        "inputs": alarms,
        "outputs_current_model": output_metrics(probs, labels, policy.confidence_threshold),
    }

    score_cols = [c for c in prod.columns if c.startswith("class_")]
    if score_cols:
        logged = prod[score_cols].to_numpy(dtype="float32")
        # The production log was scored by the inherited prototype, so use its label order.
        legacy_labels = (
            (resolve(cfg["paths"]["legacy_model_dir"]) / "vocab" / "labels.txt").read_text(encoding="utf-8").split("\n")
        )
        report["outputs_logged_legacy"] = output_metrics(logged, legacy_labels, policy.confidence_threshold)
        legacy_pred = np.array(legacy_labels)[logged.argmax(1)]
        current_pred = np.array(labels)[probs.argmax(1)]
        report["agreement_logged_vs_current"] = round(float((legacy_pred == current_pred).mean()), 4)

    out = resolve(cfg["paths"]["reports_dir"]) / "monitoring"
    out.mkdir(parents=True, exist_ok=True)
    flagged = prod.loc[profile["likely_non_english"].to_numpy(), ["id", "title", "domain"]]
    flagged.to_csv(out / "flagged_inputs.csv", index=False)
    report["flagged_domains"] = flagged["domain"].value_counts().to_dict()
    plot_oov_trend(profile, prod["id"], cfg["monitoring"], out / "oov_trend.png")
    write_json(report, out / "monitoring_report.json")
    for a in alarms["alarms"]:
        log.warning("ALARM: %s", a)
    return report
