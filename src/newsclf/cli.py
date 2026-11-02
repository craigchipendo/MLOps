"""Command-line entry point: `newsclf <command>` (or `python -m newsclf <command>`).

Lifecycle order:
    validate -> prepare -> baseline -> train / run-all -> leaderboard -> deploy
    -> serve -> canary-check -> promote -> route -> monitor
"""

from __future__ import annotations

import argparse
import json
import sys

from newsclf.config import list_experiments, load_config
from newsclf.utils import get_logger

log = get_logger("newsclf")


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_validate(args):
    from newsclf.pipelines import validate_raw

    report = validate_raw(load_config())
    _print(
        {
            k: report[k]
            for k in (
                "rows",
                "class_counts",
                "duplicate_titles",
                "titles_with_conflicting_labels",
                "collapsed_classes",
                "warnings",
            )
        }
    )


def cmd_prepare(args):
    from newsclf.data.prepare import load_manifest, prepare_dataset

    for name in args.experiment or ["E3"]:
        path = prepare_dataset(load_config(name), force=args.force)
        m = load_manifest(path)
        _print(
            {
                "experiment": name,
                "data_dir": str(path),
                "files": m["files"],
                "repair": m["repair"],
                "deduplication": m["deduplication"],
                "title_overlap": m["split_validation"]["title_overlap"],
            }
        )


def cmd_baseline(args):
    from newsclf.models.legacy import evaluate_legacy

    r = evaluate_legacy(load_config())
    _print({k: r["test"][k] for k in ("accuracy", "top2_accuracy", "macro_f1")})


def cmd_train(args):
    from newsclf.models.train import run_experiment

    for name in args.experiment:
        cfg = load_config(name)
        if args.epochs:
            cfg["training"]["epochs"] = args.epochs
        if args.no_mlflow:
            cfg["tracking"]["mlflow"]["enabled"] = False
        r = run_experiment(cfg)
        _print(
            {
                "experiment": name,
                "dev_macro_f1": r["dev"]["macro_f1"],
                "test_accuracy": r["test"]["accuracy"],
                "test_macro_f1": r["test"]["macro_f1"],
                "routing_policy": r["routing_policy"],
            }
        )


def cmd_benchmark(args):
    from newsclf.models.benchmark import run_benchmark

    r = run_benchmark(load_config("E3"))
    _print(
        {
            "dev_macro_f1": r["dev"]["macro_f1"],
            "test_accuracy": r["test"]["accuracy"],
            "test_macro_f1": r["test"]["macro_f1"],
        }
    )


def cmd_run_all(args):
    from newsclf.pipelines import run_all

    board = run_all(load_config(), args.experiment or None)
    print(board.drop(columns=["description"]).round(4).to_string(index=False))


def cmd_leaderboard(args):
    from newsclf.pipelines import leaderboard, select_champion

    board = leaderboard(load_config())
    print(board.drop(columns=["description"]).round(4).to_string(index=False))
    select_champion(load_config())


def cmd_deploy(args):
    from newsclf.pipelines import deploy

    _print(deploy(load_config(), candidate=args.candidate, reset=not args.keep))


def cmd_export(args):
    from newsclf.serving.export import export_experiment, export_legacy

    cfg = load_config()
    v = (
        export_legacy(cfg, args.label)
        if args.experiment == "E1"
        else export_experiment(cfg, args.experiment, args.label)
    )
    _print({"exported": args.experiment, "version": v, "label": args.label})


def cmd_promote(args):
    from newsclf.serving.registry import Registry

    reg = Registry(load_config())
    version = args.version or reg.version_for("canary")
    if args.label == "stable":
        reg.promote(version)
    else:
        reg.set_label(args.label, version)
    reg.save()
    _print(reg.data["labels"])


def cmd_registry(args):
    from newsclf.serving.registry import Registry

    reg = Registry(load_config())
    _print(
        {
            "labels": reg.data["labels"],
            "versions": {
                v: {k: i.get(k) for k in ("source", "input_type", "test_macro_f1", "registered_at")}
                for v, i in reg.data["versions"].items()
            },
        }
    )


def cmd_serve(args):
    import uvicorn

    from newsclf.serving.local_server import create_app

    cfg = load_config()
    port = args.port or cfg["deployment"]["local_server_port"]
    log.info("Local TF-Serving-compatible server on http://%s:%d (docs at /docs)", args.host, port)
    uvicorn.run(create_app(cfg), host=args.host, port=port, log_level="warning")


def cmd_predict(args):
    from newsclf.pipelines import make_backend
    from newsclf.serving.predictor import TopicClassifier

    registry, backend = make_backend(load_config(), args.url)
    results = TopicClassifier(registry, backend).classify(args.titles, label=args.label)
    for r in results:
        r.pop("probabilities")
    _print(results)


def cmd_canary(args):
    from newsclf.pipelines import canary_check

    r = canary_check(load_config(), url=args.url, promote=args.promote, sample=args.sample)
    _print({"stable_macro_f1": r["stable"]["macro_f1"], "canary_macro_f1": r["canary"]["macro_f1"], **r["decision"]})


def cmd_route(args):
    from newsclf.pipelines import route

    _print(route(load_config(), args.input, url=args.url, label=args.label, out_dir=args.out, limit=args.limit))


def cmd_monitor(args):
    from newsclf.pipelines import monitor

    _print(monitor(load_config(), url=args.url, label=args.label, db=args.db))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="newsclf", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)
    exps = list_experiments()

    sub.add_parser("validate", help="Validate raw data and write a findings report").set_defaults(fn=cmd_validate)

    s = sub.add_parser("prepare", help="Build versioned train/dev/test splits")
    s.add_argument("-e", "--experiment", nargs="*", choices=exps)
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_prepare)

    sub.add_parser("baseline", help="E1: re-measure the inherited prototype").set_defaults(fn=cmd_baseline)

    s = sub.add_parser("train", help="Train one or more experiments")
    s.add_argument("-e", "--experiment", nargs="+", required=True, choices=exps)
    s.add_argument("--epochs", type=int, help="override epochs (smoke tests)")
    s.add_argument("--no-mlflow", action="store_true")
    s.set_defaults(fn=cmd_train)

    sub.add_parser("benchmark", help="B1: TF-IDF + logistic regression benchmark").set_defaults(fn=cmd_benchmark)

    s = sub.add_parser("run-all", help="Baseline + all experiments + benchmark + champion selection")
    s.add_argument("-e", "--experiment", nargs="*", choices=exps)
    s.set_defaults(fn=cmd_run_all)

    sub.add_parser("leaderboard", help="Rank experiments and select the champion").set_defaults(fn=cmd_leaderboard)

    s = sub.add_parser("deploy", help="Serve legacy as stable (v1) and the champion as canary (v2)")
    s.add_argument("--candidate", choices=exps, help="override the champion")
    s.add_argument("--keep", action="store_true", help="do not wipe existing served versions")
    s.set_defaults(fn=cmd_deploy)

    s = sub.add_parser("export", help="Export one experiment as a new served version")
    s.add_argument("-e", "--experiment", required=True, choices=["E1", *exps])
    s.add_argument("--label", choices=["stable", "canary", "deprecated"])
    s.set_defaults(fn=cmd_export)

    s = sub.add_parser("promote", help="Move a version label (default: canary -> stable)")
    s.add_argument("--version", type=int)
    s.add_argument("--label", default="stable", choices=["stable", "canary", "deprecated"])
    s.set_defaults(fn=cmd_promote)

    sub.add_parser("registry", help="Show served versions and labels").set_defaults(fn=cmd_registry)

    s = sub.add_parser("serve", help="Run the local TF-Serving-compatible REST server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int)
    s.set_defaults(fn=cmd_serve)

    for name, fn, helptext in (
        ("predict", cmd_predict, "Classify titles"),
        ("canary-check", cmd_canary, "Compare canary vs stable on held-out data"),
        ("route", cmd_route, "Split titles into auto.csv / manual.csv"),
        ("monitor", cmd_monitor, "Input/output monitoring on the production log"),
    ):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--url", help="serving base URL, e.g. http://localhost:8501 (default: in-process)")
        if name != "canary-check":
            s.add_argument("--label", default="stable")
        s.set_defaults(fn=fn)
        if name == "predict":
            s.add_argument("titles", nargs="+")
        elif name == "canary-check":
            s.add_argument("--promote", action="store_true", help="promote canary if it passes")
            s.add_argument("--sample", type=int, help="evaluate on a random sample of the test split")
        elif name == "route":
            s.add_argument("--input", required=True, help="CSV with a `title` column (optional `topic`)")
            s.add_argument("--out", help="output folder (default reports/routing)")
            s.add_argument("--limit", type=int)
        elif name == "monitor":
            s.add_argument("--db", help="SQLite production log (default data/raw/news_articles.db)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
