"""Dataset preparation stage: ingest -> validate -> repair -> dedupe -> split.

Outputs are versioned by a hash of the `data` config section, so two
experiments that share data settings share the exact same split files, and a
change in data settings can never silently overwrite an earlier dataset.

    data/processed/<data_hash>/{train,dev,test}.csv
    data/processed/<data_hash>/manifest.json
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from newsclf.config import config_hash, relpath, resolve
from newsclf.data.ingest import load_raw, load_reference_titles
from newsclf.data.repair import repair_collapsed_titles
from newsclf.data.validate import normalize_title, validate_frame, validate_splits
from newsclf.utils import file_sha256, get_logger, read_json, write_json

log = get_logger(__name__)
SPLITS = ("train", "dev", "test")


def data_version(cfg: dict) -> str:
    keys = [
        "raw_files",
        "reference_titles_file",
        "repair_collapsed_titles",
        "collapse_unique_ratio",
        "deduplicate",
        "min_title_tokens",
        "split",
        "label_map",
    ]
    section = {k: cfg["data"].get(k) for k in keys}
    section["seed"] = cfg["seed"]
    return config_hash(section)


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep one row per normalized title; the label is the majority vote."""
    df = df.assign(_norm=df["title"].map(normalize_title))
    counts = df.groupby(["_norm", "topic"]).size().reset_index(name="n")
    counts = counts.sort_values(["_norm", "n", "topic"], ascending=[True, False, True])
    majority = counts.drop_duplicates("_norm").set_index("_norm")["topic"]
    conflicts = int((counts.groupby("_norm").size() > 1).sum())
    before = len(df)
    df = df.drop_duplicates("_norm").copy()
    df["topic"] = df["_norm"].map(majority)
    report = {"rows_before": before, "rows_after": len(df), "conflicting_titles_resolved": conflicts}
    log.info("De-duplicated %d -> %d rows (%d label conflicts resolved by majority)", before, len(df), conflicts)
    return df.drop(columns="_norm"), report


def stratified_split(df: pd.DataFrame, fractions: dict, seed: int, stratify: bool = True) -> dict[str, pd.DataFrame]:
    test_size = fractions["test"]
    dev_size = fractions["dev"] / (fractions["train"] + fractions["dev"])
    strat = df["topic"] if stratify else None
    train_dev, test = train_test_split(df, test_size=test_size, stratify=strat, random_state=seed)
    strat = train_dev["topic"] if stratify else None
    train, dev = train_test_split(train_dev, test_size=dev_size, stratify=strat, random_state=seed)
    return {"train": train, "dev": dev, "test": test}


def prepare_dataset(cfg: dict, force: bool = False) -> Path:
    """Build (or reuse) the processed dataset for this config. Returns its folder."""
    version = data_version(cfg)
    out_dir = resolve(cfg["paths"]["processed_dir"]) / version
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not force:
        log.info("Reusing processed dataset %s", relpath(out_dir))
        return out_dir

    dcfg = cfg["data"]
    raw = load_raw(cfg)
    raw_report = validate_frame(raw, collapse_ratio=dcfg["collapse_unique_ratio"])
    for w in raw_report["warnings"]:
        log.warning("raw data: %s", w)

    df = raw
    repair_report = None
    if dcfg["repair_collapsed_titles"]:
        reference = None
        if dcfg.get("reference_titles_file"):
            ref_path = resolve(dcfg["reference_titles_file"])
            if ref_path.exists():
                reference = load_reference_titles(ref_path)
            else:
                log.warning("reference_titles_file %s not found, using URL slugs", ref_path)
        df, repair_report = repair_collapsed_titles(df, dcfg["collapse_unique_ratio"], reference)

    dedup_report = None
    if dcfg["deduplicate"]:
        df, dedup_report = deduplicate(df)

    n_tokens = df["title"].map(lambda t: len(normalize_title(t).split()))
    short = int((n_tokens < dcfg["min_title_tokens"]).sum())
    df = df[n_tokens >= dcfg["min_title_tokens"]]

    if dcfg.get("label_map"):
        df = df.assign(topic=df["topic"].replace(dcfg["label_map"]))

    splits = stratified_split(df, dcfg["split"], cfg["seed"], dcfg["split"]["stratify"])
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in SPLITS:
        path = out_dir / f"{name}.csv"
        splits[name].to_csv(path, index=False)
        files[name] = {"rows": len(splits[name]), "sha256": file_sha256(path)}

    labels = sorted(df["topic"].unique())
    (out_dir / "labels.txt").write_text("\n".join(labels), encoding="utf-8")

    manifest = {
        "data_version": version,
        "data_config": {k: dcfg[k] for k in dcfg},
        "seed": cfg["seed"],
        "labels": labels,
        "files": files,
        "raw_validation": raw_report,
        "repair": repair_report,
        "deduplication": dedup_report,
        "dropped_short_titles": short,
        "processed_validation": validate_frame(
            df, expected_topics=set(labels), collapse_ratio=dcfg["collapse_unique_ratio"]
        ),
        "split_validation": validate_splits(splits),
    }
    write_json(manifest, manifest_path)
    log.info("Wrote dataset %s: %s", relpath(out_dir), {k: v["rows"] for k, v in files.items()})
    return out_dir


def load_splits(data_dir: Path) -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(Path(data_dir) / f"{name}.csv") for name in SPLITS}


def load_manifest(data_dir: Path) -> dict:
    return read_json(Path(data_dir) / "manifest.json")
