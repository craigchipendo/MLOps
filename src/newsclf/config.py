"""Configuration loading.

Every run is driven by `configs/base.yaml` deep-merged with one experiment
override file. The resolved config is written next to the run's artifacts so
any result can be traced back to the exact settings that produced it.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
EXPERIMENTS_DIR = CONFIG_DIR / "experiments"


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base`."""
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def find_experiment_file(name: str) -> Path:
    """Resolve `E4` or `E4_improved_model` or a path to the experiment yaml."""
    candidate = Path(name)
    if candidate.suffix in {".yaml", ".yml"} and candidate.exists():
        return candidate
    matches = sorted(EXPERIMENTS_DIR.glob(f"{name}*.yaml"))
    exact = [m for m in matches if m.stem == name or m.stem.split("_")[0] == name]
    if exact:
        return exact[0]
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No experiment config matching '{name}' in {EXPERIMENTS_DIR}")


def load_config(experiment: str | None = None, base_path: str | Path | None = None) -> dict:
    """Load base config, optionally merged with an experiment override."""
    cfg = load_yaml(base_path or CONFIG_DIR / "base.yaml")
    cfg.setdefault("name", "base")
    cfg.setdefault("description", "")
    cfg.setdefault("promotable", True)
    if experiment:
        cfg = deep_merge(cfg, load_yaml(find_experiment_file(experiment)))
    return cfg


def list_experiments() -> list[str]:
    return [p.stem.split("_")[0] for p in sorted(EXPERIMENTS_DIR.glob("E*.yaml"))]


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def relpath(path: str | Path) -> str:
    """Path relative to the project root when possible (portable across machines)."""
    p = Path(path).resolve()
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def config_hash(section: Any) -> str:
    """Stable short hash of a config section (used to version datasets)."""
    payload = json.dumps(section, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:10]


def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
