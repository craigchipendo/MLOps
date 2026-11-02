"""Raw data ingestion."""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import numpy as np
import pandas as pd

from newsclf.config import resolve

REQUIRED_COLUMNS = ["title", "link", "domain", "published_date", "topic"]


def load_raw(cfg: dict) -> pd.DataFrame:
    """Load and concatenate the raw CSV exports listed in the config.

    A `source_file` column is kept so every row can be traced to its origin.
    """
    raw_dir = resolve(cfg["paths"]["raw_dir"])
    frames = []
    for name in cfg["data"]["raw_files"]:
        df = pd.read_csv(raw_dir / name)
        df["source_file"] = name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_legacy_split(cfg: dict, name: str) -> pd.DataFrame:
    return pd.read_csv(resolve(cfg["paths"]["raw_dir"]) / name)


def load_reference_titles(path: str | Path) -> pd.DataFrame:
    """Load a trusted `link -> title` mapping (backup CSV or Kaggle export).

    The Kaggle "Topic Labeled News" export is semicolon separated, the course
    backup is comma separated, so the separator is sniffed.
    """
    df = pd.read_csv(path, sep=None, engine="python")
    missing = {"link", "title"} - set(df.columns)
    if missing:
        raise ValueError(f"Reference file {path} is missing columns: {missing}")
    return df[["link", "title"]].dropna().drop_duplicates("link")


def load_production_db(db_path: str | Path, n_classes: int = 8) -> pd.DataFrame:
    """Load the production log of scored articles.

    `class_1..class_N` hold the softmax scores the live model returned, stored
    as little-endian float32 blobs. They are decoded into a float matrix so the
    output distribution can be monitored.
    """
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql_query("SELECT * FROM news_articles ORDER BY id", con)
    score_cols = [f"class_{i}" for i in range(1, n_classes + 1) if f"class_{i}" in df.columns]
    for col in score_cols:
        df[col] = df[col].map(_decode_float32)
    return df


def _decode_float32(value) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (bytes, bytearray, memoryview)) and len(value) == 4:
        return struct.unpack("<f", bytes(value))[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan
