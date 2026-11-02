import os
import sys
from pathlib import Path

import pandas as pd
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

TOPICS = ["BUSINESS", "ENTERTAINMENT", "HEALTH", "NATION", "SCIENCE", "SPORTS", "TECHNOLOGY", "WORLD"]
WORDS = {
    "BUSINESS": "stocks profit market shares earnings revenue investors bank",
    "ENTERTAINMENT": "film movie actor singer album celebrity netflix concert",
    "HEALTH": "covid vaccine hospital doctors virus patients disease cases",
    "NATION": "council police state governor local residents town county",
    "SCIENCE": "nasa planet telescope fossil species study researchers space",
    "SPORTS": "football match league coach team season goal player",
    "TECHNOLOGY": "apple iphone app software google chip android update",
    "WORLD": "china minister president border protest election foreign un",
}


@pytest.fixture
def synthetic_df():
    """Small, learnable dataset in the raw schema, with a collapsed BUSINESS class."""
    import random

    rng = random.Random(0)
    rows = []
    for topic in TOPICS:
        vocab = WORDS[topic].split()
        n = 40 if topic == "SCIENCE" else 120
        for i in range(n):
            words = rng.sample(vocab, 5) + [f"w{i}"]
            slug = "-".join(words)
            title = " ".join(words).title()
            if topic == "BUSINESS":
                title = "Why iPhone 12 Will Be Another 'Defining Chapter' In Apple's Growth Story"
            rows.append(
                {
                    "title": title,
                    "link": f"https://news.example.com/{topic.lower()}/{slug}-1234567",
                    "domain": "news.example.com",
                    "published_date": "2020-08-01 10:00:00",
                    "topic": topic,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def tmp_project(tmp_path, synthetic_df):
    """Config pointing every path at a temporary folder with synthetic raw data."""
    from newsclf.config import load_config

    raw = tmp_path / "raw"
    raw.mkdir()
    half = len(synthetic_df) // 2
    shuffled = synthetic_df.sample(frac=1, random_state=1)
    shuffled.iloc[:half].to_csv(raw / "train_data.csv", index=False)
    shuffled.iloc[half:].to_csv(raw / "test_data.csv", index=False)
    cfg = load_config()
    cfg["paths"].update(
        {
            "raw_dir": str(raw),
            "processed_dir": str(tmp_path / "processed"),
            "artifacts_dir": str(tmp_path / "artifacts"),
            "reports_dir": str(tmp_path / "reports"),
            "serving_root": str(tmp_path / "serving"),
            "serving_registry": str(tmp_path / "serving" / "registry.json"),
        }
    )
    cfg["tracking"]["csv"] = str(tmp_path / "reports" / "tracker.csv")
    cfg["tracking"]["mlflow"]["enabled"] = False
    cfg["name"] = "T1"
    cfg["features"].update({"max_tokens": 500, "sequence_length": 8})
    cfg["model"].update({"architecture": "pooled", "embedding_dim": 16, "dense_dim": 16})
    cfg["training"].update({"epochs": 15, "batch_size": 16})
    return cfg
