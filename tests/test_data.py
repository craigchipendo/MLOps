import pytest

from newsclf.data.augment import EDAAugmenter, augment_frame
from newsclf.data.prepare import deduplicate, load_manifest, load_splits, prepare_dataset, stratified_split
from newsclf.data.validate import DataValidationError, normalize_title, validate_frame


def test_validate_flags_problems(synthetic_df):
    dup = synthetic_df.copy()
    dup.loc[0, "title"] = dup.loc[200, "title"]
    report = validate_frame(dup)
    assert "BUSINESS" in report["collapsed_classes"]
    assert report["duplicate_titles"] > 0
    assert any("Collapsed" in w for w in report["warnings"])


def test_validate_rejects_unknown_topics(synthetic_df):
    bad = synthetic_df.copy()
    bad.loc[0, "topic"] = "GOSSIP"
    with pytest.raises(DataValidationError):
        validate_frame(bad)


def test_normalize_title():
    assert normalize_title("  Apple's  iPhone-12: BIG!! ") == "apple s iphone 12 big"


def test_deduplicate_majority_label(synthetic_df):
    df = synthetic_df.iloc[200:203].copy()
    df["title"] = "Same Title"
    df["topic"] = ["HEALTH", "HEALTH", "NATION"]
    out, report = deduplicate(df)
    assert len(out) == 1 and out.iloc[0]["topic"] == "HEALTH"
    assert report["conflicting_titles_resolved"] == 1


def test_stratified_split_proportions(synthetic_df):
    splits = stratified_split(synthetic_df, {"train": 0.6, "dev": 0.2, "test": 0.2}, seed=0)
    sizes = {k: len(v) for k, v in splits.items()}
    assert abs(sizes["train"] / len(synthetic_df) - 0.6) < 0.02
    share = {k: (v.topic == "SCIENCE").mean() for k, v in splits.items()}
    assert max(share.values()) - min(share.values()) < 0.02


def test_prepare_dataset_is_leak_free_and_versioned(tmp_project):
    d1 = prepare_dataset(tmp_project)
    manifest = load_manifest(d1)
    assert set(manifest["split_validation"]["title_overlap"].values()) == {0}
    assert manifest["repair"]["collapsed_classes"] == ["BUSINESS"]
    assert prepare_dataset(tmp_project) == d1  # cached
    tmp_project["data"]["deduplicate"] = False
    assert prepare_dataset(tmp_project) != d1  # new data config -> new version
    assert set(load_splits(d1)) == {"train", "dev", "test"}


def test_eda_fallback_ops_and_synonyms():
    aug = EDAAugmenter(alpha=0.3, synonyms=None, seed=1)
    assert [op.__name__ for op in aug.ops] == ["random_swap", "random_deletion"]
    out = aug.augment("nasa telescope finds new planet orbiting star", 4)
    assert out and all(isinstance(t, str) for t in out)
    aug = EDAAugmenter(alpha=0.3, synonyms=lambda w: ["scope"] if w == "telescope" else [], seed=1)
    assert "scope" in " ".join(aug.synonym_replacement(["nasa", "telescope", "finds", "planet"]))


def test_augment_frame_only_targets_class(synthetic_df):
    out = augment_frame(synthetic_df, ["SCIENCE"], n_per_sample=2, alpha=0.2, seed=0)
    added = out[out.is_augmented]
    assert len(added) > 0 and set(added.topic) == {"SCIENCE"}
