"""
Smoke tests for the data pipeline.

All tests run without the actual dataset — they either mock the filesystem
or use tiny synthetic images. Real dataset tests are skipped automatically
when data/ is empty.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plant_disease import config
from plant_disease.data import (
    PlantDiseaseDataset,
    _eval_transform,
    _train_transform,
    get_class_to_idx,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_image(path: Path, size: tuple[int, int] = (256, 256)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=(120, 80, 60))
    img.save(path)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def test_data_dir_resolves_to_repo_root() -> None:
    data_dir = config.ROOT / "data"
    assert data_dir.name == "data"
    assert (data_dir.parent / "pyproject.toml").exists(), (
        f"ROOT ({config.ROOT}) does not contain pyproject.toml — path is misconfigured"
    )


def test_class_names_count() -> None:
    assert len(config.CLASS_NAMES) == 38


def test_class_names_sorted() -> None:
    assert config.CLASS_NAMES == sorted(config.CLASS_NAMES)


def test_class_to_idx_stable() -> None:
    mapping = get_class_to_idx()
    assert len(mapping) == 38
    # Indices must be 0..37
    assert set(mapping.values()) == set(range(38))


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def test_train_transform_output_shape() -> None:
    tf = _train_transform()
    img = Image.new("RGB", (256, 256))
    tensor = tf(img)
    assert tensor.shape == (3, config.IMAGE_SIZE, config.IMAGE_SIZE)


def test_eval_transform_output_shape() -> None:
    tf = _eval_transform()
    img = Image.new("RGB", (300, 200))  # non-square input
    tensor = tf(img)
    assert tensor.shape == (3, config.IMAGE_SIZE, config.IMAGE_SIZE)


def test_eval_transform_is_deterministic() -> None:
    tf = _eval_transform()
    img = Image.new("RGB", (256, 256), color=(10, 20, 30))
    t1 = tf(img)
    t2 = tf(img)
    assert torch.allclose(t1, t2)


# ---------------------------------------------------------------------------
# PlantDiseaseDataset
# ---------------------------------------------------------------------------

def test_dataset_len_and_item() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        img_path = tmp / "img.jpg"
        _make_fake_image(img_path)

        tf = _eval_transform()
        ds = PlantDiseaseDataset([(img_path, 5)], tf)
        assert len(ds) == 1
        tensor, label = ds[0]
        assert label == 5
        assert tensor.shape == (3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        assert tensor.dtype == torch.float32


def test_dataset_multiple_items() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        samples = []
        for i in range(5):
            p = tmp / f"img_{i}.jpg"
            _make_fake_image(p)
            samples.append((p, i % 38))

        ds = PlantDiseaseDataset(samples, _eval_transform())
        assert len(ds) == 5
        for i, (tensor, label) in enumerate(ds):
            assert label == i % 38
            assert tensor.shape == (3, config.IMAGE_SIZE, config.IMAGE_SIZE)


# ---------------------------------------------------------------------------
# prepare_splits logic (unit-tested without disk I/O)
# ---------------------------------------------------------------------------

def test_splits_file_structure() -> None:
    """splits.json must have val/test keys with lists of strings."""
    if not config.SPLITS_FILE.exists():
        pytest.skip("splits.json not found — run prepare_splits.py first")

    with open(config.SPLITS_FILE) as f:
        splits = json.load(f)

    assert "val" in splits
    assert "test" in splits
    assert isinstance(splits["val"], list)
    assert isinstance(splits["test"], list)
    assert len(splits["val"]) > 0
    assert len(splits["test"]) > 0

    # No overlap between val and test
    val_set = set(splits["val"])
    test_set = set(splits["test"])
    assert val_set.isdisjoint(test_set), "val and test share images!"


def test_splits_class_balance() -> None:
    """Each class should appear roughly equally in val and test."""
    if not config.SPLITS_FILE.exists():
        pytest.skip("splits.json not found — run prepare_splits.py first")

    with open(config.SPLITS_FILE) as f:
        splits = json.load(f)

    from collections import Counter
    val_counts = Counter(Path(p).parent.name for p in splits["val"])
    test_counts = Counter(Path(p).parent.name for p in splits["test"])

    for cls in config.CLASS_NAMES:
        v = val_counts.get(cls, 0)
        t = test_counts.get(cls, 0)
        # 50/50 split allows at most 1 image difference per class
        assert abs(v - t) <= 1, f"Class {cls}: val={v}, test={t} — imbalanced split"


# ---------------------------------------------------------------------------
# Integration: real dataset (skipped when absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not config.TRAIN_DIR.exists(),
    reason="Dataset not present — skipping integration test",
)
def test_real_dataset_sample() -> None:
    class_to_idx = get_class_to_idx()
    first_class = config.CLASS_NAMES[0]
    class_dir = config.TRAIN_DIR / first_class
    images = list(class_dir.iterdir())
    assert len(images) > 0

    ds = PlantDiseaseDataset(
        [(images[0], class_to_idx[first_class])],
        _eval_transform(),
    )
    tensor, label = ds[0]
    assert tensor.shape == (3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    assert label == 0
