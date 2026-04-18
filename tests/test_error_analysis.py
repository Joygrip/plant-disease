"""
Tests for error_analysis.py.

Uses a 5-class model with synthetic data — no real dataset, no real weights.
Images are 32×32 instead of 224×224 for speed.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plant_disease.error_analysis import (
    analyze_confused_pairs,
    run_error_analysis,
    save_confidence_histogram,
    save_per_species_rollup,
    _same_species,
    _species,
)

# ---------------------------------------------------------------------------
# Fake setup
# ---------------------------------------------------------------------------

FAKE_CLASS_NAMES = [
    "PlantA___disease1",
    "PlantA___disease2",
    "PlantB___disease1",
    "PlantB___disease2",
    "PlantC___healthy",
]
N_CLASSES = len(FAKE_CLASS_NAMES)
IMG_SIZE  = 32
N_SAMPLES = 100


class TinyModel(nn.Module):
    """Returns fixed logits based on the input sum — deterministic but not perfect."""

    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * IMG_SIZE * IMG_SIZE, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


class FakeDataset(Dataset):
    """N_SAMPLES tiny images with cycling labels, stored as real files."""

    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.samples: list[tuple[Path, int]] = []
        for i in range(N_SAMPLES):
            label = i % N_CLASSES
            img_path = root / f"img_{i:04d}.png"
            if not img_path.exists():
                Image.fromarray(
                    np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                ).save(img_path)
            self.samples.append((img_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from torchvision import transforms
        img = Image.open(self.samples[idx][0]).convert("RGB")
        tensor = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])(img)
        return tensor, self.samples[idx][1]


# ---------------------------------------------------------------------------
# Species helpers
# ---------------------------------------------------------------------------

def test_species_extraction() -> None:
    assert _species("PlantA___disease1") == "PlantA"
    assert _species("PlantA___healthy")  == "PlantA"
    assert _species("PlantC___healthy")  == "PlantC"


def test_same_species_true() -> None:
    assert _same_species("PlantA___disease1", "PlantA___disease2") is True


def test_same_species_false() -> None:
    assert _same_species("PlantA___disease1", "PlantB___disease1") is False


# ---------------------------------------------------------------------------
# Confusion-pair analysis
# ---------------------------------------------------------------------------

def test_analyze_confused_pairs_basic() -> None:
    # Labels: 0,0,0,0; Preds: 0,1,0,0 → one A→B confusion
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    preds  = [0, 1, 0, 0, 0, 1, 1, 1]
    pairs = analyze_confused_pairs(labels, preds, FAKE_CLASS_NAMES, top_k=5)
    assert len(pairs) >= 1
    # Top pair should be 0↔1 with total=2
    assert pairs[0]["total"] >= 1


def test_analyze_confused_pairs_same_species_flag() -> None:
    # Classes 0 and 1 are both PlantA — confusion should be flagged same_species
    labels = [0, 1]
    preds  = [1, 0]
    pairs = analyze_confused_pairs(labels, preds, FAKE_CLASS_NAMES, top_k=5)
    assert len(pairs) == 1
    assert pairs[0]["same_species"] is True


def test_analyze_confused_pairs_cross_species_flag() -> None:
    # Classes 0 (PlantA) and 2 (PlantB) — cross-species
    labels = [0, 2]
    preds  = [2, 0]
    pairs = analyze_confused_pairs(labels, preds, FAKE_CLASS_NAMES, top_k=5)
    assert len(pairs) == 1
    assert pairs[0]["same_species"] is False


def test_symmetric_flag_true() -> None:
    # 5 A→B and 4 B→A → ratio 4/5 = 0.8 → symmetric
    labels = [0]*5 + [1]*4
    preds  = [1]*5 + [0]*4
    pairs = analyze_confused_pairs(labels, preds, FAKE_CLASS_NAMES, top_k=5)
    assert pairs[0]["symmetric"] is True


def test_symmetric_flag_false() -> None:
    # 10 A→B but 0 B→A → ratio 0/10 = 0 → asymmetric
    labels = [0] * 10
    preds  = [1] * 10
    pairs = analyze_confused_pairs(labels, preds, FAKE_CLASS_NAMES, top_k=5)
    assert pairs[0]["symmetric"] is False


def test_no_off_diagonal_no_pairs() -> None:
    # Perfect predictions → no confused pairs
    labels = list(range(N_CLASSES))
    preds  = list(range(N_CLASSES))
    pairs = analyze_confused_pairs(labels, preds, FAKE_CLASS_NAMES, top_k=5)
    assert pairs == []


# ---------------------------------------------------------------------------
# Confidence histogram edge cases
# ---------------------------------------------------------------------------

def test_confidence_histogram_all_correct(tmp_path) -> None:
    labels = [0, 1, 2, 3, 4]
    preds  = [0, 1, 2, 3, 4]
    confs  = [0.9, 0.8, 0.7, 0.95, 0.6]
    stats = save_confidence_histogram(labels, preds, confs, tmp_path)
    assert stats["mean_confidence_correct"] > 0
    assert stats["mean_confidence_wrong"] == 0.0
    assert stats["n_wrong_high_confidence"] == 0


def test_confidence_histogram_all_wrong(tmp_path) -> None:
    labels = [0, 1, 2, 3, 4]
    preds  = [1, 2, 3, 4, 0]
    confs  = [0.95, 0.85, 0.92, 0.3, 0.4]
    stats = save_confidence_histogram(labels, preds, confs, tmp_path)
    assert stats["mean_confidence_correct"] == 0.0
    assert stats["mean_confidence_wrong"] > 0
    assert stats["n_wrong_high_confidence"] == 2  # 0.95 and 0.92 exceed 0.9; 0.85 does not
    assert (tmp_path / "confidence_histogram.png").exists()


def test_confidence_histogram_creates_file(tmp_path) -> None:
    labels = [0, 1, 0, 1]
    preds  = [0, 0, 1, 1]
    confs  = [0.9, 0.5, 0.4, 0.8]
    save_confidence_histogram(labels, preds, confs, tmp_path)
    assert (tmp_path / "confidence_histogram.png").exists()


# ---------------------------------------------------------------------------
# Per-species rollup
# ---------------------------------------------------------------------------

def test_per_species_rollup_creates_csv(tmp_path) -> None:
    labels = [0, 0, 1, 1, 2, 3, 4]
    preds  = [0, 1, 1, 0, 2, 3, 4]
    rows = save_per_species_rollup(labels, preds, FAKE_CLASS_NAMES, tmp_path)
    assert (tmp_path / "per_species.csv").exists()
    assert len(rows) == 3  # PlantA, PlantB, PlantC


def test_per_species_rollup_species_accuracy() -> None:
    # PlantA (classes 0,1): label=0, pred=1 → species correct (both PlantA)
    with tempfile.TemporaryDirectory() as tmpdir:
        labels = [0, 1]   # both PlantA
        preds  = [1, 0]   # both still predicted as PlantA variants
        rows = save_per_species_rollup(labels, preds, FAKE_CLASS_NAMES, Path(tmpdir))
        plant_a = next(r for r in rows if r["species"] == "PlantA")
        assert float(plant_a["species_accuracy"]) == 1.0
        # Disease accuracy: 0/2 (both wrong disease within PlantA)
        assert float(plant_a["disease_accuracy"]) == 0.0


def test_per_species_rollup_cross_species_penalty() -> None:
    # label=0 (PlantA), pred=2 (PlantB) → species incorrect for PlantA
    with tempfile.TemporaryDirectory() as tmpdir:
        labels = [0]
        preds  = [2]
        rows = save_per_species_rollup(labels, preds, FAKE_CLASS_NAMES, Path(tmpdir))
        plant_a = next(r for r in rows if r["species"] == "PlantA")
        assert float(plant_a["species_accuracy"]) == 0.0


# ---------------------------------------------------------------------------
# Full pipeline smoke test (tiny model, fake data, no GPU needed)
# ---------------------------------------------------------------------------

def test_run_error_analysis_smoke(tmp_path) -> None:
    data_dir = tmp_path / "images"
    ds = FakeDataset(data_dir)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    model = TinyModel(n_classes=N_CLASSES).eval()

    summary = run_error_analysis(
        model=model,
        test_loader=loader,
        class_names=FAKE_CLASS_NAMES,
        output_dir=tmp_path / "outputs",
        model_name="test_model",
    )

    assert "overall_accuracy" in summary
    assert 0.0 <= summary["overall_accuracy"] <= 1.0
    assert isinstance(summary["top_confused_pairs"], list)
    assert (tmp_path / "outputs" / "confusion_pairs.json").exists()
    assert (tmp_path / "outputs" / "confidence_histogram.png").exists()
    assert (tmp_path / "outputs" / "per_species.csv").exists()
    assert (tmp_path / "outputs" / "summary.json").exists()
