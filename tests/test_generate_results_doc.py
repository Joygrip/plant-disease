"""
Smoke tests for generate_results_doc.py.

Uses two dummy run_info.json files and pre-built artifact stubs.
No real checkpoints, no dataset required.
"""

import csv
import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import generate_results_doc as grd


# ---------------------------------------------------------------------------
# Fixture: stub artifact tree
# ---------------------------------------------------------------------------

@pytest.fixture()
def artifact_tree(tmp_path: Path, monkeypatch):
    """
    Build a minimal set of stub artifacts in tmp_path and monkey-patch
    generate_results_doc module to point at them.
    """
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    eval_baseline   = tmp_path / "outputs" / "eval" / "baseline"
    eval_mobilenet  = tmp_path / "outputs" / "eval" / "mobilenet_v2"
    ea_mobilenet    = reports_dir / "error_analysis" / "mobilenet_v2"

    for d in [models_dir, reports_dir, eval_baseline, eval_mobilenet, ea_mobilenet,
              reports_dir / "training_curves"]:
        d.mkdir(parents=True, exist_ok=True)

    # Meta files
    for model_name in ["baseline", "mobilenet_v2"]:
        (models_dir / f"{model_name}_best_meta.json").write_text(json.dumps({
            "model": model_name,
            "epoch": 15,
            "best_stage": "2" if model_name == "mobilenet_v2" else "1",
            "val_acc": 0.97 if model_name == "mobilenet_v2" else 0.94,
        }))
        (models_dir / f"{model_name}_run_info.json").write_text(json.dumps({
            "model": model_name,
            "gpu_name": "NVIDIA RTX 4090",
            "torch_version": "2.3.0",
            "python_version": "3.13.0",
            "total_training_time_s": 3600.0,
            "mean_epoch_time_s": 120.0,
            "epochs_run": 30,
            "best_epoch": 15,
            "best_val_acc": 0.97,
            "git_commit": "abc1234",
        }))

    # Eval summaries
    for model_name, acc in [("baseline", 0.94), ("mobilenet_v2", 0.97)]:
        eval_dir = eval_baseline if model_name == "baseline" else eval_mobilenet
        (eval_dir / "summary.json").write_text(json.dumps({
            "test_accuracy": acc,
            "checkpoint": f"models/{model_name}_best.pt",
        }))
        (eval_dir / "classification_report.txt").write_text(
            "              precision    recall  f1-score   support\n"
            "      class0       0.95      0.93      0.94       100\n"
        )

    # Error analysis stubs
    (ea_mobilenet / "summary.json").write_text(json.dumps({
        "model": "mobilenet_v2",
        "overall_accuracy": 0.97,
        "top_confused_pairs": [
            {
                "true_class": "Tomato___Early_blight",
                "pred_class": "Tomato___Late_blight",
                "count_ab": 5, "count_ba": 3, "total": 8,
                "symmetric": True, "same_species": True,
            }
        ],
        "same_species_confusion_pairs": 1,
        "cross_species_confusion_pairs": 0,
        "confidence_stats": {
            "mean_confidence_correct": 0.95,
            "mean_confidence_wrong": 0.6,
            "n_wrong_high_confidence": 2,
            "frac_wrong_high_confidence": 0.05,
        },
        "per_species_rows": [],
    }))

    # Per-species CSV
    per_species_path = ea_mobilenet / "per_species.csv"
    with open(per_species_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["species", "num_classes", "num_test_images",
                           "species_accuracy", "disease_accuracy"]
        )
        writer.writeheader()
        writer.writerow({
            "species": "Tomato", "num_classes": 10, "num_test_images": 500,
            "species_accuracy": 0.99, "disease_accuracy": 0.97,
        })

    # Splits file (for dataset stats)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "splits.json").write_text(json.dumps({
        "val": [f"ClassA/img_{i}.jpg" for i in range(100)],
        "test": [f"ClassA/img_{i}.jpg" for i in range(100, 200)],
        "seed": 42,
    }))

    # Dummy checkpoint files (content doesn't matter for generate_results_doc)
    (tmp_path / "baseline_best.pt").write_bytes(b"dummy")
    (tmp_path / "mobilenet_v2_best.pt").write_bytes(b"dummy")

    # Monkey-patch module globals
    monkeypatch.setattr(grd, "ROOT",        tmp_path)
    monkeypatch.setattr(grd, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(grd, "MODELS_DIR",  models_dir)

    return {
        "baseline_ckpt":   tmp_path / "baseline_best.pt",
        "mobilenet_ckpt":  tmp_path / "mobilenet_v2_best.pt",
        "baseline_eval":   eval_baseline,
        "mobilenet_eval":  eval_mobilenet,
        "output":          reports_dir / "ml_results.md",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_generate_runs_without_error(artifact_tree) -> None:
    doc = grd.generate_results_doc(
        baseline_ckpt=artifact_tree["baseline_ckpt"],
        mobilenet_ckpt=artifact_tree["mobilenet_ckpt"],
        baseline_eval_dir=artifact_tree["baseline_eval"],
        mobilenet_eval_dir=artifact_tree["mobilenet_eval"],
    )
    assert isinstance(doc, str)
    assert len(doc) > 100


def test_generate_contains_key_sections(artifact_tree) -> None:
    doc = grd.generate_results_doc(
        baseline_ckpt=artifact_tree["baseline_ckpt"],
        mobilenet_ckpt=artifact_tree["mobilenet_ckpt"],
        baseline_eval_dir=artifact_tree["baseline_eval"],
        mobilenet_eval_dir=artifact_tree["mobilenet_eval"],
    )
    assert "## Experimental Setup"  in doc
    assert "## Baseline CNN"         in doc
    assert "## MobileNetV2"          in doc
    assert "## Error Analysis"       in doc
    assert "## Limitations"          in doc


def test_generate_fills_gpu_from_run_info(artifact_tree) -> None:
    doc = grd.generate_results_doc(
        baseline_ckpt=artifact_tree["baseline_ckpt"],
        mobilenet_ckpt=artifact_tree["mobilenet_ckpt"],
        baseline_eval_dir=artifact_tree["baseline_eval"],
        mobilenet_eval_dir=artifact_tree["mobilenet_eval"],
    )
    assert "NVIDIA RTX 4090" in doc


def test_generate_fills_accuracy(artifact_tree) -> None:
    doc = grd.generate_results_doc(
        baseline_ckpt=artifact_tree["baseline_ckpt"],
        mobilenet_ckpt=artifact_tree["mobilenet_ckpt"],
        baseline_eval_dir=artifact_tree["baseline_eval"],
        mobilenet_eval_dir=artifact_tree["mobilenet_eval"],
    )
    assert "0.9400" in doc or "0.94" in doc
    assert "0.9700" in doc or "0.97" in doc


def test_generate_handles_missing_artifacts_gracefully(tmp_path, monkeypatch) -> None:
    """When artifacts are absent, TBD placeholders should appear, not exceptions."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.setattr(grd, "ROOT",        tmp_path)
    monkeypatch.setattr(grd, "REPORTS_DIR", empty_dir)
    monkeypatch.setattr(grd, "MODELS_DIR",  empty_dir)

    doc = grd.generate_results_doc(
        baseline_ckpt=tmp_path / "nonexistent_baseline.pt",
        mobilenet_ckpt=tmp_path / "nonexistent_mobilenet.pt",
        baseline_eval_dir=empty_dir,
        mobilenet_eval_dir=empty_dir,
    )
    # Should not raise; should contain TBD markers
    assert grd.TBD in doc


def test_generate_confused_pairs_table(artifact_tree) -> None:
    doc = grd.generate_results_doc(
        baseline_ckpt=artifact_tree["baseline_ckpt"],
        mobilenet_ckpt=artifact_tree["mobilenet_ckpt"],
        baseline_eval_dir=artifact_tree["baseline_eval"],
        mobilenet_eval_dir=artifact_tree["mobilenet_eval"],
    )
    assert "Tomato___Early_blight" in doc
    assert "Tomato___Late_blight"  in doc


def test_generate_species_rollup_table(artifact_tree) -> None:
    doc = grd.generate_results_doc(
        baseline_ckpt=artifact_tree["baseline_ckpt"],
        mobilenet_ckpt=artifact_tree["mobilenet_ckpt"],
        baseline_eval_dir=artifact_tree["baseline_eval"],
        mobilenet_eval_dir=artifact_tree["mobilenet_eval"],
    )
    assert "Tomato" in doc
    assert "species_accuracy" not in doc  # column name shouldn't appear raw
    assert "0.99" in doc  # species_accuracy value should appear


def test_generate_idempotent(artifact_tree) -> None:
    """Running twice produces the same content."""
    kwargs = dict(
        baseline_ckpt=artifact_tree["baseline_ckpt"],
        mobilenet_ckpt=artifact_tree["mobilenet_ckpt"],
        baseline_eval_dir=artifact_tree["baseline_eval"],
        mobilenet_eval_dir=artifact_tree["mobilenet_eval"],
    )
    doc1 = grd.generate_results_doc(**kwargs)
    doc2 = grd.generate_results_doc(**kwargs)
    assert doc1 == doc2


def test_dataset_stats_reads_splits_json(artifact_tree, monkeypatch) -> None:
    stats = grd._dataset_stats()
    assert stats["val"] == 100
    assert stats["test"] == 100
