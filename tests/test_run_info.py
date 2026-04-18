"""
Tests for _write_run_info in train.py.

Verifies that Path objects inside the args namespace are serialised as strings
(not left as WindowsPath / PosixPath) so the resulting JSON is valid.
"""

import argparse
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plant_disease.train import _write_run_info


def _make_args(output_dir: Path, config_path: Path | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        config=config_path,
        model="baseline",
        epochs=30,
        batch_size=64,
        lr=1e-3,
        weight_decay=1e-4,
        num_workers=4,
        output_dir=output_dir,
        seed=42,
        no_aug=False,
        smoke_test=False,
        finetune_schedule="single-stage",
        stage1_epochs=5,
        stage2_lr=1e-4,
        unfreeze_blocks=3,
    )


# ---------------------------------------------------------------------------
# Basic contract: file is written and parses as valid JSON
# ---------------------------------------------------------------------------

def test_write_run_info_creates_file(tmp_path):
    _write_run_info(
        output_dir=tmp_path,
        model_name="baseline",
        args=_make_args(tmp_path),
        device=torch.device("cpu"),
        total_time_s=100.0,
        epochs_run=10,
        best_epoch=5,
        best_val_acc=0.95,
    )
    assert (tmp_path / "baseline_run_info.json").exists()


def test_write_run_info_is_valid_json(tmp_path):
    _write_run_info(
        output_dir=tmp_path,
        model_name="baseline",
        args=_make_args(tmp_path),
        device=torch.device("cpu"),
        total_time_s=100.0,
        epochs_run=10,
        best_epoch=5,
        best_val_acc=0.95,
    )
    with open(tmp_path / "baseline_run_info.json") as f:
        data = json.load(f)
    assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# Path objects serialised as strings (the core bug fix)
# ---------------------------------------------------------------------------

def test_path_values_are_strings_in_json(tmp_path):
    """output_dir and config Path objects must become strings, not crash."""
    config_path = tmp_path / "configs" / "baseline.json"
    _write_run_info(
        output_dir=tmp_path,
        model_name="baseline",
        args=_make_args(tmp_path, config_path=config_path),
        device=torch.device("cpu"),
        total_time_s=100.0,
        epochs_run=10,
        best_epoch=5,
        best_val_acc=0.95,
    )
    with open(tmp_path / "baseline_run_info.json") as f:
        data = json.load(f)

    cfg = data["config"]
    assert isinstance(cfg["output_dir"], str), "output_dir must be a str in JSON"
    assert isinstance(cfg["config"], str),     "config path must be a str in JSON"


def test_path_values_round_trip_correctly(tmp_path):
    """The stringified paths should reconstruct to equivalent Path objects."""
    config_path = tmp_path / "configs" / "baseline.json"
    _write_run_info(
        output_dir=tmp_path,
        model_name="baseline",
        args=_make_args(tmp_path, config_path=config_path),
        device=torch.device("cpu"),
        total_time_s=100.0,
        epochs_run=10,
        best_epoch=5,
        best_val_acc=0.95,
    )
    with open(tmp_path / "baseline_run_info.json") as f:
        data = json.load(f)

    assert Path(data["config"]["output_dir"]) == tmp_path
    assert Path(data["config"]["config"]) == config_path


# ---------------------------------------------------------------------------
# None config path (when --config is not supplied)
# ---------------------------------------------------------------------------

def test_none_config_path_serialises_as_null(tmp_path):
    """When --config is not passed, args.config is None — must serialise to null."""
    _write_run_info(
        output_dir=tmp_path,
        model_name="baseline",
        args=_make_args(tmp_path, config_path=None),
        device=torch.device("cpu"),
        total_time_s=100.0,
        epochs_run=10,
        best_epoch=5,
        best_val_acc=0.95,
    )
    with open(tmp_path / "baseline_run_info.json") as f:
        data = json.load(f)
    assert data["config"]["config"] is None


# ---------------------------------------------------------------------------
# Top-level fields are present and have the expected types
# ---------------------------------------------------------------------------

def test_run_info_required_fields(tmp_path):
    _write_run_info(
        output_dir=tmp_path,
        model_name="baseline",
        args=_make_args(tmp_path),
        device=torch.device("cpu"),
        total_time_s=3600.0,
        epochs_run=29,
        best_epoch=24,
        best_val_acc=0.9972,
    )
    with open(tmp_path / "baseline_run_info.json") as f:
        data = json.load(f)

    assert data["model"] == "baseline"
    assert data["epochs_run"] == 29
    assert data["best_epoch"] == 24
    assert abs(data["best_val_acc"] - 0.9972) < 1e-9
    assert data["total_training_time_s"] == 3600.0
    assert "torch_version" in data
    assert "python_version" in data
    assert "platform" in data
