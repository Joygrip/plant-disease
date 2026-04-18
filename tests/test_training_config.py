"""
Smoke tests for JSON config files and the training arg parser.
No training, no dataset required.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"
REQUIRED_KEYS = {
    "model", "epochs", "batch_size", "lr", "weight_decay",
    "finetune_schedule", "stage1_epochs", "stage2_lr",
    "unfreeze_blocks", "seed",
}


# ---------------------------------------------------------------------------
# Config file structure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("config_file", ["baseline.json", "mobilenet_v2.json"])
def test_config_file_exists(config_file: str) -> None:
    assert (CONFIGS_DIR / config_file).exists(), f"{config_file} not found in configs/"


@pytest.mark.parametrize("config_file", ["baseline.json", "mobilenet_v2.json"])
def test_config_file_parses(config_file: str) -> None:
    path = CONFIGS_DIR / config_file
    with open(path) as f:
        cfg = json.load(f)
    assert isinstance(cfg, dict)


@pytest.mark.parametrize("config_file", ["baseline.json", "mobilenet_v2.json"])
def test_config_has_required_keys(config_file: str) -> None:
    path = CONFIGS_DIR / config_file
    with open(path) as f:
        cfg = json.load(f)
    missing = REQUIRED_KEYS - set(cfg.keys())
    assert not missing, f"{config_file} missing keys: {missing}"


def test_baseline_config_values() -> None:
    with open(CONFIGS_DIR / "baseline.json") as f:
        cfg = json.load(f)
    assert cfg["model"] == "baseline"
    assert cfg["finetune_schedule"] == "single-stage"
    assert cfg["epochs"] == 30
    assert cfg["seed"] == 42


def test_mobilenet_config_values() -> None:
    with open(CONFIGS_DIR / "mobilenet_v2.json") as f:
        cfg = json.load(f)
    assert cfg["model"] == "mobilenet_v2"
    assert cfg["finetune_schedule"] == "two-stage"
    assert cfg["stage1_epochs"] == 5
    assert cfg["epochs"] == 30
    assert cfg["stage1_epochs"] < cfg["epochs"], "stage1_epochs must be < total epochs"


def test_mobilenet_stage2_lr_lower_than_stage1() -> None:
    with open(CONFIGS_DIR / "mobilenet_v2.json") as f:
        cfg = json.load(f)
    assert cfg["stage2_lr"] < cfg["lr"], "stage2_lr should be lower than lr"


# ---------------------------------------------------------------------------
# Arg parser accepts configs (without actual training)
# ---------------------------------------------------------------------------

def test_parse_args_with_baseline_config(monkeypatch) -> None:
    import importlib
    monkeypatch.setattr(
        "sys.argv",
        ["train", "--config", str(CONFIGS_DIR / "baseline.json")]
    )
    # Re-import to get a clean parse
    import plant_disease.train as train_mod
    importlib.reload(train_mod)
    args = train_mod.parse_args()
    assert args.model == "baseline"
    assert args.finetune_schedule == "single-stage"
    assert args.epochs == 30


def test_parse_args_with_mobilenet_config(monkeypatch) -> None:
    import importlib
    monkeypatch.setattr(
        "sys.argv",
        ["train", "--config", str(CONFIGS_DIR / "mobilenet_v2.json")]
    )
    import plant_disease.train as train_mod
    importlib.reload(train_mod)
    args = train_mod.parse_args()
    assert args.model == "mobilenet_v2"
    assert args.finetune_schedule == "two-stage"
    assert args.stage1_epochs == 5
    assert args.stage2_lr == pytest.approx(1e-4)


def test_cli_flag_overrides_config(monkeypatch) -> None:
    """--epochs on CLI must override the value in the config file."""
    import importlib
    monkeypatch.setattr(
        "sys.argv",
        ["train", "--config", str(CONFIGS_DIR / "baseline.json"), "--epochs", "10"]
    )
    import plant_disease.train as train_mod
    importlib.reload(train_mod)
    args = train_mod.parse_args()
    assert args.epochs == 10


# ---------------------------------------------------------------------------
# build_model accepts both config model names
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_name", ["baseline", "mobilenet_v2"])
def test_build_model_accepts_config_model_name(model_name: str) -> None:
    from plant_disease.models import build_model
    model = build_model(model_name, num_classes=38)
    assert model is not None
