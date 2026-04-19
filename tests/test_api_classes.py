"""Tests for GET /classes."""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from plant_disease import config
    from plant_disease.models import build_model

    tmp = tmp_path_factory.mktemp("models_classes")
    model = build_model("baseline", num_classes=config.NUM_CLASSES)
    ckpt = tmp / "baseline_best.pt"
    meta = tmp / "baseline_best_meta.json"
    torch.save(model.state_dict(), ckpt)
    meta.write_text(json.dumps({
        "model": "baseline",
        "class_names": config.CLASS_NAMES,
        "epoch": 1,
        "val_acc": 0.5,
    }))

    import api.config as cfg
    original = cfg.settings.models_dir
    cfg.settings.models_dir = tmp

    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c:
        yield c

    cfg.settings.models_dir = original


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_classes_returns_200(client):
    resp = client.get("/classes")
    assert resp.status_code == 200


def test_classes_has_38_entries(client):
    data = client.get("/classes").json()
    assert len(data["classes"]) == 38


def test_classes_has_raw_and_display_fields(client):
    data = client.get("/classes").json()
    for entry in data["classes"]:
        assert "raw" in entry
        assert "display" in entry


def test_classes_display_has_no_triple_underscore(client):
    data = client.get("/classes").json()
    for entry in data["classes"]:
        assert "___" not in entry["display"]


def test_classes_display_has_no_underscore(client):
    data = client.get("/classes").json()
    for entry in data["classes"]:
        assert "_" not in entry["display"]


def test_classes_raw_values_are_strings(client):
    data = client.get("/classes").json()
    for entry in data["classes"]:
        assert isinstance(entry["raw"], str)
        assert isinstance(entry["display"], str)
