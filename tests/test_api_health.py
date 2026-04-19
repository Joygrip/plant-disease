"""Tests for GET /health."""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---------------------------------------------------------------------------
# Shared fixture: fake checkpoints + patched settings + TestClient
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from plant_disease import config
    from plant_disease.models import build_model

    tmp = tmp_path_factory.mktemp("models_health")
    model = build_model("mobilenet_v2", num_classes=config.NUM_CLASSES)
    ckpt = tmp / "mobilenet_v2_best.pt"
    meta = tmp / "mobilenet_v2_best_meta.json"
    torch.save(model.state_dict(), ckpt)
    meta.write_text(json.dumps({
        "model": "mobilenet_v2",
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

def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_has_all_expected_fields(client):
    data = client.get("/health").json()
    assert "status" in data
    assert "model" in data
    assert "checkpoint_age" in data
    assert "torch_version" in data
    assert "cuda_available" in data


def test_health_status_is_ok(client):
    data = client.get("/health").json()
    assert data["status"] == "ok"


def test_health_cuda_available_is_bool(client):
    data = client.get("/health").json()
    assert isinstance(data["cuda_available"], bool)


def test_health_torch_version_is_string(client):
    data = client.get("/health").json()
    assert isinstance(data["torch_version"], str)
    assert len(data["torch_version"]) > 0
