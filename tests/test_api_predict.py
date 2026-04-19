"""Tests for POST /predict."""

import io
import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fake_models_dir(tmp_path_factory):
    """Create random-weight checkpoints for both models."""
    from plant_disease import config
    from plant_disease.models import build_model

    tmp = tmp_path_factory.mktemp("models_predict")
    for model_name in ("baseline", "mobilenet_v2"):
        model = build_model(model_name, num_classes=config.NUM_CLASSES)
        ckpt = tmp / f"{model_name}_best.pt"
        meta = tmp / f"{model_name}_best_meta.json"
        torch.save(model.state_dict(), ckpt)
        meta.write_text(json.dumps({
            "model": model_name,
            "class_names": config.CLASS_NAMES,
            "epoch": 1,
            "val_acc": 0.5,
        }))
    return tmp


@pytest.fixture(scope="module")
def client(fake_models_dir):
    import api.config as cfg
    original_dir = cfg.settings.models_dir
    original_default = cfg.settings.default_model
    cfg.settings.models_dir = fake_models_dir
    cfg.settings.default_model = "mobilenet_v2"

    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c:
        yield c

    cfg.settings.models_dir = original_dir
    cfg.settings.default_model = original_default


@pytest.fixture(scope="module")
def leaf_jpeg():
    """A minimal valid JPEG image as bytes."""
    buf = io.BytesIO()
    Image.new("RGB", (300, 200), color=(34, 139, 34)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def leaf_png():
    """A minimal valid PNG image as bytes."""
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), color=(80, 140, 60)).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Happy path — schema validation
# ---------------------------------------------------------------------------

def test_predict_valid_image_returns_200(client, leaf_jpeg):
    resp = client.post("/predict", files={"image": ("leaf.jpg", leaf_jpeg, "image/jpeg")})
    assert resp.status_code == 200


def test_predict_response_has_required_fields(client, leaf_jpeg):
    data = client.post("/predict", files={"image": ("leaf.jpg", leaf_jpeg, "image/jpeg")}).json()
    for field in ("prediction", "prediction_display", "confidence", "confidence_pct",
                  "top_k", "inference_ms", "model", "image_size"):
        assert field in data, f"Missing field: {field}"


def test_predict_confidence_is_in_range(client, leaf_jpeg):
    data = client.post("/predict", files={"image": ("leaf.jpg", leaf_jpeg, "image/jpeg")}).json()
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_confidence_pct_matches_confidence(client, leaf_jpeg):
    data = client.post("/predict", files={"image": ("leaf.jpg", leaf_jpeg, "image/jpeg")}).json()
    assert abs(data["confidence_pct"] - data["confidence"] * 100) < 0.2


def test_predict_prediction_display_has_no_triple_underscore(client, leaf_jpeg):
    data = client.post("/predict", files={"image": ("leaf.jpg", leaf_jpeg, "image/jpeg")}).json()
    assert "___" not in data["prediction_display"]


def test_predict_prediction_display_has_no_underscore(client, leaf_jpeg):
    data = client.post("/predict", files={"image": ("leaf.jpg", leaf_jpeg, "image/jpeg")}).json()
    assert "_" not in data["prediction_display"]


def test_predict_image_size_fields(client, leaf_jpeg):
    data = client.post("/predict", files={"image": ("leaf.jpg", leaf_jpeg, "image/jpeg")}).json()
    assert data["image_size"]["width"] == 300
    assert data["image_size"]["height"] == 200


def test_predict_top_k_entries_have_class_display_probability(client, leaf_jpeg):
    data = client.post("/predict", files={"image": ("leaf.jpg", leaf_jpeg, "image/jpeg")}).json()
    for entry in data["top_k"]:
        assert "class" in entry
        assert "display" in entry
        assert "probability" in entry


def test_predict_accepts_png(client, leaf_png):
    resp = client.post("/predict", files={"image": ("leaf.png", leaf_png, "image/png")})
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# top_k param
# ---------------------------------------------------------------------------

def test_predict_top_k_5_returns_5_results(client, leaf_jpeg):
    resp = client.post(
        "/predict?top_k=5",
        files={"image": ("leaf.jpg", leaf_jpeg, "image/jpeg")},
    )
    assert resp.status_code == 200
    assert len(resp.json()["top_k"]) == 5


def test_predict_top_k_large_is_capped_at_10(client, leaf_jpeg):
    resp = client.post(
        "/predict?top_k=50",
        files={"image": ("leaf.jpg", leaf_jpeg, "image/jpeg")},
    )
    assert resp.status_code == 200
    assert len(resp.json()["top_k"]) <= 10


def test_predict_top_k_sorted_descending(client, leaf_jpeg):
    data = client.post(
        "/predict?top_k=5",
        files={"image": ("leaf.jpg", leaf_jpeg, "image/jpeg")},
    ).json()
    probs = [e["probability"] for e in data["top_k"]]
    assert probs == sorted(probs, reverse=True)


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def test_predict_can_request_baseline_model(client, leaf_jpeg):
    resp = client.post(
        "/predict",
        files={"image": ("leaf.jpg", leaf_jpeg, "image/jpeg")},
        data={"model": "baseline"},
    )
    assert resp.status_code == 200
    assert resp.json()["model"] == "baseline"


def test_predict_unknown_model_returns_400(client, leaf_jpeg):
    resp = client.post(
        "/predict",
        files={"image": ("leaf.jpg", leaf_jpeg, "image/jpeg")},
        data={"model": "gpt4_vision"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "UNKNOWN_MODEL"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_predict_no_file_returns_422(client):
    resp = client.post("/predict")
    assert resp.status_code == 422


def test_predict_pdf_returns_415(client):
    pdf_bytes = b"%PDF-1.4 fake pdf content"
    resp = client.post(
        "/predict",
        files={"image": ("doc.pdf", pdf_bytes, "application/pdf")},
    )
    assert resp.status_code == 415
    assert resp.json()["error"]["code"] == "INVALID_CONTENT_TYPE"


def test_predict_large_file_returns_413(client):
    big_bytes = b"\xff" * (11 * 1024 * 1024)  # 11 MB > 10 MB limit
    resp = client.post(
        "/predict",
        files={"image": ("big.jpg", big_bytes, "image/jpeg")},
    )
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "FILE_TOO_LARGE"


def test_predict_too_small_image_returns_422(client):
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), color=(100, 100, 100)).save(buf, format="PNG")
    resp = client.post(
        "/predict",
        files={"image": ("tiny.png", buf.getvalue(), "image/png")},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "IMAGE_TOO_SMALL"


def test_predict_corrupt_bytes_returns_400(client):
    corrupt = b"\xff\xd8\xff" + b"\x00" * 50  # truncated JPEG header
    resp = client.post(
        "/predict",
        files={"image": ("corrupt.jpg", corrupt, "image/jpeg")},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "INVALID_IMAGE"


# ---------------------------------------------------------------------------
# Slow: compare with CLI predict() output on a real checkpoint
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_predict_matches_cli_output(tmp_path):
    """API and CLI return the same top-1 class for the same image + checkpoint."""
    from plant_disease import config

    models_dir = Path(__file__).resolve().parents[1] / "models"
    ckpt = models_dir / "mobilenet_v2_best.pt"
    if not ckpt.exists():
        pytest.skip("Real mobilenet_v2_best.pt checkpoint not found")

    # Find any dataset image
    data_dir = Path(__file__).resolve().parents[1] / "data"
    images = list(data_dir.rglob("*.jpg"))
    if not images:
        pytest.skip("No dataset images found in data/")
    img_path = images[0]

    # CLI result
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import predict as pred_module
    cli_result = pred_module.predict(img_path, ckpt, torch.device("cpu"), top_k=3)

    # API result
    import api.config as cfg
    original = cfg.settings.models_dir
    cfg.settings.models_dir = models_dir
    cfg.settings.default_model = "mobilenet_v2"

    try:
        from fastapi.testclient import TestClient
        from api.main import app
        with TestClient(app) as c:
            img_bytes = img_path.read_bytes()
            resp = c.post(
                "/predict?top_k=3",
                files={"image": (img_path.name, img_bytes, "image/jpeg")},
                data={"model": "mobilenet_v2"},
            )
    finally:
        cfg.settings.models_dir = original

    assert resp.status_code == 200
    api_result = resp.json()
    assert api_result["prediction"] == cli_result["class_name"]
    assert abs(api_result["confidence"] - cli_result["confidence"]) < 1e-4
