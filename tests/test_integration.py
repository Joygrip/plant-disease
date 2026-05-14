"""
Integration tests — require real model checkpoints.
Run with: pytest tests/test_integration.py -v -m slow
"""

import concurrent.futures
import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SPLITS_FILE = DATA_DIR / "splits.json"


def _real_checkpoint_exists(name: str) -> bool:
    return (MODELS_DIR / f"{name}_best.pt").exists()


def _find_dataset_image() -> Path | None:
    """Return the first .jpg from the val split, or any .jpg in data/."""
    if SPLITS_FILE.exists():
        with open(SPLITS_FILE, encoding="utf-8") as f:
            splits = json.load(f)
        for rel in splits.get("val", []):
            candidate = DATA_DIR / "New Plant Diseases Dataset(Augmented)" / rel
            if candidate.exists():
                return candidate
    # fallback: any jpg in data/
    for jpg in DATA_DIR.rglob("*.jpg"):
        return jpg
    return None


@pytest.fixture(scope="module")
def real_client():
    if not _real_checkpoint_exists("mobilenet_v2"):
        pytest.skip("Real mobilenet_v2_best.pt not found")

    import api.config as cfg
    original = cfg.settings.models_dir
    cfg.settings.models_dir = MODELS_DIR

    from fastapi.testclient import TestClient
    from api.main import app

    with TestClient(app) as c:
        yield c

    cfg.settings.models_dir = original


@pytest.fixture(scope="module")
def dataset_image_bytes():
    img_path = _find_dataset_image()
    if img_path is None:
        pytest.skip("No dataset image found in data/")
    return img_path.read_bytes(), img_path.name


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_predict_returns_200_with_real_model(real_client, dataset_image_bytes):
    img_bytes, img_name = dataset_image_bytes
    resp = real_client.post(
        "/predict",
        files={"image": (img_name, img_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200


@pytest.mark.slow
def test_predict_top1_matches_cli(real_client, dataset_image_bytes):
    """API top-1 class must match scripts/predict.py for the same image."""
    img_bytes, img_name = dataset_image_bytes
    img_path = _find_dataset_image()

    import torch
    import scripts.predict as pred_module  # type: ignore[import]

    cli_result = pred_module.predict(img_path, MODELS_DIR / "mobilenet_v2_best.pt", torch.device("cpu"), top_k=3)

    resp = real_client.post(
        "/predict?top_k=3",
        files={"image": (img_name, img_bytes, "image/jpeg")},
        data={"model": "mobilenet_v2"},
    )
    assert resp.status_code == 200
    api_result = resp.json()
    assert api_result["prediction"] == cli_result["class_name"]
    assert abs(api_result["confidence"] - cli_result["confidence"]) < 1e-4


@pytest.mark.slow
def test_predict_confidence_high_for_clean_image(real_client, dataset_image_bytes):
    img_bytes, img_name = dataset_image_bytes
    resp = real_client.post(
        "/predict",
        files={"image": (img_name, img_bytes, "image/jpeg")},
    )
    assert resp.json()["confidence"] > 0.90


@pytest.mark.slow
def test_predict_inference_ms_under_500(real_client, dataset_image_bytes):
    img_bytes, img_name = dataset_image_bytes
    resp = real_client.post(
        "/predict",
        files={"image": (img_name, img_bytes, "image/jpeg")},
    )
    assert resp.json()["inference_ms"] < 500


@pytest.mark.slow
def test_predict_baseline_model_returns_different_model_used(real_client, dataset_image_bytes):
    if not _real_checkpoint_exists("baseline"):
        pytest.skip("Real baseline_best.pt not found")
    img_bytes, img_name = dataset_image_bytes
    resp = real_client.post(
        "/predict",
        files={"image": (img_name, img_bytes, "image/jpeg")},
        data={"model": "baseline"},
    )
    assert resp.status_code == 200
    assert resp.json()["model"] == "baseline"


@pytest.mark.slow
def test_predict_concurrent_10_requests_all_succeed(real_client, dataset_image_bytes):
    """Smoke-test thread safety: 10 concurrent requests must all return 200."""
    img_bytes, img_name = dataset_image_bytes

    def send_request(_: int) -> int:
        resp = real_client.post(
            "/predict",
            files={"image": (img_name, img_bytes, "image/jpeg")},
        )
        return resp.status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        statuses = list(pool.map(send_request, range(10)))

    assert all(s == 200 for s in statuses), f"Some requests failed: {statuses}"
