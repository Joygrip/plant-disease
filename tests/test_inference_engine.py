"""Tests for src/plant_disease/inference_engine.py."""

import io
import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plant_disease import config
from plant_disease.inference_engine import (
    format_class_display,
    load_model_from_checkpoint,
    preprocess_image,
    run_inference,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """Random-weight baseline checkpoint with metadata."""
    from plant_disease.models import build_model

    tmp = tmp_path_factory.mktemp("ie_ckpt")
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
    return ckpt


@pytest.fixture(scope="module")
def checkpoint_no_meta(tmp_path_factory):
    """Random-weight baseline checkpoint without metadata sidecar."""
    from plant_disease.models import build_model

    tmp = tmp_path_factory.mktemp("ie_ckpt_nometa")
    model = build_model("mobilenet_v2", num_classes=config.NUM_CLASSES)
    ckpt = tmp / "mobilenet_v2_best.pt"
    torch.save(model.state_dict(), ckpt)
    return ckpt


@pytest.fixture(scope="module")
def leaf_image():
    """Synthetic 300×200 green RGB image."""
    return Image.new("RGB", (300, 200), color=(34, 139, 34))


@pytest.fixture(scope="module")
def model_bundle(checkpoint):
    return load_model_from_checkpoint(checkpoint)


# ---------------------------------------------------------------------------
# format_class_display
# ---------------------------------------------------------------------------

def test_format_class_display_replaces_triple_underscore():
    assert "___" not in format_class_display("Tomato___Late_blight")


def test_format_class_display_replaces_single_underscore():
    assert "_" not in format_class_display("Tomato___Late_blight")


def test_format_class_display_inserts_em_dash():
    result = format_class_display("Tomato___Late_blight")
    assert "—" in result


def test_format_class_display_example():
    assert format_class_display("Tomato___Late_blight") == "Tomato — Late blight"


def test_format_class_display_healthy():
    assert format_class_display("Apple___healthy") == "Apple — healthy"


def test_format_class_display_no_underscore_input():
    assert format_class_display("Apple") == "Apple"


# ---------------------------------------------------------------------------
# load_model_from_checkpoint
# ---------------------------------------------------------------------------

def test_load_returns_dict_with_required_keys(checkpoint):
    bundle = load_model_from_checkpoint(checkpoint)
    assert "model" in bundle
    assert "class_names" in bundle
    assert "model_type" in bundle


def test_load_model_is_nn_module(checkpoint):
    bundle = load_model_from_checkpoint(checkpoint)
    assert isinstance(bundle["model"], torch.nn.Module)


def test_load_class_names_is_list(checkpoint):
    bundle = load_model_from_checkpoint(checkpoint)
    assert isinstance(bundle["class_names"], list)
    assert len(bundle["class_names"]) == config.NUM_CLASSES


def test_load_model_type_is_string(checkpoint):
    bundle = load_model_from_checkpoint(checkpoint)
    assert isinstance(bundle["model_type"], str)
    assert len(bundle["model_type"]) > 0


def test_load_model_is_in_eval_mode(checkpoint):
    bundle = load_model_from_checkpoint(checkpoint)
    assert not bundle["model"].training


def test_load_model_type_from_meta(checkpoint):
    bundle = load_model_from_checkpoint(checkpoint)
    assert bundle["model_type"] == "baseline"


def test_load_model_type_inferred_without_meta(checkpoint_no_meta):
    bundle = load_model_from_checkpoint(checkpoint_no_meta)
    assert bundle["model_type"] == "mobilenet_v2"


def test_load_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model_from_checkpoint(tmp_path / "nonexistent.pt")


def test_load_device_cpu(checkpoint):
    bundle = load_model_from_checkpoint(checkpoint, device="cpu")
    param = next(bundle["model"].parameters())
    assert param.device.type == "cpu"


# ---------------------------------------------------------------------------
# preprocess_image
# ---------------------------------------------------------------------------

def test_preprocess_returns_tensor(leaf_image):
    t = preprocess_image(leaf_image)
    assert isinstance(t, torch.Tensor)


def test_preprocess_output_shape(leaf_image):
    t = preprocess_image(leaf_image)
    assert t.shape == (1, 3, 224, 224)


def test_preprocess_dtype_is_float(leaf_image):
    t = preprocess_image(leaf_image)
    assert t.dtype == torch.float32


# ---------------------------------------------------------------------------
# run_inference
# ---------------------------------------------------------------------------

def test_run_inference_returns_dict(model_bundle, leaf_image):
    result = run_inference(model_bundle, leaf_image)
    assert isinstance(result, dict)


def test_run_inference_required_keys(model_bundle, leaf_image):
    result = run_inference(model_bundle, leaf_image)
    for key in ("prediction", "prediction_display", "confidence", "top_k", "inference_ms"):
        assert key in result, f"Missing key: {key}"


def test_run_inference_prediction_is_known_class(model_bundle, leaf_image):
    result = run_inference(model_bundle, leaf_image)
    assert result["prediction"] in config.CLASS_NAMES


def test_run_inference_confidence_is_probability(model_bundle, leaf_image):
    result = run_inference(model_bundle, leaf_image)
    assert 0.0 <= result["confidence"] <= 1.0


def test_run_inference_display_has_no_triple_underscore(model_bundle, leaf_image):
    result = run_inference(model_bundle, leaf_image)
    assert "___" not in result["prediction_display"]


def test_run_inference_display_has_no_underscore(model_bundle, leaf_image):
    result = run_inference(model_bundle, leaf_image)
    assert "_" not in result["prediction_display"]


def test_run_inference_default_top_k_length(model_bundle, leaf_image):
    result = run_inference(model_bundle, leaf_image, top_k=3)
    assert len(result["top_k"]) == 3


def test_run_inference_top_k_entry_schema(model_bundle, leaf_image):
    result = run_inference(model_bundle, leaf_image)
    for entry in result["top_k"]:
        assert "class" in entry
        assert "display" in entry
        assert "probability" in entry


def test_run_inference_top_k_sorted_descending(model_bundle, leaf_image):
    result = run_inference(model_bundle, leaf_image, top_k=5)
    probs = [e["probability"] for e in result["top_k"]]
    assert probs == sorted(probs, reverse=True)


def test_run_inference_top1_matches_prediction(model_bundle, leaf_image):
    result = run_inference(model_bundle, leaf_image)
    assert result["top_k"][0]["class"] == result["prediction"]


def test_run_inference_inference_ms_positive(model_bundle, leaf_image):
    result = run_inference(model_bundle, leaf_image)
    assert result["inference_ms"] > 0


def test_run_inference_top_k_entry_display_no_underscore(model_bundle, leaf_image):
    result = run_inference(model_bundle, leaf_image)
    for entry in result["top_k"]:
        assert "_" not in entry["display"]
