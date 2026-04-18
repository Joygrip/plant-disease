"""
Tests for scripts/predict.py.

Uses a real random-init baseline model and a synthetic PIL image — no trained
checkpoint or dataset required.  All tests call predict() directly rather than
spawning a subprocess so they stay fast and produce helpful tracebacks.
"""

import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from plant_disease import config
from plant_disease.models import build_model
import predict as pred_module


# ---------------------------------------------------------------------------
# Session-scoped fixture: one checkpoint shared across all tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """Build a random-weight baseline model once and persist it for the session."""
    tmp = tmp_path_factory.mktemp("ckpt")
    model = build_model("baseline", num_classes=config.NUM_CLASSES)
    ckpt  = tmp / "baseline_best.pt"
    meta  = tmp / "baseline_best_meta.json"
    torch.save(model.state_dict(), ckpt)
    meta.write_text(json.dumps({
        "model": "baseline",
        "class_names": config.CLASS_NAMES,
        "epoch": 1,
        "val_acc": 0.5,
    }))
    return ckpt


@pytest.fixture(scope="module")
def leaf_image(tmp_path_factory):
    """A tiny synthetic leaf-coloured JPEG image."""
    tmp  = tmp_path_factory.mktemp("img")
    path = tmp / "leaf.jpg"
    Image.new("RGB", (300, 200), color=(34, 139, 34)).save(path)
    return path


# ---------------------------------------------------------------------------
# Return-value contract
# ---------------------------------------------------------------------------

def test_predict_returns_required_keys(checkpoint, leaf_image):
    result = pred_module.predict(leaf_image, checkpoint, torch.device("cpu"), top_k=3)
    assert {"class_name", "confidence", "top_k", "inference_ms"} <= result.keys()


def test_predict_class_name_is_known(checkpoint, leaf_image):
    result = pred_module.predict(leaf_image, checkpoint, torch.device("cpu"))
    assert result["class_name"] in config.CLASS_NAMES


def test_predict_confidence_is_probability(checkpoint, leaf_image):
    result = pred_module.predict(leaf_image, checkpoint, torch.device("cpu"))
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_inference_ms_is_positive(checkpoint, leaf_image):
    result = pred_module.predict(leaf_image, checkpoint, torch.device("cpu"))
    assert result["inference_ms"] > 0


def test_predict_top_k_length(checkpoint, leaf_image):
    for k in (1, 3, 5):
        result = pred_module.predict(leaf_image, checkpoint, torch.device("cpu"), top_k=k)
        assert len(result["top_k"]) == k


def test_predict_top_k_sorted_descending(checkpoint, leaf_image):
    result = pred_module.predict(leaf_image, checkpoint, torch.device("cpu"), top_k=5)
    confs = [e["confidence"] for e in result["top_k"]]
    assert confs == sorted(confs, reverse=True)


def test_predict_top1_matches_first_top_k_entry(checkpoint, leaf_image):
    result = pred_module.predict(leaf_image, checkpoint, torch.device("cpu"), top_k=3)
    assert result["class_name"]  == result["top_k"][0]["class"]
    assert result["confidence"]  == result["top_k"][0]["confidence"]


def test_predict_full_top_k_sums_to_one(checkpoint, leaf_image):
    """Requesting all 38 classes: softmax probabilities must sum to 1."""
    result = pred_module.predict(
        leaf_image, checkpoint, torch.device("cpu"), top_k=config.NUM_CLASSES,
    )
    total = sum(e["confidence"] for e in result["top_k"])
    assert abs(total - 1.0) < 1e-4


def test_predict_top_k_entries_have_class_and_confidence_keys(checkpoint, leaf_image):
    result = pred_module.predict(leaf_image, checkpoint, torch.device("cpu"), top_k=3)
    for entry in result["top_k"]:
        assert "class"      in entry
        assert "confidence" in entry


def test_predict_top_k_classes_are_distinct(checkpoint, leaf_image):
    result = pred_module.predict(leaf_image, checkpoint, torch.device("cpu"), top_k=5)
    names = [e["class"] for e in result["top_k"]]
    assert len(names) == len(set(names))


def test_predict_top_k_classes_are_known(checkpoint, leaf_image):
    result = pred_module.predict(leaf_image, checkpoint, torch.device("cpu"), top_k=5)
    for entry in result["top_k"]:
        assert entry["class"] in config.CLASS_NAMES


# ---------------------------------------------------------------------------
# Image format tolerance
# ---------------------------------------------------------------------------

def test_predict_accepts_png(checkpoint, tmp_path):
    png = tmp_path / "leaf.png"
    Image.new("RGB", (256, 256), color=(80, 140, 60)).save(png)
    result = pred_module.predict(png, checkpoint, torch.device("cpu"))
    assert result["class_name"] in config.CLASS_NAMES


def test_predict_accepts_non_square_image(checkpoint, tmp_path):
    img = tmp_path / "wide.jpg"
    Image.new("RGB", (640, 480), color=(60, 120, 40)).save(img)
    result = pred_module.predict(img, checkpoint, torch.device("cpu"))
    assert result["class_name"] in config.CLASS_NAMES


# ---------------------------------------------------------------------------
# Error cases — exit 1 with a useful message
# ---------------------------------------------------------------------------

def test_missing_image_exits_1(checkpoint, tmp_path):
    with pytest.raises(SystemExit) as exc:
        pred_module.predict(tmp_path / "ghost.jpg", checkpoint, torch.device("cpu"))
    assert exc.value.code == 1


def test_missing_image_prints_error_to_stderr(checkpoint, tmp_path, capsys):
    with pytest.raises(SystemExit):
        pred_module.predict(tmp_path / "ghost.jpg", checkpoint, torch.device("cpu"))
    assert "ERROR" in capsys.readouterr().err


def test_missing_image_error_includes_filename(checkpoint, tmp_path, capsys):
    with pytest.raises(SystemExit):
        pred_module.predict(tmp_path / "ghost.jpg", checkpoint, torch.device("cpu"))
    assert "ghost.jpg" in capsys.readouterr().err


def test_corrupt_image_exits_1(checkpoint, tmp_path):
    bad = tmp_path / "corrupt.jpg"
    bad.write_bytes(b"\xff\xd8\xff" + b"\x00" * 50)  # truncated JPEG header
    with pytest.raises(SystemExit) as exc:
        pred_module.predict(bad, checkpoint, torch.device("cpu"))
    assert exc.value.code == 1


def test_non_image_file_exits_1(checkpoint, tmp_path):
    txt = tmp_path / "readme.txt"
    txt.write_text("this is not an image")
    with pytest.raises(SystemExit) as exc:
        pred_module.predict(txt, checkpoint, torch.device("cpu"))
    assert exc.value.code == 1


def test_missing_checkpoint_exits_1(leaf_image, tmp_path):
    with pytest.raises(SystemExit) as exc:
        pred_module.predict(leaf_image, tmp_path / "missing.pt", torch.device("cpu"))
    assert exc.value.code == 1


def test_missing_checkpoint_prints_error_to_stderr(leaf_image, tmp_path, capsys):
    with pytest.raises(SystemExit):
        pred_module.predict(leaf_image, tmp_path / "missing.pt", torch.device("cpu"))
    assert "ERROR" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Checkpoint without metadata — falls back to filename heuristic
# ---------------------------------------------------------------------------

def test_predict_without_meta_file(leaf_image, tmp_path):
    model = build_model("baseline", num_classes=config.NUM_CLASSES)
    ckpt  = tmp_path / "baseline_best.pt"
    torch.save(model.state_dict(), ckpt)
    # no _meta.json written

    result = pred_module.predict(leaf_image, ckpt, torch.device("cpu"))
    assert result["class_name"] in config.CLASS_NAMES


def test_predict_without_meta_file_warns(leaf_image, tmp_path, capsys):
    model = build_model("baseline", num_classes=config.NUM_CLASSES)
    ckpt  = tmp_path / "baseline_best.pt"
    torch.save(model.state_dict(), ckpt)

    pred_module.predict(leaf_image, ckpt, torch.device("cpu"))
    assert "WARNING" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def test_print_result_human_contains_required_labels(checkpoint, leaf_image, capsys):
    result = pred_module.predict(leaf_image, checkpoint, torch.device("cpu"), top_k=3)
    pred_module._print_result(result, as_json=False)
    out = capsys.readouterr().out
    assert "Prediction" in out
    assert "Confidence" in out
    assert "Inference"  in out
    assert "%"          in out


def test_print_result_human_shows_correct_top_k_count(checkpoint, leaf_image, capsys):
    result = pred_module.predict(leaf_image, checkpoint, torch.device("cpu"), top_k=5)
    pred_module._print_result(result, as_json=False)
    out = capsys.readouterr().out
    # Each rank line starts with "  N."
    rank_lines = [l for l in out.splitlines() if l.strip() and l.strip()[0].isdigit()]
    assert len(rank_lines) == 5


def test_print_result_json_is_valid(checkpoint, leaf_image, capsys):
    result = pred_module.predict(leaf_image, checkpoint, torch.device("cpu"), top_k=3)
    pred_module._print_result(result, as_json=True)
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["class_name"] == result["class_name"]
    assert parsed["confidence"] == pytest.approx(result["confidence"])
    assert len(parsed["top_k"]) == 3


def test_print_result_json_top_k_schema(checkpoint, leaf_image, capsys):
    result = pred_module.predict(leaf_image, checkpoint, torch.device("cpu"), top_k=3)
    pred_module._print_result(result, as_json=True)
    parsed = json.loads(capsys.readouterr().out)
    for entry in parsed["top_k"]:
        assert "class"      in entry
        assert "confidence" in entry
        assert isinstance(entry["class"],      str)
        assert isinstance(entry["confidence"], float)


def test_print_result_prediction_line_is_parseable(checkpoint, leaf_image, capsys):
    """The 'Prediction : <class>' line must be machine-parseable by splitting on ':'."""
    result = pred_module.predict(leaf_image, checkpoint, torch.device("cpu"))
    pred_module._print_result(result, as_json=False)
    out = capsys.readouterr().out
    pred_line = next(l for l in out.splitlines() if l.startswith("Prediction"))
    parsed_class = pred_line.split(":", 1)[1].strip()
    assert parsed_class == result["class_name"]


def test_print_result_confidence_line_is_parseable(checkpoint, leaf_image, capsys):
    """The 'Confidence : XX.X%' line must round-trip through float parsing."""
    result = pred_module.predict(leaf_image, checkpoint, torch.device("cpu"))
    pred_module._print_result(result, as_json=False)
    out = capsys.readouterr().out
    conf_line = next(l for l in out.splitlines() if l.startswith("Confidence"))
    pct_str   = conf_line.split(":", 1)[1].strip().rstrip("%")
    parsed_pct = float(pct_str)
    assert abs(parsed_pct - result["confidence"] * 100) < 0.1
