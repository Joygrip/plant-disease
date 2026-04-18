"""
Smoke tests for MobileNetV2Classifier.

No training, no real data — pure construction and parameter-count checks.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plant_disease.models.mobilenet_v2 import MobileNetV2Classifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trainable(model: nn.Module) -> set[str]:
    return {name for name, p in model.named_parameters() if p.requires_grad}


def _total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_construction_default_classes() -> None:
    model = MobileNetV2Classifier(num_classes=38)
    assert model is not None


def test_forward_shape() -> None:
    model = MobileNetV2Classifier(num_classes=38).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 38)


def test_total_param_count_reasonable() -> None:
    model = MobileNetV2Classifier(num_classes=38)
    total = _total_params(model)
    # MobileNetV2 features ~2.22 M + new head (Linear 1280→38) ~49 k ≈ 2.27 M
    assert 2_000_000 < total < 3_000_000, f"Unexpected total param count: {total:,}"


def test_head_architecture() -> None:
    model = MobileNetV2Classifier(num_classes=38)
    # Classifier: Dropout → Linear(1280, 38)
    assert isinstance(model.classifier[0], nn.Dropout)
    assert isinstance(model.classifier[1], nn.Linear)
    assert model.classifier[1].in_features == 1280
    assert model.classifier[1].out_features == 38


# ---------------------------------------------------------------------------
# freeze_backbone
# ---------------------------------------------------------------------------

def test_freeze_backbone_only_head_trainable() -> None:
    model = MobileNetV2Classifier(num_classes=38)
    model.freeze_backbone()
    # All frozen params must be in features
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert name.startswith("classifier."), (
                f"Expected only classifier params trainable, got: {name}"
            )


def test_freeze_backbone_param_count() -> None:
    model = MobileNetV2Classifier(num_classes=38)
    model.freeze_backbone()
    trainable = _trainable_params(model)
    # Head: Dropout (no params) + Linear(1280, 38) = 1280*38 + 38 = 48,678
    assert trainable == 1280 * 38 + 38, f"Unexpected trainable count: {trainable}"


def test_freeze_backbone_clears_bn_keep_eval() -> None:
    model = MobileNetV2Classifier(num_classes=38)
    model.unfreeze_top_blocks(3)
    assert len(model._bn_keep_eval) > 0
    model.freeze_backbone()
    assert len(model._bn_keep_eval) == 0


# ---------------------------------------------------------------------------
# unfreeze_top_blocks
# ---------------------------------------------------------------------------

def test_unfreeze_top_blocks_3_increases_trainable() -> None:
    model = MobileNetV2Classifier(num_classes=38)
    model.freeze_backbone()
    frozen_trainable = _trainable_params(model)

    model.unfreeze_top_blocks(3)
    unfrozen_trainable = _trainable_params(model)

    assert unfrozen_trainable > frozen_trainable


def test_unfreeze_top_blocks_3_adds_bn_keep_eval() -> None:
    model = MobileNetV2Classifier(num_classes=38)
    model.freeze_backbone()
    model.unfreeze_top_blocks(3)
    # There must be BN modules in the unfrozen layers
    assert len(model._bn_keep_eval) > 0
    for m in model._bn_keep_eval:
        assert isinstance(m, nn.BatchNorm2d)


def test_unfreeze_top_blocks_bn_stays_eval_in_train_mode() -> None:
    model = MobileNetV2Classifier(num_classes=38)
    model.freeze_backbone()
    model.unfreeze_top_blocks(3)
    model.train()  # triggers the override
    for m in model._bn_keep_eval:
        assert not m.training, "BN in unfrozen block should stay in eval mode during training"


def test_unfreeze_top_blocks_correct_feature_layers() -> None:
    """Last 3 feature layers (indices 16, 17, 18) must have requires_grad=True."""
    model = MobileNetV2Classifier(num_classes=38)
    model.freeze_backbone()
    model.unfreeze_top_blocks(3)

    feature_layers = list(model.features.children())
    # Last 3 should be unfrozen
    for layer in feature_layers[-3:]:
        for p in layer.parameters():
            assert p.requires_grad, "Expected last 3 feature layers to be unfrozen"
    # First 16 should remain frozen
    for layer in feature_layers[:-3]:
        for p in layer.parameters():
            assert not p.requires_grad, "Expected non-top feature layers to remain frozen"


def test_unfreeze_different_n() -> None:
    """unfreeze_top_blocks(1) should give fewer trainable params than (3)."""
    m1 = MobileNetV2Classifier(num_classes=38)
    m1.freeze_backbone()
    m1.unfreeze_top_blocks(1)

    m3 = MobileNetV2Classifier(num_classes=38)
    m3.freeze_backbone()
    m3.unfreeze_top_blocks(3)

    assert _trainable_params(m1) < _trainable_params(m3)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_build_model_factory_mobilenet() -> None:
    from plant_disease.models import build_model
    model = build_model("mobilenet_v2", num_classes=38)
    assert isinstance(model, MobileNetV2Classifier)


def test_build_model_factory_baseline() -> None:
    from plant_disease.models import build_model
    from plant_disease.models.baseline_cnn import BaselineCNN
    model = build_model("baseline", num_classes=38)
    assert isinstance(model, BaselineCNN)


def test_build_model_factory_unknown_raises() -> None:
    from plant_disease.models import build_model
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("resnet50")
