"""
Smoke tests for preflight.py — CPU only, no dataset required.

Verifies that the individual check functions run without crashing and
produce the expected pass/fail/warn results on known inputs.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

# Reset preflight accumulator state before each test
import preflight as pf


@pytest.fixture(autouse=True)
def reset_accumulators():
    """Clear _warnings and _failures between tests."""
    pf._warnings.clear()
    pf._failures.clear()
    yield
    pf._warnings.clear()
    pf._failures.clear()


# ---------------------------------------------------------------------------
# Normalization constant checks
# ---------------------------------------------------------------------------

def test_normalization_check_passes() -> None:
    pf.check_normalization_constants()
    assert len(pf._failures) == 0, f"Unexpected failures: {pf._failures}"


def test_normalization_check_produces_no_warnings() -> None:
    pf.check_normalization_constants()
    # Normalization is hardcoded correctly — should be zero warnings
    assert len(pf._warnings) == 0, f"Unexpected warnings: {pf._warnings}"


# ---------------------------------------------------------------------------
# Model forward pass (CPU, random tensors — no dataset needed)
# ---------------------------------------------------------------------------

def test_model_forward_baseline_passes() -> None:
    models = pf.check_model_forward(batch_size=4, first_batches=None)
    assert "baseline" in models
    assert len(pf._failures) == 0


def test_model_forward_mobilenet_passes() -> None:
    models = pf.check_model_forward(batch_size=4, first_batches=None)
    assert "mobilenet_v2" in models
    assert len(pf._failures) == 0


def test_model_forward_output_shape() -> None:
    models = pf.check_model_forward(batch_size=4, first_batches=None)
    for name, model in models.items():
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 38), f"{name} output shape wrong: {out.shape}"


# ---------------------------------------------------------------------------
# Gradient flow check
# ---------------------------------------------------------------------------

def test_gradient_check_passes_on_frozen_backbone() -> None:
    models = pf.check_model_forward(batch_size=4, first_batches=None)
    pf._failures.clear()  # reset failures from forward check
    pf.check_gradient_flow(models)
    assert len(pf._failures) == 0, f"Gradient check failures: {pf._failures}"


def test_gradient_check_skips_gracefully_without_mobilenet() -> None:
    pf.check_gradient_flow({})  # empty — mobilenet_v2 absent
    assert any("MobileNetV2 not built" in w for w in pf._warnings)


# ---------------------------------------------------------------------------
# Data loading (skips gracefully when dataset absent)
# ---------------------------------------------------------------------------

def test_data_loading_warns_when_dataset_absent() -> None:
    # With no real data, should warn (not fail)
    from plant_disease import config
    if not config.TRAIN_DIR.exists():
        result = pf.check_data_loading(batch_size=4, num_workers=0)
        assert result is None
        assert len(pf._warnings) > 0
        assert len(pf._failures) == 0


# ---------------------------------------------------------------------------
# GPU memory check (skips cleanly on CPU-only machine)
# ---------------------------------------------------------------------------

def test_gpu_memory_check_skips_on_cpu() -> None:
    if not torch.cuda.is_available():
        pf.check_gpu_memory(batch_size=4)
        assert any("No CUDA device" in w for w in pf._warnings)
        assert len(pf._failures) == 0


# ---------------------------------------------------------------------------
# CUDA check behaviour
# ---------------------------------------------------------------------------

def test_cuda_check_with_allow_cpu_never_fails() -> None:
    """--allow-cpu must downgrade missing CUDA to a warning, never a failure."""
    if not torch.cuda.is_available():
        pf.check_cuda(allow_cpu=True)
        assert len(pf._failures) == 0
        assert any("CPU" in w or "CUDA" in w for w in pf._warnings)


def test_cuda_check_without_allow_cpu_fails_when_no_cuda() -> None:
    if not torch.cuda.is_available():
        pf.check_cuda(allow_cpu=False)
        assert len(pf._failures) > 0


def test_cuda_check_passes_when_cuda_available() -> None:
    if torch.cuda.is_available():
        pf.check_cuda(allow_cpu=False)
        assert len(pf._failures) == 0


# ---------------------------------------------------------------------------
# _warn / _fail / _pass accumulator behaviour
# ---------------------------------------------------------------------------

def test_warn_accumulates() -> None:
    pf._warn("test warning one")
    pf._warn("test warning two")
    assert len(pf._warnings) == 2


def test_fail_accumulates() -> None:
    pf._fail("test failure one")
    pf._fail("test failure two")
    assert len(pf._failures) == 2


def test_warn_and_fail_are_independent() -> None:
    pf._warn("w")
    pf._fail("f")
    assert len(pf._warnings) == 1
    assert len(pf._failures) == 1
