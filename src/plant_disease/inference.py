"""Shared inference utilities — used by both scripts/predict.py and api/."""

import json
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from PIL import Image

from plant_disease import config
from plant_disease.data import _eval_transform
from plant_disease.models import build_model


def load_checkpoint(
    checkpoint: Path,
    device: torch.device,
    warn_callback: Callable[[str], None] | None = None,
) -> tuple[nn.Module, list[str]]:
    """
    Load (model, class_names) from a .pt file and its _meta.json sidecar.

    Raises FileNotFoundError if checkpoint is missing.
    Calls warn_callback(message) when metadata is absent and model type is inferred.
    """
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    meta_path = checkpoint.with_name(checkpoint.stem + "_meta.json")
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        model_name = meta["model"]
        class_names = meta.get("class_names", config.CLASS_NAMES)
    else:
        model_name = "baseline" if "baseline" in checkpoint.stem else "mobilenet_v2"
        class_names = config.CLASS_NAMES
        if warn_callback is not None:
            warn_callback(
                f"no metadata at {meta_path} — inferring model type as '{model_name}'"
            )

    model = build_model(model_name, num_classes=len(class_names))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model, class_names


def run_inference(
    pil_image: Image.Image,
    model: nn.Module,
    class_names: list[str],
    top_k: int = 3,
) -> dict:
    """
    Run inference on a PIL image using an already-loaded model.

    Returns::
        {
            "class_name":   str,    # top-1 predicted class
            "confidence":   float,  # 0-1 softmax probability
            "top_k": [{"class": str, "confidence": float}, ...],
            "inference_ms": float,
        }
    """
    device = next(model.parameters()).device
    tf = _eval_transform()
    tensor = tf(pil_image).unsqueeze(0).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
    inference_ms = (time.perf_counter() - t0) * 1000.0

    probs = torch.softmax(logits, dim=1)[0]
    k = min(top_k, len(class_names))
    top_probs, top_indices = probs.topk(k)

    top_list = [
        {"class": class_names[idx.item()], "confidence": prob.item()}
        for prob, idx in zip(top_probs, top_indices)
    ]

    return {
        "class_name": top_list[0]["class"],
        "confidence": top_list[0]["confidence"],
        "top_k": top_list,
        "inference_ms": round(inference_ms, 2),
    }
