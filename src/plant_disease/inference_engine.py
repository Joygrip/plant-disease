"""
High-level inference API — wraps plant_disease.inference for use by the FastAPI layer.

Provides stable, richly-typed entry points that the API and integration tests
can depend on without coupling directly to the lower-level inference primitives.
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from plant_disease.data import _eval_transform
from plant_disease.inference import load_checkpoint as _load_checkpoint
from plant_disease.inference import run_inference as _run_inference


def format_class_display(raw_class: str) -> str:
    """'Tomato___Late_blight' → 'Tomato — Late blight'"""
    return raw_class.replace("___", " — ").replace("_", " ")


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: str = "cpu",
) -> dict:
    """
    Load a checkpoint and return a model bundle.

    Returns:
        {
            "model": nn.Module (eval mode, on device),
            "class_names": list[str],
            "model_type": str,  # e.g. "mobilenet_v2" or "baseline"
        }

    Raises FileNotFoundError if the checkpoint is missing.
    """
    import json

    torch_device = torch.device(device)
    model, class_names = _load_checkpoint(checkpoint_path, torch_device)

    meta_path = checkpoint_path.with_name(checkpoint_path.stem + "_meta.json")
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        model_type = meta.get("model", checkpoint_path.stem)
    else:
        model_type = "baseline" if "baseline" in checkpoint_path.stem else "mobilenet_v2"

    return {"model": model, "class_names": class_names, "model_type": model_type}


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Apply resize + ImageNet normalisation to a PIL image.

    Returns a (1, 3, 224, 224) float tensor ready for model forward pass.
    """
    tf = _eval_transform()
    return tf(image).unsqueeze(0)


def run_inference(
    model_bundle: dict,
    image: Image.Image,
    top_k: int = 3,
) -> dict:
    """
    Run inference on a PIL image using a model bundle from load_model_from_checkpoint.

    Returns:
        {
            "prediction":         str,    # raw top-1 class name
            "prediction_display": str,    # human-readable (— instead of ___)
            "confidence":         float,  # top-1 softmax probability 0–1
            "top_k": [
                {"class": str, "display": str, "probability": float},
                ...
            ],
            "inference_ms":       float,
        }
    """
    model: nn.Module = model_bundle["model"]
    class_names: list[str] = model_bundle["class_names"]

    raw = _run_inference(image, model, class_names, top_k)

    top_k_out = [
        {
            "class": entry["class"],
            "display": format_class_display(entry["class"]),
            "probability": entry["confidence"],
        }
        for entry in raw["top_k"]
    ]

    return {
        "prediction": raw["class_name"],
        "prediction_display": format_class_display(raw["class_name"]),
        "confidence": raw["confidence"],
        "top_k": top_k_out,
        "inference_ms": raw["inference_ms"],
    }
